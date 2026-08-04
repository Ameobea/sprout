"""Temporal (future-watch) guardrail for the hybrid candidates, prod-mix scoring.

Same frozen temporal_v3 fixtures and metric functions as eval/temporal_eval.py, with:
- serve-prior support (lift-trained NN + grafts scored as logits + alpha_add*log_pop)
- scorers: nn (fresh-logq), gate/concat grafts, blend policy, ease
- the prod combined score swept over logit_weight (softmax(presence_score)^lw *
  max(rating+1,.001)^(1-lw)); ratings always from the NN-family head of the row
  (EASE/blend rows use the control NN's ratings)
- a franchise-filtered variant (same DSU components as frontier_eval) reported
  alongside, plus the franchise share of temporal targets themselves

Run inside rocm_jax: cd /jax_dir/notebooks && JAX_PLATFORMS=cpu \
  python analysis/temporal_hybrid_eval.py --out ../data/aug2026/temporal_hybrid.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "eval")

import jax.numpy as jnp

from model import CONF, make_dense_profile
from eval_harness import build_aux, load_params, preprocess
from temporal_eval import POP_TIERS, RECALL_KS, target_ranks, tier_aggregate, combined_score
from analysis.frontier_eval import build_components
from analysis.probe_value_eval import load_graft, load_prod

FIXTURES = Path("eval/fixtures/temporal_v3.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nn-weights", default="../data/aug2026/jax_model_fresh_logq.msgpack")
    ap.add_argument("--gate", default="../data/aug2026/probe/probe_graft_gate.msgpack")
    ap.add_argument("--concat", default="../data/aug2026/probe/probe_graft_concat.msgpack")
    ap.add_argument("--ease-b", default="../data/aug2026/ease_B6k_lam200.npy")
    ap.add_argument("--vectors", default="../data/aug2026/user_input_vectors_cleanup_notrust.npz")
    ap.add_argument("--corpus", default="../data/corpus_ids_aug2026.json")
    ap.add_argument("--metadata", default="../data/processed-metadata_aug2026.csv")
    ap.add_argument("--popularity", default="../data/item_popularity_dec2025.npy")
    ap.add_argument("--logit-weights", default="0.0,0.3,0.7,1.0")
    ap.add_argument("--blend-w", type=float, default=0.35)
    ap.add_argument("--blend-alpha", type=float, default=0.45)
    ap.add_argument("--blend-minctx", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    lws = [float(w) for w in args.logit_weights.split(",")]
    cs = CONF["corpus_size"]

    with open(args.corpus) as f:
        corpus_ids = json.load(f)
    id_to_idx = {aid: i for i, aid in enumerate(corpus_ids)}

    d = np.load(args.vectors)
    indices = d["indices"].astype(np.int32)
    lengths = d["lengths"].astype(np.int64)
    starts = np.zeros(len(lengths), dtype=np.int64)
    np.cumsum(lengths[:-1], out=starts[1:])
    counts = np.bincount(indices, minlength=cs).astype(np.float64)
    log_pop = np.log(np.maximum(counts, 1.0) / np.maximum(counts, 1.0).sum())
    zlp = (log_pop - log_pop.mean()) / log_pop.std()

    tier_counts = np.load(args.popularity).astype(np.float64)
    pop_rank = np.argsort(np.argsort(-tier_counts))
    tier_of = np.zeros(cs, dtype=np.int32)
    for ti, (lo, hi, _) in enumerate(POP_TIERS):
        tier_of[(pop_rank >= lo) & (pop_rank < hi)] = ti

    dsu = build_components(args.metadata)
    comp_of_corpus = np.array([dsu.find(int(a)) for a in np.array(corpus_ids, dtype=np.int64)],
                              dtype=np.int64)

    B = np.load(args.ease_b)
    rng_ref = np.random.default_rng(555)
    rng_h = np.random.default_rng(999)
    perm = rng_h.permutation(len(lengths))
    train_pool = perm[len(lengths) // 10:]
    ref_users = rng_ref.choice(train_pool, size=20000, replace=False)
    s_sum = np.zeros(cs)
    for u in ref_users:
        s0, l = starts[u], lengths[u]
        s_sum += B[indices[s0 : s0 + l]].sum(axis=0)
    mu = s_sum / len(ref_users)
    print("mu ready", flush=True)

    nn_params, nn_fwd = load_prod(args.nn_weights, 512, cs)
    gate_params, gate_fwd = load_graft(args.gate, "gate", 512, cs, B)
    concat_params, concat_fwd = load_graft(args.concat, "concat", 512, cs, B)

    with open(FIXTURES) as f:
        fixtures = json.load(f)

    SCORERS = ["nn", "gate", "concat", "blend", "ease"]
    agg = {s: {w: {"rows": [], "tiers": [[] for _ in POP_TIERS],
                   "rec10": [], "ftiers": [[] for _ in POP_TIERS], "frows": []}
               for w in lws} for s in SCORERS}
    tgt_franch_share = []

    for i, (username, p) in enumerate(sorted(fixtures.items())):
        idxs, vals, original, statuses = preprocess(p["items"], id_to_idx)
        target_idxs = np.array(sorted({id_to_idx[a] for a, _ in p["targets"] if a in id_to_idx}))
        target_idxs = target_idxs[~np.isin(target_idxs, idxs)]
        if len(idxs) < 5 or len(target_idxs) < 3:
            continue
        x = make_dense_profile(idxs, vals, build_aux(original, statuses))
        xj = jnp.asarray(x)
        nn_lg = np.array(nn_fwd(nn_params, xj))[0].astype(np.float64)
        gate_lg = np.array(gate_fwd(gate_params, xj))[0].astype(np.float64)
        concat_lg = np.array(concat_fwd(concat_params, xj))[0].astype(np.float64)
        from model import infer_outputs
        _, ratings = infer_outputs(nn_params, x)
        ratings = np.array(ratings)[0].astype(np.float64)

        s_ease = B[idxs].sum(axis=0).astype(np.float64)
        ze = s_ease - mu
        ze = (ze - ze.mean()) / (ze.std() + 1e-9)
        znn = (nn_lg - nn_lg.mean()) / (nn_lg.std() + 1e-9)
        w_eff = args.blend_w if len(idxs) >= args.blend_minctx else 0.0
        blend_sc = (1 - w_eff) * znn + w_eff * ze + args.blend_alpha * zlp

        presence = {
            "nn": nn_lg + log_pop,
            "gate": gate_lg + log_pop,
            "concat": concat_lg + log_pop,
            "blend": blend_sc,
            "ease": s_ease + 0.2167 * log_pop,
        }
        input_set = set(int(v) for v in idxs)
        cand_franch = np.isin(comp_of_corpus, np.unique(comp_of_corpus[idxs]))
        tgt_f = cand_franch[target_idxs]
        tgt_franch_share.append(float(tgt_f.mean()))

        for sname in SCORERS:
            for w in lws:
                sc = combined_score(presence[sname], ratings, w)
                ranks, cand_scores = target_ranks(sc, input_set, target_idxs, cs)
                a = agg[sname][w]
                a["rows"].append({"n_targets": len(target_idxs),
                                  "median_target_rank": float(np.median(ranks)),
                                  "mrr": float(1.0 / ranks.min()),
                                  **{f"recall@{k}": float(np.mean(ranks <= k)) for k in RECALL_KS}})
                for t, rk in zip(target_idxs, ranks):
                    a["tiers"][tier_of[t]].append(rk)
                top10 = np.argsort(cand_scores)[-10:]
                a["rec10"].append(float(np.mean(pop_rank[top10])))
                scf = np.where(cand_franch, -np.inf, sc)
                if (~tgt_f).any():
                    franks, _ = target_ranks(scf, input_set, target_idxs[~tgt_f], cs)
                    a["frows"].append({f"recall@{k}": float(np.mean(franks <= k)) for k in RECALL_KS})
                    for t, rk in zip(target_idxs[~tgt_f], franks):
                        a["ftiers"][tier_of[t]].append(rk)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(fixtures)}", flush=True)

    def agg_rows(rows, keys):
        return {k: float(np.mean([r[k] for r in rows])) for k in keys} | {"n_profiles": len(rows)}

    report = {"target_franchise_share_mean": float(np.mean(tgt_franch_share)),
              "blend_cfg": {"w": args.blend_w, "alpha_z": args.blend_alpha, "minctx": args.blend_minctx},
              "note": "nn/gate/concat at alpha_add=1.0; ease at beta=0.217; ratings from fresh-logq head everywhere",
              "scorers": {}}
    for sname in SCORERS:
        report["scorers"][sname] = {}
        for w in lws:
            a = agg[sname][w]
            report["scorers"][sname][str(w)] = {
                "overall": agg_rows(a["rows"], ["median_target_rank", "mrr"] + [f"recall@{k}" for k in RECALL_KS]),
                "by_popularity_tier": tier_aggregate(a["tiers"]),
                "rec_pop_top10": float(np.mean(a["rec10"])),
                "filtered_overall": agg_rows(a["frows"], [f"recall@{k}" for k in RECALL_KS]),
                "filtered_by_tier": tier_aggregate(a["ftiers"]),
            }
            o = report["scorers"][sname][str(w)]["overall"]
            print(f"{sname} lw={w}: r@10 {o['recall@10']:.3f} r@50 {o['recall@50']:.3f} "
                  f"med {o['median_target_rank']:.0f} | rec10pop {report['scorers'][sname][str(w)]['rec_pop_top10']:.0f}",
                  flush=True)

    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
