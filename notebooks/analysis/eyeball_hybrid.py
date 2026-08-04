"""Sentinel top-15s for the hybrid candidates, franchise-flagged.

Scorers: nn (fresh-logq, alpha_add=1.0 and z-alpha 0.45), gate/concat grafts
(alpha_add=1.0 and z-alpha 0.45), blend policy (w=0.35 unless <64 items, z-alpha
0.45), ease-lift (z-alpha 0.45). Franchise flags via the frontier DSU components.

Run inside rocm_jax: cd /jax_dir/notebooks && JAX_PLATFORMS=cpu \
  python analysis/eyeball_hybrid.py --out ../data/aug2026/eyeball_hybrid.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
import jax.numpy as jnp

from model import CONF
from analysis.frontier_eval import build_components
from analysis.probe_value_eval import load_graft, load_prod

SEASON_RELS = {"sequel", "prequel", "parent_story", "side_story"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nn-weights", default="../data/aug2026/jax_model_fresh_logq.msgpack")
    ap.add_argument("--gate", default="../data/aug2026/probe/probe_graft_gate.msgpack")
    ap.add_argument("--concat", default="../data/aug2026/probe/probe_graft_concat.msgpack")
    ap.add_argument("--ease-b", default="../data/aug2026/ease_B6k_lam200.npy")
    ap.add_argument("--vectors", default="../data/aug2026/user_input_vectors_cleanup_notrust.npz")
    ap.add_argument("--corpus", default="../data/corpus_ids_aug2026.json")
    ap.add_argument("--metadata", default="../data/processed-metadata_aug2026.csv")
    ap.add_argument("--fixtures", default="eval/fixtures")
    ap.add_argument("--profiles", default="ameo___,snapsauce")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cs = CONF["corpus_size"]

    d = np.load(args.vectors)
    indices = d["indices"].astype(np.int32)
    lengths = d["lengths"].astype(np.int64)
    starts = np.zeros(len(lengths), dtype=np.int64)
    np.cumsum(lengths[:-1], out=starts[1:])
    counts = np.bincount(indices, minlength=cs).astype(np.float64)
    log_pop = np.log(np.maximum(counts, 1.0) / np.maximum(counts, 1.0).sum())
    zlp = (log_pop - log_pop.mean()) / log_pop.std()
    pop_rank = np.argsort(np.argsort(-counts))

    with open(args.corpus) as f:
        corpus_ids = np.array(json.load(f), dtype=np.int64)
    id_to_idx = {int(a): i for i, a in enumerate(corpus_ids)}
    titles = {}
    with open(args.metadata, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            titles[int(row["id"])] = row["title_english"] or row["title"]
    dsu = build_components(args.metadata)
    comp_of_corpus = np.array([dsu.find(int(a)) for a in corpus_ids], dtype=np.int64)

    B = np.load(args.ease_b)
    rng_ref = np.random.default_rng(555)
    rng_h = np.random.default_rng(999)
    perm = rng_h.permutation(len(lengths))
    ref_users = rng_ref.choice(perm[len(lengths) // 10:], size=20000, replace=False)
    s_sum = np.zeros(cs)
    for u in ref_users:
        s0, l = starts[u], lengths[u]
        s_sum += B[indices[s0 : s0 + l]].sum(axis=0)
    mu = s_sum / len(ref_users)

    import jax
    from model import Recommender
    from analysis.train_probe_graft import GraftRecommender
    from flax import serialization

    nn_params, nn_fwd = load_prod(args.nn_weights, 512, cs)
    gate_params, gate_fwd = load_graft(args.gate, "gate", 512, cs, B)
    concat_params, concat_fwd = load_graft(args.concat, "concat", 512, cs, B)

    nn_model = Recommender()
    concat_model = GraftRecommender(mode="concat")
    Bj = jnp.asarray(B, dtype=jnp.float32)

    def nn_ratings(x):
        _, rt, _, _ = nn_model.apply({"params": nn_params}, x, training=False)
        return np.array(rt)[0].astype(np.float64)

    def concat_ratings(x):
        e = x[:, :cs] @ Bj
        e = (e - jnp.mean(e, axis=1, keepdims=True)) / (jnp.std(e, axis=1, keepdims=True) + 1e-6)
        _, rt, _, _, _ = concat_model.apply({"params": concat_params}, x, e, training=False)
        return np.array(rt)[0].astype(np.float64)

    def prod_mix(logits_with_prior, ratings, lw=0.3):
        p = np.exp(logits_with_prior - logits_with_prior.max())
        p /= p.sum()
        return np.power(p, lw) * np.power(np.maximum(ratings + 1, 0.001), 1 - lw)

    result = {}
    for prof in args.profiles.split(","):
        with open(Path(args.fixtures) / f"{prof}.json") as f:
            raw = json.load(f)
        owned = []
        vals = {}
        for e in raw:
            ls = e.get("list_status") or {}
            score = ls.get("score", 0) or 0
            if ls.get("status") == "plan_to_watch" and score == 0:
                continue
            ci = id_to_idx.get(e["node"]["id"])
            if ci is not None:
                owned.append(ci)
                vals[ci] = score
        owned = np.unique(np.array(owned, dtype=np.int64))
        rated = np.array([vals[c] for c in owned], dtype=np.float64)
        rz = np.where(rated > 0, (rated - rated[rated > 0].mean()) / (rated[rated > 0].std() + 1e-9), 0.0) \
            if (rated > 0).any() else np.zeros_like(rated)

        x = np.zeros((1, cs * 2), dtype=np.float32)
        x[0, owned] = 1.0
        x[0, cs + owned] = rz
        xj = jnp.asarray(x)
        nn_lg = np.array(nn_fwd(nn_params, xj))[0].astype(np.float64)
        gate_lg = np.array(gate_fwd(gate_params, xj))[0].astype(np.float64)
        concat_lg = np.array(concat_fwd(concat_params, xj))[0].astype(np.float64)
        s_ease = B[owned].sum(axis=0).astype(np.float64)
        ze = s_ease - mu
        ze = (ze - ze.mean()) / (ze.std() + 1e-9)
        znn = (nn_lg - nn_lg.mean()) / (nn_lg.std() + 1e-9)
        w_eff = 0.35 if len(owned) >= 64 else 0.0
        cand_franch = np.isin(comp_of_corpus, np.unique(comp_of_corpus[owned]))

        rt_nn = nn_ratings(xj)
        rt_concat = concat_ratings(xj)
        scorers = {
            "nn_prodmix_lw0.3": prod_mix(nn_lg + log_pop, rt_nn),
            "concat_prodmix_lw0.3": prod_mix(concat_lg + log_pop, rt_concat),
            "nn_a1.0": nn_lg + log_pop,
            "nn_za0.45": znn + 0.45 * zlp,
            "gate_a1.0": gate_lg + log_pop,
            "gate_za0.45": (gate_lg - gate_lg.mean()) / (gate_lg.std() + 1e-9) + 0.45 * zlp,
            "concat_a1.0": concat_lg + log_pop,
            "concat_za0.45": (concat_lg - concat_lg.mean()) / (concat_lg.std() + 1e-9) + 0.45 * zlp,
            "blend_policy_za0.45": (1 - w_eff) * znn + w_eff * ze + 0.45 * zlp,
            "ease_lift_za0.45": ze + 0.45 * zlp,
        }
        result[prof] = {"n_items": int(len(owned)), "blend_w_eff": w_eff, "lists": {}}
        for name, sc in scorers.items():
            sc = sc.copy()
            sc[owned] = -np.inf
            top = np.argsort(-sc)[:15]
            recs = [{"title": titles.get(int(corpus_ids[c]), "?"),
                     "pop_rank": int(pop_rank[c]),
                     "franchise": bool(cand_franch[c])} for c in top]
            result[prof]["lists"][name] = recs
            print(f"\n=== {prof} ({len(owned)} items) — {name} ===")
            for i, rc in enumerate(recs, 1):
                fl = " [FR]" if rc["franchise"] else ""
                print(f"  {i:2d}. #{rc['pop_rank']:<4d} {rc['title'][:56]}{fl}")

    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
