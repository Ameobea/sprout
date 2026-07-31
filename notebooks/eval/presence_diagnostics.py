"""
Presence-head diagnostics over temporal fixtures: is the softmax ordering just a
popularity prior? One forward pass per user on the pre-cutoff profile, then
decompose probs / ranks / NLL by global item popularity (training-set frequency).

Run: JAX_PLATFORMS=cpu python presence_diagnostics.py --weights ... --name ...
     [--input-channels N]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import CONF, infer_outputs, make_dense_profile
from eval_harness import build_aux, load_params, preprocess
from temporal_eval import combined_score

FIXTURES = Path(__file__).parent / "fixtures/temporal_v3.json"
REPORTS_DIR = Path(__file__).parent / "reports"
POPULARITY = Path(__file__).parent / "../../data/item_popularity_dec2025.npy"

TIERS = [(0, 50, "top50"), (50, 250, "50-250"), (250, 1000, "250-1k"),
         (1000, 3000, "1k-3k"), (3000, 6000, "3k-6k")]

ORDERINGS = {"presence_only": 1.0, "blend_0.3": 0.3, "rating_only": 0.0}


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def ranks_of(order_scores, candidate_mask, target_idxs):
    cand_scores = np.where(candidate_mask, order_scores, -np.inf)
    return np.array([1 + int(np.sum(cand_scores > cand_scores[t])) for t in target_idxs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--corpus", default="../../data/corpus_ids.json")
    ap.add_argument("--name", required=True)
    ap.add_argument("--input-channels", type=int, default=2, choices=[2, 3, 5])
    args = ap.parse_args()
    CONF["input_channels"] = args.input_channels

    with open(args.corpus) as f:
        corpus_ids = json.load(f)
    id_to_idx = {aid: i for i, aid in enumerate(corpus_ids)}
    cs = CONF["corpus_size"]

    counts = np.load(POPULARITY).astype(np.float64)
    pop_rank = np.argsort(np.argsort(-counts))
    tier_of = np.zeros(cs, dtype=np.int32)
    for ti, (lo, hi, _) in enumerate(TIERS):
        tier_of[(pop_rank >= lo) & (pop_rank < hi)] = ti
    log_pop = np.log(np.maximum(counts, 1.0))

    params = load_params(args.weights)
    with open(FIXTURES) as f:
        fixtures = json.load(f)

    per_user = []
    tier_nll_target = [[] for _ in TIERS]
    tier_nll_input = [[] for _ in TIERS]
    tier_ranks = {o: [[] for _ in TIERS] for o in ORDERINGS}

    for username, p in sorted(fixtures.items()):
        idxs, vals, original, statuses = preprocess(p["items"], id_to_idx)
        target_idxs = np.array(sorted({id_to_idx[a] for a, _ in p["targets"] if a in id_to_idx}))
        target_idxs = target_idxs[~np.isin(target_idxs, idxs)]
        if len(idxs) < 5 or len(target_idxs) < 3:
            continue
        x = make_dense_profile(idxs, vals, build_aux(original, statuses))
        logits, ratings = infer_outputs(params, x)
        logits, ratings = np.array(logits[0], dtype=np.float64), np.array(ratings[0], dtype=np.float64)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()

        input_set = np.unique(idxs)
        candidate = np.ones(cs, dtype=bool)
        candidate[input_set] = False

        nll = -np.log(np.maximum(probs, 1e-12))
        for t in target_idxs:
            tier_nll_target[tier_of[t]].append(nll[t])
        for i in input_set:
            tier_nll_input[tier_of[i]].append(nll[i])

        row = {
            "bucket": p["bucket"],
            "n_input": int(len(input_set)),
            "n_targets": int(len(target_idxs)),
            "spearman_probs_pop_all": spearman(probs, counts),
            "spearman_probs_pop_cand": spearman(probs[candidate], counts[candidate]),
            "mass_on_input": float(probs[input_set].sum()),
        }

        cand_probs = np.where(candidate, probs, -np.inf)
        cand_pop = np.where(candidate, counts, -np.inf)
        top50_presence = set(np.argsort(cand_probs)[-50:].tolist())
        top50_pop = set(np.argsort(cand_pop)[-50:].tolist())
        row["top50_overlap_with_popularity"] = len(top50_presence & top50_pop) / 50.0

        for name, lw in ORDERINGS.items():
            scores = combined_score(logits, ratings, lw)
            r = ranks_of(scores, candidate, target_idxs)
            for t, rk in zip(target_idxs, r):
                tier_ranks[name][tier_of[t]].append(rk)
            row[f"{name}_recall@50"] = float(np.mean(r <= 50))
            row[f"{name}_median_rank"] = float(np.median(r))
        per_user.append(row)

    def agg(key):
        return float(np.mean([r[key] for r in per_user]))

    tier_table = []
    for ti, (_, _, label) in enumerate(TIERS):
        entry = {
            "tier": label,
            "n_targets": len(tier_nll_target[ti]),
            "n_input_items": len(tier_nll_input[ti]),
            "target_nll": float(np.mean(tier_nll_target[ti])) if tier_nll_target[ti] else None,
            "input_recon_nll": float(np.mean(tier_nll_input[ti])) if tier_nll_input[ti] else None,
        }
        for name in ORDERINGS:
            rks = np.array(tier_ranks[name][ti])
            if len(rks):
                entry[f"{name}_recall@50"] = float(np.mean(rks <= 50))
                entry[f"{name}_median_rank"] = float(np.median(rks))
        tier_table.append(entry)

    report = {
        "name": args.name,
        "weights": str(args.weights),
        "input_channels": CONF["input_channels"],
        "n_profiles": len(per_user),
        "n_zero_count_items": int(np.sum(counts == 0)),
        "overall": {
            "spearman_probs_pop_all": agg("spearman_probs_pop_all"),
            "spearman_probs_pop_cand": agg("spearman_probs_pop_cand"),
            "mass_on_input": agg("mass_on_input"),
            "top50_overlap_with_popularity": agg("top50_overlap_with_popularity"),
            **{f"{n}_{m}": agg(f"{n}_{m}") for n in ORDERINGS for m in ["recall@50", "median_rank"]},
        },
        "by_tier": tier_table,
        "per_user": per_user,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"presence-diag-{args.name}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)

    o = report["overall"]
    print(f"\n=== presence diagnostics: {args.name} ===")
    print(f"profiles: {len(per_user)}   zero-count corpus items: {report['n_zero_count_items']}")
    print(f"spearman(probs, popularity): all={o['spearman_probs_pop_all']:.3f} "
          f"candidates-only={o['spearman_probs_pop_cand']:.3f}")
    print(f"softmax mass on input set: {o['mass_on_input']:.3f}   "
          f"top50 overlap w/ popularity list: {o['top50_overlap_with_popularity']:.3f}")
    for n in ORDERINGS:
        print(f"  {n:14s}: recall@50={o[n + '_recall@50']:.3f} med_rank={o[n + '_median_rank']:.0f}")
    hdr = f"{'tier':>8} {'n_tgt':>6} {'tgt_nll':>8} {'in_nll':>7} " + " ".join(
        f"{n[:8]:>10}" for n in ORDERINGS)
    print("\nper-tier (recall@50 by ordering):\n" + hdr)
    for e in tier_table:
        cells = " ".join(f"{e.get(n + '_recall@50', float('nan')):>10.3f}" for n in ORDERINGS)
        tn = e["target_nll"]
        inl = e["input_recon_nll"]
        print(f"{e['tier']:>8} {e['n_targets']:>6} {tn if tn is None else round(tn, 3)!s:>8} "
              f"{inl if inl is None else round(inl, 3)!s:>7} {cells}")
    print(f"report written to {out_path}")


if __name__ == "__main__":
    main()
