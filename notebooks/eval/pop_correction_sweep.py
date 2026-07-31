"""
Inference-time ranking-variant sweep on the temporal fixtures. One cached forward
pass per user, then every scoring variant is evaluated from the cache:

  - logQ popularity correction: presence probs from softmax(logits - alpha*log_pop)
  - alt z-score ranking (compute_recommendation_ranking_score_alt semantics)
  - prod niche-boost analog: score *= 1 + f*ln(1 + prob/pop_frac)

Run: JAX_PLATFORMS=cpu python pop_correction_sweep.py --weights ... --name ...
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
from temporal_eval import POP_TIERS, RECALL_KS, combined_score, tier_aggregate

FIXTURES = Path(__file__).parent / "fixtures/temporal_v3.json"
REPORTS_DIR = Path(__file__).parent / "reports"
POPULARITY = Path(__file__).parent / "../../data/item_popularity_dec2025.npy"

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
LWS = [0.1, 0.3, 0.5, 1.0]
ALT_LWS = [0.1, 0.3, 0.5, 0.7]
BOOST_FACTORS = [0.25, 0.5, 0.75, 1.0]


def corrected_score(logits, ratings, log_pop, alpha, lw):
    return combined_score(logits - alpha * log_pop, ratings, lw)


def alt_score(logits, ratings, lw):
    nl = (logits - logits.mean()) / (logits.std() + 1e-6)
    nr = (ratings - ratings.mean()) / (ratings.std() + 1e-6)
    return lw * nl + (1.0 - lw) * nr


def effective_boost(f):
    f = min(max(f, 0.0), 1.0)
    return f if f <= 0.5 else 0.5 + (f - 0.5) * np.exp(4.62 * (f - 0.5))


def boost_score(logits, ratings, pop_frac, factor, lw):
    base = combined_score(logits, ratings, lw)
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    surprise = probs / (pop_frac + 1e-9)
    return base * (1.0 + effective_boost(factor) * np.log1p(surprise))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--corpus", default="../../data/corpus_ids.json")
    ap.add_argument("--name", required=True)
    ap.add_argument("--input-channels", type=int, default=2, choices=[2, 3, 5])
    ap.add_argument("--alphas", default=None,
                    help="comma list; negative alpha ADDS popularity back (for lift-trained models)")
    ap.add_argument("--skip-alt-boost", action="store_true")
    args = ap.parse_args()
    CONF["input_channels"] = args.input_channels
    alphas = [float(a) for a in args.alphas.split(",")] if args.alphas else ALPHAS

    with open(args.corpus) as f:
        corpus_ids = json.load(f)
    id_to_idx = {aid: i for i, aid in enumerate(corpus_ids)}
    cs = CONF["corpus_size"]

    counts = np.load(POPULARITY).astype(np.float64)
    pop_rank = np.argsort(np.argsort(-counts))
    tier_of = np.zeros(cs, dtype=np.int32)
    for ti, (lo, hi, _) in enumerate(POP_TIERS):
        tier_of[(pop_rank >= lo) & (pop_rank < hi)] = ti
    log_pop = np.log(np.maximum(counts, 1.0))
    pop_frac = counts / counts.sum()

    params = load_params(args.weights)
    with open(FIXTURES) as f:
        fixtures = json.load(f)

    users = []
    for username, p in sorted(fixtures.items()):
        idxs, vals, original, statuses = preprocess(p["items"], id_to_idx)
        target_idxs = np.array(sorted({id_to_idx[a] for a, _ in p["targets"] if a in id_to_idx}))
        target_idxs = target_idxs[~np.isin(target_idxs, idxs)]
        if len(idxs) < 5 or len(target_idxs) < 3:
            continue
        x = make_dense_profile(idxs, vals, build_aux(original, statuses))
        logits, ratings = infer_outputs(params, x)
        candidate = np.ones(cs, dtype=bool)
        candidate[np.unique(idxs)] = False
        users.append((np.array(logits[0], dtype=np.float64),
                      np.array(ratings[0], dtype=np.float64), candidate, target_idxs))
    print(f"cached forward passes for {len(users)} users", flush=True)

    variants = []
    for alpha in alphas:
        for lw in LWS:
            variants.append((f"logq a={alpha} lw={lw}",
                             lambda l, r, a=alpha, w=lw: corrected_score(l, r, log_pop, a, w)))
    if not args.skip_alt_boost:
        for lw in ALT_LWS:
            variants.append((f"altz lw={lw}", lambda l, r, w=lw: alt_score(l, r, w)))
        for f in BOOST_FACTORS:
            variants.append((f"boost f={f} lw=0.3",
                             lambda l, r, ff=f: boost_score(l, r, pop_frac, ff, 0.3)))

    report_rows = {}
    for vname, fn in variants:
        pooled = []
        tiers = [[] for _ in POP_TIERS]
        mrrs, med_ranks, per_user_recalls = [], [], {k: [] for k in RECALL_KS}
        rp10, rp50, frac_tail = [], [], []
        for logits, ratings, candidate, target_idxs in users:
            scores = fn(logits, ratings)
            cand_scores = np.where(candidate, scores, -np.inf)
            tscores = cand_scores[target_idxs]
            ranks = np.array([1 + int(np.sum(cand_scores > s)) for s in tscores])
            pooled.extend(ranks)
            for t, rk in zip(target_idxs, ranks):
                tiers[tier_of[t]].append(rk)
            mrrs.append(1.0 / ranks.min())
            med_ranks.append(float(np.median(ranks)))
            for k in RECALL_KS:
                per_user_recalls[k].append(float(np.mean(ranks <= k)))
            top50 = np.argsort(cand_scores)[-50:]
            rp10.append(float(np.mean(pop_rank[top50[-10:]])))
            rp50.append(float(np.mean(pop_rank[top50])))
            frac_tail.append(float(np.mean(pop_rank[top50] >= 1000)))
        row = {
            "median_target_rank": float(np.mean(med_ranks)),
            "mrr": float(np.mean(mrrs)),
            **{f"recall@{k}": float(np.mean(per_user_recalls[k])) for k in RECALL_KS},
            "by_popularity_tier": tier_aggregate(tiers),
            "rec_popularity": {"top10": float(np.mean(rp10)), "top50": float(np.mean(rp50)),
                               "frac_tail_top50": float(np.mean(frac_tail))},
        }
        report_rows[vname] = row
        t = row["by_popularity_tier"]
        tier_cells = " ".join(f"{t[label]['recall@50']:.3f}" if label in t else "  -  "
                              for _, _, label in POP_TIERS)
        print(f"{vname:22s} r@10={row['recall@10']:.3f} r@50={row['recall@50']:.3f} "
              f"med={row['median_target_rank']:.0f} mrr={row['mrr']:.3f} | tier r@50: {tier_cells} | "
              f"pop10={row['rec_popularity']['top10']:.0f} tail50={row['rec_popularity']['frac_tail_top50']:.2f}",
              flush=True)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"pop-sweep-{args.name}.json"
    with open(out_path, "w") as f:
        json.dump({"name": args.name, "weights": str(args.weights),
                   "input_channels": CONF["input_channels"], "variants": report_rows},
                  f, indent=1, sort_keys=True)
    print(f"report written to {out_path}")


if __name__ == "__main__":
    main()
