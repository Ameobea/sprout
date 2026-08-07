"""
(alpha, k) serving-family sweep for lift-trained (logQ) models:

  serve_logits = lam * lift + alpha * log_pop,   lam_i = count_i / (count_i + k)

alpha = how much global popularity is mixed back in; k = evidence bar that mutes
thin-evidence lift. Captures, per grid point: overall/tier recall, MRR, the
popularity-rank distribution of top-50 rec slots (binned), distance from the
global watch distribution, and sentinel top-10 lists for qualitative inspection.

Run: JAX_PLATFORMS=cpu python alpha_k_sweep.py --weights ... --name ...
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import CONF, infer_outputs, make_dense_profile
from eval_harness import FIXTURES_DIR, build_aux, load_params, preprocess
from temporal_eval import POP_TIERS, combined_score

FIXTURES = Path(__file__).parent / "fixtures/temporal_v3.json"
REPORTS_DIR = Path(__file__).resolve().parents[2] / "private/eval-reports"
POPULARITY = Path(__file__).parent / "../../data/item_popularity_dec2025.npy"
METADATA = Path(__file__).parent / "../../data/processed-metadata.csv"

ALPHAS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2]
KS = [0, 500, 1000, 1500, 2250, 2750, 3500, 5000, 7500, 10000]
LW = 0.3
BIN_EDGES = [0, 25, 50, 100, 250, 500, 1000, 1500, 2250, 3000, 4000, 5000, 6000]
SENTINELS = ["ameo___", "snapsauce"]
MID_TIERS = (1, 2, 3)  # 50-250, 250-1k, 1k-3k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--popularity", default=str(POPULARITY))
    ap.add_argument("--corpus", default="../../data/corpus_ids.json")
    ap.add_argument("--name", required=True)
    ap.add_argument("--input-channels", type=int, default=2, choices=[2, 3, 5])
    args = ap.parse_args()
    CONF["input_channels"] = args.input_channels

    with open(args.corpus) as f:
        corpus_ids = json.load(f)
    id_to_idx = {aid: i for i, aid in enumerate(corpus_ids)}
    cs = CONF["corpus_size"]

    counts = np.load(args.popularity).astype(np.float64)
    log_pop = np.log(np.maximum(counts, 1.0))
    pop_rank = np.argsort(np.argsort(-counts))
    tier_of = np.zeros(cs, dtype=np.int32)
    for ti, (lo, hi, _) in enumerate(POP_TIERS):
        tier_of[(pop_rank >= lo) & (pop_rank < hi)] = ti
    bin_of = np.digitize(pop_rank, BIN_EDGES[1:-1])
    n_bins = len(BIN_EDGES) - 1

    watch_share = np.zeros(n_bins)
    np.add.at(watch_share, bin_of, counts)
    watch_share /= watch_share.sum()

    params = load_params(args.weights)
    with open(FIXTURES) as f:
        fixtures = json.load(f)

    users = []
    target_share = np.zeros(n_bins)
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
        np.add.at(target_share, bin_of[target_idxs], 1)
    target_share /= target_share.sum()
    print(f"cached {len(users)} users", flush=True)

    sent = {}
    for name in SENTINELS:
        with open(FIXTURES_DIR / f"{name}.json") as f:
            raw = json.load(f)
        items = [(e["node"]["id"], e["list_status"].get("score", 0) or 0,
                  e["list_status"].get("status", "")) for e in raw if e.get("list_status")]
        idxs, vals, original, statuses = preprocess(items, id_to_idx)
        x = make_dense_profile(idxs, vals, build_aux(original, statuses))
        logits, ratings = infer_outputs(params, x)
        candidate = np.ones(cs, dtype=bool)
        candidate[np.unique(idxs)] = False
        sent[name] = (np.array(logits[0], dtype=np.float64),
                      np.array(ratings[0], dtype=np.float64), candidate)

    variants = {}
    sentinel_recs = {name: {} for name in SENTINELS}
    used_ids = set()
    for alpha in ALPHAS:
        for k in KS:
            lam = counts / (counts + k) if k > 0 else np.ones(cs)
            key = f"a={alpha} k={k}"
            exposure = np.zeros(n_bins)
            tier_hits = np.zeros(5)
            tier_n = np.zeros(5)
            per_user_r50, per_user_r10, mrrs, top10_rank = [], [], [], []
            for logits, ratings, candidate, target_idxs in users:
                serve = lam * logits + alpha * log_pop
                scores = combined_score(serve, ratings, LW)
                cand_scores = np.where(candidate, scores, -np.inf)
                tscores = cand_scores[target_idxs]
                ranks = np.array([1 + int(np.sum(cand_scores > s)) for s in tscores])
                per_user_r50.append(float(np.mean(ranks <= 50)))
                per_user_r10.append(float(np.mean(ranks <= 10)))
                mrrs.append(1.0 / ranks.min())
                np.add.at(tier_hits, tier_of[target_idxs], (ranks <= 50).astype(float))
                np.add.at(tier_n, tier_of[target_idxs], 1)
                top50 = np.argsort(cand_scores)[-50:]
                np.add.at(exposure, bin_of[top50], 1)
                top10_rank.append(float(np.mean(pop_rank[top50[-10:]])))
            exposure /= exposure.sum()
            tier_r = tier_hits / np.maximum(tier_n, 1)
            mid_recall = tier_hits[list(MID_TIERS)].sum() / tier_n[list(MID_TIERS)].sum()
            variants[key] = {
                "alpha": alpha, "k": k,
                "recall@10": round(float(np.mean(per_user_r10)), 4),
                "recall@50": round(float(np.mean(per_user_r50)), 4),
                "mrr": round(float(np.mean(mrrs)), 4),
                "mid_recall@50": round(float(mid_recall), 4),
                "tier_recall@50": [round(float(v), 4) for v in tier_r],
                "exposure": [round(float(v), 5) for v in exposure],
                "tvd_vs_popularity": round(float(0.5 * np.abs(exposure - watch_share).sum()), 4),
                "tvd_vs_targets": round(float(0.5 * np.abs(exposure - target_share).sum()), 4),
                "mean_pop_rank_top10": round(float(np.mean(top10_rank)), 1),
            }
            for name, (logits, ratings, candidate) in sent.items():
                serve = lam * logits + alpha * log_pop
                scores = combined_score(serve, ratings, LW)
                cand_scores = np.where(candidate, scores, -np.inf)
                top10 = np.argsort(cand_scores)[-10:][::-1]
                sentinel_recs[name][key] = [
                    [int(corpus_ids[ci]), int(pop_rank[ci]), round(float(ratings[ci]), 2)]
                    for ci in top10
                ]
                used_ids.update(int(corpus_ids[ci]) for ci in top10)
        v = variants[f"a={alpha} k={KS[0]}"]
        print(f"a={alpha}: r@50={v['recall@50']:.3f} mid={v['mid_recall@50']:.3f} "
              f"tvd_pop={v['tvd_vs_popularity']:.3f}", flush=True)

    titles = {}
    with open(METADATA, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            aid = int(row["id"])
            if aid in used_ids:
                titles[aid] = row["title_english"] or row["title"]

    report = {
        "name": args.name, "weights": str(args.weights), "lw": LW,
        "alphas": ALPHAS, "ks": KS, "bin_edges": BIN_EDGES,
        "watch_share": [round(float(v), 5) for v in watch_share],
        "target_share": [round(float(v), 5) for v in target_share],
        "n_users": len(users),
        "variants": variants,
        "sentinel_recs": sentinel_recs,
        "titles": titles,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"alpha-k-{args.name}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)
    print(f"report written to {out_path}")


if __name__ == "__main__":
    main()
