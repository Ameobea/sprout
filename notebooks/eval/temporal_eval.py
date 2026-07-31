"""
Temporal (future-watch) eval: one forward pass per fixture user on their pre-cutoff
profile, then measure how the post-cutoff watched items rank among all non-input
corpus items. This is the product task; LOO reconstruction is the proxy.

Also scores a global-popularity baseline (corpus order) and supports sweeping the
logit_weight ranking blend without re-running the model.

Run: JAX_PLATFORMS=cpu python temporal_eval.py --weights ... --corpus ... --name ...
     [--input-channels N] [--logit-weights 0.1,0.3,0.5,0.7,0.9]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax

from model import CONF, infer_outputs, make_dense_profile
from eval_harness import build_aux, load_params, preprocess

FIXTURES = Path(__file__).parent / "fixtures/temporal_v3.json"
REPORTS_DIR = Path(__file__).parent / "reports"
RECALL_KS = [10, 50, 100]


def rank_metrics(order_scores, input_idx_set, target_idxs, corpus_size):
    """order_scores: 1D score per corpus item (higher = better). Ranks computed
    among items NOT in the input profile."""
    candidate = np.ones(corpus_size, dtype=bool)
    candidate[list(input_idx_set)] = False
    cand_scores = np.where(candidate, order_scores, -np.inf)
    target_scores = cand_scores[target_idxs]
    ranks = np.array([1 + int(np.sum(cand_scores > s)) for s in target_scores])
    return {
        "n_targets": len(target_idxs),
        "median_target_rank": float(np.median(ranks)),
        "mrr": float(1.0 / ranks.min()),
        **{f"recall@{k}": float(np.mean(ranks <= k)) for k in RECALL_KS},
    }


def combined_score(logits, ratings, logit_weight):
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return np.power(probs, logit_weight) * np.power(np.maximum(ratings + 1, 0.001), 1.0 - logit_weight)


def aggregate(rows):
    keys = ["median_target_rank", "mrr"] + [f"recall@{k}" for k in RECALL_KS]
    return {"n_profiles": len(rows), **{k: float(np.mean([r[k] for r in rows])) for k in keys}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--input-channels", type=int, default=2, choices=[2, 3, 5])
    ap.add_argument("--logit-weights", default="0.3")
    args = ap.parse_args()
    CONF["input_channels"] = args.input_channels
    logit_weights = [float(w) for w in args.logit_weights.split(",")]

    with open(args.corpus) as f:
        corpus_ids = json.load(f)
    id_to_idx = {aid: i for i, aid in enumerate(corpus_ids)}
    cs = CONF["corpus_size"]

    params = load_params(args.weights)
    with open(FIXTURES) as f:
        fixtures = json.load(f)

    per_profile = {w: {} for w in logit_weights}
    pop_rows = {}
    popularity_scores = -np.arange(cs, dtype=np.float32)

    for i, (username, p) in enumerate(sorted(fixtures.items())):
        idxs, vals, original, statuses = preprocess(p["items"], id_to_idx)
        target_idxs = np.array(sorted({id_to_idx[a] for a, _ in p["targets"] if a in id_to_idx}))
        target_idxs = target_idxs[~np.isin(target_idxs, idxs)]
        if len(idxs) < 5 or len(target_idxs) < 3:
            continue
        x = make_dense_profile(idxs, vals, build_aux(original, statuses))
        logits, ratings = infer_outputs(params, x)
        logits, ratings = np.array(logits[0]), np.array(ratings[0])
        input_set = set(int(v) for v in idxs)

        for w in logit_weights:
            per_profile[w][username] = {
                "bucket": p["bucket"],
                **rank_metrics(combined_score(logits, ratings, w), input_set, target_idxs, cs),
            }
        pop_rows[username] = {
            "bucket": p["bucket"],
            **rank_metrics(popularity_scores, input_set, target_idxs, cs),
        }
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(fixtures)} profiles", flush=True)

    buckets = sorted({r["bucket"] for r in pop_rows.values()})
    report = {
        "name": args.name,
        "weights": str(args.weights),
        "input_channels": CONF["input_channels"],
        "by_logit_weight": {
            str(w): {
                "overall": aggregate(list(rows.values())),
                "by_bucket": {
                    b: aggregate([r for r in rows.values() if r["bucket"] == b]) for b in buckets
                },
            }
            for w, rows in per_profile.items()
        },
        "popularity_baseline": {
            "overall": aggregate(list(pop_rows.values())),
            "by_bucket": {
                b: aggregate([r for r in pop_rows.values() if r["bucket"] == b]) for b in buckets
            },
        },
        "per_profile": {str(w): rows for w, rows in per_profile.items()},
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"{args.name}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)

    for w in logit_weights:
        o = report["by_logit_weight"][str(w)]["overall"]
        print(f"  lw={w}: recall@10={o['recall@10']:.3f} recall@50={o['recall@50']:.3f} "
              f"recall@100={o['recall@100']:.3f} med_rank={o['median_target_rank']:.0f} mrr={o['mrr']:.3f}")
    o = report["popularity_baseline"]["overall"]
    print(f"  popularity: recall@10={o['recall@10']:.3f} recall@50={o['recall@50']:.3f} "
          f"recall@100={o['recall@100']:.3f} med_rank={o['median_target_rank']:.0f} mrr={o['mrr']:.3f}")
    print(f"report written to {out_path}")


if __name__ == "__main__":
    main()
