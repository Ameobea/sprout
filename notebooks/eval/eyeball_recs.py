"""
Qualitative top-k eyeball for a sentinel fixture profile under ranking variants.
Prints title, popularity rank, presence prob, predicted rating per rec.

Run: JAX_PLATFORMS=cpu python eyeball_recs.py --weights ... [--profile ameo___]
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
from temporal_eval import combined_score

POPULARITY = Path(__file__).parent / "../../data/item_popularity_dec2025.npy"

VARIANTS = [(0.0, 0.3), (0.25, 0.3), (0.5, 0.3), (0.5, 1.0)]  # overridden by --alphas (lw fixed 0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--corpus", default="../../data/corpus_ids.json")
    ap.add_argument("--profile", default="ameo___")
    ap.add_argument("--input-channels", type=int, default=2, choices=[2, 3, 5])
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--alphas", default=None, help="comma list; each becomes (alpha, lw=0.3)")
    ap.add_argument("--popularity", default=str(POPULARITY))
    args = ap.parse_args()
    global VARIANTS
    if args.alphas:
        VARIANTS = [(float(a), 0.3) for a in args.alphas.split(",")]
    CONF["input_channels"] = args.input_channels

    with open(args.corpus) as f:
        corpus_ids = json.load(f)
    id_to_idx = {aid: i for i, aid in enumerate(corpus_ids)}

    titles = {}
    with open(Path(__file__).parent / "../../data/processed-metadata.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            titles[int(row["id"])] = row["title_english"] or row["title"]

    counts = np.load(args.popularity).astype(np.float64)
    pop_rank = np.argsort(np.argsort(-counts))
    log_pop = np.log(np.maximum(counts, 1.0))

    with open(FIXTURES_DIR / f"{args.profile}.json") as f:
        raw = json.load(f)
    items = [
        (e["node"]["id"], e["list_status"].get("score", 0) or 0, e["list_status"].get("status", ""))
        for e in raw if e.get("list_status")
    ]
    idxs, vals, original, statuses = preprocess(items, id_to_idx)
    print(f"profile {args.profile}: {len(idxs)} in-corpus items")

    params = load_params(args.weights)
    x = make_dense_profile(idxs, vals, build_aux(original, statuses))
    logits, ratings = infer_outputs(params, x)
    logits = np.array(logits[0], dtype=np.float64)
    ratings = np.array(ratings[0], dtype=np.float64)
    candidate = np.ones(CONF["corpus_size"], dtype=bool)
    candidate[np.unique(idxs)] = False
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()

    for alpha, lw in VARIANTS:
        scores = combined_score(logits - alpha * log_pop, ratings, lw)
        cand_scores = np.where(candidate, scores, -np.inf)
        top = np.argsort(cand_scores)[-args.k:][::-1]
        print(f"\n--- alpha={alpha} lw={lw} ---")
        for rank, ci in enumerate(top, 1):
            aid = corpus_ids[ci]
            zero = " [ZERO-COUNT]" if counts[ci] == 0 else ""
            print(f"  {rank:2d}. pop_rank={pop_rank[ci]:4d} prob={probs[ci]:.4f} "
                  f"pred={ratings[ci]: .2f}  {titles.get(aid, f'id={aid}')[:55]}{zero}")


if __name__ == "__main__":
    main()
