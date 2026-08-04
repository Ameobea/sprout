"""Naive EASE top-k recommendations for sentinel fixture profiles.
Recomputes B from the cached gram (lam=200), saves it, prints/exports lists."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gram-cache", required=True)
    ap.add_argument("--b-out", default="../../data/aug2026/ease_B6k_lam200.npy")
    ap.add_argument("--corpus", default="../../data/corpus_ids_aug2026.json")
    ap.add_argument("--metadata", default="../../data/processed-metadata_aug2026.csv")
    ap.add_argument("--vectors", default="../../data/aug2026/user_input_vectors_cleanup_notrust.npz")
    ap.add_argument("--fixtures", default="../eval/fixtures")
    ap.add_argument("--profiles", default="ameo___,snapsauce")
    ap.add_argument("--lam", type=float, default=200.0)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    b_path = Path(args.b_out)
    if b_path.exists():
        B = np.load(b_path)
    else:
        G = np.load(args.gram_cache)["G"]
        G[np.diag_indices(6000)] += args.lam
        P = np.linalg.inv(G)
        B = (-P / np.diag(P)[None, :]).astype(np.float32)
        np.fill_diagonal(B, 0.0)
        np.save(b_path, B)
        del G, P
    print("B ready", flush=True)

    with open(args.corpus) as f:
        corpus_ids = json.load(f)
    id_to_idx = {aid: i for i, aid in enumerate(corpus_ids)}

    titles = {}
    with open(args.metadata, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            titles[int(row["id"])] = row["title_english"] or row["title"]

    d = np.load(args.vectors)
    counts = np.bincount(d["indices"].astype(np.int32), minlength=6000).astype(np.float64)
    pop_rank = np.argsort(np.argsort(-counts))
    log_pop = np.log(np.maximum(counts, 1.0) / np.maximum(counts, 1.0).sum())

    result = {}
    for prof in args.profiles.split(","):
        with open(Path(args.fixtures) / f"{prof}.json") as f:
            raw = json.load(f)
        owned = []
        for e in raw:
            ls = e.get("list_status") or {}
            score = ls.get("score", 0) or 0
            if ls.get("status") == "plan_to_watch" and score == 0:
                continue
            ci = id_to_idx.get(e["node"]["id"])
            if ci is not None:
                owned.append(ci)
        owned = np.unique(np.array(owned, dtype=np.int64))
        s_raw = B[owned].sum(axis=0)
        result[prof] = {"n_in_corpus": int(len(owned)), "variants": {}}
        # 0.2167 = beta/tau from the rank-metric calibration (tau=3.0, beta=0.65):
        # ranking under softmax(tau*s + beta*log_pop) == ranking of s + (beta/tau)*log_pop
        for name, sc in [("as_evaluated", s_raw + 0.2167 * log_pop), ("raw", s_raw.copy())]:
            sc = sc.copy()
            sc[owned] = -np.inf
            top = np.argsort(-sc)[: args.k]
            recs = []
            for ci in top:
                aid = corpus_ids[ci]
                recs.append({"title": titles.get(aid, f"id={aid}"), "pop_rank": int(pop_rank[ci]),
                             "score": round(float(sc[ci]), 3)})
            result[prof]["variants"][name] = recs
            print(f"\n=== {prof} ({len(owned)} in-corpus) — EASE {name} ===")
            for r_i, rec in enumerate(recs, 1):
                print(f"  {r_i:2d}. pop_rank={rec['pop_rank']:4d}  {rec['title'][:55]}")

    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
