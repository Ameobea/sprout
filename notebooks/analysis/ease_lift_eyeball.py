"""Sentinel top-15s under EASE-lift serving (s - mu + alpha*z(log_pop)), with
prod-style franchise flags (union-find over season relation types)."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

SEASON_RELS = {"sequel", "prequel", "parent_story", "side_story"}


class DSU:
    def __init__(self):
        self.p = {}
    def find(self, a):
        while self.p.setdefault(a, a) != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ease-b", default="../../data/aug2026/ease_B6k_lam200.npy")
    ap.add_argument("--vectors", default="../../data/aug2026/user_input_vectors_cleanup_notrust.npz")
    ap.add_argument("--corpus", default="../../data/corpus_ids_aug2026.json")
    ap.add_argument("--metadata", default="../../data/processed-metadata_aug2026.csv")
    ap.add_argument("--fixtures", default="../eval/fixtures")
    ap.add_argument("--profiles", default="ameo___,snapsauce")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    B = np.load(args.ease_b)
    d = np.load(args.vectors)
    indices = d["indices"].astype(np.int32)
    lengths = d["lengths"].astype(np.int64)
    starts = np.zeros(len(lengths), dtype=np.int64)
    np.cumsum(lengths[:-1], out=starts[1:])
    counts = np.bincount(indices, minlength=6000).astype(np.float64)
    log_pop = np.log(np.maximum(counts, 1.0) / np.maximum(counts, 1.0).sum())
    zlp = (log_pop - log_pop.mean()) / log_pop.std()
    pop_rank = np.argsort(np.argsort(-counts))

    rng = np.random.default_rng(123)
    pool = rng.choice(len(lengths), size=25000, replace=False)
    ref_users = pool[:20000]
    s_sum = np.zeros(6000)
    for u in ref_users:
        s0, l = starts[u], lengths[u]
        s_sum += B[indices[s0 : s0 + l]].sum(axis=0)
    mu = s_sum / len(ref_users)

    with open(args.corpus) as f:
        corpus_ids = np.array(json.load(f), dtype=np.int64)
    id_to_idx = {int(a): i for i, a in enumerate(corpus_ids)}
    titles = {}
    dsu = DSU()
    with open(args.metadata, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            aid = int(row["id"])
            titles[aid] = row["title_english"] or row["title"]
            if row["related_anime"]:
                try:
                    for e in json.loads(row["related_anime"]):
                        if e.get("relation_type") in SEASON_RELS:
                            dsu.union(aid, e["node"]["id"])
                except json.JSONDecodeError:
                    pass
    comp_of_corpus = np.array([dsu.find(int(a)) for a in corpus_ids], dtype=np.int64)

    result = {}
    for prof in args.profiles.split(","):
        with open(Path(args.fixtures) / f"{prof}.json") as f:
            raw = json.load(f)
        owned = []
        for e in raw:
            ls = e.get("list_status") or {}
            if ls.get("status") == "plan_to_watch" and (ls.get("score", 0) or 0) == 0:
                continue
            ci = id_to_idx.get(e["node"]["id"])
            if ci is not None:
                owned.append(ci)
        owned = np.unique(np.array(owned, dtype=np.int64))
        prof_comps = np.array(sorted(set(comp_of_corpus[owned].tolist())), dtype=np.int64)
        cand_franch = np.isin(comp_of_corpus, prof_comps)

        s = B[owned].sum(axis=0).astype(np.float64)
        lift = s - mu
        zl = (lift - lift.mean()) / (lift.std() + 1e-9)
        result[prof] = {}
        for alpha in [0.6, 0.3]:
            sc = zl + alpha * zlp
            sc[owned] = -np.inf
            recs = []
            top = np.argsort(-sc)
            shown = 0
            for ci in top:
                if shown >= 15:
                    break
                recs.append({"title": titles.get(int(corpus_ids[ci]), "?"),
                             "pop_rank": int(pop_rank[ci]),
                             "franchise": bool(cand_franch[ci])})
                shown += 1
            result[prof][f"alpha{alpha}"] = recs
            print(f"\n=== {prof} — EASE-lift + {alpha}*z(log_pop) ===")
            for i, rc in enumerate(recs, 1):
                fl = " [FRANCHISE]" if rc["franchise"] else ""
                print(f"  {i:2d}. pop_rank={rc['pop_rank']:4d}  {rc['title'][:52]}{fl}")

    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
