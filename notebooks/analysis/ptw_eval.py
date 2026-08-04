"""Plan-to-watch eval: unrated PTW entries as intent labels.

PTW items are excluded from model input by the vectorization (intentional), so
they are unseen targets even for training users. Input = the user's full model
profile exactly as served; targets = their unrated in-corpus PTW items. Users
come from the seed-999 holdout rows; the model-input reconstruction from the raw
CSV is verified index-exact against the npz row before a user is admitted.

Run inside rocm_jax:
  JAX_PLATFORMS=cpu python analysis/ptw_eval.py --out ../data/aug2026/ptw_eval.json
"""

import argparse
import csv
import gzip
import json
import sys

import numpy as np

sys.path.insert(0, ".")
import jax
import jax.numpy as jnp
from flax import serialization

from model import CONF, Recommender
from analysis.train_probe_graft import GraftRecommender
from analysis.frontier_eval import build_components

HOLDOUT_SEED = 999
TIERS = [(0, 250), (250, 1000), (1000, 3000), (3000, 6000)]
BATCH = 256


def load_prod2(path, cs):
    model = Recommender()
    dummy = jnp.ones((1, cs * 2))
    params = model.init({"params": jax.random.PRNGKey(0), "noise": jax.random.PRNGKey(0)}, dummy)["params"]
    with open(path, "rb") as f:
        params = serialization.from_bytes(params, f.read())
    fwd = jax.jit(lambda p, x: model.apply({"params": p}, x, training=False)[:2])
    return params, fwd


def load_graft2(path, mode, cs, B):
    model = GraftRecommender(mode=mode)
    dummy_x, dummy_e = jnp.ones((1, cs * 2)), jnp.ones((1, cs))
    params = model.init({"params": jax.random.PRNGKey(0), "noise": jax.random.PRNGKey(0)}, dummy_x, dummy_e)["params"]
    with open(path, "rb") as f:
        params = serialization.from_bytes(params, f.read())
    Bj = jnp.asarray(B, dtype=jnp.float32)

    def fwd(p, x):
        e = x[:, :cs] @ Bj
        e = (e - jnp.mean(e, axis=1, keepdims=True)) / (jnp.std(e, axis=1, keepdims=True) + 1e-6)
        return model.apply({"params": p}, x, e, training=False)[:2]

    return params, jax.jit(fwd)


def collect_users(raw_csv, metrics_csv, ix_by_id, indices, starts, lengths,
                  holdout_rows, n_users, min_ptw, max_ptw, rng):
    huge = set()
    with open(metrics_csv, newline="") as f:
        reader = csv.DictReader(f)
        for rec in reader:
            if float(rec["n_entries"]) > 2000:
                huge.add(rec["username"])
    print(f"huge lists: {len(huge):,}", flush=True)

    users, row, mismatches, scanned = [], 0, 0, 0

    def handle(rows):
        nonlocal row, mismatches
        model_ix, ptw_ix = [], []
        for aid, score, status in rows:
            ix = ix_by_id.get(aid)
            if ix is None:
                continue
            if status == "plan_to_watch" and not score > 0:
                ptw_ix.append(ix)
                continue
            if status == "on_hold" and score == 0:
                continue
            model_ix.append(ix)
        if len(model_ix) < 20:
            return
        r = row
        row += 1
        if r not in holdout_rows or len(users) >= n_users:
            return
        s0, l = starts[r], lengths[r]
        if not np.array_equal(np.asarray(model_ix, dtype=np.int32), indices[s0:s0 + l]):
            mismatches += 1
            return
        ptw = np.unique(np.asarray(ptw_ix, dtype=np.int32))
        ptw = ptw[~np.isin(ptw, indices[s0:s0 + l])]
        if len(ptw) < min_ptw:
            return
        if len(ptw) > max_ptw:
            ptw = rng.choice(ptw, size=max_ptw, replace=False)
        users.append((r, ptw))

    with gzip.open(raw_csv, "rt", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        cur, rows = None, []
        for rec in reader:
            if rec[0] != cur:
                if cur is not None:
                    scanned += 1
                    if cur not in huge:
                        handle(rows)
                    if len(users) >= n_users:
                        break
                    if scanned % 100_000 == 0:
                        print(f"scanned {scanned:,} users, collected {len(users):,}", flush=True)
                cur, rows = rec[0], []
            rows.append((int(rec[1]), float(rec[2]), rec[3]))
        else:
            if cur is not None and cur not in huge:
                handle(rows)
    print(f"collected {len(users):,} eval users ({scanned:,} scanned, {mismatches} row mismatches)", flush=True)
    return users


class Agg:
    def __init__(self):
        self.ranks, self.tiers = [], []

    def add(self, ranks, tiers):
        self.ranks.extend(ranks.tolist())
        self.tiers.extend(tiers.tolist())

    def stats(self):
        rk, tt = np.asarray(self.ranks), np.asarray(self.tiers)
        s = {"n": len(rk), "median_rank": float(np.median(rk)),
             "p25_rank": float(np.percentile(rk, 25)), "p75_rank": float(np.percentile(rk, 75)),
             "r10": float((rk < 10).mean()), "r50": float((rk < 50).mean()),
             "r250": float((rk < 250).mean())}
        for t, (lo, hi) in enumerate(TIERS):
            m = tt == t
            s[f"r250_tier{lo}_{hi}"] = float((rk[m] < 250).mean()) if m.any() else None
            s[f"n_tier{lo}_{hi}"] = int(m.sum())
        return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nn-weights", default="../data/aug2026/jax_model_fresh_logq.msgpack")
    ap.add_argument("--concat", default="../data/aug2026/probe/probe_graft_concat.msgpack")
    ap.add_argument("--ease-b", default="../data/aug2026/ease_B6k_lam200.npy")
    ap.add_argument("--vectors", default="../data/aug2026/user_input_vectors_cleanup_notrust.npz")
    ap.add_argument("--raw-csv", default="../data/collected_animelists_aug2026.csv.gz")
    ap.add_argument("--metrics-csv", default="../data/aug2026-profile-metrics.csv")
    ap.add_argument("--corpus", default="../data/corpus_ids_aug2026.json")
    ap.add_argument("--metadata", default="../data/processed-metadata_aug2026.csv")
    ap.add_argument("--n-users", type=int, default=4000)
    ap.add_argument("--min-ptw", type=int, default=3)
    ap.add_argument("--max-ptw", type=int, default=25)
    ap.add_argument("--logit-weights", default="0.0,0.3,1.0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    lws = [float(w) for w in args.logit_weights.split(",")]
    cs = CONF["corpus_size"]

    with open(args.corpus) as f:
        corpus_ids = json.load(f)
    ix_by_id = {aid: i for i, aid in enumerate(corpus_ids)}

    d = np.load(args.vectors)
    indices = d["indices"].astype(np.int32)
    values = d["values"]
    lengths = d["lengths"].astype(np.int64)
    starts = np.zeros(len(lengths), dtype=np.int64)
    np.cumsum(lengths[:-1], out=starts[1:])

    counts = np.bincount(indices, minlength=cs).astype(np.float64)
    log_pop = np.log(np.maximum(counts, 1.0) / np.maximum(counts, 1.0).sum())
    rank_of_item = np.empty(cs, dtype=np.int32)
    rank_of_item[np.argsort(-counts)] = np.arange(cs)
    tier_of_item = np.zeros(cs, dtype=np.int8)
    for t, (lo, hi) in enumerate(TIERS):
        tier_of_item[(rank_of_item >= lo) & (rank_of_item < hi)] = t

    dsu = build_components(args.metadata)
    comp_of_corpus = np.array([dsu.find(int(a)) for a in np.array(corpus_ids, dtype=np.int64)], dtype=np.int64)

    rng_h = np.random.default_rng(HOLDOUT_SEED)
    perm = rng_h.permutation(len(lengths))
    holdout_rows = set(perm[: len(lengths) // 10].tolist())

    users = collect_users(args.raw_csv, args.metrics_csv, ix_by_id, indices, starts, lengths,
                          holdout_rows, args.n_users, args.min_ptw, args.max_ptw,
                          np.random.default_rng(777))

    B = np.load(args.ease_b)
    models = {"prod": load_prod2(args.nn_weights, cs),
              "hybrid": load_graft2(args.concat, "concat", cs, B)}

    SCORERS = list(models) + ["pop"]
    agg = {s: {f"{w}": (Agg(), Agg()) for w in (lws if s != "pop" else [1.0])} for s in SCORERS}
    tgt_franch, tgt_tier_counts, n_ptw_all = [], np.zeros(len(TIERS), dtype=np.int64), []

    def rank_map(sc):
        o = np.argsort(-sc)
        ro = np.empty(cs, dtype=np.int32)
        ro[o] = np.arange(cs)
        return ro

    bx = np.zeros((BATCH, cs * 2), dtype=np.float32)
    metas = []

    def flush(nb):
        outs = {}
        xj = jnp.asarray(bx[:nb])
        for mname, (params, fwd) in models.items():
            lg, rt = fwd(params, xj)
            outs[mname] = (np.asarray(lg, dtype=np.float64), np.asarray(rt, dtype=np.float64))
        for j in range(nb):
            kept, ptw = metas[j]
            cand_franch = np.isin(comp_of_corpus, np.unique(comp_of_corpus[kept]))
            tf = cand_franch[ptw]
            tgt_franch.append(float(tf.mean()))
            for t in range(len(TIERS)):
                tgt_tier_counts[t] += int((tier_of_item[ptw] == t).sum())
            n_ptw_all.append(len(ptw))
            tt = tier_of_item[ptw]
            for sname in SCORERS:
                if sname == "pop":
                    scores = {"1.0": log_pop.copy()}
                else:
                    lg, rt = outs[sname][0][j], outs[sname][1][j]
                    s = lg + log_pop
                    s -= s.max()
                    logp = s - np.log(np.exp(s).sum())
                    logr = np.log(np.maximum(rt + 1.0, 1e-3))
                    scores = {f"{w}": w * logp + (1 - w) * logr for w in lws}
                for wkey, sc in scores.items():
                    sc = sc.copy()
                    sc[kept] = -np.inf
                    au, af = agg[sname][wkey]
                    ro = rank_map(sc)
                    au.add(ro[ptw], tt)
                    scf = sc.copy()
                    scf[cand_franch] = -np.inf
                    rof = rank_map(scf)
                    ok = ~tf
                    if ok.any():
                        af.add(rof[ptw[ok]], tt[ok])
        metas.clear()

    nb = 0
    for i, (r, ptw) in enumerate(users):
        s0, l = starts[r], lengths[r]
        kept = indices[s0:s0 + l]
        bx[nb] = 0.0
        bx[nb, kept] = 1.0
        bx[nb, cs + kept] = values[s0:s0 + l]
        metas.append((kept, ptw))
        nb += 1
        if nb == BATCH:
            flush(nb)
            nb = 0
            if (i + 1) % 1024 == 0:
                print(f"{i + 1:,}/{len(users):,} users scored", flush=True)
    if nb:
        flush(nb)

    results = {s: {w: {"unfiltered": a.stats(), "filtered": f.stats()}
                   for w, (a, f) in agg[s].items()} for s in SCORERS}
    out = {"n_users": len(users), "mean_ptw_per_user": float(np.mean(n_ptw_all)),
           "target_franchise_share_mean": float(np.mean(tgt_franch)),
           "target_tier_counts": {f"{lo}_{hi}": int(tgt_tier_counts[t]) for t, (lo, hi) in enumerate(TIERS)},
           "note": "targets = unrated in-corpus PTW of holdout users, capped at "
                   f"{args.max_ptw}/user; serve = softmax(lift + log_pop) neutral knob; "
                   "combined score in log space, each model's own rating head",
           "scorers": results}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    for s in SCORERS:
        for w, r in results[s].items():
            u, fl = r["unfiltered"], r["filtered"]
            print(f"{s} lw={w}: med {u['median_rank']:.0f} r50 {u['r50']:.3f} r250 {u['r250']:.3f} "
                  f"| filt med {fl['median_rank']:.0f} r250 {fl['r250']:.3f} "
                  f"mid {fl['r250_tier1000_3000']} tail {fl['r250_tier3000_6000']}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
