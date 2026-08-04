"""Closed-form item-similarity rating predictor vs the NN rating head.

Predicts a hidden item's z-rating as the gram-cosine-weighted average of the user's
KEPT rated items' z-ratings (optionally top-K neighbors, shrunk toward the item mean).
Same corruption protocol as loss_decomposition (rate 0.4 +/- 40%, seed 123 users) so
MAE is directly comparable to the NN rating head's dropped-item 0.447 and the ladder
(item-mean 0.536, additive 0.502, floor ~0.25-0.30).

Run inside rocm_jax: cd /jax_dir/notebooks && python analysis/rating_knn_probe.py
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", default="../data/aug2026/user_input_vectors_cleanup_notrust.npz")
    ap.add_argument("--gram", default="../data/aug2026/gram6k_aug2026.npz")
    ap.add_argument("--n-users", type=int, default=20000)
    ap.add_argument("--rating-gram", default="../data/aug2026/rating_gram6k.npz")
    ap.add_argument("--fixed-drop", type=float, default=None)
    ap.add_argument("--out", default="../data/aug2026/rating_knn_probe.json")
    args = ap.parse_args()

    d = np.load(args.vectors)
    indices = d["indices"].astype(np.int32)
    values = d["values"]
    lengths = d["lengths"].astype(np.int64)
    starts = np.zeros(len(lengths), dtype=np.int64)
    np.cumsum(lengths[:-1], out=starts[1:])
    packed = d["rated_masks"]
    bits = np.unpackbits(packed)[: int(d["total_mask_bits"][0])].astype(bool)

    G = np.load(args.gram)["G"].astype(np.float64)
    dg = np.sqrt(np.maximum(np.diag(G), 1e-9))
    cos = G / dg[None, :] / dg[:, None]
    np.fill_diagonal(cos, 0.0)

    rg_path = Path(args.rating_gram)
    if rg_path.exists():
        z = np.load(rg_path)
        C, N = z["C"].astype(np.float64), z["N"].astype(np.float64)
    else:
        print("building rating gram (excl. eval users)...", flush=True)
        from scipy import sparse
        rng_ex = np.random.default_rng(123)
        excl = set(rng_ex.choice(len(lengths), size=args.n_users, replace=False).tolist())
        gid = np.repeat(np.arange(len(lengths)), lengths)
        rb = bits[: len(indices)]
        m = rb & ~np.isin(gid, np.fromiter(excl, dtype=np.int64))
        X = sparse.csr_matrix((values[m], (gid[m], indices[m])), shape=(len(lengths), 6000), dtype=np.float32)
        Xb = X.copy(); Xb.data = np.ones_like(Xb.data)
        C = np.asarray((X.T @ X).todense(), dtype=np.float64)
        N = np.asarray((Xb.T @ Xb).todense(), dtype=np.float64)
        np.savez_compressed(rg_path, C=C.astype(np.float32), N=N.astype(np.float32))
        print("rating gram built", flush=True)
    dc = np.sqrt(np.maximum(np.diag(C), 1e-9))
    rsim = C / dc[None, :] / dc[:, None]
    rsim *= N / (N + 25.0)
    np.fill_diagonal(rsim, 0.0)
    rsim = np.maximum(rsim, 0.0)

    counts = np.bincount(indices, minlength=6000).astype(np.float64)
    sums = np.zeros(6000)
    np.add.at(sums, indices[bits[: len(indices)]], values[bits[: len(indices)]])
    rated_counts = np.bincount(indices[bits[: len(indices)]], minlength=6000).astype(np.float64)
    item_mean = sums / np.maximum(rated_counts, 1.0)
    item_mean_shrunk = sums / (rated_counts + 50.0)

    rng = np.random.default_rng(123)
    users = rng.choice(len(lengths), size=args.n_users, replace=False)
    if args.fixed_drop is not None:
        rates = np.full(len(users), args.fixed_drop)
    else:
        rates = np.random.default_rng(321).uniform(0.24, 0.56, size=len(users))

    variants = {
        "cos_top20_shrunk": dict(sim="cos", topk=20, shrink=5.0),
        "rsim_all": dict(sim="rating", topk=None, shrink=0.0),
        "rsim_top20": dict(sim="rating", topk=20, shrink=0.0),
        "rsim_top20_shrunk": dict(sim="rating", topk=20, shrink=5.0),
    }
    agg = {v: [0.0, 0] for v in variants}
    agg_bases = {"zero": [0.0, 0], "item_mean": [0.0, 0], "nn_anchor_note": None}
    r = np.random.default_rng(777)

    for ui, u in enumerate(users):
        s0, l = starts[u], lengths[u]
        idx = indices[s0 : s0 + l]
        val = values[s0 : s0 + l]
        rm = bits[s0 : s0 + l]
        keep = r.random(l) > rates[ui]
        if keep.sum() == 0:
            keep[0] = True
        drop_r = (~keep) & rm
        if not drop_r.any():
            continue
        kept_r = keep & rm
        targets, tvals = idx[drop_r], val[drop_r]
        agg_bases["zero"][0] += np.abs(tvals).sum(); agg_bases["zero"][1] += len(tvals)
        agg_bases["item_mean"][0] += np.abs(tvals - item_mean_shrunk[targets]).sum()
        agg_bases["item_mean"][1] += len(tvals)
        if kept_r.sum() == 0:
            for v in variants:
                agg[v][0] += np.abs(tvals - item_mean_shrunk[targets]).sum()
                agg[v][1] += len(tvals)
            continue
        src_idx, src_val = idx[kept_r], val[kept_r]
        W_by = {"cos": cos[np.ix_(targets, src_idx)], "rating": rsim[np.ix_(targets, src_idx)]}
        for vname, cfg in variants.items():
            W = W_by[cfg["sim"]]
            Wv = W
            if cfg["topk"] and W.shape[1] > cfg["topk"]:
                thr = np.partition(W, -cfg["topk"], axis=1)[:, -cfg["topk"]][:, None]
                Wv = np.where(W >= thr, W, 0.0)
            num = Wv @ src_val + cfg["shrink"] * item_mean_shrunk[targets]
            den = Wv.sum(axis=1) + cfg["shrink"]
            pred = num / np.maximum(den, 1e-9)
            agg[vname][0] += np.abs(tvals - pred).sum()
            agg[vname][1] += len(tvals)
        if ui % 4000 == 0:
            print(f"{ui}/{len(users)}", flush=True)

    out = {"anchors": {"nn_dropped_mae": 0.447, "additive_item_user": 0.502,
                       "item_mean_ladder": 0.536, "aleatoric_floor": "0.25-0.30"}}
    for name, (s, c) in {**agg, **{k: v for k, v in agg_bases.items() if v}}.items():
        if isinstance(s, float):
            out[name] = s / max(c, 1)
            print(f"{name}: MAE {out[name]:.4f} (n={c})", flush=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
