"""Shrunk item-mean rating prior from train users (probe holdout excluded).
prior_j = (sum_j + lam*gm) / (cnt_j + lam) over rated entries, gm = global rated mean."""

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", default="../data/aug2026/user_input_vectors_cleanup_notrust.npz")
    ap.add_argument("--holdout", default="../data/aug2026/probe/holdout_users.npy")
    ap.add_argument("--lam", type=float, default=50.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = np.load(args.vectors)
    idx = d["indices"].astype(np.int32)
    vals = d["values"].astype(np.float64)
    lengths = d["lengths"].astype(np.int64)
    rated = np.unpackbits(d["rated_masks"])[: int(d["total_mask_bits"][0])].astype(bool)

    holdout = np.load(args.holdout)
    is_holdout = np.zeros(len(lengths), dtype=bool)
    is_holdout[holdout] = True
    user_of_entry = np.repeat(np.arange(len(lengths)), lengths)

    m = rated & ~is_holdout[user_of_entry]
    it, zt = idx[m], vals[m]
    gm = zt.mean()
    sums = np.bincount(it, weights=zt, minlength=6000)
    cnts = np.bincount(it, minlength=6000).astype(np.float64)
    prior = (sums + args.lam * gm) / (cnts + args.lam)

    np.save(args.out, prior.astype(np.float32))
    print(
        f"train rated entries {m.sum():,}  gm {gm:.4f}  "
        f"prior mean {prior.mean():.4f} std {prior.std():.4f} "
        f"min {prior.min():.4f} max {prior.max():.4f}  zero-count items {(cnts == 0).sum()}"
    )


if __name__ == "__main__":
    main()
