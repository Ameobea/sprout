"""Convert a ztarget-probe floors dump (z-scale predictions) to mixed units via
the per-user affine v = A_u*z + B_u so MAE strata are comparable with control.
Rho is unaffected (monotone per-user map)."""

import argparse

import numpy as np

SCORES = np.arange(1, 11, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--census", default="../../data/aug2026/rating_census.npz")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = dict(np.load(args.dump))
    hist = np.load(args.census)["hist"].astype(np.float64)[d["holdout_rows"]]
    rh = hist[:, 1:]
    n = np.maximum(rh.sum(1), 1)
    mu = (rh * SCORES).sum(1) / n
    sigma = np.sqrt((rh * (SCORES[None] - mu[:, None]) ** 2).sum(1) / n) + 1e-6
    alpha = np.clip(sigma / 2.6, 0.3, 0.8)
    A = (alpha + (1 - alpha) * sigma / 2.5).astype(np.float32)
    B = ((1 - alpha) * (mu - 5.5) / 2.5).astype(np.float32)

    for pk, uk in (("drop_pred", "drop_user"), ("kept_pred", "kept_user")):
        u = d[uk]
        d[pk] = A[u] * d[pk] + B[u]
    np.savez(args.out, **d)
    print(f"{args.out}: converted (A mean {A.mean():.3f}, B mean {B.mean():.3f})")


if __name__ == "__main__":
    main()
