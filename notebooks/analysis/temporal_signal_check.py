"""Model-free gate for the temporal channel: on the 20k floors-holdout users,
does the item-mean residual of rated entries carry within-user temporal
structure? Reports per-user Spearman(residual, temporal rank) vs a permutation
null, per-user incremental R^2 of a linear-in-rank term, and first/second-half
drift of mean residual. Run on host venv."""

import argparse
import json

import numpy as np

HOLDOUT_SEED = 999


def avgrank(x):
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x), dtype=np.float64)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    return sums[inv] / cnt[inv]


def spearman(a, b):
    ra, rb = avgrank(a), avgrank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temporal", required=True)
    ap.add_argument("--source", choices=["start_day", "upd_sec"], required=True)
    ap.add_argument("--vectors", default="/home/casey/anime-atlas/data/aug2026/user_input_vectors_cleanup_notrust.npz")
    ap.add_argument("--prior", default="/home/casey/anime-atlas/data/aug2026/rating_item_prior_lam50.npy")
    ap.add_argument("--n-users", type=int, default=20_000)
    ap.add_argument("--min-dated-rated", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    vec = np.load(args.vectors)
    indices, values, lengths = vec["indices"], vec["values"], vec["lengths"].astype(np.int64)
    masks = np.unpackbits(vec["rated_masks"])[: int(vec["total_mask_bits"][0])].astype(bool)
    starts = np.concatenate([[0], np.cumsum(lengths)])
    prior = np.load(args.prior).astype(np.float64)

    with np.load(args.temporal) as tz:
        raw = tz[args.source]
    missing = -32768 if args.source == "start_day" else -1

    rng_h = np.random.default_rng(HOLDOUT_SEED)
    perm = rng_h.permutation(len(lengths))
    hold = perm[: len(lengths) // 10][: args.n_users]

    rng = np.random.default_rng(123)
    rhos, rhos_null, r2s, drifts, drifts_null, ns = [], [], [], [], [], []
    for u in hold:
        s, e = starts[u], starts[u + 1]
        rated = masks[s:e]
        days = raw[s:e]
        m = rated & (days != missing)
        if m.sum() < args.min_dated_rated:
            continue
        resid = values[s:e][m].astype(np.float64) - prior[indices[s:e][m]]
        t = avgrank(days[m].astype(np.float64))
        rho = spearman(resid, t)
        if np.isnan(rho):
            continue
        rhos.append(rho)
        rhos_null.append(spearman(resid, rng.permutation(t)))
        tc = t - t.mean()
        beta = (tc * (resid - resid.mean())).sum() / (tc * tc).sum()
        ss_res = ((resid - resid.mean() - beta * tc) ** 2).sum()
        ss_tot = ((resid - resid.mean()) ** 2).sum()
        r2s.append(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)
        half = m.sum() // 2
        o = np.argsort(t, kind="stable")
        drifts.append(abs(resid[o[:half]].mean() - resid[o[half:]].mean()))
        po = rng.permutation(len(o))
        drifts_null.append(abs(resid[po[:half]].mean() - resid[po[half:]].mean()))
        ns.append(int(m.sum()))

    rhos, rhos_null = np.array(rhos), np.array(rhos_null)
    r2s = np.array(r2s)
    drifts, drifts_null = np.array(drifts), np.array(drifts_null)
    out = {
        "source": args.source,
        "n_users_tested": len(rhos),
        "median_dated_rated": float(np.median(ns)),
        "abs_rho_mean": float(np.abs(rhos).mean()),
        "abs_rho_null_mean": float(np.abs(rhos_null).mean()),
        "rho_mean": float(rhos.mean()),
        "frac_abs_rho_gt_0.2": float((np.abs(rhos) > 0.2).mean()),
        "frac_abs_rho_null_gt_0.2": float((np.abs(rhos_null) > 0.2).mean()),
        "linear_r2_mean": float(r2s.mean()),
        "linear_r2_median": float(np.median(r2s)),
        "drift_mean": float(drifts.mean()),
        "drift_null_mean": float(drifts_null.mean()),
        "drift_p90": float(np.percentile(drifts, 90)),
        "drift_null_p90": float(np.percentile(drifts_null, 90)),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
