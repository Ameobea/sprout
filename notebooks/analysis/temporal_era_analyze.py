"""Era-stratified error analysis on an existing floors dump: recovers each
drop row's within-user temporal rank (upd_sec), then measures (a) trusted MAE
and signed bias by target-era quintile, (b) per-user half-vs-half drift of
model errors vs the same drift of item-mean residuals (how much taste drift
the model absorbs), each against permutation nulls. Run on host venv."""

import argparse
import json

import numpy as np
from scipy.stats import rankdata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--temporal", default="/home/casey/anime-atlas/data/aug2026/temporal_upd_sec.npz")
    ap.add_argument("--vectors", default="/home/casey/anime-atlas/data/aug2026/user_input_vectors_cleanup_notrust.npz")
    ap.add_argument("--census", default="/home/casey/anime-atlas/data/aug2026/rating_census.npz")
    ap.add_argument("--prior", default="/home/casey/anime-atlas/data/aug2026/rating_item_prior_lam50.npy")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    vec = np.load(args.vectors)
    indices, lengths = vec["indices"], vec["lengths"].astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)])
    with np.load(args.temporal) as tz:
        raw = tz["upd_sec"]
    prior = np.load(args.prior).astype(np.float64)

    d = np.load(args.dump)
    holdout_rows = d["holdout_rows"]
    du, di, dt, dp = d["drop_user"], d["drop_item"], d["drop_tgt"].astype(np.float64), d["drop_pred"].astype(np.float64)

    cen = np.load(args.census)
    trusted = ~(cen["one_sitting"][holdout_rows] | cen["degenerate"][holdout_rows])

    era = np.full(len(du), np.nan)
    for uu in np.unique(du):
        row = holdout_rows[uu]
        s, e = starts[row], starts[row + 1]
        secs = raw[s:e]
        m = secs >= 0
        r = np.full(len(secs), np.nan)
        if m.sum() >= 2:
            r[m] = rankdata(secs[m], method="average") / m.sum()
        item_era = dict(zip(indices[s:e].astype(np.int64), r))
        sel = np.nonzero(du == uu)[0]
        for j in sel:
            era[j] = item_era[int(di[j])]

    ok = trusted[du] & ~np.isnan(era)
    err = dp - dt
    resid = dt - prior[di]

    out = {"n_rows": int(ok.sum())}
    edges = [0.2, 0.4, 0.6, 0.8]
    q = np.searchsorted(edges, era, side="right")
    out["by_era_quintile"] = [
        {"n": int((ok & (q == k)).sum()),
         "mae": float(np.abs(err[ok & (q == k)]).mean()),
         "bias": float(err[ok & (q == k)].mean())}
        for k in range(5)
    ]

    rng = np.random.default_rng(321)
    drift_model, drift_model_null, drift_taste, drift_taste_null = [], [], [], []
    order = np.argsort(du, kind="stable")
    duo = du[order]
    ustarts = np.searchsorted(duo, np.arange(len(holdout_rows)))
    uends = np.searchsorted(duo, np.arange(len(holdout_rows)), side="right")
    for uu in range(len(holdout_rows)):
        if not trusted[uu]:
            continue
        rows = order[ustarts[uu]:uends[uu]]
        rows = rows[~np.isnan(era[rows])]
        if len(rows) < 10:
            continue
        o = rows[np.argsort(era[rows], kind="stable")]
        half = len(o) // 2
        drift_model.append(abs(err[o[:half]].mean() - err[o[half:]].mean()))
        drift_taste.append(abs(resid[o[:half]].mean() - resid[o[half:]].mean()))
        po = rng.permutation(o)
        drift_model_null.append(abs(err[po[:half]].mean() - err[po[half:]].mean()))
        drift_taste_null.append(abs(resid[po[:half]].mean() - resid[po[half:]].mean()))

    def stats(x):
        x = np.array(x)
        return {"mean": float(x.mean()), "p90": float(np.percentile(x, 90))}

    out["n_drift_users"] = len(drift_model)
    out["model_err_drift"] = stats(drift_model)
    out["model_err_drift_null"] = stats(drift_model_null)
    out["taste_drift"] = stats(drift_taste)
    out["taste_drift_null"] = stats(drift_taste_null)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
