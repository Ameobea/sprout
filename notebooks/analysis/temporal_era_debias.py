"""Serve-side era-extrapolated debias test on an existing floors dump: per
user, fit err ~ a + b*era on KEPT rows (visible at serve), shrink b by
n/(n+lam), then correct DROP-row predictions at each row's era. Judged on
trusted rows overall and on the late-era quintile (serve proxy: candidates
score at era~1). Compares no-correction / constant-a / era-linear, MAE and
within-user rho. Run on host venv."""

import argparse
import json

import numpy as np
from scipy.stats import rankdata


def avgrank(x):
    return rankdata(x, method="average")


def spearman(a, b):
    if len(a) < 3:
        return np.nan
    ra, rb = avgrank(a).astype(np.float64), avgrank(b).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--temporal", default="/home/casey/anime-atlas/data/aug2026/temporal_upd_sec.npz")
    ap.add_argument("--vectors", default="/home/casey/anime-atlas/data/aug2026/user_input_vectors_cleanup_notrust.npz")
    ap.add_argument("--census", default="/home/casey/anime-atlas/data/aug2026/rating_census.npz")
    ap.add_argument("--lam-a", type=float, default=10.0)
    ap.add_argument("--lam-b", type=float, default=30.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    vec = np.load(args.vectors)
    indices, lengths = vec["indices"], vec["lengths"].astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)])
    with np.load(args.temporal) as tz:
        raw = tz["upd_sec"]

    d = np.load(args.dump)
    holdout_rows = d["holdout_rows"]
    du, di, dt, dp = d["drop_user"], d["drop_item"], d["drop_tgt"].astype(np.float64), d["drop_pred"].astype(np.float64)
    ku, ki, kt, kp = d["kept_user"], d["kept_item"], d["kept_tgt"].astype(np.float64), d["kept_pred"].astype(np.float64)

    cen = np.load(args.census)
    trusted = ~(cen["one_sitting"][holdout_rows] | cen["degenerate"][holdout_rows])

    def eras_for(uarr, iarr):
        out = np.full(len(uarr), np.nan)
        order = np.argsort(uarr, kind="stable")
        uo = uarr[order]
        us = np.searchsorted(uo, np.arange(len(holdout_rows)))
        ue = np.searchsorted(uo, np.arange(len(holdout_rows)), side="right")
        for uu in range(len(holdout_rows)):
            rows = order[us[uu]:ue[uu]]
            if not len(rows):
                continue
            prow = holdout_rows[uu]
            s, e = starts[prow], starts[prow + 1]
            secs = raw[s:e]
            m = secs >= 0
            r = np.full(len(secs), np.nan)
            if m.sum() >= 2:
                r[m] = rankdata(secs[m], method="average") / m.sum()
            item_era = dict(zip(indices[s:e].astype(np.int64), r))
            for j in rows:
                out[j] = item_era[int(iarr[j])]
        return out

    era_d = eras_for(du, di)
    era_k = eras_for(ku, ki)

    n_users = len(holdout_rows)
    a_u = np.zeros(n_users)
    b_u = np.zeros(n_users)
    kerr = kp - kt
    for uu in range(n_users):
        sel = np.nonzero((ku == uu) & ~np.isnan(era_k))[0]
        n = len(sel)
        if n < 3:
            continue
        e_ = era_k[sel]
        er = kerr[sel]
        a = er.mean()
        ec = e_ - e_.mean()
        denom = (ec * ec).sum()
        b = (ec * er).sum() / denom if denom > 1e-9 else 0.0
        a_u[uu] = a * n / (n + args.lam_a)
        b_u[uu] = b * n / (n + args.lam_b)

    ok = trusted[du] & ~np.isnan(era_d)
    late = ok & (era_d > 0.8)
    ek_mean = np.zeros(n_users)
    for uu in np.unique(ku):
        sel = (ku == uu) & ~np.isnan(era_k)
        if sel.sum():
            ek_mean[uu] = era_k[sel].mean()

    preds = {
        "raw": dp,
        "const_debias": dp - a_u[du],
        "era_debias": dp - (a_u[du] + b_u[du] * (era_d - ek_mean[du])),
    }

    out = {"n_trusted": int(ok.sum()), "n_late": int(late.sum()),
           "lam_a": args.lam_a, "lam_b": args.lam_b}
    for name, p in preds.items():
        rec = {}
        for tag, sel in (("all", ok), ("late", late)):
            rec[tag] = {"mae": float(np.abs(p[sel] - dt[sel]).mean()),
                        "bias": float((p[sel] - dt[sel]).mean())}
        rhos = []
        order = np.argsort(du, kind="stable")
        uo = du[order]
        us = np.searchsorted(uo, np.arange(n_users))
        ue = np.searchsorted(uo, np.arange(n_users), side="right")
        for uu in range(n_users):
            if not trusted[uu]:
                continue
            rows = order[us[uu]:ue[uu]]
            rows = rows[~np.isnan(era_d[rows])]
            if len(rows) < 8:
                continue
            rhos.append(spearman(p[rows], dt[rows]))
        rec["rho_trusted"] = float(np.nanmean(rhos))
        out[name] = rec
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
