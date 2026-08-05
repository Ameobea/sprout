"""Dated-vs-undated strata over a rating_floors_dump npz: joins each drop row
back to its profile entry via (user, item) to recover its temporal value, then
reports trusted-user MAE + within-user rho restricted to dated / undated rows,
for a probe dump and a reference dump (same corrupt protocol => same rows).
Run on host venv."""

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
    if len(a) < 3:
        return np.nan
    ra, rb = avgrank(a), avgrank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else np.nan


def user_rhos(du, dt, dp, sel, min_items=8):
    order = np.argsort(du[sel], kind="stable")
    u, t, p = du[sel][order], dt[sel][order], dp[sel][order]
    rhos = {}
    s = 0
    while s < len(u):
        e = s
        while e < len(u) and u[e] == u[s]:
            e += 1
        if e - s >= min_items:
            rhos[u[s]] = spearman(p[s:e], t[s:e])
        s = e
    return rhos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--ref-dump", required=True)
    ap.add_argument("--temporal", required=True)
    ap.add_argument("--source", choices=["start_day", "upd_sec"], required=True)
    ap.add_argument("--vectors", default="/home/casey/anime-atlas/data/aug2026/user_input_vectors_cleanup_notrust.npz")
    ap.add_argument("--census", default="/home/casey/anime-atlas/data/aug2026/rating_census.npz")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    vec = np.load(args.vectors)
    indices, lengths = vec["indices"], vec["lengths"].astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)])
    with np.load(args.temporal) as tz:
        raw = tz[args.source]
    missing = -32768 if args.source == "start_day" else -1

    d = np.load(args.dump)
    holdout_rows = d["holdout_rows"]
    du, di, dt, dp = d["drop_user"], d["drop_item"], d["drop_tgt"], d["drop_pred"].astype(np.float64)
    r = np.load(args.ref_dump)
    assert (r["holdout_rows"] == holdout_rows).all() and (r["drop_user"] == du).all() and (r["drop_item"] == di).all()
    rp = r["drop_pred"].astype(np.float64)

    dated = np.zeros(len(du), dtype=bool)
    for uu in np.unique(du):
        row = holdout_rows[uu]
        s, e = starts[row], starts[row + 1]
        item_dated = dict(zip(indices[s:e].astype(np.int64), raw[s:e] != missing))
        sel = np.nonzero(du == uu)[0]
        for j in sel:
            dated[j] = item_dated[int(di[j])]

    cen = np.load(args.census)
    trusted = ~(cen["one_sitting"][holdout_rows] | cen["degenerate"][holdout_rows])
    tr = trusted[du]

    out = {"source": args.source, "n_drop": int(len(du)), "dated_frac": float(dated.mean())}
    for tag, sel in (("dated", tr & dated), ("undated", tr & ~dated), ("all_trusted", tr)):
        e_probe = np.abs(dp[sel] - dt[sel]).mean()
        e_ref = np.abs(rp[sel] - dt[sel]).mean()
        rho_p = user_rhos(du, dt, dp, sel)
        rho_r = user_rhos(du, dt, rp, sel)
        common = set(rho_p) & set(rho_r)
        out[tag] = {
            "n": int(sel.sum()),
            "mae_probe": float(e_probe), "mae_ref": float(e_ref),
            "delta_mae": float(e_probe - e_ref),
            "rho_probe": float(np.nanmean([rho_p[u] for u in common])),
            "rho_ref": float(np.nanmean([rho_r[u] for u in common])),
            "n_rho_users": len(common),
        }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
