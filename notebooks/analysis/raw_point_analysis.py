"""Raw-score-space (1-10 points) error analysis, before/after the serve stack:
control preds vs EASE-blend preds vs blend+era-slope-debias, inverted per user
through the alpha-mix affine (v = A_u*s + B_u; exact for unclipped targets) and
compared against exact reconstructed raw targets. Stratified by user raw-score
std sigma_u. Uses the seed-2 paired dumps. Run on host venv."""

import argparse
import json

import numpy as np
from scipy.stats import rankdata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-control", default="/home/casey/anime-atlas/data/aug2026/rating_floors_dump_seed2.npz")
    ap.add_argument("--dump-blend", default="/home/casey/anime-atlas/data/aug2026/rating_floors_dump_seed2_blend.npz")
    ap.add_argument("--vectors", default="/home/casey/anime-atlas/data/aug2026/user_input_vectors_cleanup_notrust.npz")
    ap.add_argument("--census", default="/home/casey/anime-atlas/data/aug2026/rating_census.npz")
    ap.add_argument("--raw-scores", default="/home/casey/anime-atlas/data/aug2026/raw_scores_recon.npy")
    ap.add_argument("--temporal", default="/home/casey/anime-atlas/data/aug2026/temporal_upd_sec.npz")
    ap.add_argument("--lam-b", type=float, default=30.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    vec = np.load(args.vectors)
    indices, lengths = vec["indices"], vec["lengths"].astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)])
    raw_scores = np.load(args.raw_scores)
    with np.load(args.temporal) as tz:
        secs_all = tz["upd_sec"]

    dc = np.load(args.dump_control)
    db = np.load(args.dump_blend)
    holdout_rows = dc["holdout_rows"]
    assert (db["holdout_rows"] == holdout_rows).all()
    du, di, dt = dc["drop_user"], dc["drop_item"], dc["drop_tgt"].astype(np.float64)
    assert (db["drop_user"] == du).all() and (db["drop_item"] == di).all()
    dp_c = dc["drop_pred"].astype(np.float64)
    dp_b = db["drop_pred"].astype(np.float64)
    ku, ki, kt = db["kept_user"], db["kept_item"], db["kept_tgt"].astype(np.float64)
    kp_b = db["kept_pred"].astype(np.float64)

    cen = np.load(args.census)
    hist = cen["hist"].astype(np.float64)[holdout_rows]
    trusted = ~(cen["one_sitting"][holdout_rows] | cen["degenerate"][holdout_rows])
    scores = np.arange(11, dtype=np.float64)
    n_rated = hist[:, 1:].sum(axis=1)
    mu_u = (hist[:, 1:] * scores[1:]).sum(axis=1) / np.maximum(n_rated, 1)
    var_u = (hist[:, 1:] * (scores[1:][None, :] - mu_u[:, None]) ** 2).sum(axis=1) / np.maximum(n_rated, 1)
    sigma_u = np.sqrt(var_u)
    alpha_u = np.clip(sigma_u / 2.6, 0.3, 0.8)
    sig_safe = np.maximum(sigma_u, 1e-6)
    A_u = alpha_u / sig_safe + (1 - alpha_u) / 2.5
    B_u = -(alpha_u * mu_u / sig_safe + (1 - alpha_u) * 5.5 / 2.5)

    n_users = len(holdout_rows)
    raw_tgt = np.full(len(du), -1.0)
    era_d = np.full(len(du), np.nan)
    era_k = np.full(len(ku), np.nan)
    for arrs in ((du, di, True), (ku, ki, False)):
        uarr, iarr, is_drop = arrs
        order = np.argsort(uarr, kind="stable")
        uo = uarr[order]
        us = np.searchsorted(uo, np.arange(n_users))
        ue = np.searchsorted(uo, np.arange(n_users), side="right")
        for uu in range(n_users):
            rows = order[us[uu]:ue[uu]]
            if not len(rows):
                continue
            prow = holdout_rows[uu]
            s, e = starts[prow], starts[prow + 1]
            secs = secs_all[s:e]
            m = secs >= 0
            r = np.full(len(secs), np.nan)
            if m.sum() >= 2:
                r[m] = rankdata(secs[m], method="average") / m.sum()
            pos = dict(zip(indices[s:e].astype(np.int64), range(s, e)))
            for j in rows:
                p = pos[int(iarr[j])]
                if is_drop:
                    raw_tgt[j] = raw_scores[p]
                    era_d[j] = r[p - s]
                else:
                    era_k[j] = r[p - s]

    kerr = kp_b - kt
    b_u = np.zeros(n_users)
    ek_mean = np.zeros(n_users)
    for uu in range(n_users):
        sel = np.nonzero((ku == uu) & ~np.isnan(era_k))[0]
        n = len(sel)
        if n < 3:
            continue
        e_ = era_k[sel]
        er = kerr[sel]
        ec = e_ - e_.mean()
        denom = (ec * ec).sum()
        if denom > 1e-9:
            b_u[uu] = (ec * er).sum() / denom * n / (n + args.lam_b)
        ek_mean[uu] = e_.mean()

    corr = b_u[du] * (era_d - ek_mean[du])
    dp_be = dp_b - np.where(np.isnan(corr), 0.0, corr)

    ok = trusted[du] & (raw_tgt >= 1) & (raw_tgt <= 10)
    Ad, Bd = A_u[du], B_u[du]

    def raw_pred(p, sel):
        return np.clip((p[sel] - Bd[sel]) / Ad[sel], 1.0, 10.0)

    bands = [(0, 0.8), (0.8, 1.2), (1.2, 1.6), (1.6, 2.2), (2.2, 99)]
    out = {"note": "raw-score points (1-10); trusted drop rows; seed-2 dumps", "bands": []}
    sb = sigma_u[du]
    for lo, hi in bands:
        sel = ok & (sb >= lo) & (sb < hi)
        rt = raw_tgt[sel]
        rec = {"sigma_band": f"[{lo},{hi})", "n_rows": int(sel.sum()),
               "n_users": int(len(np.unique(du[sel]))),
               "mean_sigma": float(sb[sel].mean())}
        for tag, p in (("control", dp_c), ("blend", dp_b), ("blend_era", dp_be)):
            rp = raw_pred(p, sel)
            rec[tag] = {"mae_pts": float(np.abs(rp - rt).mean()),
                        "within_1pt": float((np.abs(rp - rt) <= 1.0).mean()),
                        "bias_pts": float((rp - rt).mean())}
        out["bands"].append(rec)
    sel = ok
    rt = raw_tgt[sel]
    rec = {"sigma_band": "ALL TRUSTED", "n_rows": int(sel.sum()),
           "n_users": int(len(np.unique(du[sel]))), "mean_sigma": float(sb[sel].mean())}
    for tag, p in (("control", dp_c), ("blend", dp_b), ("blend_era", dp_be)):
        rp = raw_pred(p, sel)
        rec[tag] = {"mae_pts": float(np.abs(rp - rt).mean()),
                    "within_1pt": float((np.abs(rp - rt) <= 1.0).mean()),
                    "bias_pts": float((rp - rt).mean())}
    out["bands"].append(rec)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    for b in out["bands"]:
        print(f"{b['sigma_band']:>10}  n={b['n_rows']:>8,}  sig~{b['mean_sigma']:.2f}  "
              f"ctl {b['control']['mae_pts']:.3f}pts ({b['control']['within_1pt']:.1%} w/in 1)  "
              f"blend {b['blend']['mae_pts']:.3f}  +era {b['blend_era']['mae_pts']:.3f} "
              f"({b['blend_era']['within_1pt']:.1%} w/in 1)")


if __name__ == "__main__":
    main()
