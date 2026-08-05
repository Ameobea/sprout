"""Closed-form EASE on RATING RESIDUALS (vals - shrunk item mean, rated entries,
seed-999 train users only). Gate for the rating-decoder graft: does the residual
item-item signal, standalone or blended with the NN's dump predictions, move
trusted-user rho? Saves the best-lambda B for the graft probe."""

import argparse
import json

import numpy as np
from scipy import sparse

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
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", default="../../data/aug2026/user_input_vectors_cleanup_notrust.npz")
    ap.add_argument("--prior", default="../../data/aug2026/rating_item_prior_lam50.npy")
    ap.add_argument("--census", default="../../data/aug2026/rating_census.npz")
    ap.add_argument("--dump", default="../../data/aug2026/rating_floors_dump_control.npz")
    ap.add_argument("--lams", default="200,500,1000,3000")
    ap.add_argument("--gram-cache", default="../../data/aug2026/rating_resid_gram6k.npz")
    ap.add_argument("--b-out", default="../../data/aug2026/rating_resid_B6k.npy")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prior = np.load(args.prior).astype(np.float64)

    from pathlib import Path
    if Path(args.gram_cache).exists():
        C = np.load(args.gram_cache)["C"].astype(np.float64)
        print("gram loaded from cache", flush=True)
    else:
        d = np.load(args.vectors)
        indices = d["indices"].astype(np.int32)
        values = d["values"].astype(np.float32)
        lengths = d["lengths"].astype(np.int64)
        rated = np.unpackbits(d["rated_masks"])[: int(d["total_mask_bits"][0])].astype(bool)
        n_users = len(lengths)
        perm = np.random.default_rng(HOLDOUT_SEED).permutation(n_users)
        is_hold = np.zeros(n_users, dtype=bool)
        is_hold[perm[: n_users // 10]] = True
        gid = np.repeat(np.arange(n_users), lengths)
        m = rated & ~is_hold[gid]
        resid = (values[m].astype(np.float64) - prior[indices[m]]).astype(np.float32)
        print(f"train rated entries {m.sum():,}", flush=True)
        X = sparse.csr_matrix((resid, (gid[m], indices[m])), shape=(n_users, 6000), dtype=np.float32)
        C = np.asarray((X.T @ X).todense(), dtype=np.float64)
        np.savez_compressed(args.gram_cache, C=C.astype(np.float32))
        print("gram built + cached", flush=True)

    dmp = np.load(args.dump)
    holdout_rows = dmp["holdout_rows"]
    du, di, dt = dmp["drop_user"], dmp["drop_item"], dmp["drop_tgt"].astype(np.float64)
    dp_nn = dmp["drop_pred"].astype(np.float64)
    ku, ki, kt = dmp["kept_user"], dmp["kept_item"], dmp["kept_tgt"].astype(np.float64)
    n_users = len(holdout_rows)

    K = sparse.csr_matrix(((kt - prior[ki]).astype(np.float32), (ku, ki)),
                          shape=(n_users, 6000), dtype=np.float32)

    cen = np.load(args.census)
    trusted_u = ~(cen["one_sitting"] | cen["degenerate"])[holdout_rows]
    trow = trusted_u[du]

    order = np.argsort(du, kind="stable")
    starts = np.searchsorted(du[order], np.arange(n_users))
    ends = np.searchsorted(du[order], np.arange(n_users), side="right")

    def trusted_metrics(pred):
        e = pred[trow] - dt[trow]
        po, to = pred[order], dt[order]
        rhos = []
        for u in range(n_users):
            s, ee = starts[u], ends[u]
            if ee - s >= 8 and trusted_u[u]:
                rhos.append(spearman(po[s:ee], to[s:ee]))
        return {"mae_trusted": float(np.abs(e).mean()), "rho_trusted": float(np.nanmean(rhos))}

    out = {"nn": trusted_metrics(dp_nn)}
    print("nn:", out["nn"], flush=True)

    I = np.eye(6000)
    best = (None, -np.inf, None)
    for lam in [float(x) for x in args.lams.split(",")]:
        P = np.linalg.inv(C + lam * I)
        B = -P / np.diag(P)[None, :]
        np.fill_diagonal(B, 0.0)
        S = np.asarray(K @ B.astype(np.float32), dtype=np.float64)
        pr = S[du, di]
        rec = {"standalone": trusted_metrics(prior[di] + pr)}
        cc = np.corrcoef(pr[trow], (dt - dp_nn)[trow])[0, 1]
        rec["resid_corr_with_nn_error"] = float(cc)
        for w in (0.25, 0.5, 1.0):
            rec[f"blend_w{w}"] = trusted_metrics(dp_nn + w * pr)
        out[f"lam{lam:g}"] = rec
        rho_b = max(rec[f"blend_w{w}"]["rho_trusted"] for w in (0.25, 0.5, 1.0))
        print(f"lam{lam:g}: standalone {rec['standalone']}  corr(pred, nn_err) {cc:.4f}  best blend rho {rho_b:.4f}", flush=True)
        if rho_b > best[1]:
            best = (lam, rho_b, B.astype(np.float32))

    np.save(args.b_out, best[2])
    out["best_lam"] = best[0]
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved B (lam {best[0]:g}) -> {args.b_out}", flush=True)


if __name__ == "__main__":
    main()
