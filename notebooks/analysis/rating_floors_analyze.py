"""Stratified rating-floor analysis over a rating_floors_dump npz: closed-form
baselines vs model predictions, calibration-vs-ordering decomposition, and
strata by item popularity tier, user trust flags, profile size, sigma/alpha.
Pure numpy; run on host."""

import argparse
import json

import numpy as np

TIERS = [(0, 250), (250, 1000), (1000, 3000), (3000, 6000)]
LAM_U = 10.0


def huber(err, delta=1.0):
    a = np.abs(err)
    return np.where(a <= delta, 0.5 * a * a, delta * (a - 0.5 * delta))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--vectors", default="../../data/aug2026/user_input_vectors_cleanup_notrust.npz")
    ap.add_argument("--census", default="../../data/aug2026/rating_census.npz")
    ap.add_argument("--prior", default="../../data/aug2026/rating_item_prior_lam50.npy")
    ap.add_argument("--popularity", default="../../data/aug2026/item_popularity_aug2026_cleanup_notrust.npy")
    ap.add_argument("--min-rho-items", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = np.load(args.dump)
    holdout_rows = d["holdout_rows"]
    n_users = len(holdout_rows)
    du, di, dt, dp = d["drop_user"], d["drop_item"], d["drop_tgt"], d["drop_pred"].astype(np.float64)
    ku, ki, kt, kp = d["kept_user"], d["kept_item"], d["kept_tgt"], d["kept_pred"].astype(np.float64)

    prior = np.load(args.prior).astype(np.float64)
    gm = 0.4115

    pop = np.load(args.popularity)
    pop_rank = np.empty(len(pop), dtype=np.int64)
    pop_rank[np.argsort(-pop)] = np.arange(len(pop))
    item_tier = np.searchsorted([t[1] for t in TIERS[:-1]], pop_rank[di], side="right")

    vec = np.load(args.vectors)
    lengths = vec["lengths"].astype(np.int64)

    cen = np.load(args.census)
    hist = cen["hist"].astype(np.float64)[holdout_rows]
    scores = np.arange(11, dtype=np.float64)
    n_rated_u = hist[:, 1:].sum(axis=1)
    mu_u = (hist[:, 1:] * scores[1:]).sum(axis=1) / np.maximum(n_rated_u, 1)
    var_u = (hist[:, 1:] * (scores[1:][None, :] - mu_u[:, None]) ** 2).sum(axis=1) / np.maximum(n_rated_u, 1)
    sigma_u = np.sqrt(var_u)
    alpha_u = np.clip(sigma_u / 2.6, 0.3, 0.8)
    plen_u = lengths[holdout_rows]
    ratedfrac_u = n_rated_u / plen_u
    os_u = cen["one_sitting"][holdout_rows]
    dg_u = cen["degenerate"][holdout_rows]
    trust_u = np.where(dg_u, 2, np.where(os_u, 1, 0))

    kept_off_sum = np.bincount(ku, weights=kt - prior[ki], minlength=n_users)
    kept_cnt = np.bincount(ku, minlength=n_users).astype(np.float64)
    user_off = kept_off_sum / (kept_cnt + LAM_U)
    kept_model_bias = np.bincount(ku, weights=kp - kt, minlength=n_users) / np.maximum(kept_cnt, 1)

    preds = {
        "global_mean": np.full(len(dt), gm),
        "item_mean": prior[di],
        "additive_feasible": prior[di] + user_off[du],
        "model": dp,
        "model_debias_feasible": dp - kept_model_bias[du],
    }

    user_strata = {
        "trust": (trust_u, ["trusted", "one_sitting", "degenerate"]),
        "profile_size": (np.searchsorted([60, 150, 400], plen_u, side="right"),
                         ["20-60", "60-150", "150-400", "400+"]),
        "sigma": (np.searchsorted([0.7, 1.1, 1.6], sigma_u, side="right"),
                  ["<0.7", "0.7-1.1", "1.1-1.6", ">=1.6"]),
        "alpha": (np.searchsorted([0.301, 0.55, 0.799], alpha_u, side="right"),
                  ["0.3(clip)", "0.3-0.55", "0.55-0.8", "0.8(clip)"]),
        "rated_frac": (np.searchsorted([0.25, 0.5, 0.8], ratedfrac_u, side="right"),
                       ["<0.25", "0.25-0.5", "0.5-0.8", ">=0.8"]),
    }

    tgt64 = dt.astype(np.float64)
    out = {"n_users": n_users, "n_drop_rows": len(dt), "n_kept_rows": len(kt),
           "target_var": float(tgt64.var()), "strata_labels": {}}

    def agg(err, sel=None):
        e = err if sel is None else err[sel]
        return {"n": int(len(e)), "mae": float(np.abs(e).mean()) if len(e) else None,
                "huber": float(huber(e).mean()) if len(e) else None,
                "bias": float(e.mean()) if len(e) else None}

    for name, p in preds.items():
        err = p - tgt64
        rec = {"overall": agg(err), "by_tier": [agg(err, item_tier == t) for t in range(4)]}
        for sname, (svals, labels) in user_strata.items():
            rec[f"by_{sname}"] = [agg(err, svals[du] == k) for k in range(len(labels))]
            out["strata_labels"][sname] = labels
        out[name] = rec

    order = np.argsort(du, kind="stable")
    duo, dio, dto, bounds = du[order], di[order], tgt64[order], None
    starts = np.searchsorted(duo, np.arange(n_users))
    ends = np.searchsorted(duo, np.arange(n_users), side="right")

    for name in ("model", "additive_feasible", "item_mean"):
        po = preds[name][order]
        rhos, maes_or, biases = [], [], []
        rho_user = []
        for u in range(n_users):
            s, e = starts[u], ends[u]
            if e - s < args.min_rho_items:
                continue
            t, p = dto[s:e], po[s:e]
            rhos.append(spearman(p, t))
            biases.append(float((p - t).mean()))
            maes_or.append(float(np.abs((p - t) - (p - t).mean()).mean()))
            rho_user.append(u)
        rhos = np.array(rhos)
        rho_user = np.array(rho_user)
        rec = out[name]
        rec["ordering"] = {
            "n_users": int(len(rhos)),
            "rho_mean": float(np.nanmean(rhos)),
            "rho_median": float(np.nanmedian(rhos)),
            "oracle_debiased_mae": float(np.mean(maes_or)),
            "per_user_bias_std": float(np.std(biases)),
        }
        for sname, (svals, labels) in user_strata.items():
            sv = svals[rho_user]
            rec["ordering"][f"rho_by_{sname}"] = [
                {"n": int((sv == k).sum()),
                 "rho_mean": float(np.nanmean(rhos[sv == k])) if (sv == k).any() else None}
                for k in range(len(labels))
            ]

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k]["overall"] if isinstance(out[k], dict) and "overall" in out[k] else None
                      for k in preds}, indent=1))
    for name in ("item_mean", "additive_feasible", "model"):
        o = out[name].get("ordering")
        if o:
            print(name, "rho_mean", round(o["rho_mean"], 4), "oracle_debiased_mae", round(o["oracle_debiased_mae"], 4))


if __name__ == "__main__":
    main()
