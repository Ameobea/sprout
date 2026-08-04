"""Additive baselines for the rating head: item-mean + leave-one-out user offset.
Separates per-user calibration from genuine taste-matching in the model's win."""

import argparse
import json

import numpy as np


def huber(err, delta=1.0):
    a = np.abs(err)
    return np.where(a <= delta, 0.5 * a * a, delta * (a - 0.5 * delta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", default="../../data/aug2026/user_input_vectors_cleanup_notrust.npz")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = np.load(args.vectors)
    indices = d["indices"].astype(np.int32)
    values = d["values"].astype(np.float64)
    lengths = d["lengths"].astype(np.int64)
    rated = np.unpackbits(d["rated_masks"])[: int(d["total_mask_bits"][0])].astype(bool)
    n_users = len(lengths)

    rng = np.random.default_rng(0)
    test_users = rng.random(n_users) < 0.05
    user_of_entry = np.repeat(np.arange(n_users), lengths)
    test_entry = test_users[user_of_entry] & rated
    train_entry = ~test_users[user_of_entry] & rated

    it, zt = indices[train_entry], values[train_entry]
    sums = np.bincount(it, weights=zt, minlength=6000)
    cnts = np.bincount(it, minlength=6000).astype(np.float64)
    item_mean = sums / (cnts + 50.0)

    iv, zv = indices[test_entry], values[test_entry]
    uv = user_of_entry[test_entry]

    out = {}
    resid_item = zv - item_mean[iv]

    # LOO user mean of raw z (user-mean-only baseline)
    uz_sum = np.bincount(uv, weights=zv, minlength=n_users)
    uz_cnt = np.bincount(uv, minlength=n_users).astype(np.float64)
    m = uz_cnt[uv] > 1
    loo_umean = (uz_sum[uv] - zv) / np.maximum(uz_cnt[uv] - 1, 1)
    err_um = (zv - loo_umean)[m]
    out["usermean_mae"] = float(np.abs(err_um).mean())
    out["usermean_huber"] = float(huber(err_um).mean())

    # additive: item mean + shrunk LOO user offset on residuals
    ur_sum = np.bincount(uv, weights=resid_item, minlength=n_users)
    best = None
    for lam_u in [2.0, 5.0, 10.0, 20.0, 40.0]:
        loo_n = np.maximum(uz_cnt[uv] - 1, 1)
        loo_off = (ur_sum[uv] - resid_item) / loo_n * (loo_n / (loo_n + lam_u))
        err = (resid_item - loo_off)[m]
        mae = float(np.abs(err).mean())
        rec = {"lam_u": lam_u, "mae": mae, "huber": float(huber(err).mean()),
               "var_resid": float(err.var())}
        if best is None or mae < best["mae"]:
            best = rec
    out["additive_best"] = best
    out["itemmean_mae"] = float(np.abs(resid_item[m]).mean())
    out["itemmean_huber"] = float(huber(resid_item[m]).mean())
    out["n_test_entries"] = int(m.sum())

    # per popularity tier for the additive model
    counts_all = np.bincount(indices, minlength=6000).astype(np.float64)
    order = np.argsort(-counts_all)
    rank_of_item = np.empty(6000, dtype=np.int32)
    rank_of_item[order] = np.arange(6000)
    lam_u = best["lam_u"]
    loo_n = np.maximum(uz_cnt[uv] - 1, 1)
    loo_off = (ur_sum[uv] - resid_item) / loo_n * (loo_n / (loo_n + lam_u))
    err_full = resid_item - loo_off
    tiers = []
    ir = rank_of_item[iv]
    for lo, hi in [(0, 250), (250, 1000), (1000, 3000), (3000, 6000)]:
        mm = m & (ir >= lo) & (ir < hi)
        tiers.append({"tier": f"{lo}-{hi}", "mae": float(np.abs(err_full[mm]).mean())})
    out["additive_by_tier"] = tiers

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
