"""Data-level information floors + baselines for the presence/rating losses.

Presence loss floor: per-user softmax over corpus must spread mass over the n_u
target items, so per-item NLL >= log(n_u) even for a perfect model. Popularity
NLL gives the zero-personalization reference. Rating baselines: predict-0
(= user mean) and shrunk per-item means, per-entry and per-user weighted to
match the training loss.
"""

import argparse
import json

import numpy as np


def huber(err, delta=1.0):
    a = np.abs(err)
    return np.where(a <= delta, 0.5 * a * a, delta * (a - 0.5 * delta))


def seg_mean(vals, starts, lengths):
    sums = np.add.reduceat(vals, starts)
    return sums / lengths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", default="../../data/aug2026/user_input_vectors_cleanup_notrust.npz")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = np.load(args.vectors)
    indices = d["indices"].astype(np.int32)
    values = d["values"]
    lengths = d["lengths"].astype(np.int64)
    packed = d["rated_masks"]
    total_bits = int(d["total_mask_bits"][0])
    rated = np.unpackbits(packed)[:total_bits].astype(bool)

    n_users = len(lengths)
    starts = np.zeros(n_users, dtype=np.int64)
    np.cumsum(lengths[:-1], out=starts[1:])

    out = {"n_users": n_users, "n_entries": int(lengths.sum())}

    # ---- presence floors ----
    log_n = np.log(lengths)
    out["presence_floor_E_log_n"] = float(log_n.mean())
    out["presence_floor_log_n_quantiles"] = {
        str(q): float(np.quantile(log_n, q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9]
    }
    hist, edges = np.histogram(lengths, bins=np.geomspace(5, 2000, 60))
    out["profile_size_hist"] = {"edges": edges.tolist(), "counts": hist.tolist()}
    out["profile_size_stats"] = {
        "mean": float(lengths.mean()), "median": float(np.median(lengths)),
        "p10": float(np.percentile(lengths, 10)), "p90": float(np.percentile(lengths, 90)),
    }

    counts = np.bincount(indices, minlength=6000).astype(np.float64)
    p_pop = np.maximum(counts, 1.0) / np.maximum(counts, 1.0).sum()
    out["popularity_entropy_nats"] = float(-(p_pop * np.log(p_pop)).sum())
    out["log_uniform_6000"] = float(np.log(6000.0))

    nll_pop_entry = -np.log(p_pop)[indices]
    out["presence_pop_nll_per_entry"] = float(nll_pop_entry.mean())
    per_user_pop = seg_mean(nll_pop_entry, starts, lengths)
    out["presence_pop_nll_per_user"] = float(per_user_pop.mean())
    del nll_pop_entry, per_user_pop

    # popularity NLL by popularity tier of the target item (rank by count)
    order = np.argsort(-counts)
    rank_of_item = np.empty(6000, dtype=np.int32)
    rank_of_item[order] = np.arange(6000)
    tiers = [(0, 250), (250, 1000), (1000, 3000), (3000, 6000)]
    item_rank_per_entry = rank_of_item[indices]
    tier_stats = []
    for lo, hi in tiers:
        m = (item_rank_per_entry >= lo) & (item_rank_per_entry < hi)
        tier_stats.append({
            "tier": f"{lo}-{hi}", "frac_entries": float(m.mean()),
            "pop_nll": float((-np.log(p_pop)[indices[m]]).mean()),
        })
    out["pop_nll_by_tier"] = tier_stats

    # ---- rating baselines (rated entries only) ----
    z = values[rated].astype(np.float64)
    zi = indices[rated]
    out["rated_frac"] = float(rated.mean())
    out["z_stats"] = {
        "mean": float(z.mean()), "std": float(z.std()),
        "mean_abs": float(np.abs(z).mean()),
    }
    zh, ze = np.histogram(z, bins=np.linspace(-4, 3, 71))
    out["z_hist"] = {"edges": ze.tolist(), "counts": zh.tolist()}

    out["rating_zero_mae_per_entry"] = float(np.abs(z).mean())
    out["rating_zero_huber_per_entry"] = float(huber(z).mean())

    # per-user weighted zero baseline (matches training loss weighting)
    rated_len = seg_mean(rated.astype(np.float64), starts, lengths) * lengths
    ustarts_valid = rated_len > 0
    abs_entry = np.abs(values.astype(np.float64)) * rated
    hub_entry = huber(values.astype(np.float64)) * rated
    sum_abs = np.add.reduceat(abs_entry, starts)
    sum_hub = np.add.reduceat(hub_entry, starts)
    denom = np.maximum(rated_len, 1.0)
    out["rating_zero_mae_per_user"] = float((sum_abs / denom)[ustarts_valid].mean())
    out["rating_zero_huber_per_user"] = float((sum_hub / denom)[ustarts_valid].mean())
    del abs_entry, hub_entry

    # shrunk item means, honest 95/5 user split
    rng = np.random.default_rng(0)
    test_users = rng.random(n_users) < 0.05
    test_entry = np.repeat(test_users, lengths) & rated
    train_entry = ~np.repeat(test_users, lengths) & rated

    zt = values[train_entry].astype(np.float64)
    it = indices[train_entry]
    sums = np.bincount(it, weights=zt, minlength=6000)
    cnts = np.bincount(it, minlength=6000).astype(np.float64)
    lam = 50.0
    item_mean = sums / (cnts + lam)

    zv = values[test_entry].astype(np.float64)
    iv = indices[test_entry]
    resid = zv - item_mean[iv]
    out["rating_itemmean_mae_per_entry"] = float(np.abs(resid).mean())
    out["rating_itemmean_huber_per_entry"] = float(huber(resid).mean())

    # variance decomposition on test entries
    out["rating_var_total"] = float(zv.var())
    out["rating_var_after_itemmean"] = float(resid.var())
    out["rating_itemmean_r2"] = float(1.0 - resid.var() / zv.var())

    # item-mean spread (how much "shows are just good/bad" there is)
    out["item_mean_std_weighted"] = float(
        np.sqrt(np.average((item_mean - np.average(item_mean, weights=cnts)) ** 2, weights=cnts))
    )

    # per-item residual variance by popularity tier
    tier_r = []
    iv_rank = rank_of_item[iv]
    for lo, hi in tiers:
        m = (iv_rank >= lo) & (iv_rank < hi)
        tier_r.append({
            "tier": f"{lo}-{hi}", "frac_rated_entries": float(m.mean()),
            "resid_mae": float(np.abs(resid[m]).mean()),
            "resid_var": float(resid[m].var()),
        })
    out["itemmean_resid_by_tier"] = tier_r

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
