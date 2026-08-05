"""Rating-scheme census analysis over rating_census.npz: degenerate taxonomy,
k-means archetypes of the per-user score distribution, and the exact effect of
the normalize_ratings alpha-mix on each user's target spread (ratings only take
10 support values, so per-user target stats are computable in closed form)."""

import argparse
import json

import numpy as np

SCORES = np.arange(1, 11, dtype=np.float64)


def kmeans(x, k, iters=30, seed=0):
    rng = np.random.default_rng(seed)
    c = [x[rng.integers(len(x))]]
    for _ in range(k - 1):
        d2 = np.min(((x[:, None, :] - np.array(c)[None]) ** 2).sum(-1), axis=1)
        c.append(x[rng.choice(len(x), p=d2 / d2.sum())])
    c = np.array(c)
    for _ in range(iters):
        d2 = ((x[:, None, :] - c[None]) ** 2).sum(-1)
        a = d2.argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                c[j] = x[m].mean(0)
    return a, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default="../../data/aug2026/rating_census.npz")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--min-rated", type=int, default=10)
    ap.add_argument("--sample", type=int, default=400_000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--assign-out", required=True)
    args = ap.parse_args()

    cen = np.load(args.census)
    hist = cen["hist"].astype(np.float64)
    os_f, dg_f = cen["one_sitting"], cen["degenerate"]
    n_users = len(hist)
    rh = hist[:, 1:]
    n_rated = rh.sum(1)
    n_unrated = hist[:, 0]

    mu = (rh * SCORES).sum(1) / np.maximum(n_rated, 1)
    var = (rh * (SCORES[None] - mu[:, None]) ** 2).sum(1) / np.maximum(n_rated, 1)
    sigma = np.sqrt(var)
    alpha = np.clip((sigma + 1e-6) / 2.6, 0.3, 0.8)
    p = rh / np.maximum(n_rated, 1)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(p * np.where(p > 0, np.log(p), 0)).sum(1)
    mode_frac = p.max(1)
    n_distinct = (rh > 0).sum(1)
    lowest = np.where(rh > 0, SCORES[None], np.inf).min(1)

    z = np.clip((SCORES[None] - mu[:, None]) / (sigma[:, None] + 1e-6), -3, 3)
    ab = np.clip((SCORES[None] - 5.5) / 2.5, -2.5, 2.0)
    v = np.clip(alpha[:, None] * z + (1 - alpha[:, None]) * ab, -2.5, 2.5)
    tmean = (p * v).sum(1)
    tstd = np.sqrt(np.maximum((p * (v - tmean[:, None]) ** 2).sum(1), 0))

    rated_m = n_rated >= args.min_rated
    out = {"n_users": int(n_users), "min_rated": args.min_rated,
           "n_users_rated": int(rated_m.sum()),
           "pct_one_sitting": float(os_f.mean()), "pct_degenerate": float(dg_f.mean())}

    def seg(name, m):
        mm = m & rated_m
        out.setdefault("taxonomy", {})[name] = {
            "pct_of_rated": float(mm.sum() / rated_m.sum()),
            "n": int(mm.sum()),
            "mean_sigma": float(sigma[mm].mean()) if mm.any() else None,
            "mean_target_std": float(tstd[mm].mean()) if mm.any() else None,
        }

    seg("all_one_score", n_distinct == 1)
    seg("mode_ge_090", mode_frac >= 0.9)
    seg("mean_ge_95", mu >= 9.5)
    seg("never_below_7", lowest >= 7)
    seg("never_below_5", lowest >= 5)
    seg("distinct_le_2", n_distinct <= 2)
    seg("distinct_le_3", n_distinct <= 3)
    seg("full_range_8plus", n_distinct >= 8)
    seg("uses_1_and_10", (rh[:, 0] > 0) & (rh[:, 9] > 0))
    seg("sigma_lt_05", sigma < 0.5)
    seg("sigma_gt_25", sigma > 2.5)

    for nm, arr in [("n_rated", n_rated), ("rated_frac", n_rated / (n_rated + n_unrated)),
                    ("mu", mu), ("sigma", sigma), ("alpha", alpha), ("entropy", ent),
                    ("mode_frac", mode_frac), ("n_distinct", n_distinct),
                    ("target_std", tstd), ("target_mean", tmean)]:
        a = arr[rated_m]
        out.setdefault("dists", {})[nm] = {
            "mean": float(a.mean()),
            "q": [float(np.quantile(a, q)) for q in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)],
        }

    rng = np.random.default_rng(0)
    idx_r = np.nonzero(rated_m)[0]
    samp = rng.choice(idx_r, size=min(args.sample, len(idx_r)), replace=False)
    _, cents = kmeans(p[samp], args.k, seed=0)

    d2 = ((p[rated_m][:, None, :] - cents[None]) ** 2).sum(-1)
    a_r = d2.argmin(1)
    assign = np.full(n_users, -1, dtype=np.int8)
    assign[rated_m] = a_r

    order = np.argsort([-(assign == j).sum() for j in range(args.k)])
    remap = np.empty(args.k, dtype=np.int8)
    remap[order] = np.arange(args.k, dtype=np.int8)
    assign[rated_m] = remap[a_r]
    cents = cents[order]

    clusters = []
    for j in range(args.k):
        m = assign == j
        clusters.append({
            "share": float(m.sum() / rated_m.sum()),
            "centroid": [round(float(x), 4) for x in cents[j]],
            "mean_n_rated": float(n_rated[m].mean()),
            "mean_mu": float(mu[m].mean()), "mean_sigma": float(sigma[m].mean()),
            "mean_alpha": float(alpha[m].mean()), "mean_entropy": float(ent[m].mean()),
            "mean_target_std": float(tstd[m].mean()), "mean_target_mean": float(tmean[m].mean()),
            "pct_one_sitting": float(os_f[m].mean()), "pct_degenerate": float(dg_f[m].mean()),
        })
    out["clusters"] = clusters

    np.save(args.assign_out, assign)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["taxonomy"], indent=1))
    for i, c in enumerate(clusters):
        print(f"cluster {i}: share {c['share']:.3f} mu {c['mean_mu']:.2f} sigma {c['mean_sigma']:.2f} "
              f"tstd {c['mean_target_std']:.3f} deg {c['pct_degenerate']:.2%} centroid {c['centroid']}")


if __name__ == "__main__":
    main()
