"""Estimate the irreducible (aleatoric) rating noise via near-twin profiles.

Two users whose profiles are near-identical in items AND ratings are
indistinguishable to the model; their disagreement on a shared item bounds any
model's achievable error. Pipeline:
  1. exact-duplicate presence sets via hashing (J=1 anchor)
  2. MinHash LSH (12 bands x 4) for near-twin candidates
  3. exact Jaccard + per-shared-item rating disagreement, conditioned on
     context agreement (mean |dz| on the OTHER shared rated items)

Noise conversion: dz = eps_A - eps_B, Var(eps) = Var(dz)/2,
Gaussian MAE floor = sigma * sqrt(2/pi).
"""

import argparse
import hashlib
import json
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", default="../../data/aug2026/user_input_vectors_cleanup_notrust.npz")
    ap.add_argument("--n-hashes", type=int, default=48)
    ap.add_argument("--bands", type=int, default=12)
    ap.add_argument("--max-bucket", type=int, default=200)
    ap.add_argument("--max-pairs-per-bucket", type=int, default=60)
    ap.add_argument("--min-shared-rated", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs-out", default=None)
    args = ap.parse_args()

    d = np.load(args.vectors)
    indices = d["indices"].astype(np.int32)
    values = d["values"]
    lengths = d["lengths"].astype(np.int64)
    rated = np.unpackbits(d["rated_masks"])[: int(d["total_mask_bits"][0])].astype(bool)
    n_users = len(lengths)
    starts = np.zeros(n_users, dtype=np.int64)
    np.cumsum(lengths[:-1], out=starts[1:])
    print(f"{n_users} users", flush=True)

    rng = np.random.default_rng(7)

    # ---- phase 1: exact-duplicate presence sets ----
    print("hashing exact presence sets...", flush=True)
    sig = np.empty(n_users, dtype=np.uint64)
    for u in range(n_users):
        s, l = starts[u], lengths[u]
        h = hashlib.blake2b(indices[s : s + l].tobytes(), digest_size=8).digest()
        sig[u] = np.frombuffer(h, dtype=np.uint64)[0]
    order = np.argsort(sig, kind="stable")
    sig_sorted = sig[order]
    boundaries = np.nonzero(np.diff(sig_sorted))[0] + 1
    groups = np.split(order, boundaries)
    exact_groups = [g for g in groups if len(g) > 1]
    print(f"exact-dup groups: {len(exact_groups)}, users covered: {sum(len(g) for g in exact_groups)}", flush=True)

    # ---- phase 2: minhash LSH ----
    print("computing minhashes...", flush=True)
    n_h = args.n_hashes
    minh = np.empty((n_h, n_users), dtype=np.uint32)
    hrng = np.random.default_rng(1234)
    for h in range(n_h):
        item_h = hrng.integers(0, 2**32, size=6000, dtype=np.uint32)
        entry_h = item_h[indices]
        minh[h] = np.minimum.reduceat(entry_h, starts)
        del entry_h
    print("banding...", flush=True)

    rows_per_band = n_h // args.bands
    pair_set = set()
    for b in range(args.bands):
        band = minh[b * rows_per_band : (b + 1) * rows_per_band].astype(np.uint64)
        key = band[0]
        for r in range(1, rows_per_band):
            key = key * np.uint64(1000003) + band[r]
        border = np.argsort(key, kind="stable")
        ks = key[border]
        bnd = np.nonzero(np.diff(ks))[0] + 1
        for g in np.split(border, bnd):
            gl = len(g)
            if gl < 2 or gl > args.max_bucket:
                continue
            if gl * (gl - 1) // 2 <= args.max_pairs_per_bucket:
                for i in range(gl):
                    for j in range(i + 1, gl):
                        a, c = int(g[i]), int(g[j])
                        pair_set.add((a << 21 | c) if a < c else (c << 21 | a))
            else:
                for _ in range(args.max_pairs_per_bucket):
                    i, j = rng.integers(gl), rng.integers(gl)
                    if i == j:
                        continue
                    a, c = int(g[i]), int(g[j])
                    pair_set.add((a << 21 | c) if a < c else (c << 21 | a))
        print(f"  band {b}: cumulative pairs {len(pair_set)}", flush=True)

    pairs = np.array(sorted(pair_set), dtype=np.int64)
    del pair_set
    ua = (pairs >> 21).astype(np.int64)
    ub = (pairs & ((1 << 21) - 1)).astype(np.int64)
    print(f"candidate pairs: {len(pairs)}", flush=True)

    # ---- phase 3: pair stats ----
    # bins: jaccard x context-agreement; accumulate disagreement on target items
    j_edges = np.array([0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.01])
    ctx_edges = np.array([0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 10.0])
    acc = np.zeros((len(j_edges) - 1, len(ctx_edges) - 1, 4))  # cnt, sum|dz|, sum dz^2, sum m

    pair_records = []

    def process_pair(a, b, jacc):
        sa, la = starts[a], lengths[a]
        sb, lb = starts[b], lengths[b]
        ia = indices[sa : sa + la]
        ib = indices[sb : sb + lb]
        common, ca, cb = np.intersect1d(ia, ib, assume_unique=True, return_indices=True)
        if jacc is None:
            inter = len(common)
            jacc = inter / (la + lb - inter)
            if jacc < 0.25:
                return
        ra = rated[sa : sa + la][ca]
        rb = rated[sb : sb + lb][cb]
        both = ra & rb
        m = int(both.sum())
        if m < args.min_shared_rated:
            return
        va = values[sa : sa + la][ca][both].astype(np.float64)
        vb = values[sb : sb + lb][cb][both].astype(np.float64)
        dz = va - vb
        adz = np.abs(dz)
        D = adz.sum()
        ctx = (D - adz) / (m - 1)  # leave-one-out context disagreement per target
        jb = min(np.searchsorted(j_edges, jacc, side="right") - 1, acc.shape[0] - 1)
        cb_idx = np.clip(np.searchsorted(ctx_edges, ctx, side="right") - 1, 0, acc.shape[1] - 1)
        for t in range(m):
            acc[jb, cb_idx[t], 0] += 1
            acc[jb, cb_idx[t], 1] += adz[t]
            acc[jb, cb_idx[t], 2] += dz[t] * dz[t]
            acc[jb, cb_idx[t], 3] += m
        if len(pair_records) < 300000:
            pair_records.append((round(jacc, 4), m, round(D / m, 4), int(la), int(lb)))

    print("processing exact-dup pairs...", flush=True)
    n_exact_pairs = 0
    for g in exact_groups:
        gl = len(g)
        cap = min(gl * (gl - 1) // 2, 30)
        seen = set()
        tries = 0
        while len(seen) < cap and tries < cap * 4:
            tries += 1
            i, j = rng.integers(gl), rng.integers(gl)
            if i == j:
                continue
            key = (min(g[i], g[j]), max(g[i], g[j]))
            if key in seen:
                continue
            seen.add(key)
            process_pair(key[0], key[1], 1.0)
            n_exact_pairs += 1
    print(f"exact pairs processed: {n_exact_pairs}", flush=True)

    print("processing LSH pairs...", flush=True)
    for k in range(len(pairs)):
        process_pair(ua[k], ub[k], None)
        if k % 500000 == 0:
            print(f"  {k}/{len(pairs)}", flush=True)

    out = {
        "n_users": n_users,
        "n_candidate_pairs": int(len(pairs)),
        "n_exact_groups": len(exact_groups),
        "n_exact_pairs": n_exact_pairs,
        "j_edges": j_edges.tolist(),
        "ctx_edges": ctx_edges.tolist(),
        "bins": [
            [
                {
                    "count": int(acc[j, c, 0]),
                    "mean_abs_dz": acc[j, c, 1] / acc[j, c, 0] if acc[j, c, 0] else None,
                    "var_dz": acc[j, c, 2] / acc[j, c, 0] if acc[j, c, 0] else None,
                    "mean_shared": acc[j, c, 3] / acc[j, c, 0] if acc[j, c, 0] else None,
                }
                for c in range(acc.shape[1])
            ]
            for j in range(acc.shape[0])
        ],
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)

    if args.pairs_out:
        with open(args.pairs_out, "w") as f:
            f.write("jaccard,n_shared_rated,mean_abs_dz,len_a,len_b\n")
            for r in pair_records:
                f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]}\n")

    # quick summary
    for j in range(acc.shape[0]):
        row = acc[j].sum(axis=0)
        if row[0]:
            print(
                f"J [{j_edges[j]:.2f},{j_edges[j+1]:.2f}): n={int(row[0])} "
                f"mean|dz|={row[1]/row[0]:.4f} var(dz)={row[2]/row[0]:.4f} "
                f"-> sigma_eps={np.sqrt(row[2]/row[0]/2):.4f} mae_floor={np.sqrt(row[2]/row[0]/2)*np.sqrt(2/np.pi):.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
