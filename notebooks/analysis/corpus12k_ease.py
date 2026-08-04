"""Corpus-12k experiment: sizing + EASE at doubled corpus, no NN training needed.

Pass 1: stream raw collected profiles -> full popularity counts (opportunity stats).
Pass 2: per-user item lists restricted to top-12k (cleanup_notrust-style filters).
Pass 3: EASE gram/inversion at 12k, eval on held-out users:
  (a) all dropped targets, (b) targets inside the old 6k corpus (vs 6k-EASE),
  (c) targets in the 6k-12k extension band.
"""

import argparse
import gzip
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp


def stream_chunks(path, usecols):
    return pd.read_csv(path, usecols=usecols, chunksize=8_000_000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="../../data/collected_animelists_aug2026.csv.gz")
    ap.add_argument("--corpus6k", default="../../data/corpus_ids_aug2026.json")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--lam", type=float, default=200.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = {}

    # ---- pass 1: global counts (presence definition: not unrated-PTW) ----
    print("pass 1: counting...", flush=True)
    counts = {}
    rows_total = 0
    for chunk in stream_chunks(args.raw, ["anime_id", "my_score", "status"]):
        rows_total += len(chunk)
        keep = ~((chunk["status"] == "plan_to_watch") & (chunk["my_score"] == 0))
        vc = chunk.loc[keep, "anime_id"].value_counts()
        for aid, n in vc.items():
            counts[aid] = counts.get(aid, 0) + int(n)
        print(f"  {rows_total/1e6:.0f}M rows", flush=True)
    ids = np.array(list(counts.keys()), dtype=np.int64)
    cnts = np.array([counts[i] for i in ids], dtype=np.int64)
    order = np.argsort(-cnts)
    ids, cnts = ids[order], cnts[order]
    total_entries = int(cnts.sum())
    out["n_distinct_items"] = len(ids)
    out["total_presence_entries"] = total_entries
    cum = np.cumsum(cnts) / total_entries
    for k in [1000, 3000, 6000, 9000, 12000, 18000, len(ids)]:
        if k <= len(ids):
            out[f"share_top_{k}"] = float(cum[k - 1])
    out["share_6k_12k_band"] = float(cum[min(11999, len(ids) - 1)] - cum[5999])
    out["count_at_rank"] = {str(r): int(cnts[r]) for r in [0, 999, 2999, 5999, 8999, 11999] if r < len(ids)}
    print(json.dumps({k: v for k, v in out.items() if k != "count_at_rank"}, indent=1), flush=True)

    corpus12k = ids[:12000]
    np.save(f"{args.workdir}/corpus12k_ids.npy", corpus12k)
    np.save(f"{args.workdir}/counts_full.npy", np.stack([ids, cnts]))
    sorted12k = np.sort(corpus12k)
    code_of = {int(a): i for i, a in enumerate(corpus12k)}

    with open(args.corpus6k) as f:
        c6 = json.load(f)
    in6k = np.zeros(12000, dtype=bool)
    six_set = set(c6)
    for a, i in code_of.items():
        if a in six_set:
            in6k[i] = True
    out["old6k_in_top12k"] = int(in6k.sum())
    print(f"old 6k corpus items inside new top-12k: {in6k.sum()}", flush=True)

    # ---- pass 2: per-user 12k-code lists (vectorized; file is grouped by user) ----
    print("pass 2: building per-user lists...", flush=True)
    max_id = int(ids.max())
    lookup = np.full(max_id + 2, -1, dtype=np.int32)
    lookup[corpus12k] = np.arange(12000, dtype=np.int32)

    code_chunks, kept_len_chunks, raw_len_chunks = [], [], []
    carry_user = None
    carry_raw = 0
    carry_kept = 0
    for chunk in stream_chunks(args.raw, ["username", "anime_id", "my_score", "status"]):
        uname = chunk["username"].values
        aid = chunk["anime_id"].values.astype(np.int64)
        keep = (~((chunk["status"] == "plan_to_watch") & (chunk["my_score"] == 0))).to_numpy()
        code = np.where(aid <= max_id, lookup[np.minimum(aid, max_id)], -1)
        keep = keep & (code >= 0)

        new_u = np.empty(len(uname), dtype=bool)
        first_is_new = (carry_user is None) or (uname[0] != carry_user)
        new_u[0] = first_is_new
        new_u[1:] = uname[1:] != uname[:-1]
        gid = np.cumsum(new_u)
        if first_is_new:
            gid = gid - 1  # groups 0..n-1, all starting in this chunk
        # else: group 0 continues the carried user
        n_groups = int(gid[-1]) + 1
        raw_l = np.bincount(gid, minlength=n_groups)
        kept_l = np.bincount(gid[keep], minlength=n_groups)
        code_chunks.append(code[keep].astype(np.int16))

        if not first_is_new:
            raw_l[0] += carry_raw
            kept_l[0] += carry_kept
        elif carry_user is not None:
            raw_len_chunks.append(np.array([carry_raw]))
            kept_len_chunks.append(np.array([carry_kept]))
        carry_user = uname[-1]
        carry_raw = int(raw_l[-1])
        carry_kept = int(kept_l[-1])
        raw_len_chunks.append(raw_l[:-1])
        kept_len_chunks.append(kept_l[:-1])
        print(f"  ~{sum(len(x) for x in raw_len_chunks)} users flushed", flush=True)
    raw_len_chunks.append(np.array([carry_raw]))
    kept_len_chunks.append(np.array([carry_kept]))

    raw_lengths = np.concatenate(raw_len_chunks)
    kept_lengths = np.concatenate(kept_len_chunks).astype(np.int64)
    codes_all = np.concatenate(code_chunks)
    del code_chunks
    assert kept_lengths.sum() == len(codes_all), (kept_lengths.sum(), len(codes_all))

    user_ok = (raw_lengths <= 2000) & (kept_lengths >= 20)
    entry_ok = np.repeat(user_ok, kept_lengths)
    codes = codes_all[entry_ok]
    del codes_all
    lengths = kept_lengths[user_ok]
    out["n_users_12k"] = int(len(lengths))
    out["n_entries_12k"] = int(lengths.sum())
    out["dropped_bloat_lists"] = int((raw_lengths > 2000).sum())
    np.savez(f"{args.workdir}/user_vectors_12k.npz", codes=codes, lengths=lengths)
    print(f"users {len(lengths)}, entries {lengths.sum()}", flush=True)

    starts = np.zeros(len(lengths), dtype=np.int64)
    np.cumsum(lengths[:-1], out=starts[1:])

    # ---- pass 3: EASE-12k ----
    rng = np.random.default_rng(123)
    eval_users = rng.choice(len(lengths), size=20000, replace=False)
    eval_set = set(eval_users.tolist())

    print("gram 12k...", flush=True)
    keep_user = np.ones(len(lengths), dtype=bool)
    keep_user[list(eval_set)] = False
    entry_keep = np.repeat(keep_user, lengths)
    tr_idx = codes[entry_keep].astype(np.int32)
    tr_len = lengths[keep_user]
    indptr = np.zeros(len(tr_len) + 1, dtype=np.int64)
    np.cumsum(tr_len, out=indptr[1:])
    X = sp.csr_matrix((np.ones(len(tr_idx), dtype=np.float32), tr_idx, indptr), shape=(len(tr_len), 12000))
    G = (X.T @ X).toarray().astype(np.float64)
    del X, tr_idx
    print("inverting...", flush=True)
    G[np.diag_indices(12000)] += args.lam
    P = np.linalg.inv(G)
    B = (-P / np.diag(P)[None, :]).astype(np.float32)
    np.fill_diagonal(B, 0.0)
    del P, G
    np.save(f"{args.workdir}/ease_B12k.npy", B)

    counts12 = np.bincount(codes.astype(np.int32), minlength=12000).astype(np.float64)
    log_pop12 = np.log(np.maximum(counts12, 1.0) / np.maximum(counts12, 1.0).sum())
    beta = 0.65

    def eval_ease(keep_frac, n_users_eval=8000):
        r = np.random.default_rng(int(keep_frac * 1000) + 7)
        ranks_all, ranks_in6k, ranks_band, ranks_6kpool = [], [], [], []
        pool6_idx = np.nonzero(in6k)[0]
        for u in eval_users[:n_users_eval]:
            s0, l = starts[u], lengths[u]
            idx = codes[s0 : s0 + l].astype(np.int32)
            keep = r.random(l) > (1 - keep_frac)
            if keep.sum() == 0:
                keep[r.integers(l)] = True
            dropped = idx[~keep]
            if len(dropped) == 0:
                continue
            kept = idx[keep]
            sc = B[kept].sum(axis=0) + beta * log_pop12
            sc[kept] = -np.inf
            order_r = np.argsort(-sc)
            rank_of = np.empty(12000, dtype=np.int32)
            rank_of[order_r] = np.arange(12000)
            rr = rank_of[dropped]
            ranks_all.extend(rr.tolist())
            m6 = in6k[dropped]
            ranks_in6k.extend(rr[m6].tolist())
            ranks_band.extend(rr[~m6].tolist())
            # 6k-restricted candidate pool: comparable to the 6k-EASE sweeps
            d6 = dropped[m6]
            if len(d6):
                sc6 = sc[pool6_idx]
                order6 = np.argsort(-sc6)
                rank_of6 = np.empty(len(pool6_idx), dtype=np.int32)
                rank_of6[order6] = np.arange(len(pool6_idx))
                back = np.searchsorted(pool6_idx, d6)
                ranks_6kpool.extend(rank_of6[back].tolist())
        def stats(rk):
            rk = np.asarray(rk)
            if len(rk) == 0:
                return None
            return {"n": len(rk), "median_rank": float(np.median(rk)),
                    "recall@50": float((rk < 50).mean()), "recall@250": float((rk < 250).mean())}
        return {"all": stats(ranks_all), "targets_in_6k": stats(ranks_in6k),
                "targets_6k_12k": stats(ranks_band), "targets_in_6k_6kpool": stats(ranks_6kpool)}

    for kf in [0.99, 0.6]:
        out[f"ease12k_keep_{kf}"] = eval_ease(kf)
        print(kf, json.dumps(out[f"ease12k_keep_{kf}"]), flush=True)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
