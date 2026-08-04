"""EASE (closed-form full-rank linear autoencoder) reference for the presence task.

B = argmin ||X - XB||^2 + lam||B||^2 s.t. diag(B)=0, via P=(X^T X + lam I)^-1,
B = -P/diag(P). Scores from kept items only; NLL via softmax(tau*s + beta*log_pop)
with (tau, beta) fit on a calibration user set. Directly comparable to the NN
model's dropped-item NLL from loss_decomposition.py (same eval protocol).
"""

import argparse
import json

import numpy as np
import scipy.sparse as sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", default="../../data/aug2026/user_input_vectors_cleanup_notrust.npz")
    ap.add_argument("--n-eval", type=int, default=20000)
    ap.add_argument("--n-calib", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--lams", default="200,500,1000,3000")
    ap.add_argument("--gram-cache", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cs = 6000
    d = np.load(args.vectors)
    indices = d["indices"].astype(np.int32)
    lengths = d["lengths"].astype(np.int64)
    n_users = len(lengths)
    starts = np.zeros(n_users, dtype=np.int64)
    np.cumsum(lengths[:-1], out=starts[1:])

    rng = np.random.default_rng(args.seed)
    eval_users = rng.choice(n_users, size=args.n_eval, replace=False)
    remaining = np.setdiff1d(np.arange(n_users), eval_users)
    calib_users = np.random.default_rng(args.seed + 1).choice(remaining, size=args.n_calib, replace=False)
    excluded = set(eval_users.tolist()) | set(calib_users.tolist())

    counts = np.bincount(indices, minlength=cs).astype(np.float64)
    log_pop = np.log(np.maximum(counts, 1.0) / np.maximum(counts, 1.0).sum())

    if args.gram_cache:
        try:
            G = np.load(args.gram_cache)["G"]
            print("loaded gram cache", flush=True)
        except Exception:
            G = None
    else:
        G = None
    if G is None:
        print("building sparse X (train users only)...", flush=True)
        keep_user = np.ones(n_users, dtype=bool)
        keep_user[list(excluded)] = False
        entry_keep = np.repeat(keep_user, lengths)
        tr_idx = indices[entry_keep]
        tr_len = lengths[keep_user]
        indptr = np.zeros(len(tr_len) + 1, dtype=np.int64)
        np.cumsum(tr_len, out=indptr[1:])
        X = sp.csr_matrix(
            (np.ones(len(tr_idx), dtype=np.float32), tr_idx, indptr),
            shape=(len(tr_len), cs),
        )
        print(f"X: {X.shape}, nnz {X.nnz}; computing gram...", flush=True)
        G = (X.T @ X).toarray().astype(np.float64)
        if args.gram_cache:
            np.savez_compressed(args.gram_cache, G=G)
        del X
    print("gram done", flush=True)

    def make_eval(uids, keep_frac, seed):
        r = np.random.default_rng(seed)
        rows = []
        for u in uids:
            s, l = starts[u], lengths[u]
            idx = indices[s : s + l]
            keep = r.random(l) > (1 - keep_frac)
            if keep.sum() == 0:
                keep[r.integers(l)] = True
            if (~keep).sum() == 0:
                continue
            rows.append((idx[keep], idx[~keep]))
        return rows

    def score_batch(B, rows):
        """returns list of (scores_vec, kept, dropped)"""
        out = []
        for kept, dropped in rows:
            s = B[kept].sum(axis=0)
            out.append((s, kept, dropped))
        return out

    def nll_of(scored, tau, beta):
        tot, cnt = 0.0, 0
        for s, kept, dropped in scored:
            logits = tau * s + beta * log_pop
            lse = np.logaddexp.reduce(logits)
            tot += (lse - logits[dropped]).sum()
            cnt += len(dropped)
        return tot / cnt

    def fit_calib(scored):
        best = (None, None, np.inf)
        for tau in [0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0]:
            for beta in [0.0, 0.25, 0.5, 0.75, 1.0]:
                v = nll_of(scored, tau, beta)
                if v < best[2]:
                    best = (tau, beta, v)
        tau0, beta0, _ = best
        for tau in np.geomspace(tau0 / 2, tau0 * 2, 9):
            for beta in np.linspace(max(0, best[1] - 0.25), best[1] + 0.25, 11):
                v = nll_of(scored, tau, beta)
                if v < best[2]:
                    best = (tau, beta, v)
        return best

    def rank_stats(scored, tau, beta):
        ranks = []
        for s, kept, dropped in scored:
            logits = tau * s + beta * log_pop
            logits[kept] = -np.inf
            order = np.argsort(-logits)
            rank_of = np.empty(cs, dtype=np.int32)
            rank_of[order] = np.arange(cs)
            ranks.extend(rank_of[dropped].tolist())
        ranks = np.asarray(ranks)
        return {
            "median_rank": float(np.median(ranks)),
            "recall@50": float((ranks < 50).mean()),
            "recall@250": float((ranks < 250).mean()),
        }

    results = {"lams": [], "n_eval": args.n_eval}
    lams = [float(x) for x in args.lams.split(",")]

    calib_rows = make_eval(calib_users, 0.6, args.seed + 50)
    eval_rows_60 = make_eval(eval_users, 0.6, args.seed + 100)

    best_lam, best_nll, best_B = None, np.inf, None
    for lam in lams:
        Greg = G + np.eye(cs) * lam
        P = np.linalg.inv(Greg)
        B = -P / np.diag(P)[None, :]
        np.fill_diagonal(B, 0.0)
        B = B.astype(np.float32)
        scored = score_batch(B, calib_rows)
        tau, beta, cnll = fit_calib(scored)
        print(f"lam={lam}: calib nll={cnll:.4f} (tau={tau:.3f}, beta={beta:.2f})", flush=True)
        results["lams"].append({"lam": lam, "calib_nll": cnll, "tau": tau, "beta": beta})
        if cnll < best_nll:
            best_lam, best_nll, best_B = lam, cnll, B
            best_tau, best_beta = tau, beta

    results["best_lam"] = best_lam
    print(f"best lam {best_lam}; evaluating sweep...", flush=True)

    sweep = []
    for keep_frac in [0.05, 0.1, 0.2, 0.35, 0.5, 0.6, 0.65, 0.8, 0.9, 0.95, 0.99]:
        rows = eval_rows_60 if keep_frac == 0.6 else make_eval(
            eval_users[:5000], keep_frac, args.seed + int(keep_frac * 1000)
        )
        scored = score_batch(best_B, rows)
        nll = nll_of(scored, best_tau, best_beta)
        rs = rank_stats(scored, best_tau, best_beta)
        pop_nll = np.mean([(-log_pop[dropped]).mean() for _, _, dropped in scored])
        sweep.append({"keep_frac": keep_frac, "nll_drop": nll, "pop_nll_drop": float(pop_nll), **rs})
        print(f"keep={keep_frac}: nll_drop={nll:.4f} median_rank={rs['median_rank']} r@50={rs['recall@50']:.4f}", flush=True)
    results["keep_sweep"] = sweep

    # popularity-only rank stats at LOO for reference
    pop_scored = [(np.zeros(cs), kept, dropped) for _, kept, dropped in score_batch(best_B, make_eval(eval_users[:5000], 0.99, 9999))]
    results["popularity_rank_stats"] = rank_stats(pop_scored, 0.0, 1.0)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
