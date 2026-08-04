"""Graft checkpoints on the round-3 frontier protocol: identical seed-123 eval pool,
rng-777 corruption, alpha grid, and filtered/unfiltered stats as frontier_eval.py, so
curves overlay directly onto frontier_results.json. Base score = per-user z-normed
graft logits (EASE residual included) + alpha*z(log_pop), k=0.

Run inside rocm_jax: cd /jax_dir/notebooks && python analysis/frontier_graft.py ...
"""

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, ".")
import jax.numpy as jnp

from model import CONF
from analysis.frontier_eval import build_components, spearman, TIERS
from analysis.probe_value_eval import load_graft

ALPHAS = [0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0, 1.3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gate:../data/aug2026/probe/probe_graft_gate.msgpack,"
                                        "concat:../data/aug2026/probe/probe_graft_concat.msgpack")
    ap.add_argument("--ease-b", default="../data/aug2026/ease_B6k_lam200.npy")
    ap.add_argument("--vectors", default="../data/aug2026/user_input_vectors_cleanup_notrust.npz")
    ap.add_argument("--corpus", default="../data/corpus_ids_aug2026.json")
    ap.add_argument("--metadata", default="../data/processed-metadata_aug2026.csv")
    ap.add_argument("--n-users", type=int, default=5000)
    ap.add_argument("--n-ref", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--stack-w", default="",
                    help="comma list of w for stack families: (1-w)*z(first model logits) + w*z(ease lift)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    stack_ws = [float(w) for w in args.stack_w.split(",") if w]

    cs = CONF["corpus_size"]
    d = np.load(args.vectors)
    indices = d["indices"].astype(np.int32)
    values = d["values"]
    lengths = d["lengths"].astype(np.int64)
    starts = np.zeros(len(lengths), dtype=np.int64)
    np.cumsum(lengths[:-1], out=starts[1:])

    counts = np.bincount(indices, minlength=cs).astype(np.float64)
    log_pop = np.log(np.maximum(counts, 1.0) / np.maximum(counts, 1.0).sum())
    zlp = (log_pop - log_pop.mean()) / log_pop.std()
    order = np.argsort(-counts)
    rank_of_item = np.empty(cs, dtype=np.int32)
    rank_of_item[order] = np.arange(cs)
    tier_of_item = np.zeros(cs, dtype=np.int8)
    for t, (lo, hi) in enumerate(TIERS):
        tier_of_item[(rank_of_item >= lo) & (rank_of_item < hi)] = t

    with open(args.corpus) as f:
        corpus_ids = np.array(json.load(f), dtype=np.int64)
    dsu = build_components(args.metadata)
    comp_of_corpus = np.array([dsu.find(int(a)) for a in corpus_ids], dtype=np.int64)

    B = np.load(args.ease_b)
    models = {}
    for spec in args.models.split(","):
        name, path = spec.split(":", 1)
        models[name] = load_graft(path, name.split("_")[0], 512, cs, B)

    rng = np.random.default_rng(args.seed)
    pool = rng.choice(len(lengths), size=args.n_ref + args.n_users + 5000, replace=False)
    ref_users = pool[: args.n_ref]
    eval_pool = [u for u in pool[args.n_ref:] if lengths[u] >= 24][: args.n_users]

    mu = None
    if stack_ws:
        s_sum = np.zeros(cs)
        for u in ref_users:
            s0, l = starts[u], lengths[u]
            s_sum += B[indices[s0 : s0 + l]].sum(axis=0)
        mu = s_sum / len(ref_users)
        print("mu ready for stack families", flush=True)

    rho = {}
    diag_users = eval_pool[:200]
    bx = np.zeros((len(diag_users), cs * 2), dtype=np.float32)
    for i, u in enumerate(diag_users):
        s0, l = starts[u], lengths[u]
        idx = indices[s0 : s0 + l]
        bx[i, idx] = 1.0
        bx[i, cs + idx] = values[s0 : s0 + l]
    for name, (params, fwd) in models.items():
        lg = np.asarray(fwd(params, jnp.asarray(bx)), dtype=np.float64)
        rho[name] = float(np.mean([spearman(lg[i], counts) for i in range(len(diag_users))]))
    print("spearman(logits, popularity):", rho, flush=True)

    class Agg:
        def __init__(self):
            self.r = {}
        def add(self, key, ranks, tiers):
            e = self.r.setdefault(key, {"ranks": [], "tiers": [], "top10pop": [], "top10franch": []})
            e["ranks"].extend(ranks.tolist()); e["tiers"].extend(tiers.tolist())
        def add_list(self, key, top10_pop, franch_frac):
            e = self.r[key]
            e["top10pop"].append(top10_pop); e["top10franch"].append(franch_frac)
        def stats(self):
            out = {}
            for key, e in self.r.items():
                rk = np.asarray(e["ranks"]); tt = np.asarray(e["tiers"])
                s = {"n": len(rk), "overall_r50": float((rk < 50).mean()), "overall_r250": float((rk < 250).mean()),
                     "mean_top10_poprank": float(np.mean(e["top10pop"])),
                     "franchise_share_top10": float(np.mean(e["top10franch"]))}
                for t, (lo, hi) in enumerate(TIERS):
                    m = tt == t
                    s[f"r250_tier{lo}_{hi}"] = float((rk[m] < 250).mean()) if m.any() else None
                out[key] = s
            return out

    agg, agg_f = Agg(), Agg()
    r = np.random.default_rng(777)
    bx = np.zeros((256, cs * 2), dtype=np.float32)
    metas = []

    def flush(nb):
        outs = {name: np.asarray(fwd(params, jnp.asarray(bx[:nb])), dtype=np.float64)
                for name, (params, fwd) in models.items()}
        for j in range(nb):
            kept, dropped = metas[j]
            cand_franch = np.isin(comp_of_corpus, np.unique(comp_of_corpus[kept]))
            tgt_franch = cand_franch[dropped]
            tt = tier_of_item[dropped]
            bases = {name: (lgv[j] - lgv[j].mean()) / (lgv[j].std() + 1e-9)
                     for name, lgv in outs.items()}
            if stack_ws:
                ze = B[kept].sum(axis=0).astype(np.float64) - mu
                ze = (ze - ze.mean()) / (ze.std() + 1e-9)
                first = bases[list(models)[0]]
                for w in stack_ws:
                    bases[f"stack_w{w}"] = (1 - w) * first + w * ze
            for name, zn in bases.items():
                for a in ALPHAS:
                    key = f"{name}|k0|a{a}"
                    sc = zn + a * zlp
                    sc[kept] = -np.inf
                    o = np.argsort(-sc)
                    ro = np.empty(cs, dtype=np.int32)
                    ro[o] = np.arange(cs)
                    agg.add(key, ro[dropped], tt)
                    top10 = o[:10]
                    agg.add_list(key, float(rank_of_item[top10].mean()), float(cand_franch[top10].mean()))
                    scf = sc.copy()
                    scf[cand_franch] = -np.inf
                    of = np.argsort(-scf)
                    rof = np.empty(cs, dtype=np.int32)
                    rof[of] = np.arange(cs)
                    mgt = ~tgt_franch
                    agg_f.add(key, rof[dropped[mgt]], tt[mgt])
                    agg_f.add_list(key, float(rank_of_item[of[:10]].mean()), 0.0)
        metas.clear()

    nb = 0
    done = 0
    for u in eval_pool:
        s0, l = starts[u], lengths[u]
        idx = indices[s0 : s0 + l]
        val = values[s0 : s0 + l]
        keep = r.random(l) > 0.01
        if (~keep).sum() == 0:
            keep[r.integers(l)] = False
        kept, dropped = idx[keep], idx[~keep]
        bx[nb] = 0.0
        bx[nb, kept] = 1.0
        bx[nb, cs + kept] = val[keep]
        metas.append((kept, dropped))
        nb += 1
        if nb == 256:
            flush(nb); nb = 0
            done += 256
            print(f"{done}/{len(eval_pool)}", flush=True)
    if nb:
        flush(nb)

    out = {"rho_pop": rho, "alphas": ALPHAS,
           "models": list(models) + [f"stack_w{w}" for w in stack_ws],
           "unfiltered": agg.stats(), "filtered": agg_f.stats()}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    for name in out["models"]:
        row = []
        for a in ALPHAS:
            s = out["filtered"][f"{name}|k0|a{a}"]
            row.append(f"a{a}: {s['overall_r250']:.3f}/{s['r250_tier1000_3000']:.3f}/{s['r250_tier3000_6000']:.3f}")
        print(f"{name}: " + " | ".join(row), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
