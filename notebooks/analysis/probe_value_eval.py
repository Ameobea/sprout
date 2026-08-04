"""Unified value-metric battery over probe checkpoints, EASE, and blends.

Beyond raw recall, reports the metrics that track actual recommender value:
franchise-filtered strata (prod extra-season filter simulated), novelty-stratified
recall (target distance to nearest kept item via gram cosine — adjacent-cluster
discovery), and filtered top-10 nicheness. Same eval seeds as train_probe_ease's
eval_cfg so unfiltered numbers are directly comparable to earlier rounds.

Run inside rocm_jax (CPU is fine while GPU trains):
  JAX_PLATFORMS=cpu python analysis/probe_value_eval.py --models-json ... --out ...
"""

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, ".")
import jax
import jax.numpy as jnp
from flax import serialization

from model import CONF, Recommender
from analysis.train_probe_graft import GraftRecommender
from analysis.frontier_eval import build_components

HOLDOUT_SEED = 999
TIERS = [(0, 250), (250, 1000), (1000, 3000), (3000, 6000)]
CONFIGS = [("k8", 8, 0), ("k16", 16, 0), ("keep0.6", None, 0.6),
           ("keep0.9", None, 0.9), ("keep0.99", None, 0.99)]
EASE_BETA = 0.2167


def load_prod(path, bd, cs):
    model = Recommender(bottleneck_dim=bd)
    dummy = jnp.ones((1, cs * 2))
    params = model.init({"params": jax.random.PRNGKey(0), "noise": jax.random.PRNGKey(0)}, dummy)["params"]
    with open(path, "rb") as f:
        params = serialization.from_bytes(params, f.read())
    fwd = jax.jit(lambda p, x: model.apply({"params": p}, x, training=False)[0])
    return params, fwd


def load_ease3ch(path, cs, B):
    model = Recommender()
    dummy = jnp.ones((1, cs * 3))
    params = model.init({"params": jax.random.PRNGKey(0), "noise": jax.random.PRNGKey(0)}, dummy)["params"]
    with open(path, "rb") as f:
        params = serialization.from_bytes(params, f.read())
    Bj = jnp.asarray(B, dtype=jnp.float32)

    def fwd(p, x):
        e = x[:, :cs] @ Bj
        e = (e - jnp.mean(e, axis=1, keepdims=True)) / (jnp.std(e, axis=1, keepdims=True) + 1e-6)
        return model.apply({"params": p}, jnp.concatenate([x, e], axis=1), training=False)[0]

    return params, jax.jit(fwd)


def load_graft(path, mode, bd, cs, B):
    model = GraftRecommender(mode=mode, bottleneck_dim=bd)
    dummy_x, dummy_e = jnp.ones((1, cs * 2)), jnp.ones((1, cs))
    params = model.init({"params": jax.random.PRNGKey(0), "noise": jax.random.PRNGKey(0)}, dummy_x, dummy_e)["params"]
    with open(path, "rb") as f:
        params = serialization.from_bytes(params, f.read())
    Bj = jnp.asarray(B, dtype=jnp.float32)

    def fwd(p, x):
        e = x[:, :cs] @ Bj
        e = (e - jnp.mean(e, axis=1, keepdims=True)) / (jnp.std(e, axis=1, keepdims=True) + 1e-6)
        return model.apply({"params": p}, x, e, training=False)[0]

    return params, jax.jit(fwd)


class Agg:
    def __init__(self):
        self.ranks, self.tiers, self.novs = [], [], []
        self.top10pop, self.top10franch, self.top10nov = [], [], []

    def add_targets(self, ranks, tiers, novs):
        self.ranks.extend(ranks.tolist()); self.tiers.extend(tiers.tolist()); self.novs.extend(novs.tolist())

    def add_list(self, pop, franch, nov):
        self.top10pop.append(pop); self.top10franch.append(franch); self.top10nov.append(nov)

    def stats(self, nov_bounds):
        rk = np.asarray(self.ranks); tt = np.asarray(self.tiers); nv = np.asarray(self.novs)
        s = {"n": len(rk), "median_rank": float(np.median(rk)),
             "r50": float((rk < 50).mean()), "r250": float((rk < 250).mean()),
             "mean_top10_poprank": float(np.mean(self.top10pop)),
             "franchise_share_top10": float(np.mean(self.top10franch)),
             "mean_top10_novelty": float(np.mean(self.top10nov))}
        for t, (lo, hi) in enumerate(TIERS):
            m = tt == t
            s[f"r250_tier{lo}_{hi}"] = float((rk[m] < 250).mean()) if m.any() else None
        for q in range(4):
            m = (nv >= nov_bounds[q]) & (nv < nov_bounds[q + 1])
            s[f"r250_novq{q + 1}"] = float((rk[m] < 250).mean()) if m.any() else None
        for t in range(len(TIERS)):
            for q in range(4):
                m = (tt == t) & (nv >= nov_bounds[q]) & (nv < nov_bounds[q + 1])
                s[f"x_t{t}q{q + 1}"] = [float((rk[m] < 250).mean()), int(m.sum())] if m.any() else [None, 0]
        return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-json", required=True)
    ap.add_argument("--vectors", default="../data/aug2026/user_input_vectors_cleanup_notrust.npz")
    ap.add_argument("--ease-b", default="../data/aug2026/ease_B6k_lam200.npy")
    ap.add_argument("--gram", default="../data/aug2026/gram6k_aug2026.npz")
    ap.add_argument("--metadata", default="../data/processed-metadata_aug2026.csv")
    ap.add_argument("--corpus", default="../data/corpus_ids_aug2026.json")
    ap.add_argument("--eval-n", type=int, default=3000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    EVAL_N = args.eval_n

    cs = CONF["corpus_size"]
    with open(args.models_json) as f:
        model_specs = json.load(f)

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

    G = np.load(args.gram)["G"].astype(np.float64)
    dg = np.sqrt(np.maximum(np.diag(G), 1e-9))
    cos = G / dg[None, :] / dg[:, None]
    np.fill_diagonal(cos, 1.0)

    with open(args.corpus) as f:
        corpus_ids = np.array(json.load(f), dtype=np.int64)
    dsu = build_components(args.metadata)
    comp_of_corpus = np.array([dsu.find(int(a)) for a in corpus_ids], dtype=np.int64)

    B = np.load(args.ease_b)

    rng_h = np.random.default_rng(HOLDOUT_SEED)
    perm = rng_h.permutation(len(lengths))
    n_hold = len(lengths) // 10
    holdout_idx = perm[:n_hold]
    train_pool = perm[n_hold:]

    rng_ref = np.random.default_rng(555)
    ref_users = rng_ref.choice(train_pool, size=20000, replace=False)
    s_sum = np.zeros(cs)
    for u in ref_users:
        s0, l = starts[u], lengths[u]
        s_sum += B[indices[s0 : s0 + l]].sum(axis=0)
    mu_ease = s_sum / len(ref_users)
    print("ease mu computed", flush=True)

    # ---- fixed eval sets (identical seeds to eval_cfg in earlier rounds) ----
    eval_sets = {}
    for name, k_ctx, kf in CONFIGS:
        r = np.random.default_rng((k_ctx or 0) * 977 + int(kf * 100))
        min_len = (k_ctx + 8) if k_ctx else 24
        sel = []
        for u in holdout_idx:
            if lengths[u] >= min_len:
                sel.append(u)
            if len(sel) == EVAL_N:
                break
        users = []
        all_novs = []
        for u in sel:
            s0, l = starts[u], lengths[u]
            idxs = indices[s0 : s0 + l]
            vals = values[s0 : s0 + l]
            if k_ctx:
                kp = r.choice(l, size=k_ctx, replace=False)
                keep = np.zeros(l, dtype=bool); keep[kp] = True
            else:
                keep = r.random(l) > (1 - kf)
                if keep.sum() == 0: keep[0] = True
                if (~keep).sum() == 0: keep[0] = False
            kept, dropped = idxs[keep], idxs[~keep]
            novs = 1.0 - cos[np.ix_(dropped, kept)].max(axis=1)
            cand_franch = np.isin(comp_of_corpus, np.unique(comp_of_corpus[kept]))
            users.append((kept, vals[keep], dropped, novs, cand_franch))
            all_novs.append(novs)
        pooled = np.concatenate(all_novs)
        nov_bounds = [-1e9] + [float(np.percentile(pooled, p)) for p in (25, 50, 75)] + [1e9]
        eval_sets[name] = (users, nov_bounds)
        print(f"eval set {name}: {len(users)} users, nov bounds "
              f"{[round(b, 3) for b in nov_bounds[1:4]]}", flush=True)

    def rank_map(sc):
        o = np.argsort(-sc)
        ro = np.empty(cs, dtype=np.int32)
        ro[o] = np.arange(cs)
        return o, ro

    results = {}
    for spec in model_specs:
        mname = spec["name"]
        mtype = spec["type"]
        params = fwd = None
        if mtype == "prod":
            params, fwd = load_prod(spec["path"], spec.get("bd", 512), cs)
        elif mtype == "ease3ch":
            params, fwd = load_ease3ch(spec["path"], cs, B)
        elif mtype == "graft":
            params, fwd = load_graft(spec["path"], spec["mode"], spec.get("bd", 512), cs, B)
        elif mtype == "blend":
            params, fwd = load_prod(spec["path"], spec.get("bd", 512), cs)
        results[mname] = {}

        for cfg_name, _k, _kf in CONFIGS:
            users, nov_bounds = eval_sets[cfg_name]
            agg_u, agg_f = Agg(), Agg()
            bx = np.zeros((256, cs * 2), dtype=np.float32)
            metas = []

            def flush(nb):
                if fwd is not None:
                    lg = np.asarray(fwd(params, jnp.asarray(bx[:nb])), dtype=np.float64)
                for j in range(nb):
                    kept, dropped, novs, cand_franch = metas[j]
                    if mtype == "ease":
                        s = B[kept].sum(axis=0).astype(np.float64)
                        sc = s + EASE_BETA * log_pop
                    elif mtype == "blend":
                        s = B[kept].sum(axis=0).astype(np.float64)
                        ze = s - mu_ease
                        ze = (ze - ze.mean()) / (ze.std() + 1e-9)
                        zn = (lg[j] - lg[j].mean()) / (lg[j].std() + 1e-9)
                        w = spec["w"] if len(kept) >= spec.get("min_ctx", 0) else 0.0
                        sc = (1 - w) * zn + w * ze + spec["alpha"] * zlp
                    else:
                        sc = lg[j] + log_pop
                    sc = sc.copy()
                    sc[kept] = -np.inf
                    o, ro = rank_map(sc)
                    tt = tier_of_item[dropped]
                    agg_u.add_targets(ro[dropped], tt, novs)
                    top10 = o[:10]
                    t10nov = 1.0 - cos[np.ix_(top10, kept)].max(axis=1)
                    agg_u.add_list(float(rank_of_item[top10].mean()),
                                   float(cand_franch[top10].mean()), float(t10nov.mean()))
                    scf = sc.copy()
                    scf[cand_franch] = -np.inf
                    of, rof = rank_map(scf)
                    tgt_ok = ~cand_franch[dropped]
                    agg_f.add_targets(rof[dropped[tgt_ok]], tt[tgt_ok], novs[tgt_ok])
                    t10f = of[:10]
                    t10fnov = 1.0 - cos[np.ix_(t10f, kept)].max(axis=1)
                    agg_f.add_list(float(rank_of_item[t10f].mean()), 0.0, float(t10fnov.mean()))
                metas.clear()

            nb = 0
            for kept, kvals, dropped, novs, cand_franch in users:
                if fwd is not None:
                    bx[nb] = 0.0
                    bx[nb, kept] = 1.0
                    bx[nb, cs + kept] = kvals
                metas.append((kept, dropped, novs, cand_franch))
                nb += 1
                if nb == 256:
                    flush(nb); nb = 0
            if nb:
                flush(nb)
            results[mname][cfg_name] = {"unfiltered": agg_u.stats(nov_bounds),
                                        "filtered": agg_f.stats(nov_bounds)}
            u, fl = results[mname][cfg_name]["unfiltered"], results[mname][cfg_name]["filtered"]
            print(f"{mname} {cfg_name}: unf medrank {u['median_rank']:.0f} r250 {u['r250']:.3f} "
                  f"franch10 {u['franchise_share_top10']:.2f} | filt r250 {fl['r250']:.3f} "
                  f"tail {fl['r250_tier3000_6000']} novq4 {fl['r250_novq4']} "
                  f"pop10 {fl['mean_top10_poprank']:.0f}", flush=True)

    with open(args.out, "w") as f:
        json.dump({"nov_bounds_note": "per-config quartiles of target novelty (1 - max gram-cosine to kept)",
                   "configs": [c[0] for c in CONFIGS], "models": results}, f, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
