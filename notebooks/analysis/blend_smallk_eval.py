"""NN vs EASE vs score-blends across absolute context sizes (cold-start regime)
and profile-size bins at full context. Tier-stratified rank metrics throughout.

Run inside rocm_jax: cd /jax_dir/notebooks && python analysis/blend_smallk_eval.py ...
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

TIERS = [(0, 250), (250, 1000), (1000, 3000), (3000, 6000)]
BLEND_WS = [0.2, 0.35, 0.5, 0.65, 0.8]


def load_params(path):
    model = Recommender()
    dummy = jnp.ones((1, CONF["corpus_size"] * CONF["input_channels"]))
    params = model.init({"params": jax.random.PRNGKey(0), "noise": jax.random.PRNGKey(0)}, dummy)["params"]
    with open(path, "rb") as f:
        return serialization.from_bytes(params, f.read())


@jax.jit
def forward_clean(params, x):
    logits, ratings, _, _ = Recommender().apply({"params": params}, x, training=False)
    return logits, ratings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="../data/aug2026/jax_model_fresh_logq.msgpack")
    ap.add_argument("--ease-b", default="../data/aug2026/ease_B6k_lam200.npy")
    ap.add_argument("--vectors", default="../data/aug2026/user_input_vectors_cleanup_notrust.npz")
    ap.add_argument("--n-users", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cs = CONF["corpus_size"]
    d = np.load(args.vectors)
    indices = d["indices"].astype(np.int32)
    values = d["values"]
    lengths = d["lengths"].astype(np.int64)
    starts = np.zeros(len(lengths), dtype=np.int64)
    np.cumsum(lengths[:-1], out=starts[1:])

    counts = np.bincount(indices, minlength=cs).astype(np.float64)
    log_pop = np.log(np.maximum(counts, 1.0) / np.maximum(counts, 1.0).sum())
    prior = jnp.asarray(log_pop, dtype=jnp.float32)
    order = np.argsort(-counts)
    rank_of_item = np.empty(cs, dtype=np.int32)
    rank_of_item[order] = np.arange(cs)
    tier_of_item = np.zeros(cs, dtype=np.int8)
    for t, (lo, hi) in enumerate(TIERS):
        tier_of_item[(rank_of_item >= lo) & (rank_of_item < hi)] = t

    B = np.load(args.ease_b)
    params = load_params(args.weights)

    rng = np.random.default_rng(args.seed)
    pool = rng.choice(len(lengths), size=60000, replace=False)

    class Agg:
        def __init__(self):
            self.ranks = []
            self.tiers = []
        def add(self, rr, tt):
            self.ranks.extend(rr.tolist())
            self.tiers.extend(tt.tolist())
        def stats(self):
            rk = np.asarray(self.ranks); tt = np.asarray(self.tiers)
            if len(rk) == 0:
                return None
            s = {"n": len(rk), "median_rank": float(np.median(rk)),
                 "recall@50": float((rk < 50).mean()), "recall@250": float((rk < 250).mean())}
            s["by_tier"] = [
                {"tier": f"{lo}-{hi}", "n": int((tt == t).sum()),
                 "recall@50": float((rk[tt == t] < 50).mean()) if (tt == t).any() else None,
                 "recall@250": float((rk[tt == t] < 250).mean()) if (tt == t).any() else None}
                for t, (lo, hi) in enumerate(TIERS)]
            return s

    def znorm(v):
        s = v.std()
        return (v - v.mean()) / (s + 1e-9)

    def run_config(k_ctx, keep_frac, n_users, size_bins=None):
        """k_ctx: absolute kept count (None -> use keep_frac)."""
        min_len = (k_ctx + 8) if k_ctx else 24
        users = [u for u in pool if lengths[u] >= min_len][:n_users]
        r = np.random.default_rng((k_ctx or 0) * 1000 + int(keep_frac * 100) + 5)
        variants = ["nn", "ease"] + [f"blend_{w}" for w in BLEND_WS]
        aggs = {v: Agg() for v in variants}
        bin_aggs = {}
        if size_bins:
            for v in variants:
                for b in range(len(size_bins) - 1):
                    bin_aggs[(v, b)] = Agg()

        batch_x = np.zeros((256, cs * 2), dtype=np.float32)
        metas = []

        def flush(nb):
            lg, _ = forward_clean(params, jnp.asarray(batch_x[:nb]))
            lgp = np.asarray(lg, dtype=np.float64) + log_pop[None, :]
            for j in range(nb):
                kept, dropped, u = metas[j]
                nn_s = znorm(lgp[j])
                ease_s = znorm(B[kept].sum(axis=0).astype(np.float64) + 0.2167 * log_pop)
                tt = tier_of_item[dropped]
                for v in aggs:
                    if v == "nn":
                        sc = nn_s.copy()
                    elif v == "ease":
                        sc = ease_s.copy()
                    else:
                        w = float(v.split("_")[1])
                        sc = (1 - w) * nn_s + w * ease_s
                    sc[kept] = -np.inf
                    o = np.argsort(-sc)
                    ro = np.empty(cs, dtype=np.int32)
                    ro[o] = np.arange(cs)
                    rr = ro[dropped]
                    aggs[v].add(rr, tt)
                    if size_bins:
                        b = np.searchsorted(size_bins, lengths[u], side="right") - 1
                        if 0 <= b < len(size_bins) - 1:
                            bin_aggs[(v, b)].add(rr, tt)
            metas.clear()

        nb = 0
        for u in users:
            s0, l = starts[u], lengths[u]
            idx = indices[s0 : s0 + l]
            val = values[s0 : s0 + l]
            if k_ctx:
                kept_pos = r.choice(l, size=k_ctx, replace=False)
                keep = np.zeros(l, dtype=bool)
                keep[kept_pos] = True
            else:
                keep = r.random(l) > (1 - keep_frac)
                if keep.sum() == 0:
                    keep[r.integers(l)] = True
                if (~keep).sum() == 0:
                    keep[r.integers(l)] = False
            kept, dropped = idx[keep], idx[~keep]
            batch_x[nb] = 0.0
            batch_x[nb, kept] = 1.0
            batch_x[nb, cs + kept] = val[keep]
            metas.append((kept, dropped, u))
            nb += 1
            if nb == 256:
                flush(nb); nb = 0
        if nb:
            flush(nb)

        res = {v: aggs[v].stats() for v in aggs}
        if size_bins:
            res["by_size_bin"] = {
                f"{size_bins[b]}-{size_bins[b+1]}": {v: bin_aggs[(v, b)].stats() for v in variants}
                for b in range(len(size_bins) - 1)}
        return res

    out = {"blend_ws": BLEND_WS}
    for k in [4, 8, 16, 32, 64, 128]:
        out[f"k_{k}"] = run_config(k, 0, args.n_users)
        r = out[f"k_{k}"]
        print(f"k={k}: NN {r['nn']['median_rank']:.0f}/{r['nn']['recall@50']:.3f} "
              f"EASE {r['ease']['median_rank']:.0f}/{r['ease']['recall@50']:.3f} "
              f"blend0.5 {r['blend_0.5']['median_rank']:.0f}/{r['blend_0.5']['recall@50']:.3f}", flush=True)
    out["full"] = run_config(None, 0.99, args.n_users, size_bins=[24, 60, 150, 400, 2000])
    r = out["full"]
    print(f"full: NN {r['nn']['median_rank']:.0f}/{r['nn']['recall@50']:.3f} "
          f"EASE {r['ease']['median_rank']:.0f}/{r['ease']['recall@50']:.3f} "
          f"blend0.5 {r['blend_0.5']['median_rank']:.0f}/{r['blend_0.5']['recall@50']:.3f}", flush=True)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
