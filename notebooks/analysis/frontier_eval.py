"""Matched-overall-recall frontier comparison: NN-lift vs EASE-lift vs blend,
each served as lam_k*lift + alpha*log_pop (the logQ (alpha,k) family), with
franchise-aware (prod extra-season filter) variants of every metric.

EASE lift-ization: s_j(u) - mu_j, where mu_j = mean full-profile EASE score of
item j over a reference user sample — the serve-side analog of the logQ prior.

Run inside rocm_jax: cd /jax_dir/notebooks && python analysis/frontier_eval.py ...
"""

import argparse
import csv
import json
import sys

import numpy as np

sys.path.insert(0, ".")
import jax
import jax.numpy as jnp
from flax import serialization
from model import CONF, Recommender

TIERS = [(0, 250), (250, 1000), (1000, 3000), (3000, 6000)]
SEASON_RELS = {"sequel", "prequel", "parent_story", "side_story"}


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


class DSU:
    def __init__(self):
        self.p = {}
    def find(self, a):
        while self.p.setdefault(a, a) != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def build_components(metadata_path):
    dsu = DSU()
    with open(metadata_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            aid = int(row["id"])
            rel = row["related_anime"]
            if not rel:
                continue
            try:
                entries = json.loads(rel)
            except json.JSONDecodeError:
                continue
            for e in entries:
                if e.get("relation_type") in SEASON_RELS:
                    dsu.union(aid, e["node"]["id"])
    return dsu


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="../data/aug2026/jax_model_fresh_logq.msgpack")
    ap.add_argument("--ease-b", default="../data/aug2026/ease_B6k_lam200.npy")
    ap.add_argument("--vectors", default="../data/aug2026/user_input_vectors_cleanup_notrust.npz")
    ap.add_argument("--corpus", default="../data/corpus_ids_aug2026.json")
    ap.add_argument("--metadata", default="../data/processed-metadata_aug2026.csv")
    ap.add_argument("--n-users", type=int, default=5000)
    ap.add_argument("--n-ref", type=int, default=20000)
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
    zlp = (log_pop - log_pop.mean()) / log_pop.std()
    prior = jnp.asarray(log_pop, dtype=jnp.float32)
    order = np.argsort(-counts)
    rank_of_item = np.empty(cs, dtype=np.int32)
    rank_of_item[order] = np.arange(cs)
    tier_of_item = np.zeros(cs, dtype=np.int8)
    for t, (lo, hi) in enumerate(TIERS):
        tier_of_item[(rank_of_item >= lo) & (rank_of_item < hi)] = t

    with open(args.corpus) as f:
        corpus_ids = np.array(json.load(f), dtype=np.int64)
    print("building season components...", flush=True)
    dsu = build_components(args.metadata)
    comp_of_corpus = np.array([dsu.find(int(a)) for a in corpus_ids], dtype=np.int64)

    B = np.load(args.ease_b)
    params = load_params(args.weights)

    rng = np.random.default_rng(args.seed)
    pool = rng.choice(len(lengths), size=args.n_ref + args.n_users + 5000, replace=False)
    ref_users = pool[: args.n_ref]
    eval_pool = [u for u in pool[args.n_ref:] if lengths[u] >= 24][: args.n_users]

    # ---- EASE reference stats (full-profile scores over reference users) ----
    print("EASE reference stats...", flush=True)
    s_sum = np.zeros(cs); s_sq = np.zeros(cs)
    for u in ref_users:
        s0, l = starts[u], lengths[u]
        s = B[indices[s0 : s0 + l]].sum(axis=0).astype(np.float64)
        s_sum += s; s_sq += s * s
    mu = s_sum / len(ref_users)
    sigma = np.sqrt(np.maximum(s_sq / len(ref_users) - mu * mu, 1e-9))

    # popularity-correlation diagnostics on 200 users
    rho = {"ease_raw": [], "ease_lift": [], "nn_lift": []}
    diag_users = eval_pool[:200]
    bx = np.zeros((len(diag_users), cs * 2), dtype=np.float32)
    for i, u in enumerate(diag_users):
        s0, l = starts[u], lengths[u]
        idx = indices[s0 : s0 + l]
        bx[i, idx] = 1.0
        bx[i, cs + idx] = values[s0 : s0 + l]
    lg, _ = forward_clean(params, jnp.asarray(bx))
    lg = np.asarray(lg, dtype=np.float64)
    for i, u in enumerate(diag_users):
        s0, l = starts[u], lengths[u]
        s = B[indices[s0 : s0 + l]].sum(axis=0).astype(np.float64)
        rho["ease_raw"].append(spearman(s, counts))
        rho["ease_lift"].append(spearman(s - mu, counts))
        rho["nn_lift"].append(spearman(lg[i], counts))
    rho = {k: float(np.mean(v)) for k, v in rho.items()}
    print("spearman(score, popularity):", rho, flush=True)

    # ---- frontier sweep ----
    ALPHAS = [0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0, 1.3]
    KS = [0, 2750]
    FAMILIES = ["nn", "ease", "blend0.5", "blend0.35"]

    lam = {k: (counts / (counts + k) if k else np.ones(cs)) for k in KS}

    def znorm_rows(m):
        return (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-9)

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

    agg = Agg()      # unfiltered
    agg_f = Agg()    # prod-filter simulated: same-component candidates masked, targets restricted

    r = np.random.default_rng(777)
    bx = np.zeros((256, cs * 2), dtype=np.float32)
    metas = []

    def flush(nb):
        lg, _ = forward_clean(params, jnp.asarray(bx[:nb]))
        lgv = np.asarray(lg, dtype=np.float64)
        for j in range(nb):
            kept, dropped, u = metas[j]
            lift_nn = lgv[j]
            s = B[kept].sum(axis=0).astype(np.float64)
            lift_ease = s - mu
            zn = (lift_nn - lift_nn.mean()) / (lift_nn.std() + 1e-9)
            ze = (lift_ease - lift_ease.mean()) / (lift_ease.std() + 1e-9)
            profile_comps = set(comp_of_corpus[kept].tolist()) | set(comp_of_corpus[dropped].tolist())
            cand_franch = np.isin(comp_of_corpus, np.array(sorted(set(comp_of_corpus[kept].tolist())), dtype=np.int64))
            tgt_franch = cand_franch[dropped]
            tt = tier_of_item[dropped]
            for fam in FAMILIES:
                base = zn if fam == "nn" else ze if fam == "ease" else (
                    (1 - float(fam[5:])) * zn + float(fam[5:]) * ze)
                for k in KS:
                    lifted = lam[k] * base
                    for a in ALPHAS:
                        key = f"{fam}|k{k}|a{a}"
                        sc = lifted + a * zlp
                        sc[kept] = -np.inf
                        o = np.argsort(-sc)
                        ro = np.empty(cs, dtype=np.int32)
                        ro[o] = np.arange(cs)
                        agg.add(key, ro[dropped], tt)
                        top10 = o[:10]
                        agg.add_list(key, float(rank_of_item[top10].mean()), float(cand_franch[top10].mean()))
                        # filtered variant
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
        metas.append((kept, dropped, u))
        nb += 1
        if nb == 256:
            flush(nb); nb = 0
        if len(metas) == 0 and (len(agg.r.get('nn|k0|a0.0', {'ranks': []})['ranks']) % 2000) < 3:
            print(".", end="", flush=True)
    if nb:
        flush(nb)

    out = {"rho_pop": rho, "alphas": ALPHAS, "ks": KS, "families": FAMILIES,
           "unfiltered": agg.stats(), "filtered": agg_f.stats()}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    for fam in FAMILIES:
        for k in KS:
            row = []
            for a in ALPHAS:
                s = out["unfiltered"][f"{fam}|k{k}|a{a}"]
                row.append(f"a{a}: {s['overall_r250']:.3f}/{s['r250_tier1000_3000']:.3f}/{s['r250_tier3000_6000']:.3f}")
            print(f"{fam} k={k}: " + " | ".join(row), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
