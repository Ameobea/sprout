"""
Deterministic model-quality eval over frozen fixture profiles.

Protocol: leave-one-out over every in-corpus item of every fixture profile (no RNG
anywhere). Profiles are keyed by anime_id and mapped into the corpus of the model
under eval, so any (weights, corpus, filtering, normalization) combination gets the
same treatment. For cross-version comparisons where corpora differ, pass
--restrict-corpus with the OTHER model's corpus_ids.json to score both models on the
shared item set only.

Run (CPU, safe while GPU is training):
  JAX_PLATFORMS=cpu python eval_harness.py \
    --weights ../../data/jax_model_dec2025.msgpack \
    --corpus ../../data/corpus_ids.json \
    --name dec2025-baseline
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
from flax import serialization
from jax import random

from model import (
    CONF,
    Recommender,
    batch_holdout_predict,
    compute_recommendation_ranking_score,
    create_holdout_batch,
)
from profile_preprocessing import filter_profile_entries, vectorize_entries

FIXTURES_DIR = Path(__file__).parent / "fixtures"
REPORTS_DIR = Path(__file__).parent / "reports"
SENTINELS = ["ameo___", "snapsauce"]
RECALL_KS = [10, 50, 100]


def load_fixture_profiles():
    profiles = {}
    for name in SENTINELS:
        with open(FIXTURES_DIR / f"{name}.json") as f:
            raw = json.load(f)
        items = [
            (e["node"]["id"], e["list_status"].get("score", 0) or 0, e["list_status"].get("status", ""))
            for e in raw
            if e.get("list_status")
        ]
        profiles[name] = {"bucket": "sentinel", "items": items}
    with open(FIXTURES_DIR / "sampled_profiles.json") as f:
        profiles.update(json.load(f))
    v2 = FIXTURES_DIR / "sampled_profiles_v2.json"
    if v2.exists():
        with open(v2) as f:
            profiles.update(json.load(f))
    return profiles


def preprocess(items, id_to_idx, restrict_ids=None):
    kept = filter_profile_entries(items, id_to_idx, restrict_ids)
    idxs, normalized, original, _ = vectorize_entries(kept)
    statuses = np.array([k[3] for k in kept])
    return idxs, normalized, original, statuses


def build_aux(original, statuses):
    nch = CONF["input_channels"]
    if nch <= 2:
        return None
    rows = [(original > 0).astype(np.float32)]
    if nch == 5:
        rows.append((statuses == "dropped").astype(np.float32))
        rows.append(np.isin(statuses, ("watching", "on_hold")).astype(np.float32))
    return np.stack(rows)


def load_params(weights_path, bf16_sim=False):
    model = Recommender()
    rng = random.PRNGKey(0)
    dummy = jnp.ones((1, CONF["corpus_size"] * CONF["input_channels"]))
    params = model.init({"params": rng, "noise": rng}, dummy)["params"]
    with open(weights_path, "rb") as f:
        params = serialization.from_bytes(params, f.read())
    if bf16_sim:
        params = jax.tree_util.tree_map(
            lambda a: a.astype(jnp.bfloat16).astype(a.dtype), params
        )
    return params


def load_graft(weights_path, ease_b_path, bf16_sim=False):
    # B stays f32 under bf16_sim, matching the Rust engine (packed weights bf16, B f32)
    from analysis.train_probe_graft import GraftRecommender

    cs = CONF["corpus_size"]
    model = GraftRecommender(mode="concat")
    rng = random.PRNGKey(0)
    params = model.init(
        {"params": rng, "noise": rng}, jnp.ones((1, cs * 2)), jnp.ones((1, cs))
    )["params"]
    with open(weights_path, "rb") as f:
        params = serialization.from_bytes(params, f.read())
    if bf16_sim:
        params = jax.tree_util.tree_map(
            lambda a: a.astype(jnp.bfloat16).astype(a.dtype), params
        )
    Bj = jnp.asarray(np.load(ease_b_path), dtype=jnp.float32)

    @jax.jit
    def fwd(p, x):
        e = x[:, :cs] @ Bj
        e = (e - jnp.mean(e, axis=1, keepdims=True)) / (jnp.std(e, axis=1, keepdims=True) + 1e-6)
        out = model.apply({"params": p}, x, e, training=False)
        return out[0], out[1]

    return params, fwd


def load_rc(weights_path, ease_b_path, bf16_sim=False):
    """3ch [presence | z-mix | abs] + concat graft (train_probe_rc.RCRecommender)."""
    from analysis.train_probe_rc import RCRecommender

    cs = CONF["corpus_size"]
    model = RCRecommender()
    rng = random.PRNGKey(0)
    params = model.init(
        {"params": rng, "noise": rng}, jnp.ones((1, cs * 3)), jnp.ones((1, cs))
    )["params"]
    with open(weights_path, "rb") as f:
        params = serialization.from_bytes(params, f.read())
    if bf16_sim:
        params = jax.tree_util.tree_map(
            lambda a: a.astype(jnp.bfloat16).astype(a.dtype), params
        )
    Bj = jnp.asarray(np.load(ease_b_path), dtype=jnp.float32)

    @jax.jit
    def fwd(p, x):
        e = x[:, :cs] @ Bj
        e = (e - jnp.mean(e, axis=1, keepdims=True)) / (jnp.std(e, axis=1, keepdims=True) + 1e-6)
        out = model.apply({"params": p}, x, e, training=False)
        return out[0], out[1]

    return params, fwd


def rc_abs_aux(original):
    return np.where(original > 0, (original - 5.5) / 2.5, 0.0).astype(np.float32)[None, :]


GRAFT_CHUNK = 256


def graft_fwd_chunked(graft_fwd, params, batch):
    """Fixed-shape chunks so the jit compiles once (mirrors _batch_holdout_predict_gpu;
    variable shapes recompile per profile and intermittently emit bad HSACOs on gfx1201)."""
    n, d = batch.shape
    logits, ratings = [], []
    for s in range(0, n, GRAFT_CHUNK):
        chunk = batch[s : s + GRAFT_CHUNK]
        if chunk.shape[0] < GRAFT_CHUNK:
            chunk = np.concatenate([chunk, np.zeros((GRAFT_CHUNK - chunk.shape[0], d), dtype=chunk.dtype)])
        lg, rt = graft_fwd(params, jnp.asarray(chunk))
        take = min(GRAFT_CHUNK, n - s)
        logits.append(np.asarray(lg)[:take])
        ratings.append(np.asarray(rt)[:take])
    return np.concatenate(logits), np.concatenate(ratings)


def eval_profile(params, idxs, vals, original, statuses, corpus_size, device, serve_prior=None,
                 graft_fwd=None):
    n = len(idxs)
    if graft_fwd is not None:
        aux = rc_abs_aux(original) if CONF["input_channels"] == 3 else None
        batch, _, _, _ = create_holdout_batch(idxs, vals, corpus_size, aux=aux)
        item_logits, rating_pred = graft_fwd_chunked(graft_fwd, params, np.asarray(batch))
        item_logits, rating_pred = item_logits[:n], rating_pred[:n]
    else:
        item_logits, rating_pred = batch_holdout_predict(
            params, idxs, vals, corpus_size, device=device, aux=build_aux(original, statuses)
        )
    item_logits = np.array(item_logits)
    if serve_prior is not None:
        item_logits = item_logits + serve_prior[None, :]
    rating_pred = np.array(rating_pred)

    probs = np.array(jax.nn.softmax(jnp.array(item_logits), axis=1))
    presence_probs = probs[np.arange(n), idxs]
    pred_ratings = rating_pred[np.arange(n), idxs]
    rating_errors = np.abs(pred_ratings - vals)

    # rank of each held-out item among all items outside the (reduced) input profile
    ranks = np.zeros(n, dtype=np.int32)
    profile_mask = np.zeros(corpus_size, dtype=bool)
    profile_mask[idxs] = True
    for i in range(n):
        score, _ = compute_recommendation_ranking_score(
            jnp.array(item_logits[i]), jnp.array(rating_pred[i])
        )
        score = np.array(score)
        candidate_mask = ~profile_mask
        candidate_mask[idxs[i]] = True
        held_score = score[idxs[i]]
        ranks[i] = 1 + int(np.sum(score[candidate_mask] > held_score))

    rated = original > 0
    return {
        "n_items": n,
        "n_rated": int(np.sum(rated)),
        "rating_mae": float(np.mean(rating_errors[rated])) if rated.any() else None,
        "mean_presence_prob": float(np.mean(presence_probs)),
        "median_rank": float(np.median(ranks)),
        **{f"recall@{k}": float(np.mean(ranks <= k)) for k in RECALL_KS},
    }


def aggregate(results):
    keys = ["rating_mae", "mean_presence_prob", "median_rank"] + [f"recall@{k}" for k in RECALL_KS]
    agg = {"n_profiles": len(results)}
    for k in keys:
        vals = [r[k] for r in results if r.get(k) is not None]
        agg[k] = float(np.mean(vals)) if vals else None
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--restrict-corpus", help="corpus_ids.json of another model; eval on intersection only")
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--input-channels", type=int, default=2, choices=[2, 3, 5])
    ap.add_argument("--serve-prior-alpha", type=float, default=0.0,
                    help="add alpha*log_pop to logits before ranking (lift-trained models)")
    ap.add_argument("--popularity", default=str(Path(__file__).parent / "../../data/item_popularity_dec2025.npy"))
    ap.add_argument("--bf16-sim", action="store_true",
                    help="round-trip weights through bf16 to simulate prod serving numerics")
    ap.add_argument("--graft-ease-b", help="EASE B .npy; loads weights as a concat-graft model")
    args = ap.parse_args()
    CONF["input_channels"] = args.input_channels

    serve_prior = None
    if args.serve_prior_alpha != 0.0:
        counts = np.load(args.popularity)
        serve_prior = (args.serve_prior_alpha * np.log(np.maximum(counts, 1.0))).astype(np.float32)

    with open(args.corpus) as f:
        corpus_ids = json.load(f)
    assert len(corpus_ids) == CONF["corpus_size"], f"corpus size {len(corpus_ids)} != CONF {CONF['corpus_size']}"
    id_to_idx = {aid: i for i, aid in enumerate(corpus_ids)}

    restrict_ids = None
    if args.restrict_corpus:
        with open(args.restrict_corpus) as f:
            restrict_ids = set(json.load(f)) & set(corpus_ids)
        print(f"restricting to corpus intersection: {len(restrict_ids)} items")

    graft_fwd = None
    if args.graft_ease_b and args.input_channels == 3:
        params, graft_fwd = load_rc(args.weights, args.graft_ease_b, bf16_sim=args.bf16_sim)
    elif args.graft_ease_b:
        params, graft_fwd = load_graft(args.weights, args.graft_ease_b, bf16_sim=args.bf16_sim)
    else:
        params = load_params(args.weights, bf16_sim=args.bf16_sim)
    profiles = load_fixture_profiles()

    per_profile = {}
    skipped = []
    for i, (username, p) in enumerate(sorted(profiles.items())):
        idxs, vals, original, statuses = preprocess(p["items"], id_to_idx, restrict_ids)
        if len(idxs) < 5:
            skipped.append(username)
            continue
        per_profile[username] = {
            "bucket": p["bucket"],
            "coverage": len(idxs) / max(1, len(p["items"])),
            **eval_profile(params, idxs, vals, original, statuses, CONF["corpus_size"], args.device,
                           serve_prior, graft_fwd),
        }
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(profiles)} profiles evaluated")

    buckets = sorted({r["bucket"] for r in per_profile.values()})
    report = {
        "name": args.name,
        "weights": str(args.weights),
        "input_channels": CONF["input_channels"],
        "corpus": str(args.corpus),
        "restrict_corpus": args.restrict_corpus,
        "graft_ease_b": args.graft_ease_b,
        "skipped_too_small": skipped,
        "sentinels": {u: per_profile[u] for u in SENTINELS if u in per_profile},
        "by_bucket": {
            b: aggregate([r for r in per_profile.values() if r["bucket"] == b]) for b in buckets
        },
        "overall": aggregate(list(per_profile.values())),
        "overall_v1": aggregate([r for r in per_profile.values() if not r["bucket"].startswith("v2-")]),
        "overall_v2": aggregate([r for r in per_profile.values() if r["bucket"].startswith("v2-")]),
        "per_profile": per_profile,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"{args.name}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)

    print(f"\n=== {args.name} ===")
    for u, r in report["sentinels"].items():
        print(f"  {u}: mae={r['rating_mae']:.4f} recall@50={r['recall@50']:.3f} median_rank={r['median_rank']:.0f}")
    for b, a in report["by_bucket"].items():
        print(f"  [{b}] n={a['n_profiles']} mae={a['rating_mae']:.4f} recall@50={a['recall@50']:.3f} median_rank={a['median_rank']:.0f}")
    for key in ["overall", "overall_v1", "overall_v2"]:
        o = report[key]
        if o["n_profiles"]:
            print(f"  {key}: n={o['n_profiles']} mae={o['rating_mae']:.4f} recall@50={o['recall@50']:.3f} median_rank={o['median_rank']:.0f}")
    print(f"report written to {out_path}")


if __name__ == "__main__":
    main()
