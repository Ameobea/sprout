"""Absolute-score input channel probe: standard multitask arch + protocol with
a third input channel carrying (raw_score - 5.5) / 2.5 for rated entries
(0 unrated/dropped), from raw_scores_recon.npy. Tests whether the alpha-mix
z channel loses usable input information (e.g. 1-vs-2 clipping for generous
raters). Ends with a rating_floors_dump npz. Run inside rocm_jax."""

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import jax
import jax.numpy as jnp
import optax
from flax import serialization
from jax import random
from optax import contrib

from model import CONF, Recommender
from train import TrainState, load_all_users

HOLDOUT_SEED = 999
EVAL_N = 2048
MODEL = None


@jax.jit
def train_step_t(state, batch, rated_mask, prior_logits):
    cs = CONF["corpus_size"]
    presence = batch[:, :cs]
    ratings_z = batch[:, cs : 2 * cs]

    dropout_rng, vae_rng = random.split(state.key)
    rate_variation = CONF["dropout_variation"] * CONF["dropout_rate"]
    random_rates = (
        CONF["dropout_rate"]
        + random.uniform(dropout_rng, shape=(presence.shape[0], 1)) * (2 * rate_variation)
        - rate_variation
    )
    random_rates = jnp.clip(random_rates, 0.01, 0.75)
    keep = random.bernoulli(dropout_rng, p=(1.0 - random_rates), shape=presence.shape)

    b = presence.shape[0]
    x_in = (batch.reshape(b, 3, cs) * keep[:, None, :]).reshape(b, 3 * cs)

    def loss_fn(params):
        item_logits, rating_pred, log_var_p, log_var_r = state.apply_fn(
            {"params": params}, x_in, training=True, rngs={"noise": vae_rng}
        )
        log_probs = jax.nn.log_softmax(item_logits + prior_logits[None, :], axis=1)
        counts = jnp.maximum(jnp.sum(presence, axis=1), 1.0)
        p_loss = jnp.mean(-jnp.sum(presence * log_probs, axis=1) / counts)
        per_entry = optax.huber_loss(rating_pred - ratings_z, delta=CONF["huber_delta"])
        denom = jnp.maximum(jnp.sum(rated_mask, axis=1), 1.0)
        r_loss = jnp.mean(jnp.sum(rated_mask * per_entry, axis=1) / denom)
        weighted = (jnp.exp(-log_var_p) * p_loss + log_var_p) + (
            jnp.exp(-log_var_r) * r_loss + log_var_r
        )
        return jnp.mean(weighted), (p_loss, r_loss)

    (loss, (p_loss, r_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    updates, new_opt = state.tx.update(grads, state.opt_state, state.params, value=loss)
    new_params = optax.apply_updates(state.params, updates)
    return state.replace(step=state.step + 1, params=new_params, opt_state=new_opt,
                         key=dropout_rng), loss, p_loss, r_loss


@jax.jit
def forward_t(params, x):
    logits, ratings, _, _ = MODEL.apply({"params": params}, x, training=False)
    return logits, ratings


def gen_batches(users, tvals, batch_size, cs):
    while True:
        perm = np.random.permutation(len(users))
        for b in range(0, len(users), batch_size):
            sel = perm[b : b + batch_size]
            bt = np.zeros((len(sel), cs * 3), dtype=np.float32)
            rm = np.zeros((len(sel), cs), dtype=np.float32)
            for i, ui in enumerate(sel):
                idxs, vals, rated, _st = users[ui]
                bt[i, idxs] = 1.0
                bt[i, cs + idxs] = vals
                bt[i, 2 * cs + idxs] = tvals[ui]
                rm[i, idxs] = rated.astype(np.float32)
            yield bt, rm


def eval_holdout(params, users, tvals, prior, cs, rng):
    x = np.zeros((len(users), cs * 3), dtype=np.float32)
    rows, rateds, keeps = [], [], []
    rates = rng.uniform(0.24, 0.56, size=len(users))
    for i, ((idx, vals, rated_m, _st), tv) in enumerate(zip(users, tvals)):
        x[i, idx] = 1.0
        x[i, cs + idx] = vals
        x[i, 2 * cs + idx] = tv
        rows.append(idx.astype(np.int64))
        rateds.append(rated_m.astype(bool))
        keep = rng.random(len(idx)) > rates[i]
        if keep.sum() == 0:
            keep[0] = True
        keeps.append(keep)
    xc = x.copy()
    for i, idx in enumerate(rows):
        dropped = idx[~keeps[i]]
        for ch in range(3):
            xc[i, ch * cs + dropped] = 0.0

    agg = dict(pl=0.0, n=0, nll_s=0.0, nll_c=0, mae_s=0.0, mae_c=0)
    for b in range(0, len(rows), 512):
        lg, rt = forward_t(params, jnp.asarray(xc[b : b + 512]))
        lp = np.asarray(jax.nn.log_softmax(lg + prior[None, :], axis=1), dtype=np.float64)
        rp = np.asarray(rt, dtype=np.float64)
        for j in range(lp.shape[0]):
            i = b + j
            idx = rows[i]
            keep = keeps[i]
            nll = -lp[j, idx]
            err = rp[j, idx] - x[i, cs + idx]
            agg["pl"] += nll.sum() / len(idx)
            agg["nll_s"] += nll[~keep].sum(); agg["nll_c"] += int((~keep).sum())
            m = rateds[i] & ~keep
            agg["mae_s"] += np.abs(err[m]).sum(); agg["mae_c"] += int(m.sum())
            agg["n"] += 1
    return {"presence_loss": agg["pl"] / agg["n"],
            "nll_drop_per_item": agg["nll_s"] / max(agg["nll_c"], 1),
            "mae_drop_per_item": agg["mae_s"] / max(agg["mae_c"], 1)}


def floors_dump(params, all_users, tvals_all, holdout_idx, cs, out, n_users=20_000, seed=555):
    users = [all_users[i] for i in holdout_idx[:n_users]]
    tvals = [tvals_all[i] for i in holdout_idx[:n_users]]
    rng = np.random.default_rng(seed)
    rates = rng.uniform(0.24, 0.56, size=len(users))
    keeps = []
    for i, (idx, _v, _r, _s) in enumerate(users):
        keep = rng.random(len(idx)) > rates[i]
        if keep.sum() == 0:
            keep[0] = True
        keeps.append(keep)

    d = {k: [] for k in ("drop_user", "drop_item", "drop_tgt", "drop_pred",
                          "kept_user", "kept_item", "kept_tgt", "kept_pred")}
    B = 512
    for b in range(0, len(users), B):
        chunk = users[b : b + B]
        x = np.zeros((len(chunk), cs * 3), dtype=np.float32)
        for j, (idx, vals, _r, _s) in enumerate(chunk):
            keep = keeps[b + j]
            x[j, idx[keep]] = 1.0
            x[j, cs + idx[keep]] = vals[keep]
            x[j, 2 * cs + idx[keep]] = tvals[b + j][keep]
        _, preds = forward_t(params, jnp.asarray(x))
        preds = np.asarray(preds, dtype=np.float32)
        for j, (idx, vals, rated, _s) in enumerate(chunk):
            keep = keeps[b + j]
            for pre, m in (("drop", rated & ~keep), ("kept", rated & keep)):
                d[pre + "_user"].append(np.full(m.sum(), b + j, dtype=np.int32))
                d[pre + "_item"].append(idx[m].astype(np.int32))
                d[pre + "_tgt"].append(vals[m])
                d[pre + "_pred"].append(preds[j, idx[m]])
        if (b // B) % 8 == 0:
            print(f"dump {b + len(chunk)}/{len(users)}", flush=True)
    np.savez(out, holdout_rows=np.asarray(holdout_idx[:n_users], dtype=np.int64),
             **{k: np.concatenate(v) for k, v in d.items()})
    print(f"dump saved: {out}", flush=True)


def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-scores", default="../data/aug2026/raw_scores_recon.npy")
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--vectors", default="../data/aug2026/user_input_vectors_cleanup_notrust.npz")
    ap.add_argument("--presence-prior-alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    MODEL = Recommender()
    cs = CONF["corpus_size"]
    all_users, item_counts = load_all_users(args.vectors)

    raw = np.load(args.raw_scores)
    rated_all = (raw >= 1) & (raw <= 10)
    absv = np.where(rated_all, (raw.astype(np.float32) - 5.5) / 2.5, 0.0).astype(np.float32)
    tvals_all = []
    pos = 0
    for u in all_users:
        l = len(u[0])
        tvals_all.append(absv[pos : pos + l])
        pos += l
    assert pos == len(absv)
    print(f"abs channel built, rated frac {rated_all.mean():.4f}", flush=True)

    rng_h = np.random.default_rng(HOLDOUT_SEED)
    perm = rng_h.permutation(len(all_users))
    n_hold = len(all_users) // 10
    holdout_idx = perm[:n_hold]
    train_pool = perm[n_hold:]
    train_users = [all_users[i] for i in train_pool]
    train_tvals = [tvals_all[i] for i in train_pool]
    print(f"train users: {len(train_users)}  holdout: {n_hold}", flush=True)

    clipped = np.maximum(item_counts, 1.0)
    prior = jnp.asarray(args.presence_prior_alpha * np.log(clipped / clipped.sum()), dtype=jnp.float32)

    params = MODEL.init({"params": jax.random.PRNGKey(args.seed), "noise": jax.random.PRNGKey(args.seed)},
                        jnp.ones((1, cs * 3)))["params"]
    tx = optax.chain(
        optax.adam(CONF["learning_rate"]),
        contrib.reduce_on_plateau(patience=5, cooldown=1, factor=0.5, rtol=1e-4, accumulation_size=200),
    )
    state = TrainState.create(apply_fn=MODEL.apply, params=params, tx=tx, key=jax.random.PRNGKey(args.seed))
    loader = gen_batches(train_users, train_tvals, CONF["batch_size"], cs)

    rng_e = np.random.default_rng(4242)
    hold_users = [all_users[i] for i in holdout_idx[:EVAL_N]]
    hold_tvals = [tvals_all[i] for i in holdout_idx[:EVAL_N]]

    logf = open(f"{args.out_prefix}.jsonl", "w")
    t0 = time.time()
    for step in range(args.steps):
        bt, rm = next(loader)
        state, loss, p_loss, r_loss = train_step_t(state, jnp.array(bt), jnp.array(rm), prior)
        if step % 500 == 0:
            lr_scale = optax.tree.get(state.opt_state, "scale")
            print(f"Step {step}: Loss {loss:.4f} (P {p_loss:.4f} R {r_loss:.4f}) lr {lr_scale} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            if lr_scale < 1e-6:
                print("LR decayed below 1e-6, stopping.", flush=True)
                break
        if step % 2000 == 0 and step > 0:
            ev = eval_holdout(state.params, hold_users, hold_tvals, prior, cs, np.random.default_rng(4242))
            ev["step"] = step
            logf.write(json.dumps(ev) + "\n")
            logf.flush()
            print(f"[eval @ {step}] holdout: pres {ev['presence_loss']:.4f} "
                  f"nll_drop {ev['nll_drop_per_item']:.4f} mae_drop {ev['mae_drop_per_item']:.4f}", flush=True)

    ev = eval_holdout(state.params, hold_users, hold_tvals, prior, cs, np.random.default_rng(4242))
    ev["step"] = args.steps
    logf.write(json.dumps(ev) + "\n")
    logf.close()
    with open(f"{args.out_prefix}.msgpack", "wb") as f:
        f.write(serialization.to_bytes(state.params))
    floors_dump(state.params, all_users, tvals_all, holdout_idx, cs, f"{args.out_prefix}_dump.npz")
    print(f"done: {args.out_prefix}", flush=True)


if __name__ == "__main__":
    main()
