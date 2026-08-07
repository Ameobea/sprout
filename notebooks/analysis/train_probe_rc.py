"""Release-candidate recipe, fully composed: 3ch input [presence | z-mix | abs]
+ presence-side concat EASE graft (param names match model-server weights.rs)
+ muon/cosine. --user-frac probe mode (holdout 999, jsonl presence eval +
rating floors dump) or --full for the production all-data run.
Run inside rocm_jax from notebooks/."""

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import jax
import jax.numpy as jnp
import optax
import optax.contrib as oc
from flax import linen as nn
from flax import serialization
from jax import random

from model import CONF
from train import TrainState, load_all_users

HOLDOUT_SEED = 999
EVAL_N = 2048
B_g = None
MODEL = None


def znorm(e):
    mu = jnp.mean(e, axis=1, keepdims=True)
    sd = jnp.std(e, axis=1, keepdims=True) + 1e-6
    return (e - mu) / sd


def ease_channel(presence):
    return znorm(presence @ B_g)


class RCRecommender(nn.Module):
    hidden_dim: int = CONF["hidden_dim"]
    bottleneck_dim: int = CONF["bottleneck_dim"]
    output_dim: int = CONF["corpus_size"]
    ease_proj_dim: int = 256

    @nn.compact
    def __call__(self, x, ease, training: bool = False):
        h = nn.swish(nn.Dense(self.hidden_dim)(x))
        bottleneck = nn.Dense(self.bottleneck_dim, name="bottleneck")(h)
        if training:
            rng = self.make_rng("noise")
            z = bottleneck + random.normal(rng, bottleneck.shape) * CONF["latent_noise_scale"]
        else:
            z = bottleneck

        ep = nn.swish(nn.Dense(self.ease_proj_dim, name="ease_proj")(ease))
        zc = jnp.concatenate([z, ep], axis=1)
        d1 = nn.swish(nn.Dense(self.hidden_dim // 2, name="dec_item_up1")(zc))
        d1 = nn.swish(nn.Dense(self.hidden_dim, name="dec_item_up2")(d1))
        item_logits = nn.Dense(self.output_dim, name="item_logits")(d1)

        d2 = nn.swish(nn.Dense(self.hidden_dim // 2, name="dec_rating_up1")(z))
        d2 = nn.swish(nn.Dense(self.hidden_dim, name="dec_rating_up2")(d2))
        rating_pred = nn.Dense(self.output_dim, name="rating_pred")(d2)

        log_var_presence = self.param("log_var_presence", nn.initializers.zeros, (1,))
        log_var_rating = self.param("log_var_rating", nn.initializers.zeros, (1,))
        return item_logits, rating_pred, log_var_presence, log_var_rating


@jax.jit
def train_step_rc(state, batch, rated_mask, prior_logits):
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
    ease = ease_channel(presence * keep)

    def loss_fn(params):
        item_logits, rating_pred, log_var_p, log_var_r = state.apply_fn(
            {"params": params}, x_in, ease, training=True, rngs={"noise": vae_rng}
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
    updates, new_opt = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    return state.replace(step=state.step + 1, params=new_params, opt_state=new_opt,
                         key=dropout_rng), loss, p_loss, r_loss


@jax.jit
def forward_rc(params, x3):
    cs = CONF["corpus_size"]
    ease = ease_channel(x3[:, :cs])
    logits, ratings, _, _ = MODEL.apply({"params": params}, x3, ease, training=False)
    return logits, ratings


def gen_batches(users, absv, batch_size, cs, prefetch=3):
    import queue
    import threading

    idxs_a = [u[0].astype(np.int64) for u in users]
    vals_a = [u[1] for u in users]
    rated_a = [u[2].astype(np.float32) for u in users]
    lens = np.array([len(a) for a in idxs_a], dtype=np.int64)

    def build(sel):
        n = len(sel)
        rows = np.repeat(np.arange(n, dtype=np.int64), lens[sel])
        cols = np.concatenate([idxs_a[u] for u in sel])
        bt = np.zeros(n * cs * 3, dtype=np.float32)
        base = rows * (cs * 3) + cols
        bt[base] = 1.0
        bt[base + cs] = np.concatenate([vals_a[u] for u in sel])
        bt[base + 2 * cs] = np.concatenate([absv[u] for u in sel])
        rm = np.zeros(n * cs, dtype=np.float32)
        rm[rows * cs + cols] = np.concatenate([rated_a[u] for u in sel])
        return bt.reshape(n, cs * 3), rm.reshape(n, cs)

    q = queue.Queue(maxsize=prefetch)

    def producer():
        while True:
            perm = np.random.permutation(len(users))
            for b in range(0, len(users), batch_size):
                q.put(build(perm[b : b + batch_size]))

    threading.Thread(target=producer, daemon=True).start()
    while True:
        yield q.get()


def eval_holdout(params, users, absv, prior, cs, rng):
    x = np.zeros((len(users), cs * 3), dtype=np.float32)
    rows, rateds, keeps = [], [], []
    rates = rng.uniform(0.24, 0.56, size=len(users))
    for i, ((idx, vals, rated_m, _st), av) in enumerate(zip(users, absv)):
        x[i, idx] = 1.0
        x[i, cs + idx] = vals
        x[i, 2 * cs + idx] = av
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
        lg, rt = forward_rc(params, jnp.asarray(xc[b : b + 512]))
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


def floors_dump(params, all_users, absv_all, holdout_idx, cs, out, n_users=20_000, seed=555):
    users = [all_users[i] for i in holdout_idx[:n_users]]
    avs = [absv_all[i] for i in holdout_idx[:n_users]]
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
            x[j, 2 * cs + idx[keep]] = avs[b + j][keep]
        _, preds = forward_rc(params, jnp.asarray(x))
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
    global B_g, MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=0.007)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=15_000)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vectors", default="../data/aug2026/user_input_vectors_cleanup_notrust.npz")
    ap.add_argument("--raw-scores", default="../data/aug2026/raw_scores_recon.npy")
    ap.add_argument("--ease-b", default="../data/aug2026/ease_B6k_lam200.npy")
    ap.add_argument("--presence-prior-alpha", type=float, default=1.0)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    B_g = jnp.asarray(np.load(args.ease_b), dtype=jnp.float32)
    MODEL = RCRecommender()
    cs = CONF["corpus_size"]
    all_users, item_counts = load_all_users(args.vectors)

    raw = np.load(args.raw_scores)
    rated_all = (raw >= 1) & (raw <= 10)
    absflat = np.where(rated_all, (raw.astype(np.float32) - 5.5) / 2.5, 0.0).astype(np.float32)
    absv_all = []
    pos = 0
    for u in all_users:
        l = len(u[0])
        absv_all.append(absflat[pos : pos + l])
        pos += l
    assert pos == len(absflat)

    rng_h = np.random.default_rng(HOLDOUT_SEED)
    perm = rng_h.permutation(len(all_users))
    n_hold = len(all_users) // 10
    holdout_idx = perm[:n_hold]
    if args.full:
        train_sel = np.arange(len(all_users))
        print("FULL-DATA run: training on all users", flush=True)
    else:
        train_sel = perm[n_hold:]
    train_users = [all_users[i] for i in train_sel]
    train_absv = [absv_all[i] for i in train_sel]
    print(f"train users: {len(train_users)}", flush=True)

    clipped = np.maximum(item_counts, 1.0)
    prior = jnp.asarray(args.presence_prior_alpha * np.log(clipped / clipped.sum()), dtype=jnp.float32)

    rng0 = jax.random.PRNGKey(args.seed)
    params = MODEL.init({"params": rng0, "noise": rng0}, jnp.ones((1, cs * 3)), jnp.ones((1, cs)))["params"]
    warm = min(1000, args.steps // 10)
    sched = optax.warmup_cosine_decay_schedule(0.0, args.lr, warmup_steps=warm,
                                               decay_steps=args.steps, end_value=args.lr * 0.01)
    state = TrainState.create(apply_fn=MODEL.apply, params=params, tx=oc.muon(sched), key=rng0)
    loader = gen_batches(train_users, train_absv, args.batch_size, cs)

    eval_every = max(1, round(2000 * 512 / args.batch_size))
    logf = open(f"{args.out_prefix}.jsonl", "w")
    t0 = time.time()
    for step in range(args.steps):
        bt, rm = next(loader)
        state, loss, p_loss, r_loss = train_step_rc(state, jnp.array(bt), jnp.array(rm), prior)
        if step % 500 == 0:
            print(f"Step {step}: Loss {loss:.4f} (P {p_loss:.4f} R {r_loss:.4f}) [{time.time()-t0:.0f}s]", flush=True)
        if not args.full and step % eval_every == 0 and step > 0:
            ev = eval_holdout(state.params, [all_users[i] for i in holdout_idx[:EVAL_N]],
                              [absv_all[i] for i in holdout_idx[:EVAL_N]], prior, cs,
                              np.random.default_rng(4242))
            ev["step"] = step
            logf.write(json.dumps(ev) + "\n")
            logf.flush()
            print(f"[eval @ {step}] pres {ev['presence_loss']:.4f} nll_drop {ev['nll_drop_per_item']:.4f} "
                  f"mae_drop {ev['mae_drop_per_item']:.4f}", flush=True)

    with open(f"{args.out_prefix}.msgpack", "wb") as f:
        f.write(serialization.to_bytes(state.params))
    print(f"weights saved: {args.out_prefix}.msgpack", flush=True)
    if not args.full:
        ev = eval_holdout(state.params, [all_users[i] for i in holdout_idx[:EVAL_N]],
                          [absv_all[i] for i in holdout_idx[:EVAL_N]], prior, cs,
                          np.random.default_rng(4242))
        ev["step"] = args.steps
        logf.write(json.dumps(ev) + "\n")
        print(f"[final] pres {ev['presence_loss']:.4f} nll_drop {ev['nll_drop_per_item']:.4f} "
              f"mae_drop {ev['mae_drop_per_item']:.4f}", flush=True)
        floors_dump(state.params, all_users, absv_all, holdout_idx, cs, f"{args.out_prefix}_dump.npz")
    logf.close()
    print(f"done: {args.out_prefix}", flush=True)


if __name__ == "__main__":
    main()
