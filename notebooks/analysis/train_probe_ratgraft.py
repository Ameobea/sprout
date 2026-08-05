"""EASE graft into the RATING decoder (the presence decoder stays plain, so the
probe isolates one variable vs probe_frac1.0):

  rating decoder input = [z | swish(Dense256(channel))]
  channel presence: znorm(presence_kept @ B)            (co-occurrence EASE)
  channel residual: znorm(((vals - item_mean) * rated_kept) @ B_rat)

Ends by writing a rating_floors_dump-format npz on the 20k holdout users
(corrupt seed 555) so rating_floors_analyze.py judges it directly.
Run inside rocm_jax from notebooks/."""

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax import serialization
from jax import random
from optax import contrib

from model import CONF
from train import TrainState, load_all_users

HOLDOUT_SEED = 999
B_g = None
IMEAN = None
MODEL = None
CHANNEL = "presence"


def znorm(e):
    mu = jnp.mean(e, axis=1, keepdims=True)
    sd = jnp.std(e, axis=1, keepdims=True) + 1e-6
    return (e - mu) / sd


MODE = "concat"


def make_channel(xp, xr, rk):
    if CHANNEL == "presence":
        e = xp @ B_g
    else:
        e = ((xr - IMEAN[None, :] * xp) * rk) @ B_g
    return e if MODE == "gate" else znorm(e)


class RatGraftRecommender(nn.Module):
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

        d1 = nn.swish(nn.Dense(self.hidden_dim // 2, name="dec_item_up1")(z))
        d1 = nn.swish(nn.Dense(self.hidden_dim, name="dec_item_up2")(d1))
        item_logits = nn.Dense(self.output_dim, name="item_logits")(d1)

        if MODE == "concat":
            ep = nn.swish(nn.Dense(self.ease_proj_dim, name="rat_ease_proj")(ease))
            zc = jnp.concatenate([z, ep], axis=1)
        else:
            zc = z
        d2 = nn.swish(nn.Dense(self.hidden_dim // 2, name="dec_rating_up1")(zc))
        d2 = nn.swish(nn.Dense(self.hidden_dim, name="dec_rating_up2")(d2))
        rating_pred = nn.Dense(self.output_dim, name="rating_pred")(d2)
        if MODE == "gate":
            rating_pred = rating_pred + nn.softplus(nn.Dense(1, name="rat_ease_gate")(z)) * ease

        log_var_presence = self.param("log_var_presence", nn.initializers.zeros, (1,))
        log_var_rating = self.param("log_var_rating", nn.initializers.zeros, (1,))
        return item_logits, rating_pred, log_var_presence, log_var_rating


def make_state(rng, cs):
    params = MODEL.init({"params": rng, "noise": rng}, jnp.ones((1, cs * 2)), jnp.ones((1, cs)))["params"]
    tx = optax.chain(
        optax.adam(CONF["learning_rate"]),
        contrib.reduce_on_plateau(patience=5, cooldown=1, factor=0.5, rtol=1e-4, accumulation_size=200),
    )
    return TrainState.create(apply_fn=MODEL.apply, params=params, tx=tx, key=rng)


@jax.jit
def train_step_g(state, batch, rated_mask, prior_logits):
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

    xp = presence * keep
    xr = ratings_z * keep
    x_in = jnp.concatenate([xp, xr], axis=1)
    ease = make_channel(xp, xr, rated_mask * keep)

    def loss_fn(params):
        item_logits, rating_pred, log_var_p, log_var_r = state.apply_fn(
            {"params": params}, x_in, ease, training=True, rngs={"noise": vae_rng}
        )
        log_probs = jax.nn.log_softmax(item_logits + prior_logits[None, :], axis=1)
        per_user_counts = jnp.maximum(jnp.sum(presence, axis=1), 1.0)
        presence_loss = jnp.mean(-jnp.sum(presence * log_probs, axis=1) / per_user_counts)
        err = rating_pred - ratings_z
        per_entry = optax.huber_loss(err, delta=CONF["huber_delta"])
        denom_r = jnp.maximum(jnp.sum(rated_mask, axis=1), 1.0)
        rating_loss = jnp.mean(jnp.sum(rated_mask * per_entry, axis=1) / denom_r)
        weighted = (jnp.exp(-log_var_p) * presence_loss + log_var_p) + (
            jnp.exp(-log_var_r) * rating_loss + log_var_r
        )
        return jnp.mean(weighted), (presence_loss, rating_loss)

    (loss, (p_loss, r_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    updates, new_opt = state.tx.update(grads, state.opt_state, state.params, value=loss)
    return state.replace(step=state.step + 1, params=new_params_apply(state.params, updates),
                         opt_state=new_opt, key=dropout_rng), loss, p_loss, r_loss


def new_params_apply(params, updates):
    return optax.apply_updates(params, updates)


@jax.jit
def forward_g(params, x2, rk):
    cs = CONF["corpus_size"]
    ease = make_channel(x2[:, :cs], x2[:, cs:], rk)
    _, ratings, _, _ = MODEL.apply({"params": params}, x2, ease, training=False)
    return ratings


def gen_batches(users, batch_size, cs):
    while True:
        perm = np.random.permutation(len(users))
        for b in range(0, len(users), batch_size):
            sel = perm[b : b + batch_size]
            bt = np.zeros((len(sel), cs * 2), dtype=np.float32)
            rm = np.zeros((len(sel), cs), dtype=np.float32)
            for i, ui in enumerate(sel):
                idxs, vals, rated, _st = users[ui]
                bt[i, idxs] = 1.0
                bt[i, cs + idxs] = vals
                rm[i, idxs] = rated.astype(np.float32)
            yield bt, rm


def floors_dump(state, all_users, holdout_idx, cs, out, n_users=20_000, seed=555):
    users = [all_users[i] for i in holdout_idx[:n_users]]
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
        x = np.zeros((len(chunk), cs * 2), dtype=np.float32)
        rk = np.zeros((len(chunk), cs), dtype=np.float32)
        for j, (idx, vals, rated, _s) in enumerate(chunk):
            keep = keeps[b + j]
            x[j, idx[keep]] = 1.0
            x[j, cs + idx[keep]] = vals[keep]
            rk[j, idx[keep & rated]] = 1.0
        preds = np.asarray(forward_g(state.params, jnp.asarray(x), jnp.asarray(rk)), dtype=np.float32)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", choices=["presence", "residual"], required=True)
    ap.add_argument("--mode", choices=["concat", "gate"], default="concat")
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--vectors", default="../data/aug2026/user_input_vectors_cleanup_notrust.npz")
    ap.add_argument("--ease-b", required=True)
    ap.add_argument("--item-mean", default="../data/aug2026/rating_item_prior_lam50.npy")
    ap.add_argument("--presence-prior-alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    global B_g, IMEAN, MODEL, CHANNEL, MODE
    CHANNEL = args.channel
    MODE = args.mode
    B_g = jnp.asarray(np.load(args.ease_b), dtype=jnp.float32)
    IMEAN = jnp.asarray(np.load(args.item_mean), dtype=jnp.float32)
    MODEL = RatGraftRecommender()
    print(f"channel={CHANNEL} mode={MODE} B={args.ease_b}", flush=True)

    cs = CONF["corpus_size"]
    all_users, item_counts = load_all_users(args.vectors)
    rng_h = np.random.default_rng(HOLDOUT_SEED)
    perm = rng_h.permutation(len(all_users))
    n_hold = len(all_users) // 10
    holdout_idx = perm[:n_hold]
    train_users = [all_users[i] for i in perm[n_hold:]]
    print(f"train users: {len(train_users)}  holdout: {n_hold}", flush=True)

    clipped = np.maximum(item_counts, 1.0)
    prior = jnp.asarray(args.presence_prior_alpha * np.log(clipped / clipped.sum()), dtype=jnp.float32)

    state = make_state(jax.random.PRNGKey(args.seed), cs)
    loader = gen_batches(train_users, CONF["batch_size"], cs)

    t0 = time.time()
    for step in range(args.steps):
        bt, rm = next(loader)
        state, loss, p_loss, r_loss = train_step_g(state, jnp.array(bt), jnp.array(rm), prior)
        if step % 500 == 0:
            lr_scale = optax.tree.get(state.opt_state, "scale")
            print(f"Step {step}: Loss {loss:.4f} (P {p_loss:.4f} R {r_loss:.4f}) lr {lr_scale} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            if lr_scale < 1e-6:
                print("LR decayed below 1e-6, stopping.", flush=True)
                break

    with open(f"{args.out_prefix}.msgpack", "wb") as f:
        f.write(serialization.to_bytes(state.params))
    floors_dump(state, all_users, holdout_idx, cs, f"{args.out_prefix}_dump.npz")
    print(f"done: {args.out_prefix}", flush=True)


if __name__ == "__main__":
    main()
