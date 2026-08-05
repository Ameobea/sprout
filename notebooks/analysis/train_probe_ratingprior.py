"""Item-mean prior probe for the rating head (logQ analog): a frozen shrunk
item-mean is added to the head output in-loss, so the head learns the per-user
residual. Protocol identical to train_probe.py; run inside rocm_jax from notebooks/."""

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

from model import CONF, Recommender
from train import create_train_state, data_generator, load_all_users
from train_probe import EVAL_N, HOLDOUT_SEED, SUBSAMPLE_SEED, FixedEvalSet

MODEL = None
RPRIOR = None


@jax.jit
def train_step_rp(state, batch, rated_mask, prior_logits, pop_w, rprior):
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
    present_w_sum = jnp.maximum(jnp.sum(presence * pop_w[None, :], axis=1, keepdims=True), 1e-6)
    n_present = jnp.sum(presence, axis=1, keepdims=True)
    drop_p = jnp.clip(random_rates * pop_w[None, :] * n_present / present_w_sum, 0.0, 0.95)
    keep = random.bernoulli(dropout_rng, p=(1.0 - drop_p), shape=presence.shape)

    b, cs_ = presence.shape
    nch = CONF["input_channels"]
    x_in = (batch.reshape(b, nch, cs_) * keep[:, None, :]).reshape(b, nch * cs_)

    def loss_fn(params):
        item_logits, rating_pred, lv_p, lv_r = state.apply_fn(
            {"params": params}, x_in, training=True, rngs={"noise": vae_rng}
        )
        log_probs = jax.nn.log_softmax(item_logits + prior_logits[None, :], axis=1)
        cnt = jnp.maximum(jnp.sum(presence, axis=1), 1.0)
        presence_loss = jnp.mean(-jnp.sum(presence * log_probs, axis=1) / cnt)

        err = (rating_pred + rprior[None, :]) - ratings_z
        per_entry = optax.huber_loss(err, delta=CONF["huber_delta"])
        denom = jnp.maximum(jnp.sum(rated_mask, axis=1), 1.0)
        rating_loss = jnp.mean(jnp.sum(rated_mask * per_entry, axis=1) / denom)

        weighted = (jnp.exp(-lv_p) * presence_loss + lv_p) + (
            jnp.exp(-lv_r) * rating_loss + lv_r
        )
        return jnp.mean(weighted), (presence_loss, rating_loss)

    (loss, (p_loss, r_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params, value=loss)
    new_params = optax.apply_updates(state.params, updates)
    state = state.replace(
        step=state.step + 1, params=new_params, opt_state=new_opt_state, key=dropout_rng
    )
    return state, loss, p_loss, r_loss


@jax.jit
def forward_rp(params, x):
    logits, ratings, _, _ = MODEL.apply({"params": params}, x, training=False)
    return logits, ratings + RPRIOR[None, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-frac", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--vectors", default="../data/aug2026/user_input_vectors_cleanup_notrust.npz")
    ap.add_argument("--rating-prior", required=True)
    ap.add_argument("--presence-prior-alpha", type=float, default=1.0)
    ap.add_argument("--eval-interval", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    global MODEL, RPRIOR
    MODEL = Recommender()
    RPRIOR = jnp.asarray(np.load(args.rating_prior), dtype=jnp.float32)
    print(f"rating prior: mean {float(RPRIOR.mean()):.4f} std {float(RPRIOR.std()):.4f}", flush=True)

    cs = CONF["corpus_size"]
    all_users, item_counts = load_all_users(args.vectors)

    rng_h = np.random.default_rng(HOLDOUT_SEED)
    perm = rng_h.permutation(len(all_users))
    n_hold = len(all_users) // 10
    holdout_idx = perm[:n_hold]
    train_pool = perm[n_hold:]

    rng_s = np.random.default_rng(SUBSAMPLE_SEED)
    n_train = max(int(len(train_pool) * args.user_frac), 1)
    train_idx = rng_s.choice(train_pool, size=n_train, replace=False)
    train_users = [all_users[i] for i in train_idx]
    print(f"train users: {len(train_users)}  holdout users: {n_hold}", flush=True)

    clipped = np.maximum(item_counts, 1.0)
    prior = (
        jnp.asarray(args.presence_prior_alpha * np.log(clipped / clipped.sum()), dtype=jnp.float32)
        if args.presence_prior_alpha > 0
        else jnp.zeros(cs, dtype=jnp.float32)
    )
    pop_w = jnp.ones(cs, dtype=jnp.float32)

    rng_e = np.random.default_rng(4242)
    eval_hold = FixedEvalSet([all_users[i] for i in holdout_idx[:EVAL_N]], rng_e, cs)
    eval_train = FixedEvalSet(train_users[:EVAL_N], rng_e, cs)

    state = create_train_state(jax.random.PRNGKey(args.seed), CONF["learning_rate"])
    loader = data_generator(train_users, batch_size=CONF["batch_size"])

    logf = open(f"{args.out_prefix}.jsonl", "w")
    t0 = time.time()

    def run_eval(step):
        rec = {"step": step, "seed": args.seed, "elapsed_s": round(time.time() - t0, 1)}
        rec["holdout"] = eval_hold.evaluate(state.params, prior, forward_rp)
        rec["train"] = eval_train.evaluate(state.params, prior, forward_rp)
        rec["log_var_presence"] = float(state.params["log_var_presence"][0])
        rec["log_var_rating"] = float(state.params["log_var_rating"][0])
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        h, t = rec["holdout"]["corrupt"], rec["train"]["corrupt"]
        print(
            f"[eval @ {step}] holdout: pres {h['presence_loss']:.4f} nll_drop {h['nll_drop_per_item']:.4f} "
            f"mae_drop {h['mae_drop_per_item']:.4f} | train: pres {t['presence_loss']:.4f} "
            f"nll_drop {t['nll_drop_per_item']:.4f} mae_drop {t['mae_drop_per_item']:.4f}",
            flush=True,
        )

    for step in range(args.steps):
        batch, rated_mask = next(loader)
        state, loss, p_loss, r_loss = train_step_rp(
            state, jnp.array(batch), jnp.array(rated_mask), prior, pop_w, RPRIOR
        )
        if step % 500 == 0:
            lr_scale = optax.tree.get(state.opt_state, "scale")
            print(f"Step {step}: Loss {loss:.4f} (P {p_loss:.4f} R {r_loss:.4f}) lr_scale {lr_scale}", flush=True)
            if lr_scale < 1e-6:
                print("LR decayed below 1e-6, stopping.", flush=True)
                break
        if step % args.eval_interval == 0 and step > 0:
            run_eval(step)

    run_eval(args.steps)
    with open(f"{args.out_prefix}.msgpack", "wb") as f:
        f.write(serialization.to_bytes(state.params))
    logf.close()
    print(f"done: {args.out_prefix}", flush=True)


if __name__ == "__main__":
    main()
