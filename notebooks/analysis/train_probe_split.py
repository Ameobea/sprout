"""Single-task split probes vs the shared-bottleneck multitask control
(probe_frac1.0): identical arch + protocol, but the loss keeps only one head
(raw loss, no uncertainty weighting; the dead head gets zero gradient and
stays at init). --task rating ends with a rating_floors_dump-format npz.
Run inside rocm_jax from notebooks/."""

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "analysis")
import jax
import jax.numpy as jnp
import optax
from flax import serialization
from jax import random

from model import CONF, Recommender
from train import create_train_state, data_generator, load_all_users
import train_probe as tp

HOLDOUT_SEED = 999
SUBSAMPLE_SEED = 777
EVAL_N = 2048
TASK = "rating"


@jax.jit
def train_step_split(state, batch, rated_mask, prior_logits):
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
    x_in = jnp.concatenate([presence * keep, ratings_z * keep], axis=1)

    def loss_fn(params):
        item_logits, rating_pred, _lvp, _lvr = state.apply_fn(
            {"params": params}, x_in, training=True, rngs={"noise": vae_rng}
        )
        log_probs = jax.nn.log_softmax(item_logits + prior_logits[None, :], axis=1)
        counts = jnp.maximum(jnp.sum(presence, axis=1), 1.0)
        p_loss = jnp.mean(-jnp.sum(presence * log_probs, axis=1) / counts)
        per_entry = optax.huber_loss(rating_pred - ratings_z, delta=CONF["huber_delta"])
        denom = jnp.maximum(jnp.sum(rated_mask, axis=1), 1.0)
        r_loss = jnp.mean(jnp.sum(rated_mask * per_entry, axis=1) / denom)
        loss = {"rating": r_loss, "presence": p_loss, "both_fixed": p_loss + r_loss}[TASK]
        return loss, (p_loss, r_loss)

    (loss, (p_loss, r_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    updates, new_opt = state.tx.update(grads, state.opt_state, state.params, value=loss)
    new_params = optax.apply_updates(state.params, updates)
    return state.replace(step=state.step + 1, params=new_params, opt_state=new_opt,
                         key=dropout_rng), loss, p_loss, r_loss


def floors_dump(params, all_users, holdout_idx, cs, out, n_users=20_000, seed=555):
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
        for j, (idx, vals, _r, _s) in enumerate(chunk):
            keep = keeps[b + j]
            x[j, idx[keep]] = 1.0
            x[j, cs + idx[keep]] = vals[keep]
        _, preds = tp.forward_clean(params, jnp.asarray(x))
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
    global TASK
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["presence", "rating", "both_fixed"], required=True)
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--vectors", default="../data/aug2026/user_input_vectors_cleanup_notrust.npz")
    ap.add_argument("--presence-prior-alpha", type=float, default=1.0)
    ap.add_argument("--eval-interval", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()
    TASK = args.task
    tp.MODEL = Recommender()
    print(f"task={TASK} seed={args.seed}", flush=True)

    cs = CONF["corpus_size"]
    all_users, item_counts = load_all_users(args.vectors)

    rng_h = np.random.default_rng(HOLDOUT_SEED)
    perm = rng_h.permutation(len(all_users))
    n_hold = len(all_users) // 10
    holdout_idx = perm[:n_hold]
    train_pool = perm[n_hold:]
    rng_s = np.random.default_rng(SUBSAMPLE_SEED)
    train_idx = rng_s.choice(train_pool, size=len(train_pool), replace=False)
    train_users = [all_users[i] for i in train_idx]
    print(f"train users: {len(train_users)}  holdout users: {n_hold}", flush=True)

    clipped = np.maximum(item_counts, 1.0)
    prior = jnp.asarray(args.presence_prior_alpha * np.log(clipped / clipped.sum()), dtype=jnp.float32)

    rng_e = np.random.default_rng(4242)
    eval_hold = tp.FixedEvalSet([all_users[i] for i in holdout_idx[:EVAL_N]], rng_e, cs)
    eval_train = tp.FixedEvalSet(train_users[:EVAL_N], rng_e, cs)

    state = create_train_state(jax.random.PRNGKey(args.seed), CONF["learning_rate"])
    loader = data_generator(train_users, batch_size=CONF["batch_size"])

    logf = open(f"{args.out_prefix}.jsonl", "w")
    t0 = time.time()

    def run_eval(step):
        rec = {"step": step, "task": TASK, "elapsed_s": round(time.time() - t0, 1)}
        rec["holdout"] = eval_hold.evaluate(state.params, prior, tp.forward_clean)
        rec["train"] = eval_train.evaluate(state.params, prior, tp.forward_clean)
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        h = rec["holdout"]["corrupt"]
        print(f"[eval @ {step}] holdout: pres {h['presence_loss']:.4f} nll_drop {h['nll_drop_per_item']:.4f} "
              f"mae_drop {h['mae_drop_per_item']:.4f}", flush=True)

    for step in range(args.steps):
        batch, rated_mask = next(loader)
        state, loss, p_loss, r_loss = train_step_split(
            state, jnp.array(batch), jnp.array(rated_mask), prior
        )
        if step % 500 == 0:
            lr_scale = optax.tree.get(state.opt_state, "scale")
            print(f"Step {step}: Loss {loss:.4f} (P {p_loss:.4f} R {r_loss:.4f}) lr {lr_scale} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            if lr_scale < 1e-6:
                print("LR decayed below 1e-6, stopping.", flush=True)
                break
        if step % args.eval_interval == 0 and step > 0:
            run_eval(step)

    run_eval(args.steps)
    with open(f"{args.out_prefix}.msgpack", "wb") as f:
        f.write(serialization.to_bytes(state.params))
    logf.close()
    if TASK in ("rating", "both_fixed"):
        floors_dump(state.params, all_users, holdout_idx, cs, f"{args.out_prefix}_dump.npz")
    print(f"done: {args.out_prefix}", flush=True)


if __name__ == "__main__":
    main()
