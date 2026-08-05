"""Hyperparameter/foundation sweep probe: production multitask recipe
(uncertainty weighting kept — load-bearing per both_fixed) with configurable
lr / schedule / batch size / optimizer / activation. Same judgment protocol:
jsonl eval (comparable to probe_frac1.0) + rating_floors dump at end.
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
import optax.contrib as oc
from flax import linen as nn
from flax import serialization
from jax import random

from model import CONF
from train import TrainState, data_generator, load_all_users
import train_probe as tp
from train_probe_split import floors_dump

HOLDOUT_SEED = 999
SUBSAMPLE_SEED = 777
EVAL_N = 2048
ACT = {"swish": nn.swish, "gelu": nn.gelu, "relu": nn.relu, "mish": lambda x: x * jnp.tanh(nn.softplus(x))}
PLATEAU = True
SFREE = False


class HPRecommender(nn.Module):
    act: str = "swish"
    hidden_dim: int = CONF["hidden_dim"]
    bottleneck_dim: int = CONF["bottleneck_dim"]
    output_dim: int = CONF["corpus_size"]

    @nn.compact
    def __call__(self, x, training: bool = False):
        a = ACT[self.act]
        h = a(nn.Dense(self.hidden_dim)(x))
        bottleneck = nn.Dense(self.bottleneck_dim, name="bottleneck")(h)
        if training:
            rng = self.make_rng("noise")
            z = bottleneck + random.normal(rng, bottleneck.shape) * CONF["latent_noise_scale"]
        else:
            z = bottleneck
        d1 = a(nn.Dense(self.hidden_dim // 2, name="dec_item_up1")(z))
        d1 = a(nn.Dense(self.hidden_dim, name="dec_item_up2")(d1))
        item_logits = nn.Dense(self.output_dim, name="item_logits")(d1)
        d2 = a(nn.Dense(self.hidden_dim // 2, name="dec_rating_up1")(z))
        d2 = a(nn.Dense(self.hidden_dim, name="dec_rating_up2")(d2))
        rating_pred = nn.Dense(self.output_dim, name="rating_pred")(d2)
        log_var_presence = self.param("log_var_presence", nn.initializers.zeros, (1,))
        log_var_rating = self.param("log_var_rating", nn.initializers.zeros, (1,))
        return item_logits, rating_pred, log_var_presence, log_var_rating


def make_tx(args, steps):
    lr = args.lr
    if args.schedule == "cosine":
        warm = min(1000, steps // 10)
        lr = optax.warmup_cosine_decay_schedule(0.0, args.lr, warmup_steps=warm,
                                                decay_steps=steps, end_value=args.lr * 0.01)
    opt = {
        "adam": lambda: optax.adam(lr),
        "adamw": lambda: optax.adamw(lr, weight_decay=args.wd),
        "lion": lambda: optax.lion(lr, weight_decay=args.wd),
        "ademamix": lambda: oc.ademamix(lr),
        "muon": lambda: oc.muon(lr),
        "sfree": lambda: oc.schedule_free_adamw(lr, warmup_steps=1000),
    }[args.optimizer]()
    if args.schedule == "plateau":
        return optax.chain(opt, oc.reduce_on_plateau(patience=5, cooldown=1, factor=0.5,
                                                     rtol=1e-4, accumulation_size=200))
    return opt


@jax.jit
def train_step_hp(state, batch, rated_mask, prior_logits):
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
    if PLATEAU:
        updates, new_opt = state.tx.update(grads, state.opt_state, state.params, value=loss)
    else:
        updates, new_opt = state.tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)
    return state.replace(step=state.step + 1, params=new_params, opt_state=new_opt,
                         key=dropout_rng), loss, p_loss, r_loss


def main():
    global PLATEAU, SFREE
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=CONF["learning_rate"])
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=CONF["batch_size"])
    ap.add_argument("--optimizer", choices=["adam", "adamw", "lion", "ademamix", "muon", "sfree"], default="adam")
    ap.add_argument("--schedule", choices=["plateau", "cosine", "none"], default="plateau")
    ap.add_argument("--activation", choices=list(ACT), default="swish")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--vectors", default="../data/aug2026/user_input_vectors_cleanup_notrust.npz")
    ap.add_argument("--presence-prior-alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-dump", action="store_true")
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    steps = args.steps or round(50_000 * 512 / args.batch_size)
    PLATEAU = args.schedule == "plateau"
    SFREE = args.optimizer == "sfree"
    if SFREE and PLATEAU:
        raise SystemExit("sfree needs --schedule none/cosine")
    print(f"opt={args.optimizer} lr={args.lr} sched={args.schedule} bs={args.batch_size} "
          f"act={args.activation} steps={steps} seed={args.seed}", flush=True)

    model = HPRecommender(act=args.activation)
    tp.MODEL = model
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

    rng0 = jax.random.PRNGKey(args.seed)
    params = model.init({"params": rng0, "noise": rng0}, jnp.ones((1, cs * 2)))["params"]
    tx = make_tx(args, steps)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx, key=rng0)
    loader = data_generator(train_users, batch_size=args.batch_size)

    def eval_params():
        if SFREE:
            return oc.schedule_free_eval_params(state.opt_state, state.params)
        return state.params

    logf = open(f"{args.out_prefix}.jsonl", "w")
    t0 = time.time()
    eval_every = max(1, round(2000 * 512 / args.batch_size))
    for step in range(steps):
        batch, rated_mask = next(loader)
        state, loss, p_loss, r_loss = train_step_hp(
            state, jnp.array(batch), jnp.array(rated_mask), prior
        )
        if step % 500 == 0:
            msg = f"Step {step}: Loss {loss:.4f} (P {p_loss:.4f} R {r_loss:.4f}) [{time.time()-t0:.0f}s]"
            if PLATEAU:
                lr_scale = optax.tree.get(state.opt_state, "scale")
                msg += f" lr_scale {lr_scale}"
                if lr_scale < 1e-6:
                    print(msg + "\nLR decayed below 1e-6, stopping.", flush=True)
                    break
            print(msg, flush=True)
        if step % eval_every == 0 and step > 0:
            ev = eval_hold.evaluate(eval_params(), prior, tp.forward_clean)
            rec = {"step": step, "holdout": {"corrupt": ev["corrupt"], "clean": ev["clean"]}}
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            h = ev["corrupt"]
            print(f"[eval @ {step}] pres {h['presence_loss']:.4f} nll_drop {h['nll_drop_per_item']:.4f} "
                  f"mae_drop {h['mae_drop_per_item']:.4f}", flush=True)

    fp = eval_params()
    ev = eval_hold.evaluate(fp, prior, tp.forward_clean)
    logf.write(json.dumps({"step": steps, "holdout": {"corrupt": ev["corrupt"], "clean": ev["clean"]}}) + "\n")
    logf.close()
    h = ev["corrupt"]
    print(f"[final] pres {h['presence_loss']:.4f} nll_drop {h['nll_drop_per_item']:.4f} "
          f"mae_drop {h['mae_drop_per_item']:.4f}", flush=True)
    with open(f"{args.out_prefix}.msgpack", "wb") as f:
        f.write(serialization.to_bytes(fp))
    if not args.no_dump:
        floors_dump(fp, all_users, holdout_idx, cs, f"{args.out_prefix}_dump.npz")
    print(f"done: {args.out_prefix}", flush=True)


if __name__ == "__main__":
    main()
