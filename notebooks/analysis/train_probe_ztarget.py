"""Pure-z rating target probe: input channels unchanged (mixed alpha-mix values),
but the rating loss targets clip((s - mu_u)/sigma_u, +-3) from reconstructed raw
scores — "how much more than their own average". Eval reports huber in z units
plus mae_drop converted to mixed units via the per-user affine (exact where no
clip binds) for comparability with control. Protocol = train_probe.py."""

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
from train import create_train_state, load_all_users
from train_probe import EVAL_N, HOLDOUT_SEED, SUBSAMPLE_SEED, huber

MODEL = None
SCORES = np.arange(1, 11, dtype=np.float64)


@jax.jit
def train_step_zt(state, batch, ztgt, rated_mask, prior_logits):
    cs = CONF["corpus_size"]
    presence = batch[:, :cs]

    dropout_rng, vae_rng = random.split(state.key)
    rate_variation = CONF["dropout_variation"] * CONF["dropout_rate"]
    random_rates = (
        CONF["dropout_rate"]
        + random.uniform(dropout_rng, shape=(presence.shape[0], 1)) * (2 * rate_variation)
        - rate_variation
    )
    random_rates = jnp.clip(random_rates, 0.01, 0.75)
    keep = random.bernoulli(dropout_rng, p=(1.0 - random_rates), shape=presence.shape)

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

        per_entry = optax.huber_loss(rating_pred - ztgt, delta=CONF["huber_delta"])
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
def forward_z(params, x):
    logits, ratings, _, _ = MODEL.apply({"params": params}, x, training=False)
    return logits, ratings


class FixedEvalSetZ:
    """Fixed-corruption eval with separate z targets + per-user affine to mixed units."""

    def __init__(self, users, zvals, A, B, rng, cs):
        self.cs = cs
        self.A, self.B = A, B
        self.x = np.zeros((len(users), cs * 2), dtype=np.float32)
        self.rows, self.rated, self.keeps, self.zt = [], [], [], []
        rates = rng.uniform(0.24, 0.56, size=len(users))
        for i, (idx, vals, rated_m, _st) in enumerate(users):
            self.x[i, idx] = 1.0
            self.x[i, cs + idx] = vals
            self.rows.append(idx.astype(np.int64))
            self.rated.append(rated_m.astype(bool))
            self.zt.append(zvals[i])
            keep = rng.random(len(idx)) > rates[i]
            if keep.sum() == 0:
                keep[0] = True
            self.keeps.append(keep)
        self.xc = self.x.copy()
        for i, idx in enumerate(self.rows):
            dropped = idx[~self.keeps[i]]
            self.xc[i, dropped] = 0.0
            self.xc[i, cs + dropped] = 0.0

    def evaluate(self, params, prior, batch=512):
        cs = self.cs
        res = {}
        for tag, xin, use_keep in [("corrupt", self.xc, True), ("clean", self.x, False)]:
            agg = dict(rl_kept=0.0, rl_drop=0.0, pl=0.0, n=0,
                       zmae_s=0.0, cmae_s=0.0, mae_c=0)
            for b in range(0, len(self.rows), batch):
                lg, rt = forward_z(params, jnp.asarray(xin[b : b + batch]))
                lp = np.asarray(jax.nn.log_softmax(lg + prior[None, :], axis=1), dtype=np.float64)
                rp = np.asarray(rt, dtype=np.float64)
                for j in range(lp.shape[0]):
                    i = b + j
                    idx = self.rows[i]
                    n = len(idx)
                    keep = self.keeps[i] if use_keep else np.ones(n, dtype=bool)
                    is_r = self.rated[i]
                    nr = max(is_r.sum(), 1)
                    zerr = rp[j, idx] - self.zt[i]
                    hub = huber(zerr)
                    agg["rl_kept"] += hub[is_r & keep].sum() / nr
                    agg["rl_drop"] += hub[is_r & ~keep].sum() / nr
                    agg["pl"] += -lp[j, idx].sum() / n
                    m = is_r & ~keep
                    conv = self.A[i] * rp[j, idx[m]] + self.B[i]
                    agg["zmae_s"] += np.abs(zerr[m]).sum()
                    agg["cmae_s"] += np.abs(conv - self.x[i, cs + idx[m]]).sum()
                    agg["mae_c"] += int(m.sum())
                    agg["n"] += 1
            n = agg["n"]
            res[tag] = {
                "presence_loss": agg["pl"] / n,
                "rating_loss": (agg["rl_kept"] + agg["rl_drop"]) / n,
                "rating_drop_component": agg["rl_drop"] / n,
                "zmae_drop_per_item": agg["zmae_s"] / max(agg["mae_c"], 1),
                "mae_drop_per_item_conv": agg["cmae_s"] / max(agg["mae_c"], 1),
            }
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-frac", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--vectors", default="../data/aug2026/user_input_vectors_cleanup_notrust.npz")
    ap.add_argument("--raw-scores", default="../data/aug2026/raw_scores_recon.npy")
    ap.add_argument("--census", default="../data/aug2026/rating_census.npz")
    ap.add_argument("--presence-prior-alpha", type=float, default=1.0)
    ap.add_argument("--eval-interval", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    global MODEL
    MODEL = Recommender()
    cs = CONF["corpus_size"]
    all_users, item_counts = load_all_users(args.vectors)

    raw = np.load(args.raw_scores)
    hist = np.load(args.census)["hist"].astype(np.float64)
    rh = hist[:, 1:]
    n_rated = rh.sum(1)
    mu = (rh * SCORES).sum(1) / np.maximum(n_rated, 1)
    sigma = np.sqrt((rh * (SCORES[None] - mu[:, None]) ** 2).sum(1) / np.maximum(n_rated, 1)) + 1e-6
    alpha = np.clip(sigma / 2.6, 0.3, 0.8)
    A_u = (alpha + (1 - alpha) * sigma / 2.5).astype(np.float64)
    B_u = ((1 - alpha) * (mu - 5.5) / 2.5).astype(np.float64)

    zvals_flat = np.zeros(len(raw), dtype=np.float32)
    lengths = np.array([len(u[0]) for u in all_users], dtype=np.int64)
    user_of_entry = np.repeat(np.arange(len(all_users)), lengths)
    rm = (raw >= 1) & (raw <= 10)
    zvals_flat[rm] = np.clip(
        (raw[rm].astype(np.float64) - mu[user_of_entry[rm]]) / sigma[user_of_entry[rm]], -3, 3
    ).astype(np.float32)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    zvals = [zvals_flat[s : s + l] for s, l in zip(starts, lengths)]

    rng_h = np.random.default_rng(HOLDOUT_SEED)
    perm = rng_h.permutation(len(all_users))
    n_hold = len(all_users) // 10
    holdout_idx = perm[:n_hold]
    train_pool = perm[n_hold:]
    rng_s = np.random.default_rng(SUBSAMPLE_SEED)
    n_train = max(int(len(train_pool) * args.user_frac), 1)
    train_idx = rng_s.choice(train_pool, size=n_train, replace=False)
    print(f"train users: {len(train_idx)}  holdout users: {n_hold}", flush=True)

    clipped = np.maximum(item_counts, 1.0)
    prior = (
        jnp.asarray(args.presence_prior_alpha * np.log(clipped / clipped.sum()), dtype=jnp.float32)
        if args.presence_prior_alpha > 0
        else jnp.zeros(cs, dtype=jnp.float32)
    )

    rng_e = np.random.default_rng(4242)
    eval_hold = FixedEvalSetZ(
        [all_users[i] for i in holdout_idx[:EVAL_N]], [zvals[i] for i in holdout_idx[:EVAL_N]],
        A_u[holdout_idx[:EVAL_N]], B_u[holdout_idx[:EVAL_N]], rng_e, cs)
    eval_train = FixedEvalSetZ(
        [all_users[i] for i in train_idx[:EVAL_N]], [zvals[i] for i in train_idx[:EVAL_N]],
        A_u[train_idx[:EVAL_N]], B_u[train_idx[:EVAL_N]], rng_e, cs)

    state = create_train_state(jax.random.PRNGKey(args.seed), CONF["learning_rate"])

    def loader():
        bs = CONF["batch_size"]
        while True:
            order = np.random.permutation(len(train_idx))
            for b_idx in range(0, len(order), bs):
                sel = train_idx[order[b_idx : b_idx + bs]]
                bt = np.zeros((len(sel), cs * 2), dtype=np.float32)
                zt = np.zeros((len(sel), cs), dtype=np.float32)
                rmask = np.zeros((len(sel), cs), dtype=np.float32)
                for i, u in enumerate(sel):
                    idxs, vals, rated, _st = all_users[u]
                    bt[i, idxs] = 1.0
                    bt[i, cs + idxs] = vals
                    zt[i, idxs] = zvals[u]
                    rmask[i, idxs] = rated.astype(np.float32)
                yield bt, zt, rmask

    gen = loader()
    logf = open(f"{args.out_prefix}.jsonl", "w")
    t0 = time.time()

    def run_eval(step):
        rec = {"step": step, "seed": args.seed, "elapsed_s": round(time.time() - t0, 1)}
        rec["holdout"] = eval_hold.evaluate(state.params, prior)
        rec["train"] = eval_train.evaluate(state.params, prior)
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        h = rec["holdout"]["corrupt"]
        print(
            f"[eval @ {step}] holdout: pres {h['presence_loss']:.4f} zmae_drop {h['zmae_drop_per_item']:.4f} "
            f"mae_conv {h['mae_drop_per_item_conv']:.4f}", flush=True,
        )

    for step in range(args.steps):
        bt, zt, rmask = next(gen)
        state, loss, p_loss, r_loss = train_step_zt(
            state, jnp.array(bt), jnp.array(zt), jnp.array(rmask), prior
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
