"""Graft probe: 3-channel NN where channel 3 = per-row z-normed EASE scores of the
(corrupted) presence input. Tests whether feeding the item-item signal into the NN
recovers the full-context ranking gap while keeping the NN's small-context edge.

Run inside rocm_jax: cd /jax_dir/notebooks && python analysis/train_probe_ease.py ...
"""

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

HOLDOUT_SEED = 999
SUBSAMPLE_SEED = 777

B_g = None  # set in main before jit tracing


def ease_channel(presence):
    e = presence @ B_g
    mu = jnp.mean(e, axis=1, keepdims=True)
    sd = jnp.std(e, axis=1, keepdims=True) + 1e-6
    return (e - mu) / sd


@jax.jit
def forward_clean3(params, x2):
    cs = CONF["corpus_size"]
    x3 = jnp.concatenate([x2, ease_channel(x2[:, :cs])], axis=1)
    logits, ratings, _, _ = Recommender().apply({"params": params}, x3, training=False)
    return logits, ratings


@jax.jit
def train_step3(state, batch, rated_mask, prior_logits):
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
    x_in = jnp.concatenate([xp, xr, ease_channel(xp)], axis=1)

    def loss_fn(params):
        item_logits, rating_pred, log_var_p, log_var_r = state.apply_fn(
            {"params": params}, x_in, training=True, rngs={"noise": vae_rng}
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
    new_params = optax.apply_updates(state.params, updates)
    state = state.replace(step=state.step + 1, params=new_params, opt_state=new_opt, key=dropout_rng)
    return state, loss, p_loss, r_loss


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--vectors", default="../data/aug2026/user_input_vectors_cleanup_notrust.npz")
    ap.add_argument("--ease-b", default="../data/aug2026/ease_B6k_lam200.npy")
    ap.add_argument("--presence-prior-alpha", type=float, default=1.0)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    global B_g
    B_g = jnp.asarray(np.load(args.ease_b), dtype=jnp.float32)

    CONF["input_channels"] = 3
    cs = CONF["corpus_size"]

    all_users, item_counts = load_all_users(args.vectors)
    rng_h = np.random.default_rng(HOLDOUT_SEED)
    perm = rng_h.permutation(len(all_users))
    n_hold = len(all_users) // 10
    holdout_idx = perm[:n_hold]
    train_users = [all_users[i] for i in perm[n_hold:]]
    print(f"train users: {len(train_users)}  holdout: {n_hold}", flush=True)

    clipped = np.maximum(item_counts, 1.0)
    log_pop = np.log(clipped / clipped.sum())
    prior = jnp.asarray(args.presence_prior_alpha * log_pop, dtype=jnp.float32)

    state = create_train_state(jax.random.PRNGKey(0), CONF["learning_rate"])
    loader = gen_batches(train_users, CONF["batch_size"], cs)

    t0 = time.time()
    for step in range(args.steps):
        bt, rm = next(loader)
        state, loss, p_loss, r_loss = train_step3(state, jnp.array(bt), jnp.array(rm), prior)
        if step % 500 == 0:
            lr_scale = optax.tree.get(state.opt_state, "scale")
            print(f"Step {step}: Loss {loss:.4f} (P {p_loss:.4f} R {r_loss:.4f}) lr {lr_scale} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            if lr_scale < 1e-6:
                break

    with open(f"{args.out_prefix}.msgpack", "wb") as f:
        f.write(serialization.to_bytes(state.params))
    print("saved weights", flush=True)

    # ---- final eval on holdout users: rank metrics at several context levels ----
    indices_all, starts_all, lengths_all = [], [], []
    hold_users = [all_users[i] for i in holdout_idx]
    results = {}

    def eval_cfg(k_ctx, keep_frac, n_users=3000):
        r = np.random.default_rng((k_ctx or 0) * 977 + int(keep_frac * 100))
        min_len = (k_ctx + 8) if k_ctx else 24
        sel = [u for u in hold_users if len(u[0]) >= min_len][:n_users]
        ranks, nll_sum, nll_cnt = [], 0.0, 0
        bx = np.zeros((256, cs * 2), dtype=np.float32)
        metas = []
        def flush(nb):
            nonlocal nll_sum, nll_cnt
            lg, _ = forward_clean3(state.params, jnp.asarray(bx[:nb]))
            lp = np.asarray(jax.nn.log_softmax(lg + prior[None, :], axis=1), dtype=np.float64)
            lgp = np.asarray(lg, dtype=np.float64) + log_pop[None, :]
            for j in range(nb):
                kept, dropped = metas[j]
                nll_sum += (-lp[j, dropped]).sum(); nll_cnt += len(dropped)
                sc = lgp[j].copy()
                sc[kept] = -np.inf
                o = np.argsort(-sc)
                ro = np.empty(cs, dtype=np.int32)
                ro[o] = np.arange(cs)
                ranks.extend(ro[dropped].tolist())
            metas.clear()
        nb = 0
        for idxs, vals, _rt, _st in sel:
            l = len(idxs)
            if k_ctx:
                kp = r.choice(l, size=k_ctx, replace=False)
                keep = np.zeros(l, dtype=bool); keep[kp] = True
            else:
                keep = r.random(l) > (1 - keep_frac)
                if keep.sum() == 0: keep[0] = True
                if (~keep).sum() == 0: keep[0] = False
            bx[nb] = 0.0
            bx[nb, idxs[keep]] = 1.0
            bx[nb, cs + idxs[keep]] = vals[keep]
            metas.append((idxs[keep], idxs[~keep]))
            nb += 1
            if nb == 256:
                flush(nb); nb = 0
        if nb:
            flush(nb)
        rk = np.asarray(ranks)
        return {"n": len(rk), "median_rank": float(np.median(rk)),
                "recall@50": float((rk < 50).mean()), "recall@250": float((rk < 250).mean()),
                "nll_drop": nll_sum / max(nll_cnt, 1)}

    for name, k_ctx, kf in [("k8", 8, 0), ("k16", 16, 0), ("keep0.6", None, 0.6),
                             ("keep0.9", None, 0.9), ("keep0.99", None, 0.99)]:
        results[name] = eval_cfg(k_ctx, kf)
        print(name, json.dumps(results[name]), flush=True)

    with open(f"{args.out_prefix}_eval.json", "w") as f:
        json.dump(results, f, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
