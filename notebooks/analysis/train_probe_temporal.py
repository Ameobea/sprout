"""Temporal input channel probe for the rating head: standard multitask arch +
protocol (vs probe_frac1.0 control) with a third input channel giving each
entry's within-user temporal rank (avg-rank over dated entries scaled to
(0,1]; 0 = undated). Ranks are computed on the full profile (serve-realistic)
and corrupted jointly with presence/ratings. Sources: start_day int16 from
temporal_start_days.npz (true watch dates, sparse) or upd_sec int32 from
temporal_upd_sec.npz (list-update times, ~full coverage).
Ends with a rating_floors_dump npz. Run inside rocm_jax from notebooks/."""

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
from scipy.stats import rankdata

from model import CONF, Recommender
from train import TrainState, load_all_users

HOLDOUT_SEED = 999
EVAL_N = 2048
MODEL = None


def temporal_values(days, missing):
    dated = days != missing
    out = np.zeros(len(days), dtype=np.float32)
    n = dated.sum()
    if n >= 2:
        out[dated] = rankdata(days[dated], method="average") / n
    elif n == 1:
        out[dated] = 1.0
    return out


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


class FixedEvalSetT:
    def __init__(self, users, tvals, rng, cs):
        self.cs = cs
        self.x = np.zeros((len(users), cs * 3), dtype=np.float32)
        self.rows, self.rated, self.keeps = [], [], []
        rates = rng.uniform(0.24, 0.56, size=len(users))
        for i, ((idx, vals, rated_m, _st), tv) in enumerate(zip(users, tvals)):
            self.x[i, idx] = 1.0
            self.x[i, cs + idx] = vals
            self.x[i, 2 * cs + idx] = tv
            self.rows.append(idx.astype(np.int64))
            self.rated.append(rated_m.astype(bool))
            keep = rng.random(len(idx)) > rates[i]
            if keep.sum() == 0:
                keep[0] = True
            self.keeps.append(keep)
        self.xc = self.x.copy()
        for i, idx in enumerate(self.rows):
            dropped = idx[~self.keeps[i]]
            for ch in range(3):
                self.xc[i, ch * cs + dropped] = 0.0

    def evaluate(self, params, prior, batch=512):
        cs = self.cs
        agg = dict(pl=0.0, n=0, nll_drop_s=0.0, nll_drop_c=0, mae_drop_s=0.0, mae_drop_c=0)
        for b in range(0, len(self.rows), batch):
            lg, rt = forward_t(params, jnp.asarray(self.xc[b : b + batch]))
            lp = np.asarray(jax.nn.log_softmax(lg + prior[None, :], axis=1), dtype=np.float64)
            rp = np.asarray(rt, dtype=np.float64)
            for j in range(lp.shape[0]):
                i = b + j
                idx = self.rows[i]
                keep = self.keeps[i]
                nll = -lp[j, idx]
                err = rp[j, idx] - self.x[i, cs + idx]
                agg["pl"] += nll.sum() / len(idx)
                agg["nll_drop_s"] += nll[~keep].sum(); agg["nll_drop_c"] += int((~keep).sum())
                m = self.rated[i] & ~keep
                agg["mae_drop_s"] += np.abs(err[m]).sum(); agg["mae_drop_c"] += int(m.sum())
                agg["n"] += 1
        return {"presence_loss": agg["pl"] / agg["n"],
                "nll_drop_per_item": agg["nll_drop_s"] / max(agg["nll_drop_c"], 1),
                "mae_drop_per_item": agg["mae_drop_s"] / max(agg["mae_drop_c"], 1)}


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
    ap.add_argument("--temporal", required=True)
    ap.add_argument("--source", choices=["start_day", "upd_sec"], required=True)
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--vectors", default="../data/aug2026/user_input_vectors_cleanup_notrust.npz")
    ap.add_argument("--presence-prior-alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    MODEL = Recommender()
    cs = CONF["corpus_size"]
    all_users, item_counts = load_all_users(args.vectors)

    with np.load(args.temporal) as tz:
        raw = tz[args.source]
    missing = -32768 if args.source == "start_day" else -1
    lengths = np.array([len(u[0]) for u in all_users], dtype=np.int64)
    assert lengths.sum() == len(raw), f"temporal len {len(raw)} != entries {lengths.sum()}"
    tvals_all = []
    pos = 0
    t0 = time.time()
    for l in lengths:
        tvals_all.append(temporal_values(raw[pos : pos + l], missing))
        pos += l
    cov = np.concatenate([tv > 0 for tv in tvals_all]).mean()
    print(f"temporal ranks built ({time.time()-t0:.0f}s), source={args.source}, entry coverage {cov:.4f}", flush=True)

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
    from optax import contrib
    tx = optax.chain(
        optax.adam(CONF["learning_rate"]),
        contrib.reduce_on_plateau(patience=5, cooldown=1, factor=0.5, rtol=1e-4, accumulation_size=200),
    )
    state = TrainState.create(apply_fn=MODEL.apply, params=params, tx=tx, key=jax.random.PRNGKey(args.seed))
    loader = gen_batches(train_users, train_tvals, CONF["batch_size"], cs)

    rng_e = np.random.default_rng(4242)
    eval_hold = FixedEvalSetT([all_users[i] for i in holdout_idx[:EVAL_N]],
                              [tvals_all[i] for i in holdout_idx[:EVAL_N]], rng_e, cs)

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
            ev = eval_hold.evaluate(state.params, prior)
            ev["step"] = step
            logf.write(json.dumps(ev) + "\n")
            logf.flush()
            print(f"[eval @ {step}] holdout: pres {ev['presence_loss']:.4f} "
                  f"nll_drop {ev['nll_drop_per_item']:.4f} mae_drop {ev['mae_drop_per_item']:.4f}", flush=True)

    ev = eval_hold.evaluate(state.params, prior)
    ev["step"] = args.steps
    logf.write(json.dumps(ev) + "\n")
    logf.close()
    with open(f"{args.out_prefix}.msgpack", "wb") as f:
        f.write(serialization.to_bytes(state.params))
    floors_dump(state.params, all_users, tvals_all, holdout_idx, cs, f"{args.out_prefix}_dump.npz")
    print(f"done: {args.out_prefix}", flush=True)


if __name__ == "__main__":
    main()
