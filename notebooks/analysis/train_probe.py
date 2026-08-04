"""Training probe for generalization/scaling analysis.

Trains with the production recipe on a fraction of users, holding out a fixed
10% user set that is never trained on. Logs train-user vs holdout-user losses
(corrupted kept/dropped split + clean reconstruction) every eval_interval steps.

Run inside rocm_jax: cd /jax_dir/notebooks && python analysis/train_probe.py ...
"""

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import jax
import jax.numpy as jnp
from flax import serialization

from model import CONF, Recommender
from train import create_train_state, data_generator, train_step, load_all_users

HOLDOUT_SEED = 999
SUBSAMPLE_SEED = 777
EVAL_N = 2048

MODEL = None  # set in main before jit tracing


def huber(err, delta=1.0):
    a = np.abs(err)
    return np.where(a <= delta, 0.5 * a * a, delta * (a - 0.5 * delta))


class FixedEvalSet:
    """Pre-built dense batches with a fixed corruption pattern."""

    def __init__(self, users, rng, cs):
        self.cs = cs
        self.x = np.zeros((len(users), cs * 2), dtype=np.float32)
        self.rows = []
        self.rated = []
        self.keeps = []
        rates = rng.uniform(0.24, 0.56, size=len(users))
        for i, (idx, vals, rated_m, _st) in enumerate(users):
            self.x[i, idx] = 1.0
            self.x[i, cs + idx] = vals
            self.rows.append(idx.astype(np.int64))
            self.rated.append(rated_m.astype(bool))
            keep = rng.random(len(idx)) > rates[i]
            if keep.sum() == 0:
                keep[0] = True
            self.keeps.append(keep)
        self.xc = self.x.copy()
        for i, idx in enumerate(self.rows):
            dropped = idx[~self.keeps[i]]
            self.xc[i, dropped] = 0.0
            self.xc[i, cs + dropped] = 0.0

    def evaluate(self, params, prior, forward_fn, batch=512):
        cs = self.cs
        res = {}
        for tag, xin, use_keep in [("corrupt", self.xc, True), ("clean", self.x, False)]:
            agg = dict(pl_kept=0.0, pl_drop=0.0, rl_kept=0.0, rl_drop=0.0, n=0,
                       nll_kept_s=0.0, nll_kept_c=0, nll_drop_s=0.0, nll_drop_c=0,
                       mae_drop_s=0.0, mae_drop_c=0)
            for b in range(0, len(self.rows), batch):
                lg, rt = forward_fn(params, jnp.asarray(xin[b : b + batch]))
                lp = np.asarray(jax.nn.log_softmax(lg + prior[None, :], axis=1), dtype=np.float64)
                rp = np.asarray(rt, dtype=np.float64)
                for j in range(lp.shape[0]):
                    i = b + j
                    idx = self.rows[i]
                    n = len(idx)
                    keep = self.keeps[i] if use_keep else np.ones(n, dtype=bool)
                    nll = -lp[j, idx]
                    err = rp[j, idx] - self.x[i, cs + idx]
                    hub = huber(err)
                    is_r = self.rated[i]
                    nr = max(is_r.sum(), 1)
                    agg["pl_kept"] += nll[keep].sum() / n
                    agg["pl_drop"] += nll[~keep].sum() / n
                    agg["rl_kept"] += hub[is_r & keep].sum() / nr
                    agg["rl_drop"] += hub[is_r & ~keep].sum() / nr
                    agg["nll_kept_s"] += nll[keep].sum(); agg["nll_kept_c"] += int(keep.sum())
                    agg["nll_drop_s"] += nll[~keep].sum(); agg["nll_drop_c"] += int((~keep).sum())
                    m = is_r & ~keep
                    agg["mae_drop_s"] += np.abs(err[m]).sum(); agg["mae_drop_c"] += int(m.sum())
                    agg["n"] += 1
            n = agg["n"]
            res[tag] = {
                "presence_loss": (agg["pl_kept"] + agg["pl_drop"]) / n,
                "presence_drop_component": agg["pl_drop"] / n,
                "rating_loss": (agg["rl_kept"] + agg["rl_drop"]) / n,
                "rating_drop_component": agg["rl_drop"] / n,
                "nll_kept_per_item": agg["nll_kept_s"] / max(agg["nll_kept_c"], 1),
                "nll_drop_per_item": agg["nll_drop_s"] / max(agg["nll_drop_c"], 1),
                "mae_drop_per_item": agg["mae_drop_s"] / max(agg["mae_drop_c"], 1),
            }
        return res


@jax.jit
def forward_clean(params, x):
    logits, ratings, _, _ = MODEL.apply({"params": params}, x, training=False)
    return logits, ratings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-frac", type=float, required=True)
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--vectors", default="../data/aug2026/user_input_vectors_cleanup_notrust.npz")
    ap.add_argument("--presence-prior-alpha", type=float, default=1.0)
    ap.add_argument("--eval-interval", type=int, default=2000)
    ap.add_argument("--dropout-rate", type=float, default=None)
    ap.add_argument("--dropout-variation", type=float, default=None)
    ap.add_argument("--bottleneck-dim", type=int, default=None)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()
    if args.dropout_rate is not None:
        CONF["dropout_rate"] = args.dropout_rate
    if args.dropout_variation is not None:
        CONF["dropout_variation"] = args.dropout_variation

    global MODEL
    bd = args.bottleneck_dim or CONF["bottleneck_dim"]
    MODEL = Recommender(bottleneck_dim=bd)
    if args.bottleneck_dim is not None:
        import train as train_mod
        train_mod.Recommender = lambda: Recommender(bottleneck_dim=bd)
    print(f"bottleneck_dim = {bd}", flush=True)

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
    print(f"train users: {len(train_users)}  holdout users: {n_hold}")

    # prior from the FULL dataset counts (matches production training)
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

    state = create_train_state(jax.random.PRNGKey(0), CONF["learning_rate"])
    loader = data_generator(train_users, batch_size=CONF["batch_size"])

    log_path = f"{args.out_prefix}.jsonl"
    logf = open(log_path, "w")
    t0 = time.time()

    def run_eval(step):
        rec = {"step": step, "user_frac": args.user_frac, "elapsed_s": round(time.time() - t0, 1)}
        rec["holdout"] = eval_hold.evaluate(state.params, prior, forward_clean)
        rec["train"] = eval_train.evaluate(state.params, prior, forward_clean)
        lvp = float(state.params["log_var_presence"][0])
        lvr = float(state.params["log_var_rating"][0])
        rec["log_var_presence"] = lvp
        rec["log_var_rating"] = lvr
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
        state, loss, p_loss, r_loss = train_step(
            state, jnp.array(batch), jnp.array(rated_mask), prior, pop_w
        )
        if step % 500 == 0:
            import optax
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
