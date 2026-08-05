"""Per-item rating prediction dump on probe-holdout users, corrupt-holdout
protocol (train-style corruption, matches train_probe eval). Rows are (user_row,
item, target, pred) for dropped-rated items plus kept-rated pairs for feasible
per-user debiasing. Closed-form baselines derive offline from the vectors npz.
Run inside rocm_jax from notebooks/ (JAX_PLATFORMS=cpu while the GPU trains)."""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
import jax
import jax.numpy as jnp
from flax import serialization

from model import CONF, Recommender
from train import create_train_state, load_all_users
from train_probe import HOLDOUT_SEED

MODEL = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", default="../data/aug2026/user_input_vectors_cleanup_notrust.npz")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--rating-prior")
    ap.add_argument("--n-users", type=int, default=20_000)
    ap.add_argument("--corrupt-seed", type=int, default=555)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    global MODEL
    MODEL = Recommender()
    cs = CONF["corpus_size"]
    all_users, _ = load_all_users(args.vectors)

    rng_h = np.random.default_rng(HOLDOUT_SEED)
    perm = rng_h.permutation(len(all_users))
    holdout_idx = perm[: len(all_users) // 10][: args.n_users]
    users = [all_users[i] for i in holdout_idx]

    state = create_train_state(jax.random.PRNGKey(0), CONF["learning_rate"])
    with open(args.weights, "rb") as f:
        params = serialization.from_bytes(state.params, f.read())
    rprior = (
        jnp.asarray(np.load(args.rating_prior), dtype=jnp.float32)
        if args.rating_prior
        else None
    )

    @jax.jit
    def forward(p, x):
        _, ratings, _, _ = MODEL.apply({"params": p}, x, training=False)
        return ratings + rprior[None, :] if rprior is not None else ratings

    rng = np.random.default_rng(args.corrupt_seed)
    rates = rng.uniform(0.24, 0.56, size=len(users))
    keeps = []
    for i, (idx, _v, _r, _s) in enumerate(users):
        keep = rng.random(len(idx)) > rates[i]
        if keep.sum() == 0:
            keep[0] = True
        keeps.append(keep)

    d_user, d_item, d_tgt, d_pred = [], [], [], []
    k_user, k_item, k_tgt, k_pred = [], [], [], []
    B = 512
    for b in range(0, len(users), B):
        chunk = users[b : b + B]
        x = np.zeros((len(chunk), cs * 2), dtype=np.float32)
        for j, (idx, vals, _r, _s) in enumerate(chunk):
            keep = keeps[b + j]
            x[j, idx[keep]] = 1.0
            x[j, cs + idx[keep]] = vals[keep]
        preds = np.asarray(forward(params, jnp.asarray(x)), dtype=np.float32)
        for j, (idx, vals, rated, _s) in enumerate(chunk):
            keep = keeps[b + j]
            for arrs, m in ((d_user, d_item, d_tgt, d_pred), rated & ~keep), (
                (k_user, k_item, k_tgt, k_pred),
                rated & keep,
            ):
                arrs[0].append(np.full(m.sum(), b + j, dtype=np.int32))
                arrs[1].append(idx[m].astype(np.int32))
                arrs[2].append(vals[m])
                arrs[3].append(preds[j, idx[m]])
        if (b // B) % 8 == 0:
            print(f"{b + len(chunk)}/{len(users)} users", flush=True)

    np.savez(
        args.out,
        holdout_rows=holdout_idx.astype(np.int64),
        drop_user=np.concatenate(d_user), drop_item=np.concatenate(d_item),
        drop_tgt=np.concatenate(d_tgt), drop_pred=np.concatenate(d_pred),
        kept_user=np.concatenate(k_user), kept_item=np.concatenate(k_item),
        kept_tgt=np.concatenate(k_tgt), kept_pred=np.concatenate(k_pred),
    )
    print(
        f"{args.out}: {len(users)} users, dropped-rated {sum(len(a) for a in d_item):,}, "
        f"kept-rated {sum(len(a) for a in k_item):,}", flush=True
    )


if __name__ == "__main__":
    main()
