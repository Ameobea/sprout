"""Verification of the rating-gradient masking chain (vectorize -> npz -> batch -> loss).

Stages (run separately; grad stage needs JAX, run inside rocm_jax with JAX_PLATFORMS=cpu):
  npz  - internal invariants over all users: duplicate corpus indices, per-user
         constancy of the mu-fill value on unrated entries, dropped-sentinel fill
         strictly below it, value clip range
  csv  - re-derive the first N qualifying users from the raw CSV via the exact
         vectorize_variants code path and compare npz slices bit-for-bit
  grad - with the prod checkpoint and a real batch: all param gradients of the
         exact train_step loss are bitwise invariant to arbitrary perturbation of
         rating targets at unrated positions (and NOT invariant at rated ones)
"""

import argparse
import json

import numpy as np

STATUS_DROPPED = 3


def load_npz(path):
    d = np.load(path)
    idx = d["indices"].astype(np.int32)
    vals = d["values"].astype(np.float32)
    lengths = d["lengths"].astype(np.int64)
    rated = np.unpackbits(d["rated_masks"])[: int(d["total_mask_bits"][0])].astype(bool)
    statuses = d["statuses"]
    return idx, vals, lengths, rated, statuses


def stage_npz(args, report):
    idx, vals, lengths, rated, statuses = load_npz(args.vectors)
    n_users = len(lengths)
    user_of_entry = np.repeat(np.arange(n_users, dtype=np.int64), lengths)
    report["n_users"] = int(n_users)
    report["n_entries"] = int(len(idx))
    report["rated_frac"] = float(rated.mean())

    keys = user_of_entry * 6000 + idx
    n_dup = len(keys) - len(np.unique(keys))
    report["duplicate_index_entries"] = int(n_dup)

    import pandas as pd

    def group_minmax(mask):
        df = pd.DataFrame({"u": user_of_entry[mask], "v": vals[mask]})
        g = df.groupby("u")["v"].agg(["min", "max"])
        return g

    fill = group_minmax(~rated & (statuses != STATUS_DROPPED))
    report["unrated_fill_users"] = int(len(fill))
    report["unrated_fill_nonconstant_users"] = int((fill["min"] != fill["max"]).sum())

    drop = group_minmax(~rated & (statuses == STATUS_DROPPED))
    report["dropped_fill_users"] = int(len(drop))
    report["dropped_fill_nonconstant_users"] = int((drop["min"] != drop["max"]).sum())

    both = fill.join(drop, lsuffix="_f", rsuffix="_d", how="inner")
    report["dropfill_users_compared"] = int(len(both))
    report["dropfill_not_below_users"] = int((both["max_d"] >= both["min_f"]).sum())

    report["values_out_of_clip"] = int(((vals < -2.5) | (vals > 2.5)).sum())
    report["users_below_20_entries"] = int((lengths < 20).sum())
    report["rated_but_dropped_status_entries"] = int((rated & (statuses == STATUS_DROPPED)).sum())


def stage_csv(args, report):
    from vectorize_variants import load_flags, process, users

    idx, vals, lengths, rated, statuses = load_npz(args.vectors)
    starts = np.concatenate([[0], np.cumsum(lengths)])
    untrusted, huge = load_flags(args.metrics)

    with open(args.corpus) as f:
        ix_by_id = {aid: i for i, aid in enumerate(json.load(f))}

    n_checked = n_mismatch = 0
    u = 0
    for username, rows in users(args.src):
        if username in huge:
            continue
        res = process(rows, ix_by_id, keep_rated_ptw=True)
        if res is None or len(res[0]) < 20:
            continue
        s, e = starts[u], starts[u + 1]
        ok = (
            np.array_equal(res[0].astype(np.int32), idx[s:e])
            and np.array_equal(res[1], vals[s:e])
            and np.array_equal(res[2], rated[s:e])
            and np.array_equal(res[3], statuses[s:e])
        )
        if not ok:
            n_mismatch += 1
            if n_mismatch <= 5:
                print(f"MISMATCH user #{u} ({username})")
        n_checked += 1
        u += 1
        if n_checked >= args.prefix_users:
            break
    report["csv_users_checked"] = n_checked
    report["csv_mismatches"] = n_mismatch


def stage_grad(args, report):
    import sys

    sys.path.insert(0, ".")
    import jax
    import jax.numpy as jnp
    import optax
    from flax import serialization

    from model import CONF, Recommender
    from train import create_train_state, load_all_users

    cs = CONF["corpus_size"]
    all_users, item_counts = load_all_users(args.vectors)

    rng = np.random.default_rng(7)
    pick = rng.permutation(len(all_users))[:64]
    users = [all_users[i] for i in pick]
    assert sum(1 for u in users if not u[2].all()) >= 16, "batch lacks unrated coverage"

    b = len(users)
    x = np.zeros((b, cs * 2), dtype=np.float32)
    tgt_ratings = np.zeros((b, cs), dtype=np.float32)
    rated_mask = np.zeros((b, cs), dtype=np.float32)
    for i, (uidx, uvals, urated, _st) in enumerate(users):
        x[i, uidx] = 1.0
        x[i, cs + uidx] = uvals
        tgt_ratings[i, uidx] = uvals
        rated_mask[i, uidx] = urated.astype(np.float32)
    presence = x[:, :cs]

    keep = (rng.random((b, cs)) > 0.4).astype(np.float32)
    x_in = np.concatenate([presence * keep, x[:, cs:] * keep], axis=1)

    state = create_train_state(jax.random.PRNGKey(0), CONF["learning_rate"])
    with open(args.weights, "rb") as f:
        params = serialization.from_bytes(state.params, f.read())

    clipped = np.maximum(item_counts, 1.0)
    prior = jnp.asarray(np.log(clipped / clipped.sum()), dtype=jnp.float32)
    noise_key = jax.random.PRNGKey(42)
    x_in_j = jnp.asarray(x_in)
    presence_j = jnp.asarray(presence)
    mask_j = jnp.asarray(rated_mask)

    def loss_fn(p, ratings_tgt):
        logits, rating_pred, lv_p, lv_r = Recommender().apply(
            {"params": p}, x_in_j, training=True, rngs={"noise": noise_key}
        )
        log_probs = jax.nn.log_softmax(logits + prior[None, :], axis=1)
        cnt = jnp.maximum(jnp.sum(presence_j, axis=1), 1.0)
        p_loss = jnp.mean(-jnp.sum(presence_j * log_probs, axis=1) / cnt)
        per_entry = optax.huber_loss(rating_pred - ratings_tgt, delta=CONF["huber_delta"])
        denom = jnp.maximum(jnp.sum(mask_j, axis=1), 1.0)
        r_loss = jnp.mean(jnp.sum(mask_j * per_entry, axis=1) / denom)
        return jnp.mean(
            (jnp.exp(-lv_p) * p_loss + lv_p) + (jnp.exp(-lv_r) * r_loss + lv_r)
        )

    grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=0))
    l0, g0 = grad_fn(params, jnp.asarray(tgt_ratings))

    pert = tgt_ratings + (1.0 - rated_mask) * presence * 1.234
    l1, g1 = grad_fn(params, jnp.asarray(pert))
    leaves0, leaves1 = jax.tree.leaves(g0), jax.tree.leaves(g1)
    report["grad_unrated_loss_equal"] = bool(l0 == l1)
    report["grad_unrated_all_leaves_equal"] = bool(
        all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(leaves0, leaves1))
    )

    ri, rj = np.argwhere(rated_mask > 0)[0]
    pert2 = tgt_ratings.copy()
    pert2[ri, rj] += 1.234
    l2, g2 = grad_fn(params, jnp.asarray(pert2))
    leaves2 = jax.tree.leaves(g2)
    report["grad_rated_control_differs"] = bool(
        l0 != l2
        or any(not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(leaves0, leaves2))
    )

    def rl_only(preds):
        per_entry = optax.huber_loss(preds - jnp.asarray(tgt_ratings), delta=CONF["huber_delta"])
        denom = jnp.maximum(jnp.sum(mask_j, axis=1), 1.0)
        return jnp.mean(jnp.sum(mask_j * per_entry, axis=1) / denom)

    fake_preds = jnp.asarray(rng.normal(size=(b, cs)).astype(np.float32))
    dp = np.asarray(jax.grad(rl_only)(fake_preds))
    report["dloss_dpred_nonzero_at_unrated"] = int(((rated_mask == 0) & (dp != 0)).sum())
    report["dloss_dpred_zero_at_rated"] = int(((rated_mask == 1) & (dp == 0)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["npz", "csv", "grad"])
    ap.add_argument("--vectors", default="../data/aug2026/user_input_vectors_cleanup_notrust.npz")
    ap.add_argument("--src", default="../data/collected_animelists_aug2026.csv.gz")
    ap.add_argument("--metrics", default="../data/aug2026-profile-metrics.csv")
    ap.add_argument("--corpus", default="../data/corpus_ids_aug2026.json")
    ap.add_argument("--weights", default="../data/aug2026/jax_model_fresh_logq.msgpack")
    ap.add_argument("--prefix-users", type=int, default=100_000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    report = {"stage": args.stage}
    {"npz": stage_npz, "csv": stage_csv, "grad": stage_grad}[args.stage](args, report)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
