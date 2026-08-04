"""Decompose the training loss of a trained checkpoint into:
  - kept-item vs dropped-item components under train-style corruption
  - clean-input reconstruction loss (validation-style)
  - latent-noise contribution (train pass adds N(0, 0.1) to the bottleneck)
  - popularity-baseline NLL on the same dropped items
  - keep-fraction sweep: NLL/MAE on dropped items as a function of visible context

Run inside the rocm_jax container: cd /jax_dir/notebooks && python analysis/loss_decomposition.py ...
"""

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, ".")
import jax
import jax.numpy as jnp
from flax import serialization
from model import CONF, Recommender


def load_params(path):
    model = Recommender()
    dummy = jnp.ones((1, CONF["corpus_size"] * CONF["input_channels"]))
    params = model.init({"params": jax.random.PRNGKey(0), "noise": jax.random.PRNGKey(0)}, dummy)["params"]
    with open(path, "rb") as f:
        return serialization.from_bytes(params, f.read())


@jax.jit
def forward(params, x, noise_key):
    logits, ratings, _, _ = Recommender().apply(
        {"params": params}, x, training=True, rngs={"noise": noise_key}
    )
    return logits, ratings


@jax.jit
def forward_clean(params, x):
    logits, ratings, _, _ = Recommender().apply({"params": params}, x, training=False)
    return logits, ratings


def huber(err, delta=1.0):
    a = np.abs(err)
    return np.where(a <= delta, 0.5 * a * a, delta * (a - 0.5 * delta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--vectors", default="../data/aug2026/user_input_vectors_cleanup_notrust.npz")
    ap.add_argument("--prior-alpha", type=float, default=0.0)
    ap.add_argument("--n-users", type=int, default=20000)
    ap.add_argument("--sweep-users", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--user-list", default=None, help="npy of user indices; overrides random sample")
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cs = CONF["corpus_size"]
    d = np.load(args.vectors)
    indices = d["indices"].astype(np.int32)
    values = d["values"]
    lengths = d["lengths"].astype(np.int64)
    rated = np.unpackbits(d["rated_masks"])[: int(d["total_mask_bits"][0])].astype(bool)
    starts = np.zeros(len(lengths), dtype=np.int64)
    np.cumsum(lengths[:-1], out=starts[1:])

    counts = np.bincount(indices, minlength=cs).astype(np.float64)
    clipped = np.maximum(counts, 1.0)
    log_pop = np.log(clipped / clipped.sum())
    prior = jnp.asarray(args.prior_alpha * log_pop, dtype=jnp.float32)
    log_pop_j = np.asarray(log_pop, dtype=np.float64)

    order = np.argsort(-counts)
    rank_of_item = np.empty(cs, dtype=np.int32)
    rank_of_item[order] = np.arange(cs)
    TIERS = [(0, 250), (250, 1000), (1000, 3000), (3000, 6000)]

    rng = np.random.default_rng(args.seed)
    if args.user_list:
        users = np.load(args.user_list)[: args.n_users]
    else:
        users = rng.choice(len(lengths), size=args.n_users, replace=False)

    params = load_params(args.weights)
    key = jax.random.PRNGKey(args.seed)

    def make_batch(uids):
        b = len(uids)
        x = np.zeros((b, cs * 2), dtype=np.float32)
        rm = np.zeros((b, cs), dtype=np.float32)
        rows = []
        for i, u in enumerate(uids):
            s, l = starts[u], lengths[u]
            idx = indices[s : s + l]
            x[i, idx] = 1.0
            x[i, cs + idx] = values[s : s + l]
            rm[i, idx] = rated[s : s + l]
            rows.append(idx)
        return x, rm, rows

    def corrupt(x, rows, rate_lo, rate_hi):
        b = x.shape[0]
        keep_masks = []
        xc = x.copy()
        rates = rng.uniform(rate_lo, rate_hi, size=b)
        for i, idx in enumerate(rows):
            keep = rng.random(len(idx)) > rates[i]
            if keep.sum() == 0:
                keep[rng.integers(len(idx))] = True
            dropped = idx[~keep]
            xc[i, dropped] = 0.0
            xc[i, cs + dropped] = 0.0
            keep_masks.append(keep)
        return xc, keep_masks

    def accumulate(logits, ratings_pred, x, rm, rows, keep_masks, acc):
        lp = np.asarray(jax.nn.log_softmax(logits + prior[None, :], axis=1), dtype=np.float64)
        rp = np.asarray(ratings_pred, dtype=np.float64)
        for i, idx in enumerate(rows):
            n = len(idx)
            keep = keep_masks[i] if keep_masks is not None else np.ones(n, dtype=bool)
            nll = -lp[i, idx]
            is_rated = rm[i, idx] > 0
            err = rp[i, idx] - x[i, cs + idx]  # x holds uncorrupted values here
            hub = huber(err)

            acc["n_users"] += 1
            acc["nll_kept_over_n"] += nll[keep].sum() / n
            acc["nll_drop_over_n"] += nll[~keep].sum() / n
            acc["kept_frac"] += keep.mean()
            acc["nll_kept_sum"] += nll[keep].sum(); acc["kept_cnt"] += int(keep.sum())
            acc["nll_drop_sum"] += nll[~keep].sum(); acc["drop_cnt"] += int((~keep).sum())
            acc["pop_nll_drop_sum"] += (-log_pop_j[idx[~keep]]).sum()

            rk = is_rated & keep
            rd = is_rated & ~keep
            nr = max(is_rated.sum(), 1)
            acc["hub_kept_over_nr"] += hub[rk].sum() / nr
            acc["hub_drop_over_nr"] += hub[rd].sum() / nr
            acc["hub_kept_sum"] += hub[rk].sum(); acc["rk_cnt"] += int(rk.sum())
            acc["hub_drop_sum"] += hub[rd].sum(); acc["rd_cnt"] += int(rd.sum())
            acc["mae_kept_sum"] += np.abs(err[rk]).sum()
            acc["mae_drop_sum"] += np.abs(err[rd]).sum()

            tiers = rank_of_item[idx]
            for t, (lo, hi) in enumerate(TIERS):
                m = (~keep) & (tiers >= lo) & (tiers < hi)
                acc["tier_nll_drop"][t] += nll[m].sum()
                acc["tier_drop_cnt"][t] += int(m.sum())
                mr = m & is_rated
                acc["tier_mae_drop"][t] += np.abs(err[mr]).sum()
                acc["tier_rated_drop_cnt"][t] += int(mr.sum())

    def new_acc():
        return {
            "n_users": 0, "nll_kept_over_n": 0.0, "nll_drop_over_n": 0.0, "kept_frac": 0.0,
            "nll_kept_sum": 0.0, "kept_cnt": 0, "nll_drop_sum": 0.0, "drop_cnt": 0,
            "pop_nll_drop_sum": 0.0,
            "hub_kept_over_nr": 0.0, "hub_drop_over_nr": 0.0,
            "hub_kept_sum": 0.0, "rk_cnt": 0, "hub_drop_sum": 0.0, "rd_cnt": 0,
            "mae_kept_sum": 0.0, "mae_drop_sum": 0.0,
            "tier_nll_drop": [0.0] * 4, "tier_drop_cnt": [0] * 4,
            "tier_mae_drop": [0.0] * 4, "tier_rated_drop_cnt": [0] * 4,
        }

    def finalize(acc):
        nu = max(acc["n_users"], 1)
        out = {
            "presence_loss": (acc["nll_kept_over_n"] + acc["nll_drop_over_n"]) / nu,
            "presence_kept_component": acc["nll_kept_over_n"] / nu,
            "presence_drop_component": acc["nll_drop_over_n"] / nu,
            "kept_frac": acc["kept_frac"] / nu,
            "nll_kept_per_item": acc["nll_kept_sum"] / max(acc["kept_cnt"], 1),
            "nll_drop_per_item": acc["nll_drop_sum"] / max(acc["drop_cnt"], 1),
            "pop_nll_drop_per_item": acc["pop_nll_drop_sum"] / max(acc["drop_cnt"], 1),
            "rating_loss": (acc["hub_kept_over_nr"] + acc["hub_drop_over_nr"]) / nu,
            "rating_kept_component": acc["hub_kept_over_nr"] / nu,
            "rating_drop_component": acc["hub_drop_over_nr"] / nu,
            "hub_kept_per_item": acc["hub_kept_sum"] / max(acc["rk_cnt"], 1),
            "hub_drop_per_item": acc["hub_drop_sum"] / max(acc["rd_cnt"], 1),
            "mae_kept_per_item": acc["mae_kept_sum"] / max(acc["rk_cnt"], 1),
            "mae_drop_per_item": acc["mae_drop_sum"] / max(acc["rd_cnt"], 1),
            "tiers": [
                {
                    "tier": f"{lo}-{hi}",
                    "nll_drop": acc["tier_nll_drop"][t] / max(acc["tier_drop_cnt"][t], 1),
                    "mae_drop": acc["tier_mae_drop"][t] / max(acc["tier_rated_drop_cnt"][t], 1),
                    "drop_cnt": acc["tier_drop_cnt"][t],
                }
                for t, (lo, hi) in enumerate(TIERS)
            ],
        }
        return out

    results = {"weights": args.weights, "prior_alpha": args.prior_alpha, "n_users": args.n_users}

    # --- pass 1: train-style corruption (rate 0.4 +/- 40%), with and without latent noise ---
    rate_lo, rate_hi = 0.4 - 0.16, 0.4 + 0.16
    acc_noise, acc_nonoise, acc_clean = new_acc(), new_acc(), new_acc()
    floor_sum = 0.0
    for bstart in range(0, len(users), args.batch):
        uids = users[bstart : bstart + args.batch]
        x, rm, rows = make_batch(uids)
        floor_sum += sum(np.log(len(r)) for r in rows)
        xc, keep_masks = corrupt(x, rows, rate_lo, rate_hi)

        key, k1 = jax.random.split(key)
        lg, rt = forward(params, jnp.asarray(xc), k1)
        accumulate(lg, rt, x, rm, rows, keep_masks, acc_noise)

        lg, rt = forward_clean(params, jnp.asarray(xc))
        accumulate(lg, rt, x, rm, rows, keep_masks, acc_nonoise)

        lg, rt = forward_clean(params, jnp.asarray(x))
        accumulate(lg, rt, x, rm, rows, None, acc_clean)

    results["train_corruption_with_noise"] = finalize(acc_noise)
    results["train_corruption_no_noise"] = finalize(acc_nonoise)
    results["clean_input"] = finalize(acc_clean)
    results["floor_E_log_n"] = floor_sum / len(users)

    # --- pass 2: keep-fraction sweep on a subsample, no latent noise ---
    sweep = []
    sweep_users = users[: args.sweep_users]
    for keep_frac in ([] if args.skip_sweep else [0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.99]):
        acc = new_acc()
        ranks_all = []
        for bstart in range(0, len(sweep_users), args.batch):
            uids = sweep_users[bstart : bstart + args.batch]
            x, rm, rows = make_batch(uids)
            xc, keep_masks = corrupt(x, rows, 1 - keep_frac, 1 - keep_frac)
            lg, rt = forward_clean(params, jnp.asarray(xc))
            accumulate(lg, rt, x, rm, rows, keep_masks, acc)
            lgp = np.asarray(lg) + np.asarray(prior)[None, :]
            for i, idx in enumerate(rows):
                keep = keep_masks[i]
                row_logits = lgp[i].copy()
                row_logits[idx[keep]] = -np.inf
                order_r = np.argsort(-row_logits)
                rank_of = np.empty(cs, dtype=np.int32)
                rank_of[order_r] = np.arange(cs)
                ranks_all.extend(rank_of[idx[~keep]].tolist())
        f = finalize(acc)
        f["keep_frac_target"] = keep_frac
        ranks_all = np.asarray(ranks_all)
        f["median_rank"] = float(np.median(ranks_all))
        f["recall@50"] = float((ranks_all < 50).mean())
        f["recall@250"] = float((ranks_all < 250).mean())
        sweep.append(f)
        print(f"sweep keep={keep_frac}: nll_drop={f['nll_drop_per_item']:.4f} mae_drop={f['mae_drop_per_item']:.4f} medrank={f['median_rank']} r@50={f['recall@50']:.4f}", flush=True)
    results["keep_sweep"] = sweep

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(json.dumps({k: v for k, v in results.items() if k != "keep_sweep"}, indent=1))


if __name__ == "__main__":
    main()
