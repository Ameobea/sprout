import json
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from flax import serialization
from flax.training import train_state
import optax
from optax import contrib
import csv

from model import (
    CONF,
    Recommender,
    make_dense_profile,
    infer_outputs,
    rank_by_weighted_score,
    compute_holdout_metrics,
)

EVAL_PROFILE_SEED = 2222
EVAL_PROFILE_COUNT = 200
EVAL_PROFILE_MIN_RATINGS = 50
EVAL_PROFILE_MAX_RATINGS = 300


def data_generator(all_users, batch_size=32):
    while True:
        # the list of users is shuffled each epoch
        perm = np.random.permutation(len(all_users))

        for b_idx in range(0, len(all_users), batch_size):
            batch_indices = perm[b_idx : b_idx + batch_size]

            current_batch_size = len(batch_indices)
            batch_tensor = np.zeros(
                (current_batch_size, CONF["corpus_size"] * 2), dtype=np.float32
            )
            rated_mask_tensor = np.zeros(
                (current_batch_size, CONF["corpus_size"]), dtype=np.float32
            )

            for i, u_idx in enumerate(batch_indices):
                idxs, vals, rated = all_users[u_idx]
                batch_tensor[i, idxs] = 1.0
                batch_tensor[i, CONF["corpus_size"] + idxs] = vals
                rated_mask_tensor[i, idxs] = rated.astype(np.float32)

            yield batch_tensor, rated_mask_tensor


class TrainState(train_state.TrainState):
    key: jax.Array


def create_train_state(rng, learning_rate):
    model = Recommender()

    dummy_input = jnp.ones((1, CONF["corpus_size"] * 2))
    params = model.init({"params": rng, "noise": rng}, dummy_input)["params"]

    # Number of epochs with no improvement after which learning rate will be reduced
    patience = 5
    # Number of epochs to wait before resuming normal operation after the learning rate reduction
    cooldown = 1
    # Factor by which to reduce the learning rate
    factor = 0.5
    # Relative tolerance for measuring the new optimum
    rtol = 1e-4
    # Number of iterations to accumulate an average value
    accumulation_size = 200

    tx = optax.chain(
        optax.adam(learning_rate),
        contrib.reduce_on_plateau(
            patience=patience,
            cooldown=cooldown,
            factor=factor,
            rtol=rtol,
            accumulation_size=accumulation_size,
        ),
    )
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx, key=rng)


@jax.jit
def train_step(state, batch, rated_mask):
    presence = batch[:, : CONF["corpus_size"]]
    ratings_z = batch[:, CONF["corpus_size"] :]

    dropout_rng, vae_rng = random.split(state.key)
    # keep = random.bernoulli(dropout_rng, p=(1.0 - dropout_rate), shape=presence.shape)

    # vary the dropout rate for each sample in the batch +-40% from the base rate
    rate_variation = CONF["dropout_variation"] * CONF["dropout_rate"]
    random_rates = (
        CONF["dropout_rate"]
        + random.uniform(dropout_rng, shape=(presence.shape[0], 1))
        * (2 * rate_variation)
        - rate_variation
    )
    # clamp to force the droprate rate to always be in [0.01, 0.75]
    random_rates = jnp.clip(random_rates, 0.01, 0.75)

    keep = random.bernoulli(dropout_rng, p=(1.0 - random_rates), shape=presence.shape)

    corrupted_presence = presence * keep
    corrupted_ratings = ratings_z * keep
    x_in = jnp.concatenate([corrupted_presence, corrupted_ratings], axis=1)

    x_tgt_presence = presence
    # For rating loss, only consider items that were actually rated by the user
    # (not items where we guessed the rating for unrated shows)
    rating_loss_mask = rated_mask

    def loss_fn(params):
        item_logits, rating_pred, log_var_presence, log_var_rating = state.apply_fn(
            {"params": params},
            x_in,
            training=True,
            rngs={"noise": vae_rng},
        )

        # presence multinomial loss
        log_probs = jax.nn.log_softmax(item_logits, axis=1)
        per_user_counts = jnp.maximum(jnp.sum(x_tgt_presence, axis=1), 1.0)
        recon_presence_per_user = (
            -jnp.sum(x_tgt_presence * log_probs, axis=1) / per_user_counts
        )
        presence_loss = jnp.mean(recon_presence_per_user)

        # rating huber loss, masked to only rated items
        err = rating_pred - ratings_z
        per_entry = optax.huber_loss(err, delta=CONF["huber_delta"])
        denom_r = jnp.maximum(jnp.sum(rating_loss_mask, axis=1), 1.0)
        rating_loss_per_user = jnp.sum(rating_loss_mask * per_entry, axis=1) / denom_r
        rating_loss = jnp.mean(rating_loss_per_user)

        # Uncertainty Weighting
        # from "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
        # https://arxiv.org/abs/1705.07115
        # We use exp(-log_var) as the precision multiplier
        precision_p = jnp.exp(-log_var_presence)
        precision_r = jnp.exp(-log_var_rating)

        weighted_loss = (precision_p * presence_loss + log_var_presence) + (
            precision_r * rating_loss + log_var_rating
        )

        return jnp.mean(weighted_loss), (presence_loss, rating_loss)

    (loss, (recon_presence, recon_rating)), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(state.params)

    updates, new_opt_state = state.tx.update(
        grads, state.opt_state, state.params, value=loss
    )
    new_params = optax.apply_updates(state.params, updates)

    state = state.replace(
        step=state.step + 1, params=new_params, opt_state=new_opt_state, key=dropout_rng
    )

    return state, loss, recon_presence, recon_rating


@jax.jit
def validation_step(state, batch, rated_mask):
    presence = batch[:, : CONF["corpus_size"]]
    ratings_z = batch[:, CONF["corpus_size"] :]

    # No dropout - use full input
    x_in = batch

    x_tgt_presence = presence
    rating_loss_mask = rated_mask

    item_logits, rating_pred, _log_var_presence, _log_var_rating = state.apply_fn(
        {"params": state.params}, x_in, training=False
    )

    # --- Presence multinomial loss ---
    log_probs = jax.nn.log_softmax(item_logits, axis=1)
    per_user_counts = jnp.maximum(jnp.sum(x_tgt_presence, axis=1), 1.0)
    recon_presence_per_user = (
        -jnp.sum(x_tgt_presence * log_probs, axis=1) / per_user_counts
    )
    recon_presence = jnp.mean(recon_presence_per_user)

    # --- Rating regression loss (masked Huber) ---
    err = rating_pred - ratings_z
    per_entry = optax.huber_loss(err, delta=CONF["huber_delta"])
    denom_r = jnp.maximum(jnp.sum(rating_loss_mask, axis=1), 1.0)
    recon_rating_per_user = jnp.sum(rating_loss_mask * per_entry, axis=1) / denom_r
    recon_rating = jnp.mean(recon_rating_per_user)

    return recon_presence, recon_rating


DEBUG_PROFILE_IDXS = np.array(
    # fmt: off
    [1000, 2399,  258, 1976,  806,  114,  839,  173,  233,    3,  549,
        104,  188,  548,  178,  942,  614, 2787,  445,  301,   86, 2161,
          7,  401,  509,  492,  108,  280,  421, 1027,  536,  289,  370,
        443,  623, 1053,  232, 2414,  336, 4716,  542,  628,   19,  661,
         72,  786,    1,  606,   20, 1796, 1494,  431,  619, 2267,  949],
    dtype=np.int32,
)

DEBUG_PROFILE_VALS = np.array(
    # fmt: off
    [-0.03544337, -0.03544337,  0.50305784, -0.03544337,  0.33736503,
        0.50305784, -0.03544337,  0.50305784,  1.5800602 ,  0.50305784,
       -2.3844566 , -0.03544337, -1.1124458 , -0.03544337,  0.33736503,
        1.041559  ,  0.50305784,  1.041559  ,  0.50305784,  0.50305784,
        0.50305784,  1.5800602 , -0.03544337,  1.5800602 ,  0.50305784,
        0.50305784,  0.50305784,  0.50305784,  0.50305784, -0.03544337,
        1.5800602 ,  0.50305784,  0.50305784,  0.50305784,  0.50305784,
       -0.57394457,  1.5800602 ,  1.041559  ,  0.50305784,  0.50305784,
       -0.03544337, -0.03544337, -2.3844566 ,  0.50305784,  1.041559  ,
        0.33736503, -1.1124458 ,  1.5800602 ,  0.50305784, -0.57394457,
       -0.03544337,  0.50305784,  0.50305784,  1.041559  ,  0.50305784],
    dtype=np.float32,
)


def print_top_recs(
    corpus_ids, item_logits_1d, preds_1d, already_rated_mask_1d, anime_titles, k=50
):
    logit_weight = CONF["rec_logit_weight"]
    topk_idx_w, topk_scores_w, topk_probs_w, topk_ratings_w = rank_by_weighted_score(
        item_logits_1d, preds_1d, already_rated_mask_1d, k=k, logit_weight=logit_weight
    )

    print(f"  [logit_weight={logit_weight:.2f}, rating_weight={1.0-logit_weight:.2f}]")
    for rank, (ci, score, prob, rating) in enumerate(
        zip(topk_idx_w, topk_scores_w, topk_probs_w, topk_ratings_w), start=1
    ):
        anime_id = corpus_ids[ci]
        title = anime_titles.get(anime_id, "Unknown")
        print(
            f"  {rank:2d}. corpus_idx={ci:4d}  anime_id={anime_id}  {title}  "
            f"score={score: .4f}  prob={prob: .4f}  pred_rating={rating: .4f}"
        )

    item_probs = jax.nn.softmax(item_logits_1d)
    print("\nAlready rated anime predictions:")
    rated_idxs = jnp.where(already_rated_mask_1d > 0)[0]
    rated_probs = item_probs[rated_idxs]
    rated_preds = preds_1d[rated_idxs]
    rated_idxs_np = np.array(rated_idxs)
    rated_probs_np = np.array(rated_probs)
    rated_preds_np = np.array(rated_preds)
    for ci, prob, pred in zip(rated_idxs_np, rated_probs_np, rated_preds_np):
        anime_id = corpus_ids[ci]
        title = anime_titles.get(anime_id, "Unknown")
        print(
            f"  corpus_idx={ci:4d}  anime_id={anime_id}  {title}  "
            f"prob={prob: .4f}  predicted rating={pred: .4f}"
        )


def select_eval_profiles(
    all_users,
    seed=EVAL_PROFILE_SEED,
    count=EVAL_PROFILE_COUNT,
    min_ratings=EVAL_PROFILE_MIN_RATINGS,
    max_ratings=EVAL_PROFILE_MAX_RATINGS,
):
    rng = np.random.default_rng(seed)

    eligible_indices = []
    for i, (idxs, _vals, _rated) in enumerate(all_users):
        if min_ratings <= len(idxs) <= max_ratings:
            eligible_indices.append(i)

    if len(eligible_indices) < count:
        print(
            f"Warning: Only {len(eligible_indices)} profiles have between {min_ratings} and {max_ratings} ratings, using all of them"
        )
        selected_indices = eligible_indices
    else:
        selected_indices = rng.choice(
            eligible_indices, size=count, replace=False
        ).tolist()

    return selected_indices


def print_debug_profile_holdout_validation(
    params, corpus_ids, anime_titles, corpus_size
):
    """
    Comprehensive holdout validation for the debug profile.
    """
    print("\n" + "=" * 80)
    print("DEBUG PROFILE HOLDOUT VALIDATION")
    print("=" * 80)

    metrics = compute_holdout_metrics(
        params, DEBUG_PROFILE_IDXS, DEBUG_PROFILE_VALS, corpus_size, device="gpu"
    )

    rating_errors = metrics["rating_errors"]
    presence_probs = metrics["presence_probs"]
    pred_ratings = metrics["pred_ratings"]
    held_out_indices = metrics["held_out_indices"]
    held_out_ratings = metrics["held_out_ratings"]

    # Overall stats
    print(f"\nProfile size: {len(DEBUG_PROFILE_IDXS)} items")
    print("\n--- Rating Prediction Stats ---")
    print(f"  Mean absolute error:  {np.mean(rating_errors):.4f}")
    print(f"  Std dev:              {np.std(rating_errors):.4f}")
    print(f"  Min error:            {np.min(rating_errors):.4f}")
    print(f"  Max error:            {np.max(rating_errors):.4f}")

    print("\n--- Presence Probability Stats ---")
    print(f"  Mean probability:     {np.mean(presence_probs) * 100:.4f}%")
    print(f"  Std dev:              {np.std(presence_probs) * 100:.4f}%")
    print(f"  Min probability:      {np.min(presence_probs) * 100:.4f}%")
    print(f"  Max probability:      {np.max(presence_probs) * 100:.4f}%")

    # Best predicted items (lowest rating error)
    sorted_by_rating_error = np.argsort(rating_errors)

    print("\n--- 10 Best Predicted Items (by rating error) ---")
    print(
        f"  {'Rank':<5} {'Title':<50} {'True Rating':>12} {'Pred Rating':>12} {'Error':>8} {'Prob %':>10}"
    )
    print("  " + "-" * 97)
    for rank, i in enumerate(sorted_by_rating_error[:20], 1):
        corpus_idx = held_out_indices[i]
        anime_id = corpus_ids[corpus_idx]
        title = anime_titles.get(anime_id, "Unknown")[:48]
        print(
            f"  {rank:<5} {title:<50} {held_out_ratings[i]:>12.4f} {pred_ratings[i]:>12.4f} {rating_errors[i]:>8.4f} {presence_probs[i]*100:>9.4f}%"
        )

    print("\n--- 10 Worst Predicted Items (by rating error) ---")
    print(
        f"  {'Rank':<5} {'Title':<50} {'True Rating':>12} {'Pred Rating':>12} {'Error':>8} {'Prob %':>10}"
    )
    print("  " + "-" * 97)
    for rank, i in enumerate(sorted_by_rating_error[-20:][::-1], 1):
        corpus_idx = held_out_indices[i]
        anime_id = corpus_ids[corpus_idx]
        title = anime_titles.get(anime_id, "Unknown")[:48]
        print(
            f"  {rank:<5} {title:<50} {held_out_ratings[i]:>12.4f} {pred_ratings[i]:>12.4f} {rating_errors[i]:>8.4f} {presence_probs[i]*100:>9.4f}%"
        )

    # Best/worst by presence probability
    sorted_by_prob = np.argsort(presence_probs)

    print("\n--- 10 Highest Presence Probability Items ---")
    print(
        f"  {'Rank':<5} {'Title':<50} {'True Rating':>12} {'Pred Rating':>12} {'Error':>8} {'Prob %':>10}"
    )
    print("  " + "-" * 97)
    for rank, i in enumerate(sorted_by_prob[-20:][::-1], 1):
        corpus_idx = held_out_indices[i]
        anime_id = corpus_ids[corpus_idx]
        title = anime_titles.get(anime_id, "Unknown")[:48]
        print(
            f"  {rank:<5} {title:<50} {held_out_ratings[i]:>12.4f} {pred_ratings[i]:>12.4f} {rating_errors[i]:>8.4f} {presence_probs[i]*100:>9.4f}%"
        )

    print("\n--- 10 Lowest Presence Probability Items ---")
    print(
        f"  {'Rank':<5} {'Title':<50} {'True Rating':>12} {'Pred Rating':>12} {'Error':>8} {'Prob %':>10}"
    )
    print("  " + "-" * 97)
    for rank, i in enumerate(sorted_by_prob[:20], 1):
        corpus_idx = held_out_indices[i]
        anime_id = corpus_ids[corpus_idx]
        title = anime_titles.get(anime_id, "Unknown")[:48]
        print(
            f"  {rank:<5} {title:<50} {held_out_ratings[i]:>12.4f} {pred_ratings[i]:>12.4f} {rating_errors[i]:>8.4f} {presence_probs[i]*100:>9.4f}%"
        )


def print_eval_set_holdout_validation(
    params, all_users, eval_profile_indices, corpus_ids, anime_titles, corpus_size
):
    print("\n" + "=" * 80)
    print(f"EVALUATION SET HOLDOUT VALIDATION ({len(eval_profile_indices)} profiles)")
    print("=" * 80)

    all_rating_errors = []
    all_presence_probs = []
    profile_mean_errors = []
    profile_mean_probs = []

    for profile_idx in eval_profile_indices:
        idxs, vals, _rated = all_users[profile_idx]
        metrics = compute_holdout_metrics(params, idxs, vals, corpus_size, device="gpu")

        all_rating_errors.extend(metrics["rating_errors"])
        all_presence_probs.extend(metrics["presence_probs"])
        profile_mean_errors.append(np.mean(metrics["rating_errors"]))
        profile_mean_probs.append(np.mean(metrics["presence_probs"]))

    all_rating_errors = np.array(all_rating_errors)
    all_presence_probs = np.array(all_presence_probs)
    profile_mean_errors = np.array(profile_mean_errors)
    profile_mean_probs = np.array(profile_mean_probs)

    total_items = len(all_rating_errors)

    print(f"\nTotal held-out items evaluated: {total_items}")

    print("\n--- Per-Item Rating Prediction Stats ---")
    print(f"  Mean absolute error:  {np.mean(all_rating_errors):.4f}")
    print(f"  Std dev:              {np.std(all_rating_errors):.4f}")
    print(f"  Min error:            {np.min(all_rating_errors):.4f}")
    print(f"  Max error:            {np.max(all_rating_errors):.4f}")
    print(f"  Median error:         {np.median(all_rating_errors):.4f}")

    print("\n--- Per-Item Presence Probability Stats ---")
    print(f"  Mean probability:     {np.mean(all_presence_probs) * 100:.4f}%")
    print(f"  Std dev:              {np.std(all_presence_probs) * 100:.4f}%")
    print(f"  Min probability:      {np.min(all_presence_probs) * 100:.4f}%")
    print(f"  Max probability:      {np.max(all_presence_probs) * 100:.4f}%")
    print(f"  Median probability:   {np.median(all_presence_probs) * 100:.4f}%")

    print("\n--- Per-Profile Aggregated Stats ---")
    print(f"  Mean of profile mean errors:  {np.mean(profile_mean_errors):.4f}")
    print(f"  Std dev of profile mean errors: {np.std(profile_mean_errors):.4f}")
    print(f"  Mean of profile mean probs:   {np.mean(profile_mean_probs) * 100:.4f}%")
    print(f"  Std dev of profile mean probs: {np.std(profile_mean_probs) * 100:.4f}%")

    print("=" * 80)


def load_all_users(file_path="../data/user_input_vectors.npz"):
    print(f"Loading user data from {file_path}...")
    with np.load(file_path) as data:
        indices = data["indices"]
        values = data["values"]
        lengths = data["lengths"]

        if "rated_masks" in data:
            packed_masks = data["rated_masks"]
            total_bits = int(data["total_mask_bits"][0])
            unpacked_masks = np.unpackbits(packed_masks)[:total_bits].astype(bool)
            has_rated_masks = True
            print(f"Loaded {total_bits} rating flags")
        else:
            unpacked_masks = None
            has_rated_masks = False
            print("Warning: rated_masks not found in file")

    all_users = []
    mask_start = 0
    idx_start = 0
    for l in lengths:
        idx_end = idx_start + l
        user_indices = indices[idx_start:idx_end]
        user_values = values[idx_start:idx_end]

        if has_rated_masks:
            mask_end = mask_start + l
            user_rated = unpacked_masks[mask_start:mask_end]
            mask_start = mask_end
        else:
            user_rated = np.ones(l, dtype=bool)

        all_users.append((user_indices, user_values, user_rated))
        idx_start = idx_end

    print(f"Loaded {len(all_users)} user profiles.")
    return all_users


def main(steps=50_000):
    rng = random.PRNGKey(0)
    state = create_train_state(rng, CONF["learning_rate"])

    model = Recommender()
    dummy_input = jnp.ones((1, CONF["corpus_size"] * 2))
    print(model.tabulate(rng, dummy_input))

    # Load corpus index -> anime_id mapping
    with open("../data/corpus_ids.json", "r") as f:
        corpus_ids = json.load(f)
    if len(corpus_ids) != CONF["corpus_size"]:
        raise ValueError(
            f"corpus_id.json has {len(corpus_ids)} items, expected {CONF['corpus_size']}"
        )

    print("Loading anime metadata...")
    anime_titles = {}
    with open("../data/processed-metadata.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            anime_id = int(row["id"])
            title = row["title_english"] if row["title_english"] else row["title"]
            anime_titles[anime_id] = title
    print(f"Loaded {len(anime_titles)} anime titles")

    all_users = load_all_users()

    print("Starting training...")
    loader = data_generator(all_users, batch_size=CONF["batch_size"])

    eval_profile_indices = select_eval_profiles(all_users)
    print(
        f"Selected {len(eval_profile_indices)} evaluation profiles with >= {EVAL_PROFILE_MIN_RATINGS} ratings each"
    )

    debug_profile_input_vector = make_dense_profile(
        DEBUG_PROFILE_IDXS, DEBUG_PROFILE_VALS
    )
    debug_profile_presence_mask = debug_profile_input_vector[0, : CONF["corpus_size"]]

    for step in range(steps):
        batch, rated_mask = next(loader)
        batch_jax = jnp.array(batch)
        rated_mask_jax = jnp.array(rated_mask)

        state, loss, presence_loss, rating_loss = train_step(
            state, batch_jax, rated_mask_jax
        )

        if step % 100 == 0:
            lr_scale = optax.tree.get(state.opt_state, "scale")
            print(
                f"Step {step}: Loss {loss:.4f} "
                f"(Presence: {presence_loss:.4f}, Rating: {rating_loss:.4f}), "
                f"LR scale: {lr_scale}), "
                f"log_var_presence: {state.params['log_var_presence'][0]:.4f}; log_var_rating: {state.params['log_var_rating'][0]:.4f}"
            )

            if lr_scale < 1e-6:
                print("Learning rate scale has decayed below 1e-6, stopping training.")
                break

        if step % 1000 == 0 and step > 0:
            val_batch, val_rated_mask = next(loader)
            val_batch_jax = jnp.array(val_batch)
            val_rated_mask_jax = jnp.array(val_rated_mask)
            val_presence, val_rating = validation_step(
                state, val_batch_jax, val_rated_mask_jax
            )
            print(
                f"  [Validation @ step {step}]: "
                f"(Presence: {val_presence:.4f}, Rating: {val_rating:.4f})"
            )

        if step % 5000 == 0 and step > 0:
            item_logits, rating_pred = infer_outputs(
                state.params, debug_profile_input_vector
            )
            preds = rating_pred[0]
            logits = item_logits[0]
            print(f"\n[Debug @ step {step}]")
            print_top_recs(
                corpus_ids,
                logits,
                preds,
                debug_profile_presence_mask,
                anime_titles,
                k=50,
            )
            print("")

        if step % 10000 == 0 and step > 0:
            print(f"\n{'#' * 80}")
            print(f"ENHANCED HOLDOUT VALIDATION @ STEP {step}")
            print(f"{'#' * 80}")

            print_debug_profile_holdout_validation(
                state.params, corpus_ids, anime_titles, CONF["corpus_size"]
            )

            print_eval_set_holdout_validation(
                state.params,
                all_users,
                eval_profile_indices,
                corpus_ids,
                anime_titles,
                CONF["corpus_size"],
            )

            print(f"\n{'#' * 80}\n")

    bytes_output = serialization.to_bytes(state.params)
    with open("../data/jax_model.msgpack", "wb") as f:
        f.write(bytes_output)
    print("Model parameters saved to ../data/jax_model.msgpack")


if __name__ == "__main__":
    main()
