import os
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import flax.linen as nn

CONF = {
    "corpus_size": 6000,
    "hidden_dim": 2048,
    "bottleneck_dim": 512,
    "batch_size": 512,
    "learning_rate": 0.0003,
    "dropout_rate": 0.4,  # Probability to drop an input item during training
    "dropout_variation": 0.4,  # dropout rate = base +/- base * variation
    "huber_delta": 1,  # for z-scores; 0.5–2.0 is typical
    "rec_logit_weight": 0.3,  # Weight for item logits in recommendation score (0.0 = only ratings, 1.0 = only logits)
    "latent_noise_scale": 0.1,  # scale of noise added to latent vector during training
}


def setup_jax_cpu(num_devices: int | None = None):
    """
    Configure JAX for CPU parallelism.
    Must be called before any other JAX operations (like loading the model).
    """
    # Ensure we're using CPU backend
    jax.config.update("jax_platform_name", "cpu")

    if num_devices is None:
        # Default to all available CPUs
        # os.cpu_count() returns None if undetermined
        num_devices = os.cpu_count() or 1

    # Ensure at least 1 device
    num_devices = max(1, num_devices)

    jax.config.update("jax_num_cpu_devices", num_devices)
    print(f"[model.py] Configured JAX with {num_devices} CPU devices")


def get_sharding_mesh() -> Mesh:
    """Get a mesh for data parallelism across all available devices."""
    devices = jax.devices()
    # Create a mesh where 'batch' dimension maps to all devices
    return Mesh(np.array(devices), axis_names=("batch",))


def get_num_cpu_devices() -> int:
    """Return the number of CPU devices configured."""
    return jax.device_count()


class Recommender(nn.Module):
    """
    Autoencoder with two decoder heads:
      - item_logits: used for multinomial / implicit feedback (presence)
      - rating_pred: used for explicit feedback regression (z-score ratings)
    """

    hidden_dim: int = CONF["hidden_dim"]
    bottleneck_dim: int = CONF["bottleneck_dim"]
    output_dim: int = CONF["corpus_size"]

    @nn.compact
    def __call__(self, x, training: bool = False):
        # --- Encoder ---

        h = nn.Dense(self.hidden_dim)(x)
        h = nn.swish(h)

        # this doesn't seem to provide much benefit and makes training harder
        # h = nn.Dense(self.hidden_dim // 2)(h)
        # h = nn.swish(h)

        bottleneck = nn.Dense(self.bottleneck_dim, name="bottleneck")(h)

        if training:
            rng = self.make_rng("noise")
            noise = (
                random.normal(rng, shape=bottleneck.shape) * CONF["latent_noise_scale"]
            )
            z = bottleneck + noise
        else:
            z = bottleneck

        # decoder head 1: predicts entity presence
        d1 = nn.Dense(self.hidden_dim // 2, name="dec_item_up1")(z)
        d1 = nn.swish(d1)
        d1 = nn.Dense(self.hidden_dim, name="dec_item_up2")(d1)
        d1 = nn.swish(d1)
        item_logits = nn.Dense(self.output_dim, name="item_logits")(d1)

        # decoder head 2: predicts per-entity ratings (regression)
        d2 = nn.Dense(self.hidden_dim // 2, name="dec_rating_up1")(z)
        d2 = nn.swish(d2)
        d2 = nn.Dense(self.hidden_dim, name="dec_rating_up2")(d2)
        d2 = nn.swish(d2)
        rating_pred = nn.Dense(self.output_dim, name="rating_pred")(d2)

        # learnable log variances for uncertainty weighting
        log_var_presence = self.param("log_var_presence", nn.initializers.zeros, (1,))
        log_var_rating = self.param("log_var_rating", nn.initializers.zeros, (1,))

        return item_logits, rating_pred, log_var_presence, log_var_rating


def make_dense_profile(idxs: np.ndarray, vals: np.ndarray) -> jnp.ndarray:
    """
    Convert sparse profile representation to dense input vector.

    Args:
        idxs: Array of corpus indices for items the user has rated
        vals: Array of normalized rating values corresponding to idxs

    Returns:
        Dense input vector of shape (1, corpus_size * 2) where:
        - First half is presence indicators (0 or 1)
        - Second half is normalized ratings
    """
    x = np.zeros((CONF["corpus_size"] * 2,), dtype=np.float32)
    x[idxs] = 1.0
    x[CONF["corpus_size"] + idxs] = vals
    return jnp.asarray(x[None, :])


@jax.jit
def infer_outputs(params, x):
    """
    Run inference on a single input (or batch).

    Args:
        params: Model parameters
        x: Input tensor of shape (batch_size, corpus_size * 2)

    Returns:
        tuple of (item_logits, rating_pred) each of shape (batch_size, corpus_size)
    """
    item_logits, rating_pred, _, _ = Recommender().apply(
        {"params": params}, x, training=False
    )
    return item_logits, rating_pred


# Alias for clarity when used explicitly for batch inference
batch_infer_outputs = infer_outputs


def sharded_batch_infer_outputs(params, x):
    """
    Run inference on a sharded batch across multiple devices.

    This function expects 'x' to be already sharded using jax.device_put
    and 'params' to be already replicated.

    Because this just calls the JIT-compiled function, JAX's GSPMD
    (General Sharded Parallel Motion Data) system handles the parallel execution.
    """
    return batch_infer_outputs(params, x)


@jax.jit
def infer_bottleneck(params, x):
    h = jnp.dot(x, params["Dense_0"]["kernel"]) + params["Dense_0"]["bias"]
    h = jax.nn.swish(h)
    bottleneck = (
        jnp.dot(h, params["bottleneck"]["kernel"]) + params["bottleneck"]["bias"]
    )
    return bottleneck


def compute_recommendation_ranking_score(
    presence_logits_1d: jnp.ndarray,
    predicted_ratings_1d: jnp.ndarray,
    logit_weight: float | None = None,  # defaults to `CONF["rec_logit_weight"]`
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute combined recommendation score using weighted combination of presence probability and rating.

    Args:
        item_logits_1d: Raw logits from the model (1D array of shape corpus_size)
        preds_1d: Predicted ratings (1D array of shape corpus_size)
        logit_weight: Weight for probability in the combined score.
                      0.0 = only ratings, 1.0 = only probability.
                      If None, uses CONF["rec_logit_weight"].

    Returns:
        tuple of (combined_score, item_probs) - both 1D arrays of shape corpus_size
    """
    if logit_weight is None:
        logit_weight = CONF["rec_logit_weight"]

    item_probs = jax.nn.softmax(presence_logits_1d)
    prob_power = jnp.power(item_probs, logit_weight)
    # ratings can be negative and negative doesn't necessarily mean bad recommendation
    rating_power = jnp.power(
        jnp.maximum(predicted_ratings_1d + 1, 0.001), 1.0 - logit_weight
    )
    combined_score = prob_power * rating_power

    return combined_score, item_probs

def compute_recommendation_ranking_score_alt(
    presence_logits_1d: jnp.ndarray,
    predicted_ratings_1d: jnp.ndarray,
    logit_weight: float | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute ranking score by mixing normalized logits and ratings.
    This provides a linear response to the weighting parameter.
    """
    if logit_weight is None:
        logit_weight = CONF["rec_logit_weight"]

    # 1. Normalize Presence Logits (Z-Score)
    # We use raw logits, not softmax. This removes the "sum to 1" penalty
    # for niche items while preserving the relative ordering of likelihood.
    p_mean = jnp.mean(presence_logits_1d)
    p_std = jnp.std(presence_logits_1d) + 1e-6
    norm_logits = (presence_logits_1d - p_mean) / p_std

    # 2. Normalize Predicted Ratings (Z-Score)
    # Assuming predicted_ratings_1d are roughly -2 to 2.
    # We normalize them to match the scale of the logits.
    r_mean = jnp.mean(predicted_ratings_1d)
    r_std = jnp.std(predicted_ratings_1d) + 1e-6
    norm_ratings = (predicted_ratings_1d - r_mean) / r_std

    # 3. Linear Mixing
    # Since we are adding standardized scores, the magnitude of the
    # combined score doesn't have a direct physical meaning (like probability),
    # but the RANKING order is statistically robust.
    combined_score = (logit_weight * norm_logits) + ((1.0 - logit_weight) * norm_ratings)

    # We still return probabilities for display/debug purposes if needed
    item_probs = jax.nn.softmax(presence_logits_1d)

    return combined_score, item_probs


def rank_by_weighted_score(
    item_logits_1d: jnp.ndarray,
    preds_1d: jnp.ndarray,
    already_rated_mask_1d: jnp.ndarray,
    k: int = 50,
    logit_weight: float | None = None,
    use_alt_ranking: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Rank items using weighted combination of presence probability and rating.

    Args:
        item_logits_1d: Raw logits from the model (1D array of shape corpus_size)
        preds_1d: Predicted ratings (1D array of shape corpus_size)
        already_rated_mask_1d: Binary mask indicating already-rated items (1 = rated)
        k: Number of top items to return
        logit_weight: Weight for probability in the combined score

    Returns:
        tuple of (topk_indices, topk_scores, topk_probs, topk_ratings)
    """
    ranking_fn = (
        compute_recommendation_ranking_score_alt
        if use_alt_ranking
        else compute_recommendation_ranking_score
    )
    combined_score, item_probs = ranking_fn(
        item_logits_1d, preds_1d, logit_weight
    )

    masked = jnp.where(already_rated_mask_1d > 0, -jnp.inf, combined_score)
    topk_idx = jnp.argsort(masked)[-k:][::-1]

    return (
        np.array(topk_idx),
        np.array(combined_score[topk_idx]),
        np.array(item_probs[topk_idx]),
        np.array(preds_1d[topk_idx]),
    )


def create_holdout_batch(
    idxs: np.ndarray,
    vals: np.ndarray,
    corpus_size: int,
    pad_to_size: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """
    Create a batch where each row holds out one item from the profile.

    Args:
        idxs: Array of corpus indices for items the user has rated
        vals: Array of normalized rating values
        corpus_size: Size of the corpus
        pad_to_size: If provided, pads the batch to this size.
                     Otherwise, size will be exactly len(idxs).

    Returns:
        - input_batch: (actual_size or pad_to_size, corpus_size * 2)
        - held_out_indices: (actual_size or pad_to_size,)
        - held_out_ratings: (actual_size or pad_to_size,)
        - actual_size: int - the number of items held out
    """
    n_items = len(idxs)
    actual_size = n_items
    batch_size = pad_to_size if pad_to_size is not None else actual_size

    if batch_size < actual_size:
        # If explicitly told to pad to a smaller size, we truncate
        actual_size = batch_size

    input_batch = np.zeros((batch_size, corpus_size * 2), dtype=np.float32)
    held_out_indices = np.zeros(batch_size, dtype=np.int32)
    held_out_ratings = np.zeros(batch_size, dtype=np.float32)

    for i in range(actual_size):
        mask = np.ones(n_items, dtype=bool)
        mask[i] = False

        remaining_idxs = idxs[mask]
        remaining_vals = vals[mask]

        input_batch[i, remaining_idxs] = 1.0
        input_batch[i, corpus_size + remaining_idxs] = remaining_vals

        held_out_indices[i] = idxs[i]
        held_out_ratings[i] = vals[i]

    return (
        jnp.array(input_batch),
        jnp.array(held_out_indices),
        jnp.array(held_out_ratings),
        actual_size,
    )


# Fixed batch size for GPU holdout prediction to avoid JIT recompilation.
# On GPU, recompiling for every unique batch size is extremely slow.
_GPU_HOLDOUT_BATCH_SIZE = 256


def _batch_holdout_predict_gpu(
    params,
    idxs: np.ndarray,
    vals: np.ndarray,
    corpus_size: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    GPU-optimized holdout prediction.

    Uses a fixed batch size to avoid JIT recompilation overhead.
    GPU recompilation is very expensive, so we process in fixed-size
    chunks to ensure the JIT-compiled function is reused.
    """
    n_items = len(idxs)
    batch_size = _GPU_HOLDOUT_BATCH_SIZE

    # Create the full holdout batch (unpadded)
    full_input_batch, _, _, _ = create_holdout_batch(
        idxs, vals, corpus_size, pad_to_size=None
    )

    # Process in fixed-size chunks
    all_item_logits = []
    all_rating_pred = []

    for start_idx in range(0, n_items, batch_size):
        end_idx = min(start_idx + batch_size, n_items)
        chunk_size = end_idx - start_idx

        # Pad chunk to fixed batch size if needed
        if chunk_size < batch_size:
            padded_chunk = jnp.zeros((batch_size, corpus_size * 2), dtype=jnp.float32)
            padded_chunk = padded_chunk.at[:chunk_size].set(
                full_input_batch[start_idx:end_idx]
            )
        else:
            padded_chunk = full_input_batch[start_idx:end_idx]

        item_logits, rating_pred = batch_infer_outputs(params, padded_chunk)

        # Only keep the actual results (not padding)
        all_item_logits.append(item_logits[:chunk_size])
        all_rating_pred.append(rating_pred[:chunk_size])

    return jnp.concatenate(all_item_logits, axis=0), jnp.concatenate(
        all_rating_pred, axis=0
    )


def _batch_holdout_predict_cpu(
    params,
    idxs: np.ndarray,
    vals: np.ndarray,
    corpus_size: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    CPU-optimized holdout prediction.

    Parallelizes across available CPU devices. Pads to multiples of device
    count for efficient sharding. CPU JIT compilation is faster, so we
    can be more flexible with batch sizes.
    """
    num_devices = get_num_cpu_devices()
    n_items = len(idxs)

    # Pad to multiple of num_devices for efficient sharding
    padded_batch_size = ((n_items + num_devices - 1) // num_devices) * num_devices

    input_batch, _, _, _ = create_holdout_batch(
        idxs, vals, corpus_size, pad_to_size=padded_batch_size
    )

    if num_devices > 1:
        mesh = get_sharding_mesh()
        input_sharding = NamedSharding(mesh, P("batch", None))
        sharded_input = jax.device_put(input_batch, input_sharding)

        item_logits, rating_pred = sharded_batch_infer_outputs(params, sharded_input)
    else:
        item_logits, rating_pred = batch_infer_outputs(params, input_batch)

    # Return only the actual items (remove padding)
    return item_logits[:n_items], rating_pred[:n_items]


def batch_holdout_predict(
    params,
    idxs: np.ndarray,
    vals: np.ndarray,
    corpus_size: int,
    device: str = "cpu",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Run inference for every possible single-item holdout in the profile.

    Args:
        params: Model parameters
        idxs: Array of corpus indices for items the user has rated
        vals: Array of normalized rating values
        corpus_size: Size of the corpus
        device: Device to optimize for - "cpu" or "gpu"

    Returns:
        tuple of (item_logits, rating_pred) for each holdout
    """
    if device == "gpu":
        return _batch_holdout_predict_gpu(params, idxs, vals, corpus_size)
    else:
        return _batch_holdout_predict_cpu(params, idxs, vals, corpus_size)


def compute_holdout_metrics(
    params,
    idxs: np.ndarray,
    vals: np.ndarray,
    corpus_size: int,
    logit_weight: float | None = None,
    baseline_top50_indices: np.ndarray | None = None,
    baseline_top50_scores: np.ndarray | None = None,
    device: str = "cpu",
    use_alt_ranking: bool = False,
) -> dict:
    """
    Compute holdout metrics for a single user profile.

    For each item in the profile, holds it out and predicts its rating/presence
    using the remaining items. Uses data parallelism across CPU devices when available.

    Args:
        params: Model parameters (can be already replicated)
        idxs: Array of corpus indices for items the user has rated
        vals: Array of normalized rating values
        corpus_size: Size of the corpus
        logit_weight: Weight for logits in recommendation score (None uses default)
        baseline_top50_indices: Corpus indices of top 50 recommendations from full profile
                                (required for impact score computation)
        baseline_top50_scores: Scores of top 50 recommendations from full profile
                               (required for impact score computation)
        device: Device to optimize for - "cpu" or "gpu"

    Returns:
        dict with keys:
            - rating_errors: absolute errors between predicted and true ratings
            - pred_ratings: predicted ratings for held-out items
            - presence_probs: predicted presence probabilities for held-out items
            - held_out_indices: corpus indices of held-out items
            - held_out_ratings: true normalized ratings of held-out items
            - recommendation_scores: weighted recommendation scores for held-out items
            - impact_scores: sum of absolute score changes for top-50 items when each item is held out
                             (only if baseline_top50_indices/scores provided)
    """
    n_items = len(idxs)

    # Use batched inference
    item_logits, rating_pred = batch_holdout_predict(
        params, idxs, vals, corpus_size, device=device
    )

    item_logits_np = np.array(item_logits)
    rating_pred_np = np.array(rating_pred)

    held_out_indices_np = idxs
    held_out_ratings_np = vals

    pred_ratings = np.array(
        [rating_pred_np[i, held_out_indices_np[i]] for i in range(n_items)]
    )

    probs = jax.nn.softmax(item_logits, axis=1)
    probs_np = np.array(probs)
    presence_probs = np.array(
        [probs_np[i, held_out_indices_np[i]] for i in range(n_items)]
    )

    rating_errors = np.abs(pred_ratings - held_out_ratings_np)

    ranking_fn = (
        compute_recommendation_ranking_score_alt
        if use_alt_ranking
        else compute_recommendation_ranking_score
    )

    # Compute weighted recommendation scores for each held-out item
    recommendation_scores = np.zeros(n_items, dtype=np.float32)
    for i in range(n_items):
        score, _ = ranking_fn(
            jnp.array(item_logits_np[i]),
            jnp.array(rating_pred_np[i]),
            logit_weight,
        )
        recommendation_scores[i] = float(score[held_out_indices_np[i]])

    result = {
        "rating_errors": rating_errors,
        "pred_ratings": pred_ratings,
        "presence_probs": presence_probs,
        "held_out_indices": held_out_indices_np,
        "held_out_ratings": held_out_ratings_np,
        "recommendation_scores": recommendation_scores,
    }

    # Compute impact scores if baseline top-50 data is provided
    if baseline_top50_indices is not None and baseline_top50_scores is not None:
        impact_scores = np.zeros(n_items, dtype=np.float32)
        # We need to compute scores for all recommendations for each holdout
        # This can still be slow if done in a loop, but the model inference part is done.

        # Optimize impact score calculation using JAX if possible,
        # but let's keep it simple for now as it's only top-50.
        for i in range(n_items):
            # Compute scores for the baseline top-50 items using this holdout's predictions
            holdout_scores, _ = ranking_fn(
                jnp.array(item_logits_np[i]),
                jnp.array(rating_pred_np[i]),
                logit_weight,
            )
            # Get scores for the specific top-50 indices
            holdout_top50_scores = np.array(holdout_scores[baseline_top50_indices])
            # Sum of absolute differences
            impact_scores[i] = np.sum(
                np.abs(baseline_top50_scores - holdout_top50_scores)
            )
        result["impact_scores"] = impact_scores

    return result
