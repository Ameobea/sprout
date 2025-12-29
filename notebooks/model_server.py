"""
HTTP API server for the anime recommendation model.

Endpoints:
    POST /recommend - Get recommendations for a user profile
    GET /health - Health check endpoint

Run with: python model_server.py
    or:   uvicorn model_server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import os
import json
import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from model import (
    CONF,
    Recommender,
    make_dense_profile,
    infer_outputs,
    compute_recommendation_ranking_score,
    compute_recommendation_ranking_score_alt,
    rank_by_weighted_score,
    compute_holdout_metrics,
    setup_jax_cpu,
    get_sharding_mesh,
    get_num_cpu_devices,
    batch_holdout_predict,
)

# Configure JAX for CPU parallelism
setup_jax_cpu()

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from flax import serialization
from normalize_ratings import normalize_ratings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class LRUCache:
    """Simple LRU cache with fixed maximum size."""

    def __init__(self, max_size: int = 500):
        self.cache = OrderedDict()
        self.max_size = max_size

    def _make_key(self, request: "RecommendRequest") -> str:
        """
        Create a cache key from a request.

        Uses Pydantic's model_dump with mode='json' for consistent serialization.
        This ensures that new fields added to the model in the future will
        automatically be included in the cache key.
        """
        # Get JSON-serializable dict, then convert to sorted tuple for hashing
        req_dict = request.model_dump(mode="json")
        # Convert to a stable string representation
        return json.dumps(req_dict, sort_keys=True)

    def get(self, request: "RecommendRequest") -> dict | None:
        """Get cached response for a request, or None if not found."""
        key = self._make_key(request)
        if key in self.cache:
            # Move to end (mark as recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, request: "RecommendRequest", response: dict) -> None:
        """Store a response in the cache."""
        key = self._make_key(request)

        # If key exists, remove it first (will be re-added at the end)
        if key in self.cache:
            del self.cache[key]
        # If cache is full, remove oldest item
        elif len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)

        self.cache[key] = response

    def clear(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()

    def __len__(self) -> int:
        return len(self.cache)


class UserAnimeEntry(TypedDict):
    anime_id: int
    rating: float  # 0-10 scale, 0 means unrated, -2 means dropped
    watch_status: str  # "completed", "watching", "dropped", "plan_to_watch", etc.


@dataclass
class RecommenderModel:
    """Holds the loaded model and corpus mapping."""

    params: dict
    corpus_ids: list[int]
    anime_id_to_corpus_idx: dict[int, int]
    corpus_size: int
    popularity_distribution: np.ndarray | None = (
        None  # Normalized popularity for each corpus item
    )


# Global model instance - loaded once at startup
_model: RecommenderModel | None = None
# Single-thread executor to serialize model inference (JAX uses all CPU threads internally)
_inference_executor = ThreadPoolExecutor(max_workers=1)
# Global cache for recommendation responses
_cache = LRUCache(max_size=500)


def load_model(
    model_path: str = "/opt/model/jax_model.msgpack",
    corpus_ids_path: str = "/opt/model/corpus_ids.json",
    metadata_path: str | None = None,
) -> RecommenderModel:
    """Load the model and corpus mapping from disk."""
    logger.info(f"Loading corpus mapping from {corpus_ids_path}")
    with open(corpus_ids_path, "r") as f:
        corpus_ids = json.load(f)

    corpus_size = CONF["corpus_size"]
    if len(corpus_ids) != corpus_size:
        raise ValueError(
            f"corpus_ids.json has {len(corpus_ids)} items, expected {corpus_size}"
        )

    anime_id_to_corpus_idx = {anime_id: idx for idx, anime_id in enumerate(corpus_ids)}

    # Load metadata for popularity distribution
    popularity_distribution = None
    if metadata_path is None:
        metadata_path = os.environ.get("METADATA_PATH")

    if metadata_path and os.path.exists(metadata_path):
        logger.info(f"Loading metadata from {metadata_path}")
        import csv

        # Map anime_id to rating_count
        anime_id_to_rating_count = {}
        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                anime_id = int(row["id"])
                rating_count = int(row["rating_count"])
                anime_id_to_rating_count[anime_id] = rating_count

        # Build popularity array for corpus items
        rating_counts = np.zeros(corpus_size, dtype=np.float32)
        for idx, anime_id in enumerate(corpus_ids):
            if anime_id in anime_id_to_rating_count:
                rating_counts[idx] = anime_id_to_rating_count[anime_id]
            else:
                logger.warning(f"No metadata found for anime_id {anime_id} in corpus")

        # Normalize to probability distribution (sum to 1.0)
        total_rating_count = rating_counts.sum()
        if total_rating_count > 0:
            popularity_distribution = rating_counts / total_rating_count
            logger.info(
                f"Loaded popularity distribution. Min: {popularity_distribution.min():.6f}, Max: {popularity_distribution.max():.6f}, Mean: {popularity_distribution.mean():.6f}"
            )
        else:
            logger.warning(
                "No valid rating counts found, niche_boost_factor will be disabled"
            )
    else:
        logger.warning(
            f"Metadata file not found at {metadata_path}, niche_boost_factor will be disabled"
        )

    logger.info(f"Loading model from {model_path}")
    model = Recommender()
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, corpus_size * 2))
    params = model.init({"params": rng, "noise": rng}, dummy_input)["params"]

    with open(model_path, "rb") as f:
        saved_bytes = f.read()

    params = serialization.from_bytes(params, saved_bytes)

    # Replicate params across devices if we have multiple CPUs
    num_devices = get_num_cpu_devices()
    if num_devices > 1:
        logger.info(f"Replicating model parameters across {num_devices} devices...")
        mesh = get_sharding_mesh()
        # Replicate params (empty tuple for PartitionSpec means replicated on all axes)
        replicated_sharding = NamedSharding(mesh, P())
        params = jax.tree.map(lambda x: jax.device_put(x, replicated_sharding), params)
        logger.info("Model parameters replicated.")

    logger.info("Model loaded successfully")

    return RecommenderModel(
        params=params,
        corpus_ids=corpus_ids,
        anime_id_to_corpus_idx=anime_id_to_corpus_idx,
        corpus_size=corpus_size,
        popularity_distribution=popularity_distribution,
    )


def get_model() -> RecommenderModel:
    """Get the loaded model, initializing if necessary."""
    global _model
    if _model is None:
        model_path = os.environ.get("MODEL_PATH", "/opt/model/jax_model.msgpack")
        corpus_path = os.environ.get("CORPUS_PATH", "/opt/model/corpus_ids.json")
        _model = load_model(model_path, corpus_path)
    return _model


def preprocess_profile(
    model: RecommenderModel, user_profile: list[UserAnimeEntry]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], dict]:
    """
    Preprocess user profile into model input format.

    Returns:
        - corpus_indices: indices into the corpus for valid entries
        - normalized_ratings: normalized rating values
        - original_ratings: original 0-10 ratings
        - valid_entries: list of valid entry dicts with corpus_idx added
        - norm_stats: normalization statistics
    """
    valid_entries = []
    for entry in user_profile:
        anime_id = entry["anime_id"]
        if anime_id not in model.anime_id_to_corpus_idx:
            continue

        rating = entry.get("rating", 0) or 0
        watch_status = entry.get("watch_status", "")

        if rating > 0 or watch_status in ("completed", "watching", "dropped"):
            valid_entries.append(
                {
                    "corpus_idx": model.anime_id_to_corpus_idx[anime_id],
                    "anime_id": anime_id,
                    "rating": rating,
                    "watch_status": watch_status,
                }
            )

    if not valid_entries:
        raise ValueError("No valid entries in user profile")

    valid_entries.sort(key=lambda x: x["corpus_idx"])

    corpus_indices = np.array([e["corpus_idx"] for e in valid_entries], dtype=np.int32)
    original_ratings = np.array(
        [
            (
                -2
                if e.get("watch_status") == "dropped" and e.get("rating", 0) == 0
                else e["rating"]
            )
            for e in valid_entries
        ],
        dtype=np.float32,
    )

    normalized_ratings, norm_stats = normalize_ratings(original_ratings)

    return (
        corpus_indices,
        normalized_ratings,
        original_ratings,
        valid_entries,
        norm_stats,
    )


def get_recommendations_simple(
    model: RecommenderModel,
    corpus_indices: np.ndarray,
    normalized_ratings: np.ndarray,
    top_k: int,
    logit_weight: float | None,
    use_alt_ranking: bool = False,
) -> list[dict]:
    """Get recommendations without any holdout analysis."""
    input_vector = make_dense_profile(corpus_indices, normalized_ratings)
    item_logits, rating_pred = infer_outputs(model.params, input_vector)

    presence_mask = input_vector[0, : model.corpus_size]
    topk_idx, topk_scores, topk_probs, topk_ratings = rank_by_weighted_score(
        item_logits[0],
        rating_pred[0],
        presence_mask,
        k=top_k,
        logit_weight=logit_weight,
        use_alt_ranking=use_alt_ranking,
    )

    recommendations = []
    for i in range(len(topk_idx)):
        corpus_idx = int(topk_idx[i])
        recommendations.append(
            {
                "anime_id": model.corpus_ids[corpus_idx],
                "corpus_idx": corpus_idx,
                "score": float(topk_scores[i]),
                "probability": float(topk_probs[i]),
                "predicted_rating": float(topk_ratings[i]),
            }
        )

    return recommendations


def apply_niche_boost(
    recommendations: list[dict],
    model: RecommenderModel,
    niche_boost_factor: float,
) -> list[dict]:
    """
    Apply niche boost to recommendation scores based on model probability vs. popularity.

    The niche boost rewards items where the model's predicted probability is higher than
    expected given the item's global popularity. This helps surface niche items that are
    particularly well-suited for the user, even if they have low overall popularity.

    Args:
        recommendations: List of recommendation dicts with 'corpus_idx', 'score', and 'probability'
        model: RecommenderModel with popularity_distribution
        niche_boost_factor: Float in [0, 1] controlling boost strength (0 = no boost)

    Returns:
        Updated list of recommendations with adjusted scores, sorted by new scores
    """
    if niche_boost_factor <= 0 or model.popularity_distribution is None:
        # No boost or no popularity data
        return recommendations

    # Clamp niche_boost_factor to [0, 1]
    niche_boost_factor = max(0.0, min(1.0, niche_boost_factor))

    # Apply non-linear mapping to boost factor:
    # - Linear from 0 to ~0.5
    # - Exponential for values > 0.5, so high values (0.9-1.0) have dramatic effects
    # We use a piecewise function:
    #   f(x) = x                              for x <= 0.5
    #   f(x) = 0.5 + (x - 0.5) * exp(k*(x-0.5))   for x > 0.5
    # Where k controls the exponential growth rate
    if niche_boost_factor <= 0.5:
        effective_boost = niche_boost_factor
    else:
        # Exponential scaling for values above 0.5
        # k=3 gives good exponential growth: f(0.5)=0.5, f(1.0)≈1.32
        k = 4.62
        excess = niche_boost_factor - 0.5
        effective_boost = 0.5 + excess * np.exp(k * excess)

    # Apply boost
    for rec in recommendations:
        corpus_idx = rec["corpus_idx"]
        model_prob = rec["probability"]
        popularity = model.popularity_distribution[corpus_idx]

        # Compute "surprise factor": how much more the model likes this vs. its popularity
        # Add small epsilon to avoid division by zero
        epsilon = 1e-9
        surprise_ratio = model_prob / (popularity + epsilon)

        # Convert surprise ratio to a boost multiplier
        # Use log to compress the range (surprise_ratio can be very large for niche items)
        # Apply effective_boost to control the strength
        # Formula: boost = 1 + effective_boost * log(1 + surprise_ratio)
        # This gives boost >= 1.0, with higher values for higher surprise
        log_surprise = np.log1p(surprise_ratio)  # log(1 + x)
        boost_multiplier = 1.0 + effective_boost * log_surprise

        # Apply boost to the score
        original_score = rec["score"]
        boosted_score = original_score * boost_multiplier
        logger.info(
            f"Niche boost applied to anime_id {rec['anime_id']}: "
            f"original_score={original_score:.4f}, "
            f"model_prob={model_prob:.6f}, "
            f"popularity={popularity:.6f}, "
            f"surprise_ratio={surprise_ratio:.4f}, "
            f"boost_multiplier={boost_multiplier:.4f}, "
            f"boosted_score={boosted_score:.4f}"
        )
        rec["score"] = float(boosted_score)

    # Re-sort by new scores
    recommendations.sort(key=lambda x: x["score"], reverse=True)

    return recommendations


def compute_profile_holdout_analysis(
    model: RecommenderModel,
    corpus_indices: np.ndarray,
    normalized_ratings: np.ndarray,
    original_ratings: np.ndarray,
    logit_weight: float | None,
    baseline_top50_indices: np.ndarray,
    baseline_top50_scores: np.ndarray,
) -> dict:
    """Compute holdout analysis for items in the user's profile."""
    metrics = compute_holdout_metrics(
        model.params,
        corpus_indices,
        normalized_ratings,
        model.corpus_size,
        logit_weight=logit_weight,
        baseline_top50_indices=baseline_top50_indices,
        baseline_top50_scores=baseline_top50_scores,
    )

    rating_errors = metrics["rating_errors"]
    pred_ratings = metrics["pred_ratings"]
    presence_probs = metrics["presence_probs"]
    held_out_indices_np = metrics["held_out_indices"]
    held_out_ratings_np = metrics["held_out_ratings"]
    recommendation_scores = metrics["recommendation_scores"]
    impact_scores = metrics.get("impact_scores", np.zeros(len(rating_errors)))
    actual_size = len(held_out_indices_np)

    items = []
    for i in range(actual_size):
        corpus_idx = int(held_out_indices_np[i])
        items.append(
            {
                "anime_id": model.corpus_ids[corpus_idx],
                "corpus_idx": corpus_idx,
                "true_rating": float(original_ratings[i]),
                "true_normalized_rating": float(held_out_ratings_np[i]),
                "predicted_rating": float(pred_ratings[i]),
                "rating_error": float(rating_errors[i]),
                "presence_probability": float(presence_probs[i]),
                "recommendation_score": float(recommendation_scores[i]),
                "impact_score": float(impact_scores[i]),
            }
        )

    return {
        "items": items,
        "mean_rating_error": float(np.mean(rating_errors)),
        "std_rating_error": float(np.std(rating_errors)),
        "mean_presence_prob": float(np.mean(presence_probs)),
        "std_presence_prob": float(np.std(presence_probs)),
    }


def compute_recommendation_contributions(
    model: RecommenderModel,
    corpus_indices: np.ndarray,
    normalized_ratings: np.ndarray,
    recommendations: list[dict],
    top_n_contributors: int,
    logit_weight: float | None,
    use_alt_ranking: bool = False,
) -> list[dict]:
    """
    For each recommendation, compute which profile items contribute most to its score.

    This works by holding out each profile item one at a time and measuring how much
    the recommendation's score drops.
    """
    rec_corpus_indices = [r["corpus_idx"] for r in recommendations]
    n_profile_items = len(corpus_indices)
    n_recs = len(recommendations)

    if n_profile_items == 0 or n_recs == 0:
        return recommendations

    # Get baseline scores for all recommendations
    input_vector = make_dense_profile(corpus_indices, normalized_ratings)
    item_logits, rating_pred = infer_outputs(model.params, input_vector)
    ranking_fn = (
        compute_recommendation_ranking_score_alt
        if use_alt_ranking
        else compute_recommendation_ranking_score
    )
    baseline_scores, _ = ranking_fn(item_logits[0], rating_pred[0], logit_weight)
    baseline_scores_for_recs = np.array(
        [float(baseline_scores[idx]) for idx in rec_corpus_indices]
    )

    # Batched inference for all holdouts
    # ho_logits: (n_profile_items, corpus_size)
    # ho_ratings: (n_profile_items, corpus_size)
    ho_logits, ho_ratings = batch_holdout_predict(
        model.params, corpus_indices, normalized_ratings, model.corpus_size
    )

    # Convert to numpy for processing (or use JAX for vectorized score computation)
    ho_logits_np = np.array(ho_logits)
    ho_ratings_np = np.array(ho_ratings)

    # For each profile item, compute scores with that item held out
    score_drops = np.zeros((n_profile_items, n_recs), dtype=np.float32)

    for i in range(n_profile_items):
        # Compute scores for all recommendations using this holdout's predictions
        # Use JAX for the score computation part to keep it relatively fast
        holdout_scores, _ = ranking_fn(
            jnp.array(ho_logits_np[i]),
            jnp.array(ho_ratings_np[i]),
            logit_weight,
        )

        for j, rec_idx in enumerate(rec_corpus_indices):
            score_drops[i, j] = baseline_scores_for_recs[j] - float(
                holdout_scores[rec_idx]
            )

    # For each recommendation, find the top N contributors
    enriched_recs = []
    for j, rec in enumerate(recommendations):
        drops_for_rec = score_drops[:, j]
        # Sort by score drop (descending) - items that cause biggest drop are top contributors
        top_contributor_indices = np.argsort(drops_for_rec)[::-1][:top_n_contributors]

        contributors = []
        for idx in top_contributor_indices:
            corpus_idx = int(corpus_indices[idx])
            drop = float(drops_for_rec[idx])
            # Only include if there's a positive contribution
            if drop > 0:
                contributors.append(
                    {
                        "anime_id": model.corpus_ids[corpus_idx],
                        "corpus_idx": corpus_idx,
                        "score_contribution": drop,
                    }
                )

        enriched_rec = rec.copy()
        enriched_rec["top_contributors"] = contributors
        enriched_recs.append(enriched_rec)

    return enriched_recs


# Pydantic models for request/response validation
class ProfileEntry(BaseModel):
    anime_id: int
    rating: float = 0
    watch_status: str = ""


class RecommendRequest(BaseModel):
    profile: list[ProfileEntry]
    top_k: int = 50
    logit_weight: float | None = None
    include_profile_holdout: bool = False
    include_contribution_analysis: bool = False
    top_contributors: int = 3
    use_alt_ranking: bool = False
    niche_boost_factor: float = 0.0


# FastAPI app
app = FastAPI(title="Anime Recommendation Model Server")


@app.on_event("startup")
async def startup_event():
    """Load the model at startup."""
    logger.info("Loading model at startup...")
    get_model()
    logger.info("Model loaded successfully")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


def _run_inference(
    model: RecommenderModel,
    corpus_indices: np.ndarray,
    normalized_ratings: np.ndarray,
    original_ratings: np.ndarray,
    top_k: int,
    logit_weight: float | None,
    include_contribution_analysis: bool,
    include_profile_holdout: bool,
    top_contributors: int,
    use_alt_ranking: bool = False,
    niche_boost_factor: float = 0.0,
) -> tuple[list[dict], dict | None]:
    """Run model inference (called in executor to serialize access)."""
    # If niche boost is enabled, fetch more recommendations to allow composition changes
    # This ensures the boost can promote items that wouldn't have made the initial top_k
    if niche_boost_factor > 0 and model.popularity_distribution is not None:
        # Fetch top_k * 3 recommendations (capped at 500) for niche boosting
        expanded_k = min(top_k * 3, 500)
    else:
        expanded_k = top_k

    recommendations = get_recommendations_simple(
        model,
        corpus_indices,
        normalized_ratings,
        expanded_k,
        logit_weight,
        use_alt_ranking,
    )

    # Apply niche boost as final step (after all other scoring is done)
    # This will re-sort and potentially change the composition of top recommendations
    recommendations = apply_niche_boost(recommendations, model, niche_boost_factor)

    # Truncate to requested top_k after boost is applied
    recommendations = recommendations[:top_k]

    if include_contribution_analysis:
        recommendations = compute_recommendation_contributions(
            model,
            corpus_indices,
            normalized_ratings,
            recommendations,
            top_contributors,
            logit_weight,
            use_alt_ranking,
        )

    profile_holdout = None
    if include_profile_holdout:
        # Extract top-50 (or top_k) indices and scores for impact calculation
        # Use up to 50 recommendations for impact scoring
        num_for_impact = min(50, len(recommendations))
        baseline_top50_indices = np.array(
            [r["corpus_idx"] for r in recommendations[:num_for_impact]], dtype=np.int32
        )
        baseline_top50_scores = np.array(
            [r["score"] for r in recommendations[:num_for_impact]], dtype=np.float32
        )

        profile_holdout = compute_profile_holdout_analysis(
            model,
            corpus_indices,
            normalized_ratings,
            original_ratings,
            logit_weight,
            baseline_top50_indices,
            baseline_top50_scores,
        )

    return recommendations, profile_holdout


@app.post("/recommend")
async def recommend(req: RecommendRequest):
    """Get recommendations for a user profile."""
    try:
        # Check cache first
        cached_response = _cache.get(req)
        if cached_response is not None:
            logger.debug("Cache hit for request")
            return cached_response

        if not req.profile:
            raise HTTPException(
                status_code=400, detail="profile must be a non-empty list"
            )

        # Convert pydantic models to dicts for existing code
        profile = [entry.model_dump() for entry in req.profile]

        model = get_model()

        try:
            (
                corpus_indices,
                normalized_ratings,
                original_ratings,
                valid_entries,
                norm_stats,
            ) = preprocess_profile(model, profile)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Run inference in executor to serialize model access
        loop = asyncio.get_event_loop()
        recommendations, profile_holdout = await loop.run_in_executor(
            _inference_executor,
            _run_inference,
            model,
            corpus_indices,
            normalized_ratings,
            original_ratings,
            req.top_k,
            req.logit_weight,
            req.include_contribution_analysis,
            req.include_profile_holdout,
            req.top_contributors,
            req.use_alt_ranking,
            req.niche_boost_factor,
        )

        # Clean up normalization stats for JSON serialization
        clean_norm_stats = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in norm_stats.items()
        }

        response = {
            "recommendations": recommendations,
            "profile_holdout": profile_holdout,
            "normalization_stats": clean_norm_stats,
        }

        # Store in cache
        _cache.put(req, response)
        logger.debug(f"Cached response (cache size: {len(_cache)})")

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error processing recommendation request")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/corpus")
async def get_corpus():
    """Return the list of anime IDs in the corpus (for debugging/validation)."""
    model = get_model()
    return {"corpus_ids": model.corpus_ids, "corpus_size": model.corpus_size}


@app.get("/cache/stats")
async def cache_stats():
    """Return cache statistics."""
    return {
        "size": len(_cache),
        "max_size": _cache.max_size,
    }


@app.post("/cache/clear")
async def clear_cache():
    """Clear the recommendation cache."""
    _cache.clear()
    logger.info("Cache cleared")
    return {"status": "ok", "message": "Cache cleared"}


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Model server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument(
        "--model-path",
        type=str,
        default="../data/jax_model.msgpack",
        help="Path to model file",
    )
    parser.add_argument(
        "--corpus-path",
        type=str,
        default="../data/corpus_ids.json",
        help="Path to corpus IDs file",
    )
    args = parser.parse_args()

    os.environ["MODEL_PATH"] = args.model_path
    os.environ["CORPUS_PATH"] = args.corpus_path

    uvicorn.run(app, host=args.host, port=args.port)
