import json
from dataclasses import dataclass
from typing import TypedDict

import numpy as np


import jax
import jax.numpy as jnp
from flax import serialization

from model import (
    CONF,
    Recommender,
    make_dense_profile,
    infer_outputs,
    infer_bottleneck,
    rank_by_weighted_score,
    compute_holdout_metrics,
    setup_jax_cpu,
)

# Configure JAX for CPU for local inference
setup_jax_cpu()
from normalize_ratings import normalize_ratings


class UserAnimeEntry(TypedDict):
    anime_id: int
    rating: float  # 0-10 scale, 0 means unrated, -2 means dropped
    watch_status: str  # "completed", "watching", "dropped", "plan_to_watch", etc.


@dataclass
class RecommendationResult:
    anime_id: int
    corpus_idx: int
    score: float  # Combined weighted score
    probability: float  # Presence probability from model
    predicted_rating: float  # Predicted normalized rating


@dataclass
class HoldoutResult:
    anime_id: int
    corpus_idx: int
    true_rating: float  # Original 0-10 rating
    true_normalized_rating: float  # Normalized rating used by model
    predicted_rating: float  # Model's predicted rating
    rating_error: float  # Absolute error
    presence_probability: float  # Model's probability for this item
    recommendation_score: float  # Weighted recommendation score for this item
    impact_score: (
        float  # Sum of absolute score changes for top-50 when this item is held out
    )


@dataclass
class HoldoutAnalysis:
    item_results: list[HoldoutResult]
    mean_rating_error: float
    std_rating_error: float
    mean_presence_prob: float
    std_presence_prob: float


@dataclass
class InferenceResult:
    recommendations: list[RecommendationResult]
    holdout_analysis: HoldoutAnalysis | None
    normalization_stats: dict

    def to_dict(self) -> dict:
        recommendations = [
            {
                "anime_id": r.anime_id,
                "corpus_idx": r.corpus_idx,
                "score": r.score,
                "probability": r.probability,
                "predicted_rating": r.predicted_rating,
            }
            for r in self.recommendations
        ]

        holdout = None
        if self.holdout_analysis:
            ha = self.holdout_analysis
            holdout = {
                "items": [
                    {
                        "anime_id": h.anime_id,
                        "corpus_idx": h.corpus_idx,
                        "true_rating": h.true_rating,
                        "true_normalized_rating": h.true_normalized_rating,
                        "predicted_rating": h.predicted_rating,
                        "rating_error": h.rating_error,
                        "presence_probability": h.presence_probability,
                        "recommendation_score": h.recommendation_score,
                        "impact_score": h.impact_score,
                    }
                    for h in ha.item_results
                ],
                "mean_rating_error": ha.mean_rating_error,
                "std_rating_error": ha.std_rating_error,
                "mean_presence_prob": ha.mean_presence_prob,
                "std_presence_prob": ha.std_presence_prob,
            }

        return {
            "recommendations": recommendations,
            "holdout_analysis": holdout,
            "normalization_stats": {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in self.normalization_stats.items()
            },
        }


class RecommenderInference:
    def __init__(
        self,
        model_path: str = "../data/jax_model.msgpack",
        corpus_ids_path: str = "../data/corpus_ids.json",
    ):
        self.corpus_size = CONF["corpus_size"]

        self._load_corpus_mapping(corpus_ids_path)
        self._load_model(model_path)

    def _load_corpus_mapping(self, corpus_ids_path: str):
        with open(corpus_ids_path, "r") as f:
            self.corpus_ids = json.load(f)

        if len(self.corpus_ids) != self.corpus_size:
            raise ValueError(
                f"corpus_ids.json has {len(self.corpus_ids)} items, expected {self.corpus_size}"
            )

        self.anime_id_to_corpus_idx = {
            anime_id: idx for idx, anime_id in enumerate(self.corpus_ids)
        }

    def _load_model(self, model_path: str):
        model = Recommender()
        rng = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((1, self.corpus_size * 2))
        params = model.init({"params": rng, "noise": rng}, dummy_input)["params"]

        with open(model_path, "rb") as f:
            saved_bytes = f.read()

        self.params = serialization.from_bytes(params, saved_bytes)
        print(f"Loaded model from {model_path}")

    def _preprocess_profile(
        self, user_profile: list[UserAnimeEntry]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        valid_entries = []
        for entry in user_profile:
            anime_id = entry["anime_id"]
            if anime_id not in self.anime_id_to_corpus_idx:
                continue

            # Include items that have been rated (rating > 0)
            # Also include completed/dropped items even without rating for presence signal
            rating = entry.get("rating", 0) or 0
            watch_status = entry.get("watch_status", "")

            if rating > 0 or watch_status in ("completed", "watching", "dropped"):
                valid_entries.append(
                    {
                        "corpus_idx": self.anime_id_to_corpus_idx[anime_id],
                        "anime_id": anime_id,
                        "rating": rating,
                        "watch_status": watch_status,
                    }
                )

        if not valid_entries:
            raise ValueError("No valid entries in user profile (no items in corpus)")

        valid_entries.sort(key=lambda x: x["corpus_idx"])

        corpus_indices = np.array(
            [e["corpus_idx"] for e in valid_entries], dtype=np.int32
        )
        original_ratings = np.array(
            [
                # -2 is a special flag used to indicate unrated + dropped
                #
                # it will be set to a low value wrt. the user's rating distribution during normalization
                -2 if e.get("watch_status") == "dropped" else e["rating"]
                for e in valid_entries
            ],
            dtype=np.float32,
        )

        normalized_ratings, norm_stats = normalize_ratings(original_ratings)

        return corpus_indices, normalized_ratings, original_ratings, norm_stats

    def get_recommendations(
        self,
        user_profile: list[UserAnimeEntry],
        top_k: int = 50,
        logit_weight: float | None = None,
        include_holdout_analysis: bool = False,
    ) -> InferenceResult:
        corpus_indices, normalized_ratings, original_ratings, norm_stats = (
            self._preprocess_profile(user_profile)
        )

        input_vector = make_dense_profile(corpus_indices, normalized_ratings)

        item_logits, rating_pred = infer_outputs(self.params, input_vector)

        presence_mask = input_vector[0, : self.corpus_size]

        topk_idx, topk_scores, topk_probs, topk_ratings = rank_by_weighted_score(
            item_logits[0],
            rating_pred[0],
            presence_mask,
            k=top_k,
            logit_weight=logit_weight,
        )

        recommendations = []
        for i in range(len(topk_idx)):
            corpus_idx = int(topk_idx[i])
            recommendations.append(
                RecommendationResult(
                    anime_id=self.corpus_ids[corpus_idx],
                    corpus_idx=corpus_idx,
                    score=float(topk_scores[i]),
                    probability=float(topk_probs[i]),
                    predicted_rating=float(topk_ratings[i]),
                )
            )

        holdout_analysis = None
        if include_holdout_analysis:
            holdout_analysis = self._compute_holdout_analysis(
                corpus_indices,
                normalized_ratings,
                original_ratings,
                logit_weight,
                topk_idx,
                topk_scores,
            )

        return InferenceResult(
            recommendations=recommendations,
            holdout_analysis=holdout_analysis,
            normalization_stats=norm_stats,
        )

    def _compute_holdout_analysis(
        self,
        corpus_indices: np.ndarray,
        normalized_ratings: np.ndarray,
        original_ratings: np.ndarray,
        logit_weight: float | None,
        baseline_top50_indices: np.ndarray,
        baseline_top50_scores: np.ndarray,
    ) -> HoldoutAnalysis:
        metrics = compute_holdout_metrics(
            self.params,
            corpus_indices,
            normalized_ratings,
            self.corpus_size,
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

        item_results = []
        for i in range(actual_size):
            corpus_idx = int(held_out_indices_np[i])
            item_results.append(
                HoldoutResult(
                    anime_id=self.corpus_ids[corpus_idx],
                    corpus_idx=corpus_idx,
                    true_rating=float(original_ratings[i]),
                    true_normalized_rating=float(held_out_ratings_np[i]),
                    predicted_rating=float(pred_ratings[i]),
                    rating_error=float(rating_errors[i]),
                    presence_probability=float(presence_probs[i]),
                    recommendation_score=float(recommendation_scores[i]),
                    impact_score=float(impact_scores[i]),
                )
            )

        return HoldoutAnalysis(
            item_results=item_results,
            mean_rating_error=float(np.mean(rating_errors)),
            std_rating_error=float(np.std(rating_errors)),
            mean_presence_prob=float(np.mean(presence_probs)),
            std_presence_prob=float(np.std(presence_probs)),
        )

    def get_anime_id_for_corpus_idx(self, corpus_idx: int) -> int:
        return self.corpus_ids[corpus_idx]

    def get_corpus_idx_for_anime_id(self, anime_id: int) -> int | None:
        return self.anime_id_to_corpus_idx.get(anime_id)

    def is_anime_in_corpus(self, anime_id: int) -> bool:
        return anime_id in self.anime_id_to_corpus_idx


def main(print_bottleneck: bool = False):
    test_profile: list[UserAnimeEntry] = [
        {"anime_id": 54309, "rating": 9, "watch_status": "completed"},
        {"anime_id": 35557, "rating": 8, "watch_status": "completed"},
        {
            "anime_id": 33089,
            "rating": 10,
            "watch_status": "completed",
        },
        {
            "anime_id": 66,
            "rating": 9,
            "watch_status": "completed",
        },
        {
            "anime_id": 477,
            "rating": 8,
            "watch_status": "completed",
        },
        {
            "anime_id": 999999999,  # not in corpus
            "rating": 8,
            "watch_status": "completed",
        },
    ]

    print(f"Initializing recommender...")
    recommender = RecommenderInference()

    print(f"\nTest profile has {len(test_profile)} entries")

    print("\nGetting recommendations...")
    result = recommender.get_recommendations(
        test_profile, top_k=20, include_holdout_analysis=True
    )

    if print_bottleneck:
        print("\n" + "=" * 60)
        print("BOTTLENECK LAYER ACTIVATIONS")
        print("=" * 60)

        corpus_indices, normalized_ratings, _, _ = recommender._preprocess_profile(
            test_profile
        )
        input_vector = make_dense_profile(corpus_indices, normalized_ratings)

        bottleneck = infer_bottleneck(recommender.params, input_vector)
        bottleneck_np = np.array(bottleneck[0])

        print(f"Bottleneck shape: {bottleneck_np.shape}")
        print(f"Min: {bottleneck_np.min():.6f}")
        print(f"Max: {bottleneck_np.max():.6f}")
        print(f"Mean: {bottleneck_np.mean():.6f}")
        print(f"Std: {bottleneck_np.std():.6f}")
        print(f"Num zeros: {np.sum(bottleneck_np == 0)}")
        print(f"Num positive: {np.sum(bottleneck_np > 0)}")
        print(f"Num negative: {np.sum(bottleneck_np < 0)}")

        percentiles = [0, 10, 25, 50, 75, 90, 100]
        pct_values = np.percentile(bottleneck_np, percentiles)
        print("\nPercentile distribution:")
        for p, v in zip(percentiles, pct_values):
            print(f"  {p:3d}%: {v:10.6f}")

        print(f"\nRaw bottleneck values ({len(bottleneck_np)} dims):")
        if len(bottleneck_np) <= 100:
            for i, v in enumerate(bottleneck_np):
                print(f"  [{i:3d}] {v:12.6f}")
        else:
            print("  First 50:")
            for i, v in enumerate(bottleneck_np[:50]):
                print(f"    [{i:3d}] {v:12.6f}")
            print("  ...")
            print("  Last 50:")
            for i, v in enumerate(bottleneck_np[-50:], start=len(bottleneck_np) - 50):
                print(f"    [{i:3d}] {v:12.6f}")

    print(f"\nNormalization stats: {result.normalization_stats}")

    print("\n" + "=" * 60)
    print("TOP 20 RECOMMENDATIONS")
    print("=" * 60)
    for i, rec in enumerate(result.recommendations, 1):
        print(
            f"{i:2d}. anime_id={rec.anime_id:5d}  "
            f"score={rec.score:.4f}  "
            f"prob={rec.probability:.4f}  "
            f"pred_rating={rec.predicted_rating:.4f}"
        )

    if result.holdout_analysis:
        ha = result.holdout_analysis
        print("\n" + "=" * 60)
        print("HOLDOUT ANALYSIS")
        print("=" * 60)
        print(f"Mean rating error: {ha.mean_rating_error:.4f}")
        print(f"Std rating error:  {ha.std_rating_error:.4f}")
        print(f"Mean presence prob: {ha.mean_presence_prob * 100:.4f}%")
        print(f"Std presence prob:  {ha.std_presence_prob * 100:.4f}%")

        print("\nPer-item holdout results:")
        for h in ha.item_results:
            print(
                f"  anime_id={h.anime_id:5d}  "
                f"true={h.true_rating:.1f}  "
                f"norm={h.true_normalized_rating:.4f}  "
                f"pred={h.predicted_rating:.4f}  "
                f"err={h.rating_error:.4f}  "
                f"prob={h.presence_probability * 100:.4f}%"
            )

    print("\n" + "=" * 60)
    print("=" * 60)
    json_result = result.to_dict()
    print(json_result)
    print(json.dumps(json_result, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="infer.py")
    parser.add_argument(
        "--bottleneck",
        action="store_true",
        help="Print raw bottleneck layer activations for analysis",
    )
    args = parser.parse_args()

    main(print_bottleneck=args.bottleneck)
