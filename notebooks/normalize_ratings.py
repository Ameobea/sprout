import numpy as np


def normalize_ratings(
    scores: np.ndarray,
    sigma_divisor=2.6,
    neutral_point=5.5,
):
    """
    Normalizes user ratings from the base 1-10 scale to a normalized scale suitable for model input

    Uses a mix between Z-score normalization (to help smooth over per-user differences in rating scale)
    and an absolute scale (to help preserve information about absolute rating magnitude).

    The goal is to encode information about both how much more users like one entity compared to others in
    their profile while still letting the model tell the difference between high and low ratings (which can
    be important for estimating ratings, especially for sparser profiles).

    Args:
        scores: numpy array of rating scores (float32).  Scores of 0 are treated as unrated.  Scores of -2
                       indicate unrated dropped shows as a special case.
        sigma_divisor: The divisor for sigma when computing adaptive alpha.
                       Higher values -> lower alpha -> more absolute normalization.
        neutral_point: The neutral point for the absolute rating scale which maps to 0.0
    """
    scores = np.asarray(scores, dtype=np.float32).copy()

    if len(scores) <= 1:
        return np.zeros_like(scores), {
            "mu": 0.0,
            "sigma": 0.0,
            "alpha": 0.0,
            "zscore_norm": np.zeros_like(scores),
            "absolute_norm": np.zeros_like(scores),
        }

    rated_scores = scores[scores > 0]
    mu = np.mean(rated_scores) if len(rated_scores) > 0 else 5.0

    # TODO: filling unrated shows with the mean might not be the best way to do it, but
    # idk of a better alternative
    scores[scores == 0] = mu
    sigma = np.std(rated_scores) + 1e-6

    # not much else we can do in this case
    if len(rated_scores) == 0:
        mu = 5.0
        scores[scores == 0] = mu
        sigma = 2.0

    # fill in scores for unrated dropped shows 1.5 stddevs below the mean
    scores[scores == -2] = mu - 1.5 * sigma

    # z-score
    zscore_norm = (scores - mu) / sigma
    zscore_norm = np.clip(zscore_norm, -3.0, 3.0)

    # modified absolute scale
    absolute_norm = (scores - neutral_point) / 2.5
    absolute_norm = np.clip(absolute_norm, -2.5, 2.0)

    # mixing between the relative and absolute scales based on variance in the user's ratings
    #
    # low variance -> lower alpha (preserve absolute)
    # high variance -> higher alpha (use z-score)
    alpha = np.clip(sigma / sigma_divisor, 0.3, 0.8)

    norm_ratings = alpha * zscore_norm + (1 - alpha) * absolute_norm
    norm_ratings = np.clip(norm_ratings, -2.5, 2.5)

    stats = {
        "mu": float(mu),
        "sigma": float(sigma),
        "alpha": float(alpha) if np.isscalar(alpha) else float(alpha),
        "zscore_norm": zscore_norm,
        "absolute_norm": absolute_norm,
    }

    return norm_ratings, stats
