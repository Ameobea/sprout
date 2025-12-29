/**
 * Utilities for converting between normalized model ratings and the 1-10 rating scale.
 *
 * The model uses a custom normalization that mixes z-score and absolute normalization.
 * These utilities allow inverting that to recover approximate 1-10 ratings.
 */

export interface NormalizationStats {
  mu: number;
  sigma: number;
  alpha: number;
  zscore_norm: number[];
  absolute_norm: number[];
}

const NEUTRAL_POINT = 5.5;
const ABSOLUTE_SCALE_DIVISOR = 2.5;

/**
 * Inverts the normalization algorithm to convert from normalized rating back to 1-10 scale.
 *
 * The original normalization:
 *   zscore_norm = (rating - mu) / sigma
 *   absolute_norm = (rating - neutral_point) / 2.5
 *   norm_rating = alpha * zscore_norm + (1 - alpha) * absolute_norm
 *
 * Inverting:
 *   norm_rating = alpha * (rating - mu) / sigma + (1 - alpha) * (rating - neutral_point) / 2.5
 *   norm_rating = rating * (alpha / sigma + (1 - alpha) / 2.5) - alpha * mu / sigma - (1 - alpha) * neutral_point / 2.5
 *
 * Let A = alpha / sigma + (1 - alpha) / 2.5
 * Let B = alpha * mu / sigma + (1 - alpha) * neutral_point / 2.5
 *
 * norm_rating = rating * A - B
 * rating = (norm_rating + B) / A
 */
export const denormalizeRating = (normalizedRating: number, stats: NormalizationStats): number => {
  const { mu, sigma, alpha } = stats;

  // Avoid division by zero
  const safeSigma = Math.max(sigma, 0.001);

  const A = alpha / safeSigma + (1 - alpha) / ABSOLUTE_SCALE_DIVISOR;
  const B = (alpha * mu) / safeSigma + ((1 - alpha) * NEUTRAL_POINT) / ABSOLUTE_SCALE_DIVISOR;

  const rating = (normalizedRating + B) / A;

  return Math.min(10, Math.max(1, rating));
};

export const denormalizeRatings = (normalizedRatings: number[], stats: NormalizationStats): number[] =>
  normalizedRatings.map((r) => denormalizeRating(r, stats));
