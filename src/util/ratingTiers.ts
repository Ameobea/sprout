export type RatingTier = 'high' | 'mid' | 'low';

export interface RatingTierStats {
  mean: number;
  stdDev: number;
  ratedCount: number;
}

const MIN_STD_DEV_FOR_TIERS = 0.75;
const MIN_RATED_COUNT_FOR_TIERS = 10;

/**
 * Tiers a predicted rating relative to the user's own rating distribution.
 * Returns null (no coloring) for outlier raters: too few ratings for the stats to be
 * trustworthy, or a stdDev so small that half a σ spans less than half a rating point.
 */
export const getRatingTier = (predictedRating: number, stats: RatingTierStats): RatingTier | null => {
  if (stats.stdDev < MIN_STD_DEV_FOR_TIERS || stats.ratedCount < MIN_RATED_COUNT_FOR_TIERS) {
    return null;
  }
  if (predictedRating >= stats.mean + stats.stdDev / 2) {
    return 'high';
  }
  if (predictedRating <= stats.mean - stats.stdDev / 2) {
    return 'low';
  }
  return 'mid';
};
