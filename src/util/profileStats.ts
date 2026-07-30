import { AnimeListStatusCode } from 'src/malAPI';

export interface ProfileRatingStats {
  /** Number of entries with a rating > 0 */
  ratedCount: number;
  /** Number of entries excluding plan-to-watch */
  consideredCount: number;
  /** ratedCount / consideredCount */
  ratedFraction: number;
  mean: number;
  stdDev: number;
  isNonRater: boolean;
}

export const NON_RATER_RATED_FRACTION_THRESHOLD = 0.5;

/**
 * Canonical non-rater detection + rating distribution stats, shared between the
 * recommendations pipeline and the profile stats page. Plan-to-watch entries are
 * excluded from the denominator since they're always unrated.
 */
export const computeProfileRatingStats = (
  profile: { list_status: { score: number; status: string } }[]
): ProfileRatingStats => {
  const considered = profile.filter((entry) => entry.list_status.status !== AnimeListStatusCode.PlanToWatch);
  const ratedScores = considered.map((entry) => entry.list_status.score).filter((score) => score > 0);

  const ratedCount = ratedScores.length;
  const consideredCount = considered.length;
  const ratedFraction = consideredCount === 0 ? 0 : ratedCount / consideredCount;

  let mean = 0;
  let stdDev = 0;
  if (ratedCount > 0) {
    mean = ratedScores.reduce((acc, score) => acc + score, 0) / ratedCount;
    stdDev = Math.sqrt(ratedScores.reduce((acc, score) => acc + (score - mean) ** 2, 0) / ratedCount);
  }

  return {
    ratedCount,
    consideredCount,
    ratedFraction,
    mean,
    stdDev,
    isNonRater: ratedFraction < NON_RATER_RATED_FRACTION_THRESHOLD,
  };
};
