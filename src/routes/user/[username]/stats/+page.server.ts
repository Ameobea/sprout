import * as R from 'ramda';
import { isLeft } from 'fp-ts/lib/Either.js';
import { fetchUserRankings } from 'src/helpers';

import { AnimeListStatusCode, type AnimeBasicDetails, type AnimeListStatus } from 'src/malAPI';
import { getOfflineMetadataDB, type OfflineAnimeMetadatum } from 'src/offlineMetadataDB';
import { DEFAULT_PROFILE_SOURCE } from 'src/components/recommendation/conf';
import { typify, type Typify } from 'src/components/recommendation/utils';
import {
  ProfileSourceValidator,
  type HoldoutItem,
  type ModelServerInput,
  type ModelServerOutput,
  type NormalizationStats,
} from 'src/routes/recommendation/recommendation/recommendation';
import { MODEL_SERVER_URL } from 'src/conf';
import { denormalizeRating } from 'src/util/ratingNormalization';
import type { PageServerLoad } from './$types';

export type PartialStatsAnimeMetadatum = Pick<OfflineAnimeMetadatum, 'tags' | 'title'>;
export interface PartialStatsMALUserAnimeListItem {
  node: Pick<AnimeBasicDetails, 'id'>;
  list_status: Pick<AnimeListStatus, 'score'>;
}

export interface ProfileAnalysis {
  mostImpactfulRatings: HoldoutItem[];
  mostSurprisingItems: HoldoutItem[];
  normalizationStats: NormalizationStats;
}

export type UserStatsLoadProps = {
  animeData: { [id: number]: PartialStatsAnimeMetadatum };
  profileRes: { type: 'ok'; profile: Typify<PartialStatsMALUserAnimeListItem[]> } | { type: 'error'; error: string };
  profileAnalysis: ProfileAnalysis | null;
};

async function fetchProfileAnalysis(
  userProfile: { node: { id: number }; list_status: { score: number; status: string } }[]
): Promise<ProfileAnalysis | null> {
  try {
    const requestBody: ModelServerInput = {
      profile: userProfile.map((entry) => ({
        anime_id: entry.node.id,
        rating: entry.list_status.score,
        watch_status: entry.list_status.status,
      })),
      top_k: 50,
      include_profile_holdout: true,
      include_contribution_analysis: false,
    };

    const response = await fetch(`${MODEL_SERVER_URL}/recommend`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      console.error(`Model server request failed with status ${response.status}`);
      return null;
    }

    const output: ModelServerOutput = await response.json();

    if (!output.profile_holdout) {
      console.error('Model server did not return profile holdout data');
      return null;
    }

    // Sort holdout items by impact score (descending) for most impactful
    const mostImpactfulRatings = [...output.profile_holdout.items]
      .sort((a, b) => b.impact_score - a.impact_score)
      .slice(0, 10);

    // For surprising ratings: filter to only rated items, compute diff in 1-10 scale, sort by absolute diff
    const normStats = output.normalization_stats;
    const mostSurprisingItems = [...output.profile_holdout.items]
      .filter((item) => item.true_rating > 0) // Exclude unrated items
      .map((item) => {
        const predictedRating = denormalizeRating(item.predicted_rating, normStats);
        const ratingDiff = Math.abs(item.true_rating - predictedRating);
        return { item, ratingDiff };
      })
      .sort((a, b) => b.ratingDiff - a.ratingDiff)
      .slice(0, 10)
      .map(({ item }) => item);

    return {
      mostImpactfulRatings,
      mostSurprisingItems,
      normalizationStats: output.normalization_stats,
    };
  } catch (error) {
    console.error('Error fetching profile analysis:', error);
    return null;
  }
}

export const load: PageServerLoad = async ({ params, url }): Promise<UserStatsLoadProps> => {
  const username = params.username;
  const rawProfileSource = url.searchParams.get('source') ?? DEFAULT_PROFILE_SOURCE;
  const profileSourceParseRes = ProfileSourceValidator.decode(rawProfileSource);
  if (isLeft(profileSourceParseRes)) {
    return {
      animeData: {},
      profileRes: { type: 'error' as const, error: 'Invalid `source` query param' },
      profileAnalysis: null,
    };
  }
  const profileSource = profileSourceParseRes.right;

  const rankingsRes = await fetchUserRankings(username, profileSource)();
  if (isLeft(rankingsRes)) {
    return {
      animeData: {},
      profileRes: { type: 'error' as const, error: typify(rankingsRes.left.body) },
      profileAnalysis: null,
    };
  }
  const { profile: userProfile } = rankingsRes.right;

  // Fetch profile analysis from model server
  const profileAnalysis = await fetchProfileAnalysis(userProfile);

  // Collect anime IDs that need metadata (profile + holdout items)
  const animeIdsNeedingMetadata = new Set(userProfile.map((item) => item.node.id));
  if (profileAnalysis) {
    for (const item of profileAnalysis.mostImpactfulRatings) {
      animeIdsNeedingMetadata.add(item.anime_id);
    }
    for (const item of profileAnalysis.mostSurprisingItems) {
      animeIdsNeedingMetadata.add(item.anime_id);
    }
  }

  const offlineMetadataDB = await getOfflineMetadataDB();
  const animeData: { [malID: number]: PartialStatsAnimeMetadatum } = {};
  for (const id of animeIdsNeedingMetadata) {
    const details = await offlineMetadataDB.byMALID.get(id);
    if (!details) {
      console.error(`Could not find offline metadata for anime ID ${id}`);
      continue;
    }
    animeData[id] = R.pick(['tags', 'title'], details);
  }

  const profile: PartialStatsMALUserAnimeListItem[] = userProfile
    .filter((rating) => {
      switch (rating.list_status.status) {
        case AnimeListStatusCode.Completed:
        case AnimeListStatusCode.Watching:
        case AnimeListStatusCode.OnHold:
        case AnimeListStatusCode.Dropped:
          return true;
        case AnimeListStatusCode.PlanToWatch:
          return false;
        default:
          console.error(`Unknown anime list status code ${rating.list_status.status}`);
          return true;
      }
    })
    .map((item) => ({
      node: R.pick(['id'], item.node),
      list_status: R.pick(['score'], item.list_status),
    }));

  return {
    animeData: typify(animeData),
    profileRes: { type: 'ok', profile: typify(profile) },
    profileAnalysis: typify(profileAnalysis),
  };
};
