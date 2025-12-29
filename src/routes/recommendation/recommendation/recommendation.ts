import { type Either, isLeft, right } from 'fp-ts/lib/Either.js';
import * as t from 'io-ts';

import { ModelName, ProfileSource, RECOMMENDATION_MODEL_CORPUS_SIZE } from 'src/components/recommendation/conf';
import { loadEmbedding } from 'src/embedding';
import { AnimeListStatusCode, AnimeMediaType, AnimeRelationType, getAnimesByID, type AnimeDetails } from 'src/malAPI';
import { EmbeddingName } from 'src/types';
import type { Embedding } from 'src/routes/embedding';
import { fetchUserRankings } from 'src/helpers';
import type { CompatAnimeListEntry } from 'src/anilistAPI';
import { MODEL_SERVER_URL } from 'src/conf';

export interface Recommendation {
  id: number;
  score: number;
  /**
   * These IDs will be negative to indicate a negative rating contributing to a positive recommendation.  Absolute
   * value should be used to get IDs to look up.
   */
  topRatingContributorsIds?: number[];
  planToWatch?: boolean;
}

interface GetRecommendationsArgs {
  dataSource:
    | { type: 'username'; username: string }
    | { type: 'rawProfile'; profile: { animeID: number; score: number }[] };
  count: number;
  computeContributions: boolean;
  modelName: ModelName;
  excludedRankingAnimeIDs: Set<number>;
  excludedGenreIDs: Set<number>;
  includeExtraSeasons: boolean;
  includeONAsOVAsSpecials: boolean;
  includeMovies: boolean;
  includeMusic: boolean;
  profileSource: ProfileSource;
  filterPlanToWatch: boolean;
  /**
   * Controls how the recommendation score is computed from both the presence probability
   * and the predicted rating (0.0 = only ratings, 1.0 = only presence probability, default = 0.4)
   */
  logitWeight: number;
  /**
   * Controls how much to boost less popular shows in the rankings (0.0 = no boost, 1.0 = maximum boost)
   */
  nicheBoostFactor: number;
}

let CachedEmbeddingMetadata: (AnimeDetails | null)[] | null = null;
let CachedGenresDB: Map<number, string> | null = null;

export const getEmbeddingMetadata = async (embedding: Embedding): Promise<(AnimeDetails | null)[]> => {
  if (CachedEmbeddingMetadata) {
    return CachedEmbeddingMetadata;
  }

  console.log('Loading embedding metadata for recommendations...');
  const fetched = await getAnimesByID(embedding.map(({ metadata }) => metadata.id));
  CachedEmbeddingMetadata = fetched.map((item, i) => {
    if (!item) {
      console.error(`Could not find metadata for anime ID ${embedding[i].metadata.id}`);
      return null;
    }
    return item;
  });
  console.log('Done loading embedding metadata for recommendations.');
  return CachedEmbeddingMetadata;
};

export const getGenresDB = async (): Promise<Map<number, string>> => {
  if (CachedGenresDB) {
    return CachedGenresDB;
  }

  const embedding = (await loadEmbedding(EmbeddingName.Model)).slice(0, RECOMMENDATION_MODEL_CORPUS_SIZE);
  const embeddingMetadata = await getEmbeddingMetadata(embedding);
  const genresDB = new Map<number, string>();
  for (const metadatum of embeddingMetadata) {
    if (!metadatum || !metadatum.genres) {
      continue;
    }

    for (const genre of metadatum.genres) {
      genresDB.set(genre.id, genre.name);
    }
  }
  CachedGenresDB = genresDB;
  return CachedGenresDB;
};

const MAX_TRAVERSAL_DEPTH = 18;

const buildSeasonRelationshipsByAnimeID = async (
  animeIDsToCheck: number[],
  seasonRelationshipsByAnimeID: Map<number, Set<number>> = new Map(),
  processedAnimeIDs: Set<number> = new Set(),
  forceRetainedAnimeIDs: Set<number> = new Set(),
  depth = 0
): Promise<{ seasonRelationshipsByAnimeID: Map<number, Set<number>>; forceRetainedAnimeIDs: Set<number> }> => {
  const metadata = await getAnimesByID(animeIDsToCheck, true);

  let nextAnimeIDsToCheck: number[] = [];
  for (let i = 0; i < metadata.length; i++) {
    const datum = metadata[i];
    processedAnimeIDs.add(animeIDsToCheck[i]);
    if (!datum) {
      console.warn(`Could not find metadata for anime ${animeIDsToCheck[i]}`);
      continue;
    }
    if (!datum.related_anime) {
      continue;
    }
    // Movies, music, ONAs, OVAs, and specials are handled by separate filters; don't filter them with extra seasons
    if (
      datum.media_type === AnimeMediaType.Movie ||
      datum.media_type === AnimeMediaType.Music ||
      datum.media_type === AnimeMediaType.ONA ||
      datum.media_type === AnimeMediaType.OVA ||
      datum.media_type === AnimeMediaType.Special
    ) {
      forceRetainedAnimeIDs.add(datum.id);
      // Do not continue so that we can continue crawling relationships from this anime, because for some series
      // the only link between different seasons is ONAs/OVAs/Movies/etc.
    }

    const extraSeasonIDs = new Set(
      datum.related_anime
        .filter((entry) => {
          switch (entry.relation_type) {
            case AnimeRelationType.Sequel:
            case AnimeRelationType.Prequel:
            case AnimeRelationType.ParentStory:
            case AnimeRelationType.SideStory:
            case AnimeRelationType.Other:
              return true;
            default:
              return false;
          }
        })
        .map((entry) => entry.node.id)
    );
    seasonRelationshipsByAnimeID.set(datum.id, extraSeasonIDs);
    for (const extraSeasonID of extraSeasonIDs) {
      if (!seasonRelationshipsByAnimeID.has(extraSeasonID)) {
        seasonRelationshipsByAnimeID.set(extraSeasonID, new Set());
      }
      seasonRelationshipsByAnimeID.get(extraSeasonID)!.add(datum.id);
      nextAnimeIDsToCheck.push(extraSeasonID);
    }
  }

  nextAnimeIDsToCheck = [...nextAnimeIDsToCheck].filter((id) => !processedAnimeIDs.has(id));
  if (depth >= MAX_TRAVERSAL_DEPTH || nextAnimeIDsToCheck.length === 0) {
    return { forceRetainedAnimeIDs, seasonRelationshipsByAnimeID };
  }

  return buildSeasonRelationshipsByAnimeID(
    nextAnimeIDsToCheck,
    seasonRelationshipsByAnimeID,
    processedAnimeIDs,
    forceRetainedAnimeIDs,
    depth + 1
  );
};

const getIsExtraSeasonOfRatedAnime = (
  embedding: Embedding,
  animeIdToEmbeddingIndex: Map<number, number>,
  allWatchedAnimeIDs: Set<number>,
  seasonRelationshipsByAnimeID: Map<number, Set<number>>,
  animeID: number,
  checkedIDs: Set<number> = new Set(),
  depth = 0
) => {
  const extraSeasonIDs = seasonRelationshipsByAnimeID.get(animeID);
  if (!extraSeasonIDs) {
    return false;
  }

  const embeddingIx = animeIdToEmbeddingIndex.get(animeID);
  const metadatum = typeof embeddingIx === 'number' ? embedding[embeddingIx].metadata : null;
  if (
    !metadatum ||
    metadatum.media_type === AnimeMediaType.Movie ||
    metadatum.media_type === AnimeMediaType.Music ||
    metadatum.media_type === AnimeMediaType.ONA ||
    metadatum.media_type === AnimeMediaType.OVA ||
    metadatum.media_type === AnimeMediaType.Special
  ) {
    return false;
  }

  for (const extraSeasonAnimeID of extraSeasonIDs) {
    if (allWatchedAnimeIDs.has(extraSeasonAnimeID)) {
      return true;
    }
  }

  if (depth >= MAX_TRAVERSAL_DEPTH) {
    return false;
  }

  checkedIDs.add(animeID);
  for (const extraSeasonAnimeID of extraSeasonIDs) {
    if (checkedIDs.has(extraSeasonAnimeID)) {
      continue;
    }

    const childIsRelated = getIsExtraSeasonOfRatedAnime(
      embedding,
      animeIdToEmbeddingIndex,
      allWatchedAnimeIDs,
      seasonRelationshipsByAnimeID,
      extraSeasonAnimeID,
      checkedIDs,
      depth + 1
    );
    if (childIsRelated) {
      return true;
    }
  }

  return false;
};

export interface ModelServerInput {
  profile: {
    anime_id: number;
    rating: number;
    watch_status: string;
  }[];
  top_k: number;
  /**
   * Controls how the recommendation score is computed from both the presence probability
   * and the predicted rating (0.0 = only ratings, 1.0 = only presence probability, default = 0.4)
   */
  logit_weight?: number;
  include_profile_holdout: boolean;
  include_contribution_analysis: boolean;
  /**
   * Number of top contributors to return per recommendation (if contribution analysis is enabled).
   *
   * Defaults to 3.
   */
  top_contributors?: number;
  /**
   * Defaults to false; this is pretty much obsolete and a failed experiment
   */
  use_alt_ranking?: boolean;
  /**
   * Optional number in [0, 1] to boost less popular shows in the rankings.  It compares the absolute
   * popularity of a show to the probability distribution from the model output and boosts shows that
   * the model favors more than their popularity would suggest.
   *
   * Defaults to 0 (no boost).
   */
  niche_boost_factor?: number;
}

export interface ModelServerOutput {
  recommendations: ModelServerRecommendation[];
  profile_holdout?: ProfileHoldout;
  normalization_stats: NormalizationStats;
}

export interface ModelServerRecommendation {
  anime_id: number;
  corpus_idx: number;
  score: number;
  probability: number;
  predicted_rating: number;
  top_contributors?: TopContributor[];
}

export interface TopContributor {
  anime_id: number;
  corpus_idx: number;
  score_contribution: number;
}

export interface ProfileHoldout {
  items: HoldoutItem[];
  mean_rating_error: number;
  std_rating_error: number;
  mean_presence_prob: number;
  std_presence_prob: number;
}

export interface HoldoutItem {
  anime_id: number;
  corpus_idx: number;
  true_rating: number;
  true_normalized_rating: number;
  predicted_rating: number;
  rating_error: number;
  presence_probability: number;
  /**
   * Weighted recommendation score (using `logit_weight` from root request) for this item if it
   * were not in the profile
   */
  recommendation_score: number;
  /**
   * Sum of absolute score changes for top-50 when this item is held out
   */
  impact_score: number;
}

export interface NormalizationStats {
  mu: number;
  sigma: number;
  alpha: number;
  zscore_norm: number[];
  absolute_norm: number[];
}

const performInferrence = async ({
  modelName: _modelName,
  profile,
  count,
  computeContributions,
  logitWeight,
  nicheBoostFactor,
}: {
  modelName: ModelName;
  profile: CompatAnimeListEntry[];
  count: number;
  computeContributions: boolean;
  logitWeight: number;
  nicheBoostFactor: number;
}): Promise<ModelServerOutput> => {
  // Request extra recommendations to account for filtering later
  const requestCount = count * 3;

  const useAltRanking = false; // TODO: testing
  if (!useAltRanking) {
    // since the original rating system works with presence probabilities after passing them
    // through softmax, this makes the lower values more extreme, while compressing the higher values.
    const before = logitWeight;
    logitWeight = logitWeight * logitWeight * logitWeight * (2 / 3) + (1 / 3) * logitWeight + 0.01;
    console.log(`logit weight ${before} -> ${logitWeight}`);
  }

  const requestBody: ModelServerInput = {
    profile: profile.map((entry) => ({
      anime_id: entry.node.id,
      rating: entry.list_status.score,
      watch_status: entry.list_status.status,
    })),
    top_k: requestCount,
    logit_weight: logitWeight,
    include_profile_holdout: false,
    include_contribution_analysis: computeContributions,
    use_alt_ranking: useAltRanking,
    niche_boost_factor: nicheBoostFactor,
  };

  const response = await fetch(`${MODEL_SERVER_URL}/recommend`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(requestBody),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Model server request failed with status ${response.status}: ${errorText}`);
  }

  return response.json();
};

export const getRecommendations = async ({
  dataSource,
  count,
  computeContributions,
  modelName,
  excludedRankingAnimeIDs,
  excludedGenreIDs,
  includeExtraSeasons,
  includeONAsOVAsSpecials,
  includeMovies,
  includeMusic,
  profileSource,
  filterPlanToWatch,
  logitWeight,
  nicheBoostFactor,
}: GetRecommendationsArgs): Promise<Either<{ status: number; body: string }, Recommendation[]>> => {
  const embedding = (await loadEmbedding(EmbeddingName.Model)).slice(0, RECOMMENDATION_MODEL_CORPUS_SIZE);

  let profileData: {
    profile: CompatAnimeListEntry[];
    username: string;
  };
  if (dataSource.type === 'username') {
    const rankingsRes = await fetchUserRankings(dataSource.username, profileSource)();
    if (isLeft(rankingsRes)) {
      return rankingsRes;
    }
    profileData = { ...rankingsRes.right, username: dataSource.username };
  } else if (dataSource.type === 'rawProfile') {
    const profile = dataSource.profile.map(({ animeID, score }) => ({
      node: { id: animeID },
      list_status: { status: AnimeListStatusCode.Completed, score },
    }));

    profileData = {
      profile,
      username: '<raw profile>',
    };
  } else {
    throw new Error(`Unknown data source type ${(dataSource as any).type}`);
  }
  const { profile, username } = profileData;

  if (excludedRankingAnimeIDs.size > 0) {
    console.log(`Excluding ${excludedRankingAnimeIDs.size} rankings`);
  }

  console.log(`Generating recommendations for user ${username} with ${profile.length} profile entries...`);
  const output = await performInferrence({
    modelName,
    profile: profile.filter((entry) => !excludedRankingAnimeIDs.has(entry.node.id)),
    count,
    computeContributions,
    logitWeight,
    nicheBoostFactor,
  });

  // Build a map from anime_id to embedding index for filtering
  const animeIdToEmbeddingIndex = new Map<number, number>();
  for (let i = 0; i < embedding.length; i++) {
    animeIdToEmbeddingIndex.set(embedding[i].metadata.id, i);
  }

  // Build valid media types set for filtering
  const validAnimeMediaTypes = new Set<AnimeMediaType>();
  validAnimeMediaTypes.add(AnimeMediaType.TV);
  validAnimeMediaTypes.add(AnimeMediaType.Unknown);
  if (includeONAsOVAsSpecials) {
    validAnimeMediaTypes.add(AnimeMediaType.OVA);
    validAnimeMediaTypes.add(AnimeMediaType.Special);
    validAnimeMediaTypes.add(AnimeMediaType.ONA);
  }
  if (includeMovies) {
    validAnimeMediaTypes.add(AnimeMediaType.Movie);
  }
  if (includeMusic) {
    validAnimeMediaTypes.add(AnimeMediaType.Music);
  }

  let seasonRelationshipsByAnimeID: Map<number, Set<number>> | null = null;
  let excludedFromSeasonRelationshipExclusionAnimeIDs: Set<number> | null = null;
  if (!includeExtraSeasons) {
    const ratedAnimeIDs = profile.map((entry) => entry.node.id);
    const res = await buildSeasonRelationshipsByAnimeID(ratedAnimeIDs);
    seasonRelationshipsByAnimeID = res.seasonRelationshipsByAnimeID;
    excludedFromSeasonRelationshipExclusionAnimeIDs = res.forceRetainedAnimeIDs;
  }

  const embeddingMetadata = excludedGenreIDs.size > 0 ? await getEmbeddingMetadata(embedding) : [];

  const allWatchedAnimeIDs = new Set(
    profile
      .filter((entry) => entry.list_status.status !== AnimeListStatusCode.PlanToWatch)
      .map((entry) => entry.node.id)
  );
  const planToWatchAnimeIDs = new Set(
    profile
      .filter((entry) => entry.list_status.status === AnimeListStatusCode.PlanToWatch)
      .map((entry) => entry.node.id)
  );

  // Filter recommendations based on media type, genre exclusion, and season relationships
  const filteredRecommendations = output.recommendations.filter((rec) => {
    const embeddingIndex = animeIdToEmbeddingIndex.get(rec.anime_id);
    if (embeddingIndex === undefined) {
      return false;
    }

    // Filter out plan to watch items if requested
    if (filterPlanToWatch && planToWatchAnimeIDs.has(rec.anime_id)) {
      return false;
    }

    const datum = embedding[embeddingIndex];
    const animeMediaType = datum.metadata.media_type;
    if (animeMediaType && !validAnimeMediaTypes.has(animeMediaType)) {
      return false;
    }

    if (excludedGenreIDs.size > 0) {
      const metadatum = embeddingMetadata[embeddingIndex];
      if (metadatum && metadatum.genres?.some((genre) => excludedGenreIDs.has(genre.id))) {
        return false;
      }
    }

    if (seasonRelationshipsByAnimeID) {
      if (!excludedFromSeasonRelationshipExclusionAnimeIDs?.has(datum.metadata.id)) {
        const isExtraSeason = getIsExtraSeasonOfRatedAnime(
          embedding,
          animeIdToEmbeddingIndex,
          allWatchedAnimeIDs,
          seasonRelationshipsByAnimeID,
          datum.metadata.id
        );
        if (isExtraSeason) {
          return false;
        }
      }
    }

    return true;
  });

  // Build a map of profile entries by anime ID for contributor sign determination
  const profileAnimeByID = new Map(profile.map((entry) => [entry.node.id, entry]));

  // Determine if user is a non-rater (most scores are 0)
  const ratedCount = profile.filter((entry) => entry.list_status.score > 0).length;
  const userIsNonRater = ratedCount < profile.length * 0.5;

  return right(
    filteredRecommendations.slice(0, count).map((rec) => {
      const reco: Recommendation = { id: rec.anime_id, score: rec.score };

      if (rec.top_contributors) {
        reco.topRatingContributorsIds = rec.top_contributors.map((contrib) => {
          const rating = profileAnimeByID.get(contrib.anime_id);
          // Positive contribution if score >= 6, or if user is a non-rater
          const isPositive = (rating?.list_status.score ?? 10) >= 6;
          return isPositive || userIsNonRater ? contrib.anime_id : -contrib.anime_id;
        });
      }

      if (planToWatchAnimeIDs.has(rec.anime_id)) {
        reco.planToWatch = true;
      }

      return reco;
    })
  );
};

const AllProfileSources: { [key in ProfileSource]: ProfileSource } = {
  [ProfileSource.AniList]: ProfileSource.AniList,
  [ProfileSource.MyAnimeList]: ProfileSource.MyAnimeList,
};

export const ProfileSourceValidator = t.keyof(AllProfileSources);
