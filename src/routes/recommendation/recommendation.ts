import type { RequestHandler } from '@sveltejs/kit';
import { type Either, isLeft, right } from 'fp-ts/lib/Either.js';
import { performance } from 'perf_hooks';
import * as t from 'io-ts';
import { PathReporter } from 'io-ts/lib/PathReporter.js';

import {
  getIsModelScoresWeighted,
  ModelName,
  ProfileSource,
  RECOMMENDATION_MODEL_CORPUS_SIZE,
  validateModelName,
} from 'src/components/recommendation/conf';
import { loadEmbedding } from 'src/embedding';
import { AnimeListStatusCode, AnimeMediaType, AnimeRelationType, getAnimesByID, type AnimeDetails } from 'src/malAPI';
import { DataContainer, scoreRating } from 'src/training/data';
import { EmbeddingName } from 'src/types';
import type { Embedding } from '../embedding';
import { performInferrence } from './inferrenceThreadPool';
import { convertMALProfileToTrainingData, type TrainingDatum } from './training/trainingData';
import { fetchUserRankings } from 'src/helpers';
import type { CompatAnimeListEntry } from 'src/anilistAPI';
import { typify } from 'src/components/recommendation/utils';

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

interface RecommendationWithIndex extends Recommendation {
  animeIx: number;
}

const maxK = 3;

const computeRecommendationContributionsInner = async (
  modelName: ModelName,
  combosToCheck: number[][],
  minOutputByRecommendationIx: number[],
  minComboByRecommendationIx: (number[] | null)[],
  input: number[],
  recommendations: RecommendationWithIndex[],
  validRatings: TrainingDatum[],
  fastMode: boolean
) => {
  const outputs: Float32Array[] = [];

  const batchSize = 1024 * 16;
  const batches = Math.ceil(combosToCheck.length / batchSize);
  for (let batchIx = 0; batchIx < batches; batchIx++) {
    const start = batchIx * batchSize;
    const end = Math.min(start + batchSize, combosToCheck.length);
    const batchCombos = combosToCheck.slice(start, end);

    const batchInputs = new Float32Array(batchCombos.length * RECOMMENDATION_MODEL_CORPUS_SIZE);
    for (let comboIx = 0; comboIx < batchCombos.length; comboIx++) {
      const combo = batchCombos[comboIx];
      for (let i = 0; i < RECOMMENDATION_MODEL_CORPUS_SIZE; i++) {
        batchInputs[comboIx * RECOMMENDATION_MODEL_CORPUS_SIZE + i] = input[i];
      }

      // Hold out ratings according to the combo
      combo.forEach((ratingIx) => {
        const animeIx = validRatings[ratingIx].animeIx;
        batchInputs[comboIx * RECOMMENDATION_MODEL_CORPUS_SIZE + animeIx] = 0;
      });
    }

    console.log(`Running batch ${batchIx + 1}/${batches} with ${batchCombos.length} inputs...`);
    const now = performance.now();
    const batchOutputs = await performInferrence(modelName, batchInputs);
    const elapsed = performance.now() - now;
    console.log(`Model ran in ${elapsed.toFixed(2)}ms`);
    console.log(`Batch ${batchIx + 1}/${batches} complete`);

    if (fastMode) {
      outputs.push(batchOutputs);
    } else {
      for (let comboIx = 0; comboIx < batchCombos.length; comboIx++) {
        const combo = batchCombos[comboIx];
        const output = batchOutputs.subarray(
          comboIx * RECOMMENDATION_MODEL_CORPUS_SIZE,
          (comboIx + 1) * RECOMMENDATION_MODEL_CORPUS_SIZE
        );

        recommendations.forEach(({ animeIx }, recommendationIx) => {
          const animeOutput = output[animeIx];
          if (animeOutput < minOutputByRecommendationIx[recommendationIx]) {
            minOutputByRecommendationIx[recommendationIx] = animeOutput;
            minComboByRecommendationIx[recommendationIx] = combo;
          }
        });
      }
    }
  }

  if (fastMode) {
    recommendations.forEach(({ animeIx }, recommendationIx) => {
      const outputsForRecommendation = outputs
        .flatMap((batchOutputs) => {
          const batchSize = batchOutputs.length / RECOMMENDATION_MODEL_CORPUS_SIZE;
          // make sure batchSize is an integer as a sanity check
          if (batchSize % 1 !== 0) {
            throw new Error('Unexpected batch size');
          }

          const outputsForRecommendation: number[] = [];
          for (let predIx = 0; predIx < batchSize; predIx++) {
            const output = batchOutputs[predIx * RECOMMENDATION_MODEL_CORPUS_SIZE + animeIx];
            outputsForRecommendation.push(output);
          }
          return outputsForRecommendation;
        })
        .map((output, comboIx) => ({ output, comboIx }));

      // Find `maxK` combos which produced lowest outputs for this recommendation
      const topK = outputsForRecommendation.sort((a, b) => a.output - b.output).slice(0, maxK);
      const combo = topK.map(({ comboIx }) => {
        const combo = combosToCheck[comboIx];
        if (combo.length !== 1) {
          throw new Error('Unexpected combo length');
        }
        return combo[0];
      });
      minComboByRecommendationIx[recommendationIx] = combo;
      minOutputByRecommendationIx[recommendationIx] = topK[0].output;
    });
  }
};

/**
 * For each recommended anime, find the top k ratings from the user's anime list that contribute the most to it.
 * This is accomplished by re-running the recommendation model on the user's anime list and holding out all sets
 * of `k` ratings from the user's anime list and finding which ratings contribute the most to the predicted rating.
 */
const computeRecommendationContributions = async (
  modelName: ModelName,
  embedding: Embedding,
  input: number[],
  recommendations: RecommendationWithIndex[],
  validRatings: TrainingDatum[],
  fastMode: boolean
) => {
  if (recommendations.length === 0 || validRatings.length < 3) {
    return;
  }

  // Run once with single-rating combos.  Although it's technically possible for the lowest score for a k-item combo
  // to not include the rating from the the lowest 1-item combo, it is unlikely and the performance cost is too high
  // to check them all.
  const firstRoundCombos = new Array(validRatings.length).fill(null).map((_, i) => [i]);

  const minOutputByRecommendationIx: number[] = new Array(recommendations.length).fill(1);
  const minComboByRecommendationIx: (number[] | null)[] = new Array(recommendations.length).fill(null);

  await computeRecommendationContributionsInner(
    modelName,
    firstRoundCombos,
    minOutputByRecommendationIx,
    minComboByRecommendationIx,
    input,
    recommendations,
    validRatings,
    fastMode
  );

  if (!fastMode) {
    for (let k = 2; k <= maxK; k++) {
      const hashCombo = (combo: number[]): string => combo.join('-');
      const combosToCheck: number[][] = [];
      const allComboHashes: Set<string> = new Set();

      for (const minCombo of minComboByRecommendationIx) {
        if (!minCombo) {
          continue;
        }

        validRatings.forEach((_rating, ratingIx) => {
          if (minCombo.includes(ratingIx)) {
            return;
          }
          const newCombo = [...minCombo, ratingIx];
          const comboHash = hashCombo(newCombo);
          if (allComboHashes.has(comboHash)) {
            return;
          }
          allComboHashes.add(comboHash);
          combosToCheck.push(newCombo);
        });
      }

      await computeRecommendationContributionsInner(
        modelName,
        combosToCheck,
        minOutputByRecommendationIx,
        minComboByRecommendationIx,
        input,
        recommendations,
        validRatings,
        false
      );
    }
  }

  return recommendations.map((_, recoIx) => {
    const minCombo = minComboByRecommendationIx[recoIx];
    if (!minCombo) {
      return [];
    }
    return minCombo.map((ratingIx) => embedding[validRatings[ratingIx].animeIx].metadata.id);
  });
};

/**
 * Weights ratings of less-popular anime more heavily.
 */
const attenuateRecommendationOutputs = (outputs: Float32Array, boostFactor: number): Float32Array => {
  for (let i = 0; i < outputs.length; i++) {
    // May need to tweak this.
    const boost = Math.log(Math.E + i * boostFactor);
    outputs[i] *= boost;
  }
  return outputs;
};

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
  popularityAttenuationFactor: number;
  profileSource: ProfileSource;
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

  const embedding = (await loadEmbedding(EmbeddingName.PyMDE_3D_40N)).slice(0, RECOMMENDATION_MODEL_CORPUS_SIZE);
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
  allInputAnimeIDs: Set<number>,
  seasonRelationshipsByAnimeID: Map<number, Set<number>>,
  animeID: number,
  checkedIDs: Set<number> = new Set(),
  depth = 0
) => {
  const extraSeasonIDs = seasonRelationshipsByAnimeID.get(animeID);
  if (!extraSeasonIDs) {
    return false;
  }

  for (const extraSeasonAnimeID of extraSeasonIDs) {
    if (allInputAnimeIDs.has(extraSeasonAnimeID)) {
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
      allInputAnimeIDs,
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
  popularityAttenuationFactor,
  profileSource,
}: GetRecommendationsArgs): Promise<Either<{ status: number; body: string }, Recommendation[]>> => {
  const embedding = (await loadEmbedding(EmbeddingName.PyMDE_3D_40N)).slice(0, RECOMMENDATION_MODEL_CORPUS_SIZE);

  let profileData: {
    profile: CompatAnimeListEntry[];
    ratings: TrainingDatum[];
    userIsNonRater: boolean;
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
    const { ratings, userIsNonRater } = (await convertMALProfileToTrainingData([profile]))[0];

    profileData = {
      profile,
      ratings,
      userIsNonRater,
      username: '<raw profile>',
    };
  } else {
    throw new Error(`Unknown data source type ${(dataSource as any).type}`);
  }
  const { profile, ratings: rawRatings, userIsNonRater, username } = profileData;

  if (excludedRankingAnimeIDs.size > 0) {
    console.log(`Excluding ${excludedRankingAnimeIDs.size} rankings`);
  }
  const ratings = rawRatings.filter((rating) => {
    const animeID = embedding[rating.animeIx]?.metadata.id;
    return !!animeID && !excludedRankingAnimeIDs.has(animeID);
  });
  const profileAnimeByID = new Map<number, CompatAnimeListEntry>();
  const planToWatchAnimeIDs = new Set<number>();
  for (const entry of profile) {
    if (entry?.node?.id) {
      profileAnimeByID.set(entry.node.id, entry);
    }
    if (entry?.list_status.status === AnimeListStatusCode.PlanToWatch) {
      planToWatchAnimeIDs.add(entry.node.id);
    }
  }

  const weightScores = getIsModelScoresWeighted(modelName);
  const { input, validIndices } = DataContainer.buildModelInput(
    ratings,
    RECOMMENDATION_MODEL_CORPUS_SIZE,
    weightScores
  );

  console.log(`Running recommendation model for user ${username} with ${validIndices.length} anime in input...`, {
    posRatingCount: input.filter((score) => score > 0).length,
    negRatingCount: input.filter((score) => score < 0).length,
  });
  const now = performance.now();
  let output = await performInferrence(modelName, new Float32Array(input));
  const elapsed = performance.now() - now;
  console.log(`Model ran in ${elapsed.toFixed(2)}ms`);

  if (popularityAttenuationFactor) {
    output = attenuateRecommendationOutputs(output, popularityAttenuationFactor);
  }

  const allInputAnimeIndices = new Set(validIndices.map((ixIx) => ratings[ixIx].animeIx));
  const allInputAnimeIDs = new Set(validIndices.map((ixIx) => embedding[ratings[ixIx].animeIx].metadata.id));

  // Sort output by indices of top recommendations from highest to lowest
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

  const sortedOutput = [...output]
    .map((score, animeIx) => {
      const datum = embedding[animeIx];
      const animeMediaType = datum.metadata.media_type;
      if (animeMediaType && !validAnimeMediaTypes.has(animeMediaType)) {
        return { score: -Infinity, animeIx };
      }

      if (excludedGenreIDs.size > 0) {
        const metadatum = embeddingMetadata[animeIx];
        if (metadatum && metadatum.genres?.some((genre) => excludedGenreIDs.has(genre.id))) {
          return { score: -Infinity, animeIx };
        }
      }

      if (seasonRelationshipsByAnimeID) {
        if (!excludedFromSeasonRelationshipExclusionAnimeIDs?.has(datum.metadata.id)) {
          const isExtraSeason = getIsExtraSeasonOfRatedAnime(
            allInputAnimeIDs,
            seasonRelationshipsByAnimeID,
            datum.metadata.id
          );
          if (isExtraSeason) {
            return { score: -Infinity, animeIx };
          }
        }
      }

      return { score, animeIx };
    })
    .sort((a, b) => b.score - a.score);

  const recommendations: RecommendationWithIndex[] = sortedOutput
    .filter(({ animeIx, score }) => !allInputAnimeIndices.has(animeIx) && score > 0)
    .map(({ animeIx, score }) => ({ animeIx, id: embedding[animeIx].metadata.id, score: +score.toFixed(3) }))
    .filter(({ id }) => {
      const entry = profileAnimeByID.get(id);
      if (!entry) {
        return true;
      }
      const watchStatus = entry.list_status.status;
      switch (watchStatus) {
        case AnimeListStatusCode.Completed:
        case AnimeListStatusCode.Dropped:
        case AnimeListStatusCode.Watching:
          return false;
        case AnimeListStatusCode.OnHold:
        case AnimeListStatusCode.PlanToWatch:
          return true;
        default:
          console.error(`Unknown watch status ${watchStatus}`);
          return false;
      }
    })
    .slice(0, count);

  let contributions: number[][] = [];
  if (computeContributions) {
    contributions =
      (await computeRecommendationContributions(
        modelName,
        embedding,
        input,
        recommendations,
        validIndices.map((ratingIx) => ratings[ratingIx]),
        true
      )) ?? [];
  }

  return right(
    recommendations.map(({ score, id }, i) => {
      const reco: Recommendation = { id, score };
      const contribs = contributions[i];
      if (contribs) {
        reco.topRatingContributorsIds = contribs.map((contribAnimeID) => {
          const rating = profileAnimeByID.get(contribAnimeID);
          const isPositive = scoreRating(rating?.list_status.score ?? 10, weightScores) > 0;
          return isPositive || userIsNonRater ? contribAnimeID : -contribAnimeID;
        });
      }
      if (planToWatchAnimeIDs.has(id)) {
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

const RecommendationRequest = t.type({
  availableAnimeMetadataIDs: t.array(t.number),
  dataSource: t.union([
    t.type({
      type: t.literal('username'),
      username: t.string,
    }),
    t.type({
      type: t.literal('rawProfile'),
      profile: t.array(t.type({ animeID: t.number, score: t.number })),
    }),
  ]),
  excludedRankingAnimeIDs: t.array(t.number),
  excludedGenreIDs: t.array(t.number),
  modelName: t.string,
  includeContributors: t.boolean,
  includeExtraSeasons: t.boolean,
  includeONAsOVAsSpecials: t.boolean,
  includeMovies: t.boolean,
  includeMusic: t.boolean,
  popularityAttenuationFactor: t.number,
  profileSource: ProfileSourceValidator,
});

export const post: RequestHandler = async ({ request }) => {
  const parsed = RecommendationRequest.decode(await request.json());
  if (isLeft(parsed)) {
    const errors = PathReporter.report(parsed);
    return { status: 400, body: errors.join(', ') };
  }
  const req = parsed.right;

  const modelName = validateModelName(req.modelName);
  if (!modelName) {
    return { status: 400, body: 'Invalid modelName' };
  }
  const {
    dataSource,
    includeExtraSeasons,
    includeONAsOVAsSpecials,
    includeMovies,
    includeMusic,
    popularityAttenuationFactor,
    profileSource,
  } = req;

  const recommendationsRes = await getRecommendations({
    dataSource,
    count: 50,
    computeContributions: req.includeContributors,
    modelName,
    excludedRankingAnimeIDs: new Set(req.excludedRankingAnimeIDs),
    excludedGenreIDs: new Set(req.excludedGenreIDs),
    includeExtraSeasons,
    includeONAsOVAsSpecials,
    includeMovies,
    includeMusic,
    popularityAttenuationFactor,
    profileSource,
  });
  if (isLeft(recommendationsRes)) {
    return recommendationsRes.left;
  }
  const recommendationsList = recommendationsRes.right;

  const alreadyFetchedAnimeIDs = new Set(req.availableAnimeMetadataIDs);
  const idsToFetch = [
    ...new Set(
      recommendationsList
        .flatMap(({ id, topRatingContributorsIds }) =>
          topRatingContributorsIds ? [id, ...topRatingContributorsIds.map(Math.abs)] : id
        )
        .filter((id) => !alreadyFetchedAnimeIDs.has(id))
    ),
  ];
  const animeData = (await getAnimesByID(idsToFetch)).reduce((acc, details) => {
    if (!details) {
      return acc;
    }

    acc[details.id] = details;
    return acc;
  }, {} as { [id: number]: AnimeDetails });

  return { status: 200, body: { recommendations: typify(recommendationsList), animeData: typify(animeData) } };
};
