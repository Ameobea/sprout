import type { RequestHandler } from '@sveltejs/kit';
import * as tf from '@tensorflow/tfjs-node';
import { type Either, isLeft, right } from 'fp-ts/lib/Either.js';
import { tryCatchK } from 'fp-ts/lib/TaskEither.js';
import { performance } from 'perf_hooks';
import * as t from 'io-ts';
import { PathReporter } from 'io-ts/lib/PathReporter.js';
import { Worker, isMainThread } from 'node:worker_threads';

import { ModelName, RECOMMENDATION_MODEL_CORPUS_SIZE, validateModelName } from 'src/components/recommendation/conf';
import { loadEmbedding } from 'src/embedding';
import {
  AnimeListStatusCode,
  AnimeMediaType,
  getAnimeByID,
  getUserAnimeList,
  type AnimeDetails,
  type MALUserAnimeListItem,
} from 'src/malAPI';
import { DataContainer, scoreRating } from 'src/training/data';
import { EmbeddingName } from 'src/types';
import type { Embedding } from '../embedding';
import { loadRecommendationModel } from './model';
import { convertMALProfileToTrainingData, type TrainingDatum } from './training/trainingData';

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

const fetchUserRankings = tryCatchK(
  async (username: string): Promise<{ profile: MALUserAnimeListItem[]; ratings: TrainingDatum[] }> => {
    const profile: MALUserAnimeListItem[] = await getUserAnimeList(username);
    if (!Array.isArray(profile)) {
      console.error('Unexpected response from /mal-profile', profile);
      throw new Error('Failed to fetch user profile from MyAnimeList');
    }

    const ratings = (await convertMALProfileToTrainingData([profile]))[0];
    return { profile, ratings };
  },
  (err: Error) => {
    console.error('Failed to fetch user rankings', err);
    return { status: 500, body: err.message };
  }
);

const computeRecommendationContributionsInner = async (
  model: tf.LayersModel,
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

    const batchInputTensor = tf.tensor(batchInputs, [batchCombos.length, RECOMMENDATION_MODEL_CORPUS_SIZE]);
    console.log(`Running batch ${batchIx + 1}/${batches} with ${batchCombos.length} inputs...`);
    const now = performance.now();
    const batchOutputsTensor = model.predict(batchInputTensor) as tf.Tensor;
    const elapsed = performance.now() - now;
    console.log(`Model ran in ${elapsed}ms`);
    console.log(`Batch ${batchIx + 1}/${batches} complete`);

    const batchOutputs = (await batchOutputsTensor.data()) as Float32Array;
    tf.dispose(batchInputTensor);
    tf.dispose(batchOutputsTensor);

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
  model: tf.LayersModel,
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
    model,
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
        model,
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

interface GetRecommendationsArgs {
  username: string;
  count: number;
  computeContributions: boolean;
  modelName: ModelName;
  excludedRankingAnimeIDs: Set<number>;
  includeONAsOVAsSpecials: boolean;
  includeMovies: boolean;
  includeMusic: boolean;
}

export const getRecommendations = async ({
  username,
  count,
  computeContributions,
  modelName,
  excludedRankingAnimeIDs,
  includeONAsOVAsSpecials,
  includeMovies,
  includeMusic,
}: GetRecommendationsArgs): Promise<Either<{ status: number; body: string }, Recommendation[]>> => {
  const embedding = (await loadEmbedding(EmbeddingName.PyMDE_3D_40N)).slice(0, RECOMMENDATION_MODEL_CORPUS_SIZE);
  const model = await loadRecommendationModel(embedding, modelName);

  const rankingsRes = await fetchUserRankings(username)();
  if (isLeft(rankingsRes)) {
    return rankingsRes;
  }
  const { profile, ratings: rawRatings } = rankingsRes.right;
  if (excludedRankingAnimeIDs.size > 0) {
    console.log(`Excluding ${excludedRankingAnimeIDs.size} rankings`);
  }
  const ratings = rawRatings.filter((rating) => {
    const animeID = embedding[rating.animeIx]?.metadata.id;
    return !!animeID && !excludedRankingAnimeIDs.has(animeID);
  });
  const profileAnimeByID = new Map<number, MALUserAnimeListItem>();
  const planToWatchAnimeIDs = new Set<number>();
  for (const entry of profile) {
    if (entry?.node?.id) {
      profileAnimeByID.set(entry.node.id, entry);
    }
    if (entry?.list_status.status === AnimeListStatusCode.PlanToWatch) {
      planToWatchAnimeIDs.add(entry.node.id);
    }
  }

  const { input, validIndices } = DataContainer.buildModelInput(ratings, RECOMMENDATION_MODEL_CORPUS_SIZE);
  const inputTensor = tf.tensor([input], [1, RECOMMENDATION_MODEL_CORPUS_SIZE]);

  console.log(`Running recommendation model for user ${username} with ${validIndices.length} anime in input...`, {
    posRatingCount: input.filter((score) => score > 0).length,
    negRatingCount: input.filter((score) => score < 0).length,
  });
  const now = performance.now();
  const outputTensor = (await model.predict(inputTensor)) as tf.Tensor;
  const elapsed = performance.now() - now;
  console.log(`Model ran in ${elapsed}ms`);
  console.log(outputTensor.shape);

  const output = (await outputTensor.data()) as Float32Array;
  tf.dispose(inputTensor);
  tf.dispose(outputTensor);
  const allInputAnimeIndices = new Set(validIndices.map((ixIx) => ratings[ixIx].animeIx));

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
  const sortedOutput = [...output]
    .map((score, animeIx) => {
      const animeMediaType = embedding[animeIx].metadata.media_type;
      if (animeMediaType && !validAnimeMediaTypes.has(animeMediaType)) {
        return { score: -2, animeIx };
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
        model,
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
          const isPositive = scoreRating(rating?.list_status.score ?? 10) > 0;
          return isPositive ? contribAnimeID : -contribAnimeID;
        });
      }
      if (planToWatchAnimeIDs.has(id)) {
        reco.planToWatch = true;
      }
      return reco;
    })
  );
};

const RecommendationRequest = t.type({
  availableAnimeMetadataIDs: t.array(t.number),
  username: t.string,
  excludedRankingAnimeIDs: t.array(t.number),
  modelName: t.string,
  includeContributors: t.boolean,
  includeONAsOVAsSpecials: t.boolean,
  includeMovies: t.boolean,
  includeMusic: t.boolean,
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
  const { includeONAsOVAsSpecials, includeMovies, includeMusic } = req;

  const recommendationsRes = await getRecommendations({
    username: req.username,
    count: 20,
    computeContributions: req.includeContributors,
    modelName,
    excludedRankingAnimeIDs: new Set(req.excludedRankingAnimeIDs),
    includeONAsOVAsSpecials,
    includeMovies,
    includeMusic,
  });
  if (isLeft(recommendationsRes)) {
    return recommendationsRes.left;
  }
  const recommendationsList = recommendationsRes.right;

  // TODO: Use DB rather than MAL API
  const alreadyFetchedAnimeIDs = new Set(req.availableAnimeMetadataIDs);
  const animeData = (
    await Promise.all(
      [
        ...new Set(
          recommendationsList
            .flatMap(({ id, topRatingContributorsIds }) =>
              topRatingContributorsIds ? [id, ...topRatingContributorsIds.map(Math.abs)] : id
            )
            .filter((id) => !alreadyFetchedAnimeIDs.has(id))
        ),
      ].map((id) => getAnimeByID(id))
    )
  ).reduce((acc, details) => {
    acc[details.id] = details;
    return acc;
  }, {} as { [id: number]: AnimeDetails });

  return { status: 200, body: { recommendations: recommendationsList, animeData } };
};
