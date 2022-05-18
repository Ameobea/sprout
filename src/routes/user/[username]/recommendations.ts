import type { RequestHandler } from '@sveltejs/kit';
import { isLeft } from 'fp-ts/lib/Either.js';
import * as t from 'io-ts';
import { PathReporter } from 'io-ts/lib/PathReporter.js';

import { ModelName, validateModelName } from 'src/components/recommendation/conf';
import { getAnimeByID, type AnimeDetails } from 'src/malAPI';
import { getRecommendations, type Recommendation } from '../../recommendation/recommendation';

export type RecommendationsResponse =
  | {
      type: 'ok';
      recommendations: Recommendation[];
      animeData: { [id: number]: AnimeDetails };
    }
  | { type: 'error'; error: string };

const ExcludedAnimeIDs = t.array(t.number);

export const get: RequestHandler = async ({ params, url }) => {
  const username = params.username;
  if (username === '_') {
    return {
      status: 200,
      body: {
        initialRecommendations: {
          type: 'ok',
          recommendations: [],
          animeData: {},
        },
      },
    };
  }

  const modelName = validateModelName(url.searchParams.get('model') ?? ModelName.Model_4K_V2);
  if (!modelName) {
    return { status: 400, body: 'Invalid model name' };
  }

  const rawExcludedRankingAnimeIDs = url.searchParams.getAll('eid');
  const parsedExcludedRankingAnimeIDs = ExcludedAnimeIDs.decode(
    Array.isArray(rawExcludedRankingAnimeIDs) ? rawExcludedRankingAnimeIDs.map((x) => +x) : rawExcludedRankingAnimeIDs
  );
  if (isLeft(parsedExcludedRankingAnimeIDs)) {
    console.error(`Invalid \`eid\` params: ${rawExcludedRankingAnimeIDs}`);
    const errors = PathReporter.report(parsedExcludedRankingAnimeIDs);
    return { status: 400, body: errors.join(', ') };
  }
  const excludedRankingAnimeIDs = parsedExcludedRankingAnimeIDs.right;

  const recommendationsRes = await getRecommendations({
    username,
    count: 20,
    computeContributions: false,
    modelName,
    excludedRankingAnimeIDs: new Set(excludedRankingAnimeIDs),
  });
  if (isLeft(recommendationsRes)) {
    return { status: 500, body: { recommendations: { type: 'error', error: recommendationsRes.left.body } } };
  }
  const recommendationsList = recommendationsRes.right;

  const animeIdsNeedingMetadata = new Set(
    recommendationsList.flatMap(({ id, topRatingContributorsIds }) => [
      id,
      ...(topRatingContributorsIds ?? []).map(Math.abs),
    ])
  );
  for (const animeID of excludedRankingAnimeIDs) {
    animeIdsNeedingMetadata.add(animeID);
  }

  // TODO: Use DB rather than MAL API
  const animeData = (await Promise.all([...animeIdsNeedingMetadata].map((id) => getAnimeByID(id)))).reduce(
    (acc, details) => {
      acc[details.id] = details;
      return acc;
    },
    {} as { [id: number]: AnimeDetails }
  );

  const recommendations: RecommendationsResponse = {
    type: 'ok',
    recommendations: recommendationsList.filter((reco) => !!animeData[reco.id]),
    animeData,
  };

  return {
    status: 200,
    body: { initialRecommendations: recommendations },
  };
};
