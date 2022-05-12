import type { RequestHandler } from '@sveltejs/kit';
import { isLeft } from 'fp-ts/lib/Either.js';
import { ModelName, validateModelName } from 'src/components/recommendation/conf';

import { getAnimeByID, type AnimeDetails } from 'src/malAPI';
import { getRecommendations, type Recommendation } from './recommendation';

export type RecommendationsResponse =
  | {
      type: 'ok';
      recommendations: Recommendation[];
      animeData: { [id: number]: AnimeDetails };
    }
  | { type: 'error'; error: string };

export const get: RequestHandler = async ({ params, url }) => {
  const username = params.username;
  const modelName = validateModelName(url.searchParams.get('model') ?? ModelName.Model_4K_V2);
  if (!modelName) {
    return { status: 400, message: 'Invalid model name' };
  }

  const recommendationsRes = await getRecommendations(username, 20, true, modelName);
  if (isLeft(recommendationsRes)) {
    return { status: 500, body: { recommendations: { type: 'error', error: recommendationsRes.left.body } } };
  }
  const recommendationsList = recommendationsRes.right;

  // TODO: Use DB rather than MAL API
  const animeData = (
    await Promise.all(
      [
        ...new Set(
          recommendationsList.flatMap(({ id, topRatingContributorsIds }) => [id, ...topRatingContributorsIds])
        ),
      ].map((id) => getAnimeByID(id))
    )
  ).reduce((acc, details) => {
    acc[details.id] = details;
    return acc;
  }, {} as { [id: number]: AnimeDetails });

  const recommendations: RecommendationsResponse = {
    type: 'ok',
    recommendations: recommendationsList.filter((reco) => !!animeData[reco.id]),
    animeData,
  };

  return {
    status: 200,
    body: { recommendations } as any,
  };
};
