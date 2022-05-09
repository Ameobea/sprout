import type { RequestHandler } from '@sveltejs/kit';
import { isLeft } from 'fp-ts/Either';
import { getAnimeByID, type AnimeDetails } from 'src/malAPI';

import { getRecommendations } from './recommendation';

export const get: RequestHandler = async ({ params }) => {
  const username = params.username;
  const recommendationsRes = await getRecommendations(username, true);
  if (isLeft(recommendationsRes)) {
    return { status: 500, body: { recommendations: { type: 'error', error: recommendationsRes.left.body } } };
  }
  const recommendations = recommendationsRes.right.slice(0, 20);

  // TODO: Use DB rather than MAL API
  const animeData = (
    await Promise.all(
      recommendations.flatMap(({ id, topRatingContributorsIds }) =>
        [id, ...topRatingContributorsIds].map((id) => getAnimeByID(id))
      )
    )
  ).reduce((acc, details) => {
    acc[details.id] = details;
    return acc;
  }, {} as { [id: number]: AnimeDetails });

  return {
    status: 200,
    body: {
      recommendations: {
        type: 'ok',
        recommendations: recommendations.filter((reco) => !!animeData[reco.id]),
        animeData,
      },
    } as any,
  };
};
