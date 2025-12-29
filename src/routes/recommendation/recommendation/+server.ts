import { error, json, type RequestHandler } from '@sveltejs/kit';
import { isLeft } from 'fp-ts/lib/Either.js';
import * as t from 'io-ts';
import { PathReporter } from 'io-ts/lib/PathReporter.js';

import { typify } from 'src/components/recommendation/utils';
import { getRecommendations, ProfileSourceValidator } from './recommendation';
import { validateModelName } from 'src/components/recommendation/conf';
import { getAnimesByID, type AnimeDetails } from 'src/malAPI';

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
  profileSource: ProfileSourceValidator,
  filterPlanToWatch: t.boolean,
  logitWeight: t.number,
  nicheBoostFactor: t.number,
});

export const POST: RequestHandler = async ({ request }) => {
  const parsed = RecommendationRequest.decode(await request.json());
  if (isLeft(parsed)) {
    const errors = PathReporter.report(parsed);
    error(400, errors.join(', '));
  }
  const req = parsed.right;

  const modelName = validateModelName(req.modelName);
  if (!modelName) {
    error(400, 'Invalid modelName');
  }
  const {
    dataSource,
    includeExtraSeasons,
    includeONAsOVAsSpecials,
    includeMovies,
    includeMusic,
    profileSource,
    filterPlanToWatch,
    logitWeight,
    nicheBoostFactor,
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
    profileSource,
    filterPlanToWatch,
    logitWeight: Math.max(0, Math.min(1, logitWeight)),
    nicheBoostFactor: Math.max(0, Math.min(1, nicheBoostFactor)),
  });
  if (isLeft(recommendationsRes)) {
    return error(recommendationsRes.left.status, recommendationsRes.left.body);
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
  const animeData = (await getAnimesByID(idsToFetch)).reduce(
    (acc, details) => {
      if (!details) {
        return acc;
      }

      acc[details.id] = details;
      return acc;
    },
    {} as { [id: number]: AnimeDetails }
  );

  return json({ recommendations: typify(recommendationsList), animeData: typify(animeData) });
};
