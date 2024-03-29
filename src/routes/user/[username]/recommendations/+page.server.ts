import { error } from '@sveltejs/kit';
import { type Either, isLeft, mapLeft } from 'fp-ts/lib/Either.js';
import * as t from 'io-ts';
import { PathReporter } from 'io-ts/lib/PathReporter.js';

import {
  DEFAULT_MODEL_NAME,
  DEFAULT_POPULARITY_ATTENUATION_FACTOR,
  DEFAULT_PROFILE_SOURCE,
  validateModelName,
} from 'src/components/recommendation/conf';
import { typify } from 'src/components/recommendation/utils';
import { getAnimesByID, type AnimeDetails } from 'src/malAPI';
import {
  getGenresDB,
  getRecommendations,
  ProfileSourceValidator,
  type Recommendation,
} from 'src/routes/recommendation/recommendation/recommendation';
import type { PageServerLoad } from './$types';

export type RecommendationsResponse =
  | {
      type: 'ok';
      recommendations: Recommendation[];
      animeData: { [id: number]: AnimeDetails };
    }
  | { type: 'error'; error: string };

const ExcludedIDs = t.array(t.number);

const parseExcludedIDs = (searchParamKey: string, url: URL): Either<{ status: number; body: string }, number[]> => {
  const rawExcludedIDs = url.searchParams.getAll(searchParamKey);
  const parsedExcludedIDs = ExcludedIDs.decode(
    Array.isArray(rawExcludedIDs) ? rawExcludedIDs.map((x) => +x) : rawExcludedIDs
  );
  return mapLeft<any, { status: number; body: string }>((_err) => {
    console.error(`Invalid \`${searchParamKey}\` params: ${rawExcludedIDs}`);
    const errors = PathReporter.report(parsedExcludedIDs);
    return { status: 400, body: errors.join(', ') };
  })(parsedExcludedIDs);
};

export const load: PageServerLoad = async ({ params, url }) => {
  const username = params.username;
  if (!username || username === '_') {
    return {
      initialRecommendations: {
        type: 'ok',
        recommendations: [],
        animeData: {},
      },
    };
  }

  const modelName = validateModelName(url.searchParams.get('model') ?? DEFAULT_MODEL_NAME);
  if (!modelName) {
    error(400, 'Invalid model name');
  }

  const includeExtraSeasons = url.searchParams.get('exs') === 'true';
  const includeONAsOVAsSpecials = url.searchParams.get('specials') === 'true';
  const includeMovies = url.searchParams.get('movies') === 'true';
  const includeMusic = url.searchParams.get('music') === 'true';
  const popularityAttenuationFactorRaw = url.searchParams.get('apops');
  const popularityAttenuationFactor = popularityAttenuationFactorRaw
    ? +popularityAttenuationFactorRaw
    : DEFAULT_POPULARITY_ATTENUATION_FACTOR;
  if (isNaN(popularityAttenuationFactor)) {
    error(400, 'Invalid popularityAttenuationFactor');
  }
  const rawProfileSource = url.searchParams.get('source') ?? DEFAULT_PROFILE_SOURCE;
  const profileSourceParseRes = ProfileSourceValidator.decode(rawProfileSource);
  if (isLeft(profileSourceParseRes)) {
    error(400, 'Invalid `source` query param');
  }
  const profileSource = profileSourceParseRes.right;

  const excludedRankingAnimeIDsRes = parseExcludedIDs('eid', url);
  if (isLeft(excludedRankingAnimeIDsRes)) {
    return error(excludedRankingAnimeIDsRes.left.status, excludedRankingAnimeIDsRes.left.body);
  }
  const excludedRankingAnimeIDs = excludedRankingAnimeIDsRes.right;

  const excludedGenreIDsRes = parseExcludedIDs('egid', url);
  if (isLeft(excludedGenreIDsRes)) {
    return error(excludedGenreIDsRes.left.status, excludedGenreIDsRes.left.body);
  }
  const excludedGenreIDs = excludedGenreIDsRes.right;

  const recommendationsRes = await getRecommendations({
    dataSource: { type: 'username', username },
    count: 20,
    computeContributions: false,
    modelName,
    excludedRankingAnimeIDs: new Set(excludedRankingAnimeIDs),
    excludedGenreIDs: new Set(excludedGenreIDs),
    includeExtraSeasons,
    includeONAsOVAsSpecials,
    includeMovies,
    includeMusic,
    popularityAttenuationFactor,
    profileSource,
  });
  if (isLeft(recommendationsRes)) {
    return { initialRecommendations: { type: 'error', error: recommendationsRes.left.body } };
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

  const animeData = (await getAnimesByID(Array.from(animeIdsNeedingMetadata))).reduce(
    (acc, details) => {
      if (details) {
        acc[details.id] = details;
      }
      return acc;
    },
    {} as { [id: number]: AnimeDetails }
  );

  const recommendations: RecommendationsResponse = {
    type: 'ok',
    recommendations: recommendationsList.filter((reco) => !!animeData[reco.id]),
    animeData,
  };

  const genreNames: { [id: number]: string } = {};
  if (excludedGenreIDs.length > 0) {
    const genresDB = await getGenresDB();
    for (const genreID of excludedGenreIDs) {
      const genreName = genresDB.get(genreID);
      if (!genreName) {
        console.error(`Could not find genre name for genre ID ${genreID}`);
        continue;
      }
      genreNames[genreID] = genreName;
    }
  }

  return { initialRecommendations: typify(recommendations), genreNames: typify(genreNames) };
};
