import { browser } from '$app/environment';
import { replaceState } from '$app/navigation';

/**
 * This is a workaround for an annoying SvelteKit/TypeScript issue:
 * https://github.com/sveltejs/kit/issues/1997
 */
export type Typify<T> = { [K in keyof T]: Typify<T[K]> };

/**
 * This is a workaround for an annoying SvelteKit/TypeScript issue:
 * https://github.com/sveltejs/kit/issues/1997
 */
export const typify = <T>(obj: T): Typify<T> => obj;

import {
  DEFAULT_MODEL_NAME,
  DEFAULT_POPULARITY_ATTENUATION_FACTOR,
  DEFAULT_PROFILE_SOURCE,
  ModelName,
  PopularityAttenuationFactor,
  ProfileSource,
} from './conf';

export interface RecommendationControlParams {
  modelName: ModelName;
  excludedRankingAnimeIDs: number[];
  excludedGenreIDs: number[];
  includeExtraSeasons: boolean;
  includeONAsOVAsSpecials: boolean;
  includeMovies: boolean;
  includeMusic: boolean;
  popularityAttenuationFactor: PopularityAttenuationFactor;
  profileSource: ProfileSource;
}

export const getDefaultRecommendationControlParams = (): RecommendationControlParams => {
  const queryParams = new URLSearchParams(browser ? window.location.search : '');
  return {
    modelName: (queryParams.get('model') as any) ?? DEFAULT_MODEL_NAME,
    excludedRankingAnimeIDs: Array.from(new Set(queryParams.getAll('eid').map((eid) => +eid))),
    excludedGenreIDs: Array.from(new Set(queryParams.getAll('egid').map((egid) => +egid))),
    includeExtraSeasons: queryParams.get('exs') === 'true',
    includeONAsOVAsSpecials: queryParams.get('specials') === 'true',
    includeMovies: queryParams.get('movies') === 'true',
    includeMusic: queryParams.get('music') === 'true',
    popularityAttenuationFactor: +(queryParams.get('apops') ?? DEFAULT_POPULARITY_ATTENUATION_FACTOR),
    profileSource: (queryParams.get('source') as ProfileSource | null) ?? ProfileSource.MyAnimeList,
  };
};

export const updateQueryParams = (params: RecommendationControlParams) => {
  if (!browser) {
    return;
  }

  const url = new URL(window.location.toString());
  url.search = '';
  const oldSearchParams = new URLSearchParams(window.location.search).toString();

  if (params.profileSource !== DEFAULT_PROFILE_SOURCE) {
    url.searchParams.set('source', params.profileSource);
  }
  for (const animeID of new Set(params.excludedRankingAnimeIDs)) {
    url.searchParams.append('eid', animeID.toString());
  }
  for (const genreID of new Set(params.excludedGenreIDs)) {
    url.searchParams.append('egid', genreID.toString());
  }
  if (params.modelName !== DEFAULT_MODEL_NAME) {
    url.searchParams.set('model', params.modelName);
  }
  if (params.includeExtraSeasons) {
    url.searchParams.set('exs', 'true');
  }
  if (params.includeONAsOVAsSpecials) {
    url.searchParams.set('specials', 'true');
  }
  if (params.includeMovies) {
    url.searchParams.set('movies', 'true');
  }
  if (params.includeMusic) {
    url.searchParams.set('music', 'true');
  }
  if (params.popularityAttenuationFactor !== DEFAULT_POPULARITY_ATTENUATION_FACTOR) {
    url.searchParams.set('apops', params.popularityAttenuationFactor.toString());
  }

  const newSearchParams = url.searchParams.toString();
  if (newSearchParams !== oldSearchParams) {
    replaceState(url, {});
  }
};

export const filterNils = <T>(arr: (T | null | undefined)[]): T[] =>
  arr.filter((x) => x !== null && x !== undefined) as T[];
