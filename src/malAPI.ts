import { MAL_API_BASE_URL, MAL_CLIENT_ID } from './conf';
import { delay } from './util';
import NodeCache from 'node-cache';
import { DbPool } from './dbUtil';
import { AsyncSemaphore } from './util/asyncSemaphore';

export class MALAPIError extends Error {
  public statusCode: number;

  constructor(message: string, statusCode: number) {
    super(message);
    this.name = 'MALAPIError';
    this.statusCode = statusCode;
  }
}

const MAX_CONCURRENT_MAL_API_REQUESTS = 4;
const MALAPIConcurrencyLimiter = new AsyncSemaphore(MAX_CONCURRENT_MAL_API_REQUESTS);

const makeMALRequestInner = async (url: string, retryCount?: number) => {
  const res = await fetch(url, { headers: { 'X-MAL-CLIENT-ID': MAL_CLIENT_ID } });
  if (res.status === 429 || (res.status >= 500 && res.status < 600)) {
    console.log(`Retryable error ${res.status} for ${url}`);
    const retryAfter = res.headers.get('Retry-After');
    if (!retryAfter) {
      await delay(1000);
    } else {
      await delay(+retryAfter * 1000);
    }
    return makeMALRequestInner(url, (retryCount ?? 0) + 1);
  } else if (!res.ok) {
    throw new MALAPIError(`MAL API returned ${res.status}: ${await res.text()}`, res.status);
  }
  return res.json();
};

const makeMALRequest = MALAPIConcurrencyLimiter.wrap(makeMALRequestInner);

export enum AnimeListStatusCode {
  Watching = 'watching',
  Completed = 'completed',
  OnHold = 'on_hold',
  Dropped = 'dropped',
  PlanToWatch = 'plan_to_watch',
}

export interface AnimeBasicDetails {
  id: number;
  title: string;
  main_picture: {
    medium: string;
    large: string;
  };
}

export interface AnimeListStatus {
  status: AnimeListStatusCode;
  score: number;
  num_episodes_watched: number;
  is_rewatching: boolean;
  updated_at: string;
}

export interface MALUserAnimeListItem {
  node: AnimeBasicDetails;
  list_status: AnimeListStatus;
}

export interface MALUserAnimeListResponse {
  data: MALUserAnimeListItem[];
  paging: { next?: string | null };
}

export enum AnimeMediaType {
  Unknown = 'unknown',
  TV = 'tv',
  OVA = 'ova',
  Movie = 'movie',
  Special = 'special',
  ONA = 'ona',
  Music = 'music',
}

export enum MangaListStatusCode {
  Reading = 'reading',
  Completed = 'completed',
  OnHold = 'on_hold',
  Dropped = 'dropped',
  PlanToRead = 'plan_to_read',
}

interface MangaListStatus {
  status: MangaListStatusCode;
  score: number;
  num_volumes_read: number;
  num_chapters_read: number;
  is_rereading: boolean;
  start_date?: string | null;
  finish_date?: string | null;
  updated_at: string;
}

export interface MALUserMangaListItem {
  node: AnimeBasicDetails;
  list_status: MangaListStatus;
}

export interface MALUserMangaListResponse {
  data: MALUserMangaListItem[];
  paging: { next?: string | null };
}

// Cache for 5 minutes
const UserAnimeListCache = new NodeCache({ stdTTL: 60 });
const UserMangaListCache = new NodeCache({ stdTTL: 60 });

export const getUserAnimeList = async (username: string): Promise<MALUserAnimeListItem[]> => {
  const cached: MALUserAnimeListItem[] | undefined = UserAnimeListCache.get(username);
  if (cached) {
    console.log('Found cached user anime list for ' + username);
    return cached;
  }

  const data: MALUserAnimeListItem[] = [];

  let i = 0;
  const pageSize = 1000;
  for (;;) {
    const url = `${MAL_API_BASE_URL}/users/${username}/animelist?nsfw=true&offset=${
      i * pageSize
    }&limit=${pageSize}&fields=list_status`;
    i += 1;
    console.log(`Fetching page ${i}...`, url);

    const res: MALUserAnimeListResponse = await makeMALRequest(url);
    data.push(...res.data);
    if (!res.paging.next) {
      break;
    }
  }
  UserAnimeListCache.set(username, data);

  return data;
};

export const getUserMangaList = async (username: string): Promise<MALUserMangaListItem[]> => {
  const cached: MALUserMangaListItem[] | undefined = UserMangaListCache.get(username);
  if (cached) {
    console.log('Found cached user manga list for ' + username);
    return cached;
  }

  const data: MALUserMangaListItem[] = [];

  let i = 0;
  const pageSize = 1000;
  for (;;) {
    const url = `${MAL_API_BASE_URL}/users/${username}/mangalist?nsfw=true&offset=${
      i * pageSize
    }&limit=${pageSize}&fields=list_status`;
    i += 1;
    console.log(`Fetching page ${i}...`, url);

    const res: MALUserMangaListResponse = await makeMALRequest(url);
    data.push(...res.data);
    if (!res.paging.next) {
      break;
    }
  }
  UserMangaListCache.set(username, data);

  return data;
};

interface Genre {
  id: number;
  name: string;
}

export enum AnimeRelationType {
  Sequel = 'sequel',
  Prequel = 'prequel',
  AlternativeSetting = 'alternative_setting',
  AlternativeVersion = 'alternative_version',
  SideStory = 'side_story',
  ParentStory = 'parent_story',
  Summary = 'summary',
  FullStory = 'full_story',
  Other = 'other',
}

export interface AnimeDetails {
  id: number;
  title: string;
  main_picture: {
    medium: string;
    large: string;
  };
  alternative_titles: {
    synonyms: string[];
    en?: string | null;
    ja?: string | null;
  };
  start_date: string;
  end_date: string;
  synopsis: string;
  recommendations?: {
    node: AnimeBasicDetails;
    num_recommendations: number;
  }[];
  media_type: AnimeMediaType;
  related_anime?: {
    node: AnimeBasicDetails;
    relation_type: AnimeRelationType;
    relation_type_formatted: string;
  }[];
  genres?: Genre[];
}

const AnimeDetailsCache = new NodeCache({ stdTTL: 24 * 60 * 60 * 1000 });

export const fetchAnimeFromMALAPI = async (id: number): Promise<AnimeDetails | null> => {
  const fieldsToFetch = [
    'main_picture',
    'alternative_titles',
    'start_date',
    'end_date',
    'synopsis',
    'genres',
    'recommendations',
    'related_anime',
    'media_type',
  ];
  const url = `${MAL_API_BASE_URL}/anime/${id}?nsfw=true&fields=${fieldsToFetch.join(',')}`;
  console.log('Fetching anime...', url);
  const details = (await makeMALRequest(url).catch((err) => {
    if (err instanceof MALAPIError && err.statusCode === 404) {
      // Don't delete entry from the database if it's there, but do mark it as having been updated to prevent the refresher
      // from trying to fetch it again.
      DbPool.query('UPDATE `anime-metadata` SET update_timestamp = NOW() WHERE id = ?', [id], (err) => {
        if (err) {
          console.error('Failed to update update_timestamp for anime that 404s from MAL API:', id, err);
        }
      });

      return null;
    }
    throw err;
  })) as AnimeDetails | null;
  if (!details) {
    return null;
  }

  AnimeDetailsCache.set(id, details);

  // Update DB in the background
  DbPool.query(
    'INSERT INTO `anime-metadata` (id, metadata) VALUES (?, ?) ON DUPLICATE KEY UPDATE metadata = VALUES(metadata), update_timestamp = NOW()',
    [id, JSON.stringify(details)],
    (err) => {
      if (err) {
        console.error('Failed to save anime metadata', err);
      }
    }
  );

  return details;
};

const fetchAnimesFromDB = async (ids: number[]): Promise<(AnimeDetails | null)[]> => {
  if (ids.length === 0) {
    return [];
  }

  const entries = await new Promise<{ id: number; metadata: string; update_timestamp: string }[]>((resolve, reject) => {
    const replacers = ids.map(() => '?').join(',');
    // Re-fetch from MAL if older than 30 days
    const query = `SELECT id, metadata, update_timestamp FROM \`anime-metadata\` WHERE update_timestamp >= now() - interval 30 DAY AND id IN (${replacers})`;

    DbPool.query(query, ids, (err, res) => {
      if (err) {
        reject(err);
      } else {
        resolve(res);
      }
    });
  });

  const entriesByID = entries.reduce(
    (acc, entry) => {
      acc[entry.id] = entry;
      return acc;
    },
    {} as { [id: number]: { id: number; metadata: string; update_timestamp: string } }
  );
  return ids.map((id) => {
    const entry = entriesByID[id];
    if (entry) {
      return JSON.parse(entry.metadata);
    } else {
      return null;
    }
  });
};

const fetchAnimesByID = async (ids: number[]): Promise<(AnimeDetails | null)[]> => {
  const fromDB = await fetchAnimesFromDB(ids);
  const missingIDs = ids.filter((_id, i) => fromDB[i] === null);
  if (missingIDs.length === 0) {
    return fromDB as AnimeDetails[];
  }

  console.log(`Missing ${missingIDs.length} anime from DB. Fetching from MAL...`);
  const fromMAL = await Promise.all(missingIDs.map(fetchAnimeFromMALAPI));
  return fromDB.map((entry) => entry || fromMAL.shift() || null);
};

export const getAnimesByID = async (
  ids: number[],
  includeRecommendationsAndRelated = false
): Promise<(AnimeDetails | null)[]> => {
  const cached: (AnimeDetails | undefined | null)[] = ids.map((id) => AnimeDetailsCache.get(id));
  const uncachedIds: number[] = cached
    .map((entry, i) => [i, entry] as const)
    .filter((entry) => entry[1] === null || entry[1] === undefined)
    .map(([i]) => ids[i]);

  if (uncachedIds.length === 0) {
    return (cached as AnimeDetails[]).map((cachedDatum) => {
      const datum = { ...cachedDatum };
      if (!includeRecommendationsAndRelated) {
        delete datum.recommendations;
        delete datum.related_anime;
      }
      return datum;
    });
  }

  const fetched = await fetchAnimesByID(uncachedIds);
  const joined = cached.map((entry) => {
    let datum = entry || fetched.shift() || null;
    if (!datum) {
      return datum;
    }

    AnimeDetailsCache.set(datum.id, datum);

    datum = { ...datum };
    if (!includeRecommendationsAndRelated) {
      delete datum.recommendations;
      delete datum.related_anime;
    }

    return datum;
  });
  return joined;
};
