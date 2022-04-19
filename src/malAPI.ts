import { MAL_API_BASE_URL, MAL_CLIENT_ID } from './conf';
import { delay } from './util';
import TimedCache from 'timed-cache';
import { getConn } from './dbUtil';

export class MALAPIError extends Error {
  public statusCode: number;

  constructor(message: string, statusCode: number) {
    super(message);
    this.name = 'MALAPIError';
    this.statusCode = statusCode;
  }
}

const makeMALRequest = async (url: string, retryCount?: number) => {
  const res = await fetch(url, { headers: { 'X-MAL-CLIENT-ID': MAL_CLIENT_ID } });
  if (res.status === 429 || (res.status >= 500 && res.status < 600)) {
    console.log(`Retryable error ${res.status} for ${url}`);
    const retryAfter = res.headers.get('Retry-After');
    if (!retryAfter) {
      await delay(1000);
    } else {
      await delay(+retryAfter * 1000);
    }
    return makeMALRequest(url, (retryCount ?? 0) + 1);
  } else if (!res.ok) {
    throw new MALAPIError(`MAL API returned ${res.status}: ${await res.text()}`, res.status);
  }
  return res.json();
};

interface AnimeBasicDetails {
  id: number;
  title: string;
  main_picture: {
    medium: string;
    large: string;
  };
}

export interface MALUserAnimeListItem {
  node: AnimeBasicDetails;
  list_status: ListStatus;
}

export interface MALUserAnimeListResponse {
  data: MALUserAnimeListItem[];
  paging: { next?: string | null };
}

export enum ListStatusCode {
  Watching = 'watching',
  Completed = 'completed',
  OnHold = 'on_hold',
  Dropped = 'dropped',
  PlanToWatch = 'plan_to_watch',
}

interface ListStatus {
  status: ListStatusCode;
  score: number;
  num_episodes_watched: number;
  is_rewatching: boolean;
  updated_at: string;
}

// Cache for 24 hours
const UserAnimeListCache = new TimedCache({ defaultTtl: 60 * 60 * 24 * 1000 });

export const getUserAnimeList = async (username: string): Promise<MALUserAnimeListItem[]> => {
  const cached = UserAnimeListCache.get(username);
  if (cached) {
    console.log('Found cached user anime list for ' + username);
    return cached;
  }

  const data: MALUserAnimeListItem[] = [];

  let i = 0;
  const pageSize = 1000;
  for (;;) {
    const url = `${MAL_API_BASE_URL}/users/${username}/animelist?offset=${
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
  UserAnimeListCache.put(username, data);

  return data;
};

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
  related_anime?: {
    node: AnimeBasicDetails;
    relation_type: string;
    relation_type_formatted: string;
  }[];
}

const AnimeDetailsCache: Map<number, AnimeDetails> = new Map();

export const getAnimeByID = async (id: number, includeRecommendationsAndRelated = false) => {
  if (AnimeDetailsCache.has(id)) {
    const details = { ...AnimeDetailsCache.get(id) } as AnimeDetails;
    if (!includeRecommendationsAndRelated) {
      delete details.recommendations;
      delete details.related_anime;
    }
    return details;
  }

  const fieldsToFetch = [
    'main_picture',
    'alternative_titles',
    'start_date',
    'end_date',
    'synopsis',
    'genres',
    'recommendations',
    'related_anime',
  ];
  const url = `${MAL_API_BASE_URL}/anime/${id}?fields=${fieldsToFetch.join(',')}`;
  console.log('Fetching anime...', url);
  const details = (await makeMALRequest(url)) as AnimeDetails;
  AnimeDetailsCache.set(id, details);

  const conn = getConn();
  await new Promise((resolve) =>
    conn.query(
      'INSERT INTO `anime-metadata` (id, metadata) VALUES (?, ?) ON DUPLICATE KEY UPDATE metadata = VALUES(metadata)',
      [id, JSON.stringify(details)],
      (err) => {
        if (err) {
          console.error('Failed to save anime metadata', err);
        }
        resolve(undefined);
      }
    )
  );
  if (!includeRecommendationsAndRelated) {
    delete details.recommendations;
    delete details.related_anime;
  }

  return details;
};
