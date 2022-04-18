import { MAL_API_BASE_URL, MAL_CLIENT_ID } from './conf';
import { delay } from './util';
import TimedCache from 'timed-cache';

const makeMALRequest = async (url: string, retryCount?: number) => {
  const res = await fetch(url, { headers: { 'X-MAL-CLIENT-ID': MAL_CLIENT_ID } });
  if (res.status === 429) {
    console.log('Rate limited by MAL API, waiting...');
    const retryAfter = res.headers.get('Retry-After');
    if (!retryAfter) {
      await delay(1000);
    } else {
      await delay(+retryAfter * 1000);
    }
    return makeMALRequest(url, (retryCount ?? 0) + 1);
  } else if (!res.ok) {
    throw new Error(`MAL API returned ${res.status}: ${await res.text()}`);
  }
  return res.json();
};

export interface MALUserAnimeListItem {
  node: {
    id: number;
    title: string;
    main_picture: {
      medium: string;
      large: string;
    };
  };
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
}

const AnimeDetailsCache: Map<number, AnimeDetails> = new Map();

export const getAnimeByID = async (id: number) => {
  if (AnimeDetailsCache.has(id)) {
    return AnimeDetailsCache.get(id);
  }

  const url = `${MAL_API_BASE_URL}/anime/${id}?fields=main_picture,alternative_titles,start_date,end_date,synopsis,genres`;
  console.log('Fetching anime...', url);
  const details = (await makeMALRequest(url)) as AnimeDetails;
  AnimeDetailsCache.set(id, details);
  return details;
};
