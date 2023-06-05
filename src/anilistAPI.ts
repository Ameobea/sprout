import * as anilist from 'anilist-node';
import NodeCache from 'node-cache';
import * as R from 'ramda';
import { performance } from 'perf_hooks';

import { AnimeListStatusCode } from './malAPI';
import { AsyncSemaphore } from './util/asyncSemaphore';

const AnilistClient = new anilist.default();

const AnilistAPIConcurrencyLimiter = new AsyncSemaphore(1);

export interface CompatAnimeListEntry {
  node: { id: number };
  list_status: { score: number; status: AnimeListStatusCode };
}

const convertAniListMediaStatusToMALFormat = (
  status:
    | anilist.MediaStatus
    | 'PLANNING'
    | 'COMPLETED'
    | 'DROPPED'
    | 'PAUSED'
    | 'REPEAT'
    | 'WATCHING'
    | 'REPEATING'
    | 'CURRENT'
): AnimeListStatusCode => {
  switch (status) {
    case 'FINISHED':
    case 'COMPLETED':
    case 'REPEATING':
    case 'REPEAT':
      return AnimeListStatusCode.Completed;
    case 'CANCELLED':
    case 'DROPPED':
      return AnimeListStatusCode.Dropped;
    case 'RELEASING':
    case 'WATCHING':
    case 'CURRENT':
      return AnimeListStatusCode.Watching;
    case 'NOT_YET_RELEASED':
      return AnimeListStatusCode.PlanToWatch;
    case 'HIATUS':
    case 'PAUSED':
      return AnimeListStatusCode.OnHold;
    case 'PLANNING':
      return AnimeListStatusCode.PlanToWatch;
    default:
      console.error(`Unknown \`MediaStatus\` from AniList API: ${status}`);
      return AnimeListStatusCode.Watching;
  }
};

const convertAniListAnimeListToMALFormat = (entries: anilist.ListEntry[]): CompatAnimeListEntry[] => {
  const scoreDivisor = (() => {
    const maxScore = Math.max(...entries.map((entry) => entry.score || 0));
    if (maxScore === 0) {
      return 1;
    } else if (maxScore > 10) {
      return 10;
    } else if (maxScore <= 5) {
      return maxScore / 10;
    }
    return 1;
  })();
  const addedIDs = new Set<number>();

  return entries
    .filter((entry) => {
      if (addedIDs.has(entry.media.idMal)) {
        return false;
      }
      addedIDs.add(entry.media.idMal);
      return !R.isNil(entry.media.idMal);
    })
    .map((entry) => ({
      node: { id: entry.media.idMal },
      list_status: {
        score: Math.round((entry.score || 0) / scoreDivisor),
        status: convertAniListMediaStatusToMALFormat(entry.status),
      },
    }));
};

const AnilistUserAnimeListCache = new NodeCache({ stdTTL: 5 * 60 * 1000 });

type GetAniListUserAnimeListResponse =
  | { type: 'ok'; data: CompatAnimeListEntry[] }
  | { type: 'error'; message?: string; status: number };

const getAniListUserAnimeListInner = async (username: string): Promise<GetAniListUserAnimeListResponse> => {
  try {
    const cached: CompatAnimeListEntry[] | undefined = AnilistUserAnimeListCache.get(username);
    if (cached) {
      console.log(`Using cached AniList user anime list for ${username}`);
      return { type: 'ok', data: cached };
    }

    console.log(`Fetching AniList user anime list for ${username}...`);
    const start = performance.now();
    const userAnimeListsRes = (await AnilistClient.lists.anime(username)) as
      | anilist.UserList[]
      | { message?: string; status: number }[];

    if (typeof userAnimeListsRes[0].status === 'number') {
      const err = userAnimeListsRes[0] as { message?: string; status: number };
      console.error(`Error fetching AniList user anime list for ${username}: ${err.message}`);
      return { type: 'error', message: err.message, status: err.status };
    }
    const end = performance.now();
    console.log(`Fetched AniList user anime list for ${username} in ${(end - start).toFixed(2)}ms`);

    const userAnimeLists = userAnimeListsRes as anilist.UserList[];
    // Concatenate all the lists together
    const allEntries = userAnimeLists.reduce((acc, list) => acc.concat(list.entries), [] as anilist.ListEntry[]);
    const converted = convertAniListAnimeListToMALFormat(allEntries);
    AnilistUserAnimeListCache.set(username, converted);
    return { type: 'ok' as const, data: converted };
  } catch (err) {
    console.error(`Error getting Anilist user profile for ${username}: `, err);
    return { type: 'error' as const, message: err instanceof Error ? err.message : '', status: 500 };
  }
};

export const getAnilistUserAnimeList = AnilistAPIConcurrencyLimiter.wrap(getAniListUserAnimeListInner);
