import { tryCatchK } from 'fp-ts/lib/TaskEither.js';

import { getAnilistUserAnimeList, type CompatAnimeListEntry } from './anilistAPI';
import { getUserAnimeList as getUserMALAnimeList, MALAPIError } from 'src/malAPI';
import { ProfileSource } from './components/recommendation/conf';

export type ProfileFetchErrorKind = 'not_found' | 'upstream_5xx' | 'private' | 'unknown' | 'internal';

export interface ProfileFetchError {
  status: number;
  body: string;
  kind: ProfileFetchErrorKind;
}

export const fetchUserRankings = tryCatchK(
  async (username: string, profileSource: ProfileSource): Promise<{ profile: CompatAnimeListEntry[] }> => {
    let profile: CompatAnimeListEntry[] = [];
    switch (profileSource) {
      case ProfileSource.MyAnimeList:
        profile = await getUserMALAnimeList(username);
        break;
      case ProfileSource.AniList: {
        const res = await getAnilistUserAnimeList(username);
        switch (res.type) {
          case 'ok':
            profile = res.data;
            break;
          case 'error':
            throw new MALAPIError(res.message ?? 'Unknown error', res.status);
        }
        break;
      }
      default:
        throw new Error(`Unknown profile source: ${profileSource}`);
    }
    if (!Array.isArray(profile)) {
      console.error('Unexpected response from /mal-profile', profile);
      throw new Error('Failed to fetch user profile from source');
    }

    return { profile };
  },
  (err: Error): ProfileFetchError => {
    console.error('Failed to fetch user rankings', err);
    if (!(err instanceof MALAPIError)) {
      return {
        status: 500,
        body: 'An internal error occured while fetching user profile from source',
        kind: 'internal',
      };
    }

    if (err.statusCode === 404) {
      return { status: 500, body: 'User not found.  Check the username you entered and try again.', kind: 'not_found' };
    } else if (err.statusCode >= 500) {
      return {
        status: 500,
        body: 'Error received from source API when fetching profile; their servers are probably overloaded.  Please try again later',
        kind: 'upstream_5xx',
      };
    } else if (err.statusCode >= 400) {
      return { status: 500, body: 'User profile is private or could not be accessed', kind: 'private' };
    }

    console.error('Unknown error received from source API when fetching profile', err);
    return { status: 500, body: 'An unknown error occurred while fetching user profile', kind: 'unknown' };
  }
);
