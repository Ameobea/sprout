import { tryCatchK } from 'fp-ts/lib/TaskEither.js';

import { getAnilistUserAnimeList, type CompatAnimeListEntry } from './anilistAPI';
import { getUserAnimeList as getUserMALAnimeList, MALAPIError } from 'src/malAPI';
import { ProfileSource } from './components/recommendation/conf';
import {
  convertMALProfileToTrainingData,
  type TrainingDatum,
} from './routes/recommendation/training/trainingData/trainingData';

export const fetchUserRankings = tryCatchK(
  async (
    username: string,
    profileSource: ProfileSource
  ): Promise<{ profile: CompatAnimeListEntry[]; ratings: TrainingDatum[]; userIsNonRater: boolean }> => {
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

    const { ratings, userIsNonRater } = (await convertMALProfileToTrainingData([profile]))[0];
    return { profile, ratings, userIsNonRater };
  },
  (err: Error) => {
    console.error('Failed to fetch user rankings', err);
    const body =
      err instanceof MALAPIError
        ? (() => {
            switch (err.statusCode) {
              case 404:
                return 'User not found.  Check the username you entered and try again.';
              default:
                if (err.statusCode >= 500) {
                  return 'Error received from source API when fetching profile; their servers are probably overloaded.  Please try again later';
                } else if (err.statusCode >= 400) {
                  return 'User profile is private or could not be accessed';
                } else {
                  console.error('Unknown error received from source API when fetching profile', err);
                  return 'An unknown error occurred while fetching user profile';
                }
            }
          })()
        : 'An internal error occured while fetching user profile from source';
    return { status: 500, body };
  }
);
