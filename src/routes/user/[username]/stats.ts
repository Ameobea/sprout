import type { RequestHandler } from '@sveltejs/kit';
import * as R from 'ramda';

import {
  AnimeListStatusCode,
  getUserAnimeList,
  MALAPIError,
  type AnimeBasicDetails,
  type AnimeListStatus,
  type MALUserAnimeListItem,
} from 'src/malAPI';
import { getOfflineMetadataDB, type OfflineAnimeMetadatum } from 'src/offlineMetadataDB';

export type PartialStatsAnimeMetadatum = Pick<OfflineAnimeMetadatum, 'tags'>;
export interface PartialStatsMALUserAnimeListItem {
  node: Pick<AnimeBasicDetails, 'id'>;
  list_status: Pick<AnimeListStatus, 'score'>;
}

export type UserStatsLoadProps = {
  animeData: { [id: number]: PartialStatsAnimeMetadatum };
  profileRes: { type: 'ok'; profile: PartialStatsMALUserAnimeListItem[] } | { type: 'error'; error: string };
};

export const get: RequestHandler = async ({ params }): Promise<{ status: number; body: UserStatsLoadProps }> => {
  let userProfile: MALUserAnimeListItem[] = [];
  try {
    userProfile = await getUserAnimeList(params.username);
  } catch (err) {
    console.error('Error fetching user animelist: ', err);
    let errMessage: string;
    if (err instanceof MALAPIError) {
      switch (err.statusCode) {
        case 404:
          errMessage = 'User not found; double-check the username';
          break;
        case 403:
          errMessage = "User's anime list is not public";
          break;
        default:
          if (err.statusCode >= 500) {
            errMessage = 'MyAnimeList API is having trouble; try again later';
          } else {
            errMessage = 'An unknown error occurred fetching MyAnimeList profile';
          }
      }
    }

    return { status: 200, body: { animeData: {}, profileRes: { type: 'error', error: errMessage } } };
  }

  const animeIdsNeedingMetadata = new Set(userProfile.map((item) => item.node.id));

  const offlineMetadataDB = await getOfflineMetadataDB();
  const animeData: { [malID: number]: PartialStatsAnimeMetadatum } = {};
  for (const id of animeIdsNeedingMetadata) {
    const details = await offlineMetadataDB.byMALID.get(id);
    if (!details) {
      console.error(`Could not find offline metadata for anime ID ${id}`);
      continue;
    }
    animeData[id] = R.pick(['tags'], details);
  }

  const profile: PartialStatsMALUserAnimeListItem[] = userProfile
    .filter((rating) => {
      switch (rating.list_status.status) {
        case AnimeListStatusCode.Completed:
        case AnimeListStatusCode.Watching:
        case AnimeListStatusCode.OnHold:
        case AnimeListStatusCode.Dropped:
          return true;
        case AnimeListStatusCode.PlanToWatch:
          return false;
        default:
          console.error(`Unknown anime list status code ${rating.list_status.status}`);
          return true;
      }
    })
    .map((item) => ({
      node: R.pick(['id'], item.node),
      list_status: R.pick(['score'], item.list_status),
    }));

  return { status: 200, body: { animeData, profileRes: { type: 'ok', profile } } };
};
