import * as R from 'ramda';
import { isLeft } from 'fp-ts/lib/Either.js';
import { fetchUserRankings } from 'src/helpers';

import { AnimeListStatusCode, type AnimeBasicDetails, type AnimeListStatus } from 'src/malAPI';
import { getOfflineMetadataDB, type OfflineAnimeMetadatum } from 'src/offlineMetadataDB';
import { DEFAULT_PROFILE_SOURCE } from 'src/components/recommendation/conf';
import { typify, type Typify } from 'src/components/recommendation/utils';
import { ProfileSourceValidator } from 'src/routes/recommendation/recommendation/recommendation';
import type { PageServerLoad } from './$types';

export type PartialStatsAnimeMetadatum = Pick<OfflineAnimeMetadatum, 'tags'>;
export interface PartialStatsMALUserAnimeListItem {
  node: Pick<AnimeBasicDetails, 'id'>;
  list_status: Pick<AnimeListStatus, 'score'>;
}

export type UserStatsLoadProps = {
  animeData: { [id: number]: PartialStatsAnimeMetadatum };
  profileRes: { type: 'ok'; profile: Typify<PartialStatsMALUserAnimeListItem[]> } | { type: 'error'; error: string };
};

export const load: PageServerLoad = async ({ params, url }): Promise<UserStatsLoadProps> => {
  const username = params.username;
  const rawProfileSource = url.searchParams.get('source') ?? DEFAULT_PROFILE_SOURCE;
  const profileSourceParseRes = ProfileSourceValidator.decode(rawProfileSource);
  if (isLeft(profileSourceParseRes)) {
    return { animeData: {}, profileRes: { type: 'error' as const, error: 'Invalid `source` query param' } };
  }
  const profileSource = profileSourceParseRes.right;

  const rankingsRes = await fetchUserRankings(username, profileSource)();
  if (isLeft(rankingsRes)) {
    return { animeData: {}, profileRes: { type: 'error' as const, error: typify(rankingsRes.left.body) } };
  }
  const { profile: userProfile } = rankingsRes.right;

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

  return { animeData: typify(animeData), profileRes: { type: 'ok', profile: typify(profile) } };
};
