import type { RequestHandler } from '@sveltejs/kit';

import type { CompatAnimeListEntry } from 'src/anilistAPI';
import { typify } from 'src/components/recommendation/utils';
import { getUserAnimeList } from '../malAPI';

export const get: RequestHandler = async ({ url }) => {
  const username = url.searchParams.get('username');
  if (!username) {
    return { status: 400, body: 'Missing username param' };
  }

  try {
    const profile = await getUserAnimeList(username);
    const compatProfile: CompatAnimeListEntry[] = profile.map((entry) => ({
      node: { id: entry.node.id },
      list_status: { status: entry.list_status.status, score: entry.list_status.score },
    }));
    return { body: typify(compatProfile) };
  } catch (err) {
    return { status: 500, body: 'Unable to fetch profile due to internal error' };
  }
};
