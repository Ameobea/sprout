import { error, json, type RequestHandler } from '@sveltejs/kit';

import type { CompatAnimeListEntry } from 'src/anilistAPI';
import { typify } from 'src/components/recommendation/utils';
import { getUserAnimeList } from '../../malAPI';

export const GET: RequestHandler = async ({ url }) => {
  const username = url.searchParams.get('username');
  if (!username) {
    error(400, 'Missing username param');
  }

  try {
    const profile = await getUserAnimeList(username);
    const compatProfile: CompatAnimeListEntry[] = profile.map((entry) => ({
      node: { id: entry.node.id },
      list_status: { status: entry.list_status.status, score: entry.list_status.score },
    }));
    return json(typify(compatProfile));
  } catch (err) {
    error(500, 'Unable to fetch profile due to internal error');
  }
};
