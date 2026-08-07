import { error, json, type RequestHandler } from '@sveltejs/kit';

import type { CompatAnimeListEntry } from 'src/anilistAPI';
import { typify } from 'src/components/recommendation/utils';
import { getUserAnimeList, MALAPIError } from '../../malAPI';

export const GET: RequestHandler = async ({ url }) => {
  const username = url.searchParams.get('username');
  if (!username) {
    error(400, 'Missing username param');
  }

  try {
    const profile = await getUserAnimeList(username);
    const compatProfile: CompatAnimeListEntry[] = profile.map((entry) => ({
      node: { id: entry.node.id },
      list_status: {
        status: entry.list_status.status,
        score: entry.list_status.score,
        updated_at: entry.list_status.updated_at,
      },
    }));
    return json(typify(compatProfile));
  } catch (err) {
    if (err instanceof MALAPIError) {
      if (err.statusCode === 404) {
        error(404, `No MyAnimeList user found with the username "${username}"`);
      } else if (err.statusCode === 403) {
        error(403, `The anime list for MyAnimeList user "${username}" is private`);
      }
      error(502, 'The MyAnimeList API is unavailable right now; try again in a bit');
    }
    console.error(`Error fetching MAL profile for ${username}: `, err);
    error(500, 'Unable to fetch profile due to internal error');
  }
};
