import type { RequestHandler } from '@sveltejs/kit';

import { getUserAnimeList } from '../malAPI';

export const get: RequestHandler = async ({ url }) => {
  const username = url.searchParams.get('username');
  if (!username) {
    return { status: 400, body: 'Missing username param' };
  }

  try {
    const profile = await getUserAnimeList(username);
    return { body: profile };
  } catch (err) {
    return { status: 500, body: 'Unable to fetch profile due to internal error' };
  }
};
