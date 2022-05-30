import type { RequestHandler } from '@sveltejs/kit';

import { getAnilistUserAnimeList } from '../anilistAPI';

export const get: RequestHandler = async ({ url }) => {
  const username = url.searchParams.get('username');
  if (!username) {
    return { status: 400, body: 'Missing username param' };
  }

  try {
    const res = await getAnilistUserAnimeList(username);
    if (res.type === 'error') {
      return { status: res.status, body: res.message ?? 'Unable to fetch profile due to internal error' };
    }
    const compatProfile = res.data;
    return { body: compatProfile };
  } catch (err) {
    return { status: 500, body: 'Unable to fetch profile due to internal error' };
  }
};
