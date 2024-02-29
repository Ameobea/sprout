import { error, json, type RequestHandler } from '@sveltejs/kit';

import { getAnilistUserAnimeList } from '../../anilistAPI';

export const GET: RequestHandler = async ({ url }) => {
  const username = url.searchParams.get('username');
  if (!username) {
    error(400, 'Missing username param');
  }

  try {
    const res = await getAnilistUserAnimeList(username);
    if (res.type === 'error') {
      error(res.status, res.message ?? 'Unable to fetch profile due to internal error');
    }
    const compatProfile = res.data;
    return json(compatProfile);
  } catch (err) {
    return error(500, 'Unable to fetch profile due to internal error');
  }
};
