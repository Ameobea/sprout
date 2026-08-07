import { error, isHttpError, json, type RequestHandler } from '@sveltejs/kit';

import { getAnilistUserAnimeList } from '../../anilistAPI';

export const GET: RequestHandler = async ({ url }) => {
  const username = url.searchParams.get('username');
  if (!username) {
    error(400, 'Missing username param');
  }

  try {
    const res = await getAnilistUserAnimeList(username);
    if (res.type === 'error') {
      const status = res.status >= 400 && res.status <= 599 ? res.status : 502;
      error(
        status,
        status === 404
          ? `No AniList user found with the username "${username}"`
          : 'Unable to fetch profile from AniList right now; try again in a bit'
      );
    }
    return json(res.data);
  } catch (err) {
    if (isHttpError(err)) {
      throw err;
    }
    console.error(`Error fetching AniList profile for ${username}: `, err);
    error(500, 'Unable to fetch profile due to internal error');
  }
};
