import type { RequestHandler } from '@sveltejs/kit';

import { getAnimesByID } from 'src/malAPI';

export const get: RequestHandler = async ({ url }) => {
  const id = url.searchParams.get('id');
  if (!id) {
    return { status: 400, body: 'Missing id param' };
  }
  if (isNaN(+id)) {
    return { status: 400, body: 'Invalid id param' };
  }

  try {
    const [anime] = await getAnimesByID([+id]);
    if (anime) {
      return { body: anime };
    }
    return { status: 404, body: 'Anime not found' };
  } catch (err) {
    console.error('Error getting anime: ', err);
    return { status: 500, body: 'Unable to fetch anime due to internal error' };
  }
};
