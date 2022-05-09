import type { RequestHandler } from '@sveltejs/kit';

import { getAnimeByID } from 'src/malAPI';

export const get: RequestHandler = async ({ url }) => {
  const id = url.searchParams.get('id');
  if (!id) {
    return { status: 400, body: 'Missing id param' };
  }
  if (isNaN(+id)) {
    return { status: 400, body: 'Invalid id param' };
  }

  try {
    const anime = await getAnimeByID(+id);
    return { body: anime };
  } catch (err) {
    console.error('Error getting anime: ', err);
    return { status: 500, body: 'Unable to fetch anime due to internal error' };
  }
};
