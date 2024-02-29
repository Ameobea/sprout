import { error, json, type RequestHandler } from '@sveltejs/kit';

import { getAnimesByID } from 'src/malAPI';

export const GET: RequestHandler = async ({ url }) => {
  const id = url.searchParams.get('id');
  if (!id) {
    error(400, 'Missing id param');
  }
  if (isNaN(+id)) {
    error(400, 'Invalid id param');
  }

  try {
    const [anime] = await getAnimesByID([+id]);
    if (anime) {
      return json(anime);
    }
    error(404, 'Anime not found');
  } catch (err) {
    console.error('Error getting anime: ', err);
    error(500, 'Unable to fetch anime due to internal error');
  }
};
