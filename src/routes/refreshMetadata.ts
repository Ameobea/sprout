import type { RequestHandler } from '@sveltejs/kit';

import { DbPool } from 'src/dbUtil';
import { fetchAnimeFromMALAPI } from 'src/malAPI';

export const post: RequestHandler = async () => {
  const query =
    'SELECT id FROM `anime-metadata` WHERE update_timestamp < now() - interval 20 DAY ORDER BY update_timestamp DESC LIMIT 4';
  const ids = await new Promise<number[]>((resolve, reject) => {
    DbPool.query(query, [], (err, rows) => {
      if (err) {
        reject(err);
      } else {
        resolve(rows.map((row) => row.id));
      }
    });
  });

  if (ids.length) {
    console.log(`Refreshing metadata for ${ids.length} anime from MAL...`);
  } else {
    console.log('No anime to refresh metadata for.');
    return { status: 204 };
  }

  for (const id of ids) {
    await fetchAnimeFromMALAPI(id);
  }
  console.log('Refreshed all metadata successfully');

  return { status: 204 };
};
