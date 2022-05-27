import type { RequestHandler } from '@sveltejs/kit';

import { loadEmbedding } from 'src/embedding';
import { EmbeddingName } from 'src/types';
import { DbPool } from 'src/dbUtil';
import { fetchAnimeFromMALAPI } from 'src/malAPI';

let CachedQuery: { query: string; binds: number[] } | null = null;

const getIDQuery = async () => {
  if (CachedQuery) {
    return CachedQuery;
  }

  const embedding = await loadEmbedding(EmbeddingName.PyMDE_3D_40N);
  const replacers = embedding.map(() => '?').join(',');
  const query =
    'SELECT id FROM `anime-metadata` WHERE id IN (' +
    replacers +
    ') AND update_timestamp < now() - interval 20 DAY ORDER BY update_timestamp DESC LIMIT 4';
  CachedQuery = { query, binds: embedding.map((d) => d.metadata.id) };
  return CachedQuery;
};

export const post: RequestHandler = async () => {
  const { query, binds } = await getIDQuery();
  const ids = await new Promise<number[]>((resolve, reject) => {
    DbPool.query(query, binds, (err, rows) => {
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
