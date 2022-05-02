import fs from 'fs';
import type { RequestHandler } from '@sveltejs/kit';

import { DbPool } from '../dbUtil';
import { ADMIN_API_TOKEN } from '../conf';
import { getAnimeByID, MALAPIError } from '../malAPI';

const fillFromScratch = async () => {
  const allAnimeIDs: number[] = await new Promise((resolve) =>
    fs.readFile('/home/casey/mal-graph/data/all-anime-ids.json', 'utf8', (err, data) => resolve(JSON.parse(data)))
  );

  await new Promise((resolve, reject) => {
    DbPool.query('INSERT IGNORE INTO `anime-metadata` (id) VALUES ?', [allAnimeIDs.map((id) => [id])], (err) => {
      if (err) {
        reject(err);
      } else {
        resolve(undefined);
      }
    });
  });
};

export const post: RequestHandler = async ({ url }) => {
  const token = url.searchParams.get('token');
  if (!token) {
    return { status: 400, body: 'Missing token' };
  }

  if (token !== ADMIN_API_TOKEN) {
    return { status: 403, body: 'Invalid token' };
  }

  const populateNulls = url.searchParams.get('populateNulls');
  if (populateNulls) {
    await fillFromScratch();
    return { status: 200, body: 'Successfully populated table' };
  }

  try {
    const idToFetch = await new Promise<number | null>((resolve, reject) =>
      DbPool.query('SELECT id FROM `anime-metadata` WHERE metadata IS NULL LIMIT 1', (err, results) => {
        if (err) {
          reject(err);
        } else {
          if (results.length === 0) {
            resolve(null);
          } else {
            resolve(results[0].id);
          }
        }
      })
    );

    if (idToFetch === null) {
      console.warn('No anime IDs remain to fetch');
      return { status: 204 };
    }

    try {
      await getAnimeByID(idToFetch);
    } catch (err) {
      if (err instanceof MALAPIError) {
        const statusCode = err.statusCode;
        if (statusCode === 404) {
          console.warn(`Anime ID ${idToFetch} not found`);
          await new Promise((resolve, reject) =>
            DbPool.query('DELETE FROM `anime-metadata` WHERE id = ?', [idToFetch], (err) => {
              if (err) {
                reject(err);
              } else {
                resolve(undefined);
              }
            })
          );
          return { status: 200, body: `Anime ID ${idToFetch} not found; deleting placeholder from table` };
        }
        console.error(`Anime ID ${idToFetch} returned status code ${statusCode}: ${err.message}`);
      }
      return { status: 500, body: `Anime ID ${idToFetch} failed: ${err.message}` };
    }
    return { status: 200, body: `Successfully fetched anime id=${idToFetch}` };
  } catch (err) {
    console.error(err);
    return { status: 500, body: `${err}` };
  }
};
