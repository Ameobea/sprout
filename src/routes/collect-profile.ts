import type { RequestHandler } from '@sveltejs/kit';

import { DbPool } from '../dbUtil';
import { ADMIN_API_TOKEN } from '../conf';
import { getUserAnimeList, getUserMangaList, MALAPIError } from '../malAPI';

enum CollectionType {
  Anime = 'anime',
  Manga = 'manga',
}

const getTableNames = (collectionType: CollectionType) => {
  switch (collectionType) {
    case CollectionType.Anime:
      return { listsTableName: '`mal-user-animelists`', collectedStatusColumnName: 'collected' };
    case CollectionType.Manga:
      return { listsTableName: '`mal-user-mangalists`', collectedStatusColumnName: '`collected-manga`' };
    default:
      throw new Error('Unknown collection type');
  }
};

export const get: RequestHandler = async ({ url }) => {
  const token = url.searchParams.get('token');
  if (!token) {
    return { status: 400, body: 'Missing token' };
  } else if (token !== ADMIN_API_TOKEN) {
    return { status: 403, body: 'Invalid token' };
  }

  const type = url.searchParams.get('type');
  let collectionType = CollectionType.Anime;
  if (type === 'manga') {
    collectionType = CollectionType.Manga;
  }

  const { listsTableName, collectedStatusColumnName } = getTableNames(collectionType);

  try {
    const username: string | null = await new Promise((resolve, reject) =>
      DbPool.query(
        `SELECT username FROM \`usernames-to-collect\` WHERE ${collectedStatusColumnName} = 0 LIMIT 100`,
        (err, results) => {
          if (err) {
            reject(err);
          } else {
            if (results.length === 0) {
              resolve(null);
            } else {
              // pick random one
              const randomIndex = Math.floor(Math.random() * results.length);
              resolve(results[randomIndex].username);
            }
          }
        }
      )
    );

    if (username === null) {
      console.warn('No users remain to collect');
      return { status: 204 };
    }

    try {
      const fetcherFn = collectionType === CollectionType.Anime ? getUserAnimeList : getUserMangaList;
      const animeList = await fetcherFn(username);

      await new Promise((resolve, reject) => {
        DbPool.query(
          `INSERT INTO ${listsTableName} (username, animelist_json) VALUES (?, ?) ON DUPLICATE KEY UPDATE animelist_json = VALUES(animelist_json)`,
          [username, JSON.stringify(animeList)],
          (err) => {
            if (err) {
              reject(err);
            } else {
              resolve(undefined);
            }
          }
        );
      });
      console.log('Successfully wrote anime list to DB');

      await new Promise((resolve, reject) => {
        DbPool.query(
          `UPDATE \`usernames-to-collect\` SET ${collectedStatusColumnName} = 200 WHERE username = ?`,
          [username],
          (err) => {
            if (err) {
              reject(err);
            } else {
              resolve(undefined);
            }
          }
        );
      });

      const body = `Successfully collected ${collectionType} list for ${username}`;
      console.log(body);
      return { body, status: 200 };
    } catch (err) {
      console.error(`Error fetching ${collectionType} list for ${username}: `, err);
      const status = err instanceof MALAPIError ? err.statusCode : 500;

      await new Promise((resolve, reject) => {
        DbPool.query(
          `UPDATE \`usernames-to-collect\` SET ${collectedStatusColumnName} = ? WHERE username = ?`,
          [status, username],
          (err) => {
            if (err) {
              reject(err);
            } else {
              resolve(undefined);
            }
          }
        );
      });
      return { status: 500, body: `Unable to fetch ${collectionType} list for ${username}` };
    }
  } catch (err) {
    console.error(`Error getting ${collectionType} list for username`, err);
    return { status: 500, body: 'Error getting username; DB error likely' };
  }
};
