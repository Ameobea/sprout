import type { RequestHandler } from '@sveltejs/kit';

import { ADMIN_API_TOKEN } from '../conf';
import { getUserAnimeList, MALAPIError } from '../malAPI';
import { getConn } from '../dbUtil';

export const get: RequestHandler = async ({ url }) => {
  const token = url.searchParams.get('token');
  if (!token) {
    return { status: 400, body: 'Missing token' };
  }

  if (token !== ADMIN_API_TOKEN) {
    return { status: 403, body: 'Invalid token' };
  }

  try {
    const conn = getConn();

    const username: string | null = await new Promise((resolve, reject) =>
      conn.query(
        'SELECT username FROM `usernames-to-collect` WHERE collected = 0 ORDER BY RAND() LIMIT 1',
        (err, results) => {
          if (err) {
            reject(err);
          } else {
            if (results.length === 0) {
              resolve(null);
            } else {
              resolve(results[0].username);
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
      const animeList = await getUserAnimeList(username);

      await new Promise((resolve, reject) => {
        conn.query(
          'INSERT INTO `mal-user-animelists` (username, animelist_json) VALUES (?, ?) ON DUPLICATE KEY UPDATE animelist_json = VALUES(animelist_json)',
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

      await new Promise((resolve, reject) => {
        conn.query(
          'UPDATE `usernames-to-collect` SET collected = 200 WHERE username = ?',
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

      const body = `Successfully collected profile for ${username}`;
      console.log(body);
      return { body, status: 200 };
    } catch (err) {
      console.error(`Error fetching anime list for ${username}: `, err);
      const status = err instanceof MALAPIError ? err.statusCode : 500;

      await new Promise((resolve, reject) => {
        conn.query(
          'UPDATE `usernames-to-collect` SET collected = ? WHERE username = ?',
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
      return { status: 500, body: `Unable to fetch anime list for ${username}` };
    }
  } catch (err) {
    console.error('Error getting animelist for username', err);
    return { status: 500, body: 'Error getting username; DB error likely' };
  }
};
