import mysql from 'mysql';
import type { RequestHandler } from '@sveltejs/kit';

import { ADMIN_API_TOKEN, MYSQL_DATABASE, MYSQL_HOST, MYSQL_PASSWORD, MYSQL_USER } from '../conf';
import { getUserAnimeList } from 'src/malAPI';

const conn = mysql.createConnection({
  host: MYSQL_HOST,
  user: MYSQL_USER,
  password: MYSQL_PASSWORD,
  database: MYSQL_DATABASE,
});
conn.connect();

export const get: RequestHandler = async ({ url }) => {
  const token = url.searchParams.get('token');
  if (!token) {
    return { status: 400, body: 'Missing token' };
  }

  if (token !== ADMIN_API_TOKEN) {
    return { status: 403, body: 'Invalid token' };
  }

  try {
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
          'INSERT INTO `mal-user-animelists` (username, animelist_json) VALUES (?, ?)',
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
          'UPDATE `usernames-to-collect` SET collected = 1 WHERE username = ?',
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

      return { body: 'OK', status: 204 };
    } catch (err) {
      console.error(`Error fetching anime list for ${username}: `, err);
      await new Promise((resolve, reject) => {
        conn.query(
          'UPDATE `usernames-to-collect` SET collected = 2 WHERE username = ?',
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
      return { status: 500, body: 'Unable to fetch anime list' };
    }
  } catch (err) {
    console.error('Error getting animelist for username', err);
    return { status: 500, body: 'Error getting username' };
  }
};
