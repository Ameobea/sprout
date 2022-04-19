import type { RequestHandler } from '@sveltejs/kit';

import { getConn } from '../dbUtil';
import { ADMIN_API_TOKEN } from '../conf';

export const addUsernames = async (usernames: string[]) => {
  const conn = getConn();
  try {
    const insertedRowCount = await new Promise((resolve, reject) =>
      conn.query(
        'INSERT IGNORE INTO `usernames-to-collect` (username) VALUES ?',
        [usernames.map((username) => [username])],
        (err, results) => {
          if (err) {
            reject(err);
          } else {
            resolve(results.affectedRows);
          }
        }
      )
    );
    return { status: 200, body: `Successfully inserted ${insertedRowCount} usernames` };
  } catch (err) {
    console.error('Error inserting usernames: ', err);
    return { status: 500, body: 'Error inserting usernames' };
  }
};

export const post: RequestHandler = async ({ url, request }) => {
  const token = url.searchParams.get('token');
  if (!token) {
    return { status: 400, body: 'Missing token' };
  }

  if (token !== ADMIN_API_TOKEN) {
    return { status: 403, body: 'Invalid token' };
  }

  const body = await request.json();
  if (!Array.isArray(body)) {
    return { status: 400, body: 'Body must be an array of usernames' };
  }

  return addUsernames(body);
};
