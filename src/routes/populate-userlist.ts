import fs from 'fs';
import type { RequestHandler } from '@sveltejs/kit';

import { ADMIN_API_TOKEN } from '../conf';
import { DbPool } from '../dbUtil';

export const get: RequestHandler = async ({ url }) => {
  const token = url.searchParams.get('token');
  if (!token) {
    return { status: 400, body: 'Missing token' };
  }

  if (token !== ADMIN_API_TOKEN) {
    return { status: 403, body: 'Invalid token' };
  }

  try {
    const usernamesJSON = fs.readFileSync('/home/casey/mal-graph/data/all_usernames.json', {
      encoding: 'utf-8',
    });
    const allUsernames = JSON.parse(usernamesJSON);
    // Insert into DB in chunks of 1000
    const chunkSize = 1;
    for (let chunkIx = 0; chunkIx < allUsernames.length; chunkIx += chunkSize) {
      const usernames = allUsernames.slice(chunkIx, chunkIx + chunkSize);
      console.log(`Inserting chunk of ${chunkSize} usernames...`);
      await new Promise((resolve, reject) => {
        DbPool.query('INSERT IGNORE INTO `usernames-to-collect` (username) VALUES (?)', [usernames[0]], (err) => {
          if (err) {
            reject(err);
          } else {
            resolve(undefined);
          }
        });
      });
      console.log(`Successfully inserted chunk of ${chunkSize} usernames chunk=${chunkIx}`);
    }
    return { status: 200, body: 'Success' };
  } catch (err) {
    console.error('Error populating usernames', err);
    return { status: 500, body: 'Error populating tbale' };
  }
};
