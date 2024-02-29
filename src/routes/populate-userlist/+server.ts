import fs from 'fs';
import { error, text, type RequestHandler } from '@sveltejs/kit';

import { ADMIN_API_TOKEN } from '../../conf';
import { DbPool } from '../../dbUtil';

export const GET: RequestHandler = async ({ url }) => {
  const token = url.searchParams.get('token');
  if (!token) {
    error(400, 'Missing token');
  }

  if (token !== ADMIN_API_TOKEN) {
    error(403, 'Invalid token');
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
    return text('Success');
  } catch (err) {
    console.error('Error populating usernames', err);
    error(500, 'Error populating tbale');
  }
};
