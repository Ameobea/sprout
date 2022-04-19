import fs from 'fs';
import type { RequestHandler } from '@sveltejs/kit';

import { ADMIN_API_TOKEN } from '../conf';
import { getConn } from '../dbUtil';

// fetch('http://localhost:3080/add-usernames?token=asdf', {method: 'POST', body: JSON.stringify([...document.querySelector('#content > table > tbody > tr > td:nth-child(1) > table > tbody').children].flatMap(tr => [...tr.children]).map(td => td.children[0].innerText))}).then(res=>res.text()).then(console.log)

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
        conn.query(
          'INSERT IGNORE INTO `usernames-to-collect` (username) VALUES (?)',
          [usernames[0]],
          (err) => {
            if (err) {
              reject(err);
            } else {
              resolve(undefined);
            }
          }
        );
      });
      console.log(`Successfully inserted chunk of ${chunkSize} usernames chunk=${chunkIx}`);
    }
  } catch (err) {
    console.error('Error populating usernames', err);
    return { status: 500, body: 'Error populating tbale' };
  }
};
