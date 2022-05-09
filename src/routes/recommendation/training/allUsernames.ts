import type { RequestHandler } from '@sveltejs/kit';
import TimedCache from 'timed-cache';

import { LocalAnimelistsDB } from './localAnimelistsDB';

const AllUsernamesCache = new TimedCache({ defaultTtl: 60 * 60 * 1000 });

const getAllUsernames = async (): Promise<string[]> => {
  const cached = AllUsernamesCache.get('allUsernames');
  if (cached) {
    return cached;
  }

  const stmt = LocalAnimelistsDB.prepare('SELECT username FROM `mal-user-animelists`');
  const allUsernames = await new Promise<string[]>((resolve, reject) =>
    stmt.all((err, rows) => {
      if (err) {
        reject(err);
      } else {
        resolve(rows.map((row) => row.username));
      }
    })
  );
  AllUsernamesCache.put('allUsernames', [...allUsernames]);
  return allUsernames;
};

export const get: RequestHandler = async () => {
  const allUsernames = await getAllUsernames();
  return { body: { usernames: allUsernames } };
};
