import type { RequestHandler } from '@sveltejs/kit';
import NodeCache from 'node-cache';

import { getLocalAnimelistsDB } from './localAnimelistsDB';

const AllUsernamesCache = new NodeCache({ stdTTL: 60 * 60 * 1000 });

const getAllUsernames = async (): Promise<string[]> => {
  const cached: string[] | undefined = AllUsernamesCache.get('allUsernames');
  if (cached) {
    return cached;
  }

  const stmt = getLocalAnimelistsDB().prepare('SELECT username FROM `mal-user-animelists`');
  const allUsernames = await new Promise<string[]>((resolve, reject) =>
    stmt.all((err, rows) => {
      if (err) {
        reject(err);
      } else {
        resolve(rows.map((row) => row.username));
      }
    })
  );
  AllUsernamesCache.set('allUsernames', [...allUsernames]);
  return allUsernames;
};

export const get: RequestHandler = async () => {
  const allUsernames = await getAllUsernames();
  return { body: { usernames: allUsernames } };
};
