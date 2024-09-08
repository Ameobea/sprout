import { json, type RequestHandler } from '@sveltejs/kit';
import NodeCache from 'node-cache';

import { getLocalAnimelistsDB } from '../localAnimelistsDB';

const AllUsernamesCache = new NodeCache({ stdTTL: 60 * 60 * 1000 });

const getAllUsernames = async (): Promise<string[]> => {
  const cached: string[] | undefined = AllUsernamesCache.get('allUsernames');
  if (cached) {
    return cached;
  }

  const stmt = getLocalAnimelistsDB().prepare(
    'SELECT username FROM `mal-user-animelists` WHERE json_array_length(animelist_json) > 8'
  );
  const allUsernames = await new Promise<string[]>((resolve, reject) =>
    stmt.all<{ username: string }>((err, rows) => {
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

export const GET: RequestHandler = async () => {
  const allUsernames = await getAllUsernames();
  return json({ usernames: allUsernames });
};
