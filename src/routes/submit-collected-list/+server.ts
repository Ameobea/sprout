import { error, text, type RequestHandler } from '@sveltejs/kit';

import { CollectionType, getTableNames } from '../../collection';
import { ADMIN_API_TOKEN } from '../../conf';
import { dbQuery } from '../../dbUtil';

export const POST: RequestHandler = async ({ url, request }) => {
  const token = url.searchParams.get('token');
  if (!token) {
    error(400, 'Missing token');
  } else if (token !== ADMIN_API_TOKEN) {
    error(403, 'Invalid token');
  }

  const { username, type, status, list } = await request.json();
  if (typeof username !== 'string' || typeof status !== 'number') {
    error(400, 'Expected { username: string, status: number, type?: string, list?: unknown[] }');
  }
  if (status === 200 && !Array.isArray(list)) {
    error(400, 'list must be an array when status is 200');
  }
  const collectionType = type === 'manga' ? CollectionType.Manga : CollectionType.Anime;
  const { listsTableName, collectedStatusColumnName } = getTableNames(collectionType);

  try {
    if (status === 200) {
      await dbQuery(
        `INSERT INTO ${listsTableName} (username, animelist_json) VALUES (?, ?) ON DUPLICATE KEY UPDATE animelist_json = VALUES(animelist_json)`,
        [username, JSON.stringify(list)]
      );
    }
    await dbQuery(`UPDATE \`usernames-to-collect\` SET ${collectedStatusColumnName} = ? WHERE username = ?`, [
      status,
      username,
    ]);
    return text(`Recorded ${collectionType} list for ${username} with status ${status}`);
  } catch (err) {
    console.error(`Error recording collected ${collectionType} list for ${username}: `, err);
    error(500, 'DB error recording collected list');
  }
};
