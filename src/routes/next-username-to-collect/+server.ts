import { error, text, type RequestHandler } from '@sveltejs/kit';

import { CollectionType, getTableNames } from '../../collection';
import { ADMIN_API_TOKEN } from '../../conf';
import { dbQuery } from '../../dbUtil';

export const GET: RequestHandler = async ({ url }) => {
  const token = url.searchParams.get('token');
  if (!token) {
    error(400, 'Missing token');
  } else if (token !== ADMIN_API_TOKEN) {
    error(403, 'Invalid token');
  }

  const collectionType = url.searchParams.get('type') === 'manga' ? CollectionType.Manga : CollectionType.Anime;
  const { collectedStatusColumnName } = getTableNames(collectionType);

  try {
    const rows: { username: string }[] = await dbQuery(
      `SELECT username FROM \`usernames-to-collect\` WHERE ${collectedStatusColumnName} = 0 LIMIT 100`
    );
    if (rows.length === 0) {
      return new Response(undefined, { status: 204 });
    }
    return text(rows[Math.floor(Math.random() * rows.length)].username);
  } catch (err) {
    console.error('Error fetching next username to collect: ', err);
    error(500, 'DB error fetching next username to collect');
  }
};
