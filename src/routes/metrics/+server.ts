import { error, text, type RequestHandler } from '@sveltejs/kit';

import { ADMIN_API_TOKEN } from '../../conf';
import { dbQuery } from '../../dbUtil';

export const GET: RequestHandler = async ({ url }) => {
  const token = url.searchParams.get('token');
  if (!token) {
    error(400, 'Missing token');
  } else if (token !== ADMIN_API_TOKEN) {
    error(403, 'Invalid token');
  }

  try {
    const rows: { collected: number; count: number }[] = await dbQuery(
      'SELECT collected, COUNT(*) AS count FROM `usernames-to-collect` GROUP BY collected'
    );
    const lines = [
      '# HELP anime_atlas_usernames_by_collected_total Count of usernames-to-collect rows by collected status',
      '# TYPE anime_atlas_usernames_by_collected_total gauge',
      ...rows.map((row) => `anime_atlas_usernames_by_collected_total{collected="${row.collected}"} ${row.count}`),
    ];
    return text(lines.join('\n') + '\n');
  } catch (err) {
    console.error('Error building metrics: ', err);
    error(500, 'DB error building metrics');
  }
};
