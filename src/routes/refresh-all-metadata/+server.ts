import { error, type RequestHandler } from '@sveltejs/kit';

import { ADMIN_API_TOKEN } from 'src/conf';
import { refreshAllMetadataInDB } from 'src/malAPI';

let running = false;

export const POST: RequestHandler = async ({ url }) => {
  const token = url.searchParams.get('token');
  if (!token) {
    error(400, 'Missing token');
  }
  if (token !== ADMIN_API_TOKEN) {
    error(403, 'Invalid token');
  }
  if (running) {
    return new Response('refresh already in progress', { status: 409 });
  }

  running = true;
  // Detached: the walk takes hours and must survive the HTTP connection closing.
  void refreshAllMetadataInDB()
    .catch((err) => console.error('refresh-all-metadata run failed:', err))
    .finally(() => {
      running = false;
    });

  return new Response('refresh started', { status: 202 });
};
