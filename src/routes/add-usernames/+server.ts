import { error, text, type RequestHandler } from '@sveltejs/kit';

import { ADMIN_API_TOKEN } from '../../conf';
import { addUsernames } from './addUsernames';

export const POST: RequestHandler = async ({ url, request }) => {
  const token = url.searchParams.get('token');
  if (!token) {
    error(400, 'Missing token');
  }

  if (token !== ADMIN_API_TOKEN) {
    error(403, 'Invalid token');
  }

  const body = await request.json();
  if (!Array.isArray(body)) {
    error(400, 'Body must be an array of usernames');
  }

  const resText = await addUsernames(body);
  return text(resText);
};
