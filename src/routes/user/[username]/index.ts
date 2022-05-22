import type { RequestHandler } from '@sveltejs/kit';

export const get: RequestHandler = ({ params }) => ({
  status: 302,
  headers: { Location: `/user/${params.username}/recommendations` },
});
