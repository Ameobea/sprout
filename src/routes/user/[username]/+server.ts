import type { RequestHandler } from '@sveltejs/kit';

export const GET: RequestHandler = async ({ params }) =>
  new Response(undefined, {
    status: 302,
    headers: { Location: `/user/${params.username}/recommendations` },
  });
