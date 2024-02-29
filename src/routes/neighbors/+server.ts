import { error, type RequestHandler } from '@sveltejs/kit';

import { loadNeighbors, validateEmbeddingName } from '../../embedding';

export const GET: RequestHandler = async ({ url }) => {
  const embeddingName = validateEmbeddingName(url.searchParams.get('embedding') ?? 'pymde');
  if (!embeddingName) {
    error(400, 'Invalid embedding name');
  }

  const neighbors = await loadNeighbors(embeddingName);
  return new Response(JSON.stringify(neighbors), {
    headers: { 'content-type': 'application/json', 'cache-control': 'public, max-age=800' },
  });
};
