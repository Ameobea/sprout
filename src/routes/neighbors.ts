import type { RequestHandler } from '@sveltejs/kit';

import { loadNeighbors, validateEmbeddingName } from '../embedding';

export const get: RequestHandler = async ({ url }) => {
  const embeddingName = validateEmbeddingName(url.searchParams.get('embedding') ?? 'pymde');
  const neighbors = await loadNeighbors(embeddingName);
  return { body: { neighbors } };
};
