import type { RequestHandler } from '@sveltejs/kit';

import { EmbeddingName } from '../types';
import { loadEmbedding } from '../embedding';

export const get: RequestHandler<Record<string, never>> = async () => {
  const embeddingName = EmbeddingName.PyMDE;
  const embedding = await loadEmbedding(embeddingName);
  return { body: { embedding, embeddingName } };
};
