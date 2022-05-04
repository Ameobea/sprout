import type { RequestHandler } from '@sveltejs/kit';

import { EmbeddingName } from '../types';
import { loadEmbedding } from '../embedding';

export const get: RequestHandler<Record<string, never>> = async () => {
  const embeddingName = EmbeddingName.PyMDE_4D_40N;
  const embedding = await loadEmbedding(embeddingName);
  return { body: { embedding, embeddingName } };
};
