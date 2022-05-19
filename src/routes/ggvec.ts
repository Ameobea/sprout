import type { RequestHandler } from '@sveltejs/kit';

import { EmbeddingName } from '../types';
import { loadEmbedding } from '../embedding';

export const get: RequestHandler<Record<string, never>> = async () => {
  const embeddingName = EmbeddingName.GGVec_10D_40N_Order2;
  const embedding = await loadEmbedding(embeddingName);
  return { body: { embedding, embeddingName } };
};
