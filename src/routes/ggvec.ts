import type { RequestHandler } from '@sveltejs/kit';

import { EmbeddingName } from '../types';
import { loadEmbedding } from '../embedding';

export const get: RequestHandler<Record<string, never>> = async () => {
  const embedding = await loadEmbedding(EmbeddingName.GGVec);
  return { body: { embedding } };
};
