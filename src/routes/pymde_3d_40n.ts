import type { RequestHandler } from '@sveltejs/kit';

import { EmbeddingName } from '../types';
import { loadEmbedding } from '../embedding';
import { typify } from 'src/components/recommendation/utils';

export const get: RequestHandler<Record<string, never>> = async () => {
  const embeddingName = EmbeddingName.PyMDE_3D_40N;
  const embedding = await loadEmbedding(embeddingName);
  return { body: typify({ embedding, embeddingName }) };
};
