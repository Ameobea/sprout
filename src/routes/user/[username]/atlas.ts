import type { RequestHandler } from '@sveltejs/kit';

import { typify } from 'src/components/recommendation/utils';
import { loadEmbedding } from 'src/embedding';
import { EmbeddingName } from 'src/types';

export const get: RequestHandler = async () => {
  const embeddingName = EmbeddingName.PyMDE_4D_40N;
  const embedding = await loadEmbedding(embeddingName);

  return { status: 200, body: typify({ embeddingName, embedding }) };
};
