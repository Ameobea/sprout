import type { RequestHandler } from '@sveltejs/kit';
import { loadEmbedding } from 'src/embedding';
import { EmbeddingName } from 'src/types';

export const get: RequestHandler = async ({ params }) => {
  // TODO: param
  const embeddingName = EmbeddingName.PyMDE_3D_40N;
  const embedding = await loadEmbedding(embeddingName);

  return { status: 200, body: { embeddingName, embedding } };
};
