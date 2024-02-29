import { error, json, type RequestHandler } from '@sveltejs/kit';

import { loadEmbedding, type Metadatum, validateEmbeddingName } from '../../embedding';

export interface EmbeddedPoint {
  vector: { x: number; y: number };
  metadata: Metadatum;
}
export type Embedding = EmbeddedPoint[];

export const GET: RequestHandler<Record<string, never>> = async ({ url }) => {
  const embeddingName = validateEmbeddingName(url.searchParams.get('embedding'));
  if (!embeddingName) {
    error(400, 'Missing or invalid `embedding` param');
  }
  const embedding = await loadEmbedding(embeddingName);
  return json({ embedding, embeddingName });
};
