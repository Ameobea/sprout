import type { RequestHandler } from '@sveltejs/kit';

import { loadEmbedding, type Metadatum, validateEmbeddingName } from '../embedding';

export interface EmbeddedPoint {
  vector: { x: number; y: number };
  metadata: Metadatum;
}
export type Embedding = EmbeddedPoint[];

export const get: RequestHandler<Record<string, never>> = async ({ url }) => {
  const embeddingName = validateEmbeddingName(url.searchParams.get('embedding'));
  if (!embeddingName) {
    return { status: 400, body: 'Missing or invalid `embedding` param' };
  }
  const embedding = await loadEmbedding(embeddingName);
  return { body: { embedding, embeddingName } };
};
