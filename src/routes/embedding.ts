import type { RequestHandler } from '@sveltejs/kit';

import { loadEmbedding, type Metadatum, validateEmbeddingName } from '../embedding';

export interface EmbeddedPoint {
  vector: { x: number; y: number };
  metadata: Metadatum;
}
export type Embedding = EmbeddedPoint[];

export const get: RequestHandler<Record<string, never>> = async ({ url }) => {
  const embeddingName = validateEmbeddingName(url.searchParams.get('embedding'));
  const embedding = await loadEmbedding(embeddingName);
  return { body: { embedding, embeddingName } };
};
