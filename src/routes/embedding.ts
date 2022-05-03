import type { RequestHandler } from '@sveltejs/kit';

import { loadEmbedding, validateEmbeddingName, type Metadatum } from '../embedding';

export interface EmbeddedPoint {
  vector: { x: number; y: number };
  metadata: Metadatum;
}
export type Embedding = EmbeddedPoint[];

export const get: RequestHandler<Record<string, never>, { embedding: Embedding }> = async ({ url }) => {
  const embeddingName = validateEmbeddingName(url.searchParams.get('embedding') ?? 'pymde');
  const embedding = await loadEmbedding(embeddingName);
  return { body: { embedding } };
};
