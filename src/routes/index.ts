import type { RequestHandler } from '@sveltejs/kit';

import { loadEmbedding, type Metadatum } from '../embedding';

export interface EmbeddedPoint {
  vector: { x: number; y: number };
  metadata: Metadatum;
}
export type Embedding = EmbeddedPoint[];

export const get: RequestHandler<Record<string, never>, { embedding: Embedding }> = async () => {
  const embedding = await loadEmbedding();
  return { body: { embedding } };
};
