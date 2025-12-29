import type { Metadatum } from 'src/embedding';

export interface EmbeddedPoint {
  vector: { x: number; y: number };
  metadata: Metadatum;
}
export type Embedding = EmbeddedPoint[];
