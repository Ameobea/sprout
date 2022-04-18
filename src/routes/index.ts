import type { RequestHandler } from '@sveltejs/kit';
import fs from 'fs';

import { DATA_DIR } from '../conf';

interface RawEmbedding {
  points: { [index: string]: { x: number; y: number } };
  neighbors: { [index: string]: number[] };
}

interface Metadatum {
  id: number;
  title: string;
  title_english: string;
  rating_count: number;
  average_rating: number;
  aired_from_year: number;
}

export interface EmbeddedPoint {
  vector: number[];
  metadata: Metadatum;
}
export type Embedding = EmbeddedPoint[];

let cachedEmbedding: Embedding | null = null;

const loadEmbedding = async (): Promise<Embedding> => {
  if (cachedEmbedding) {
    return cachedEmbedding;
  }

  const metadata: Metadatum[] = await new Promise((resolve) => {
    fs.readFile(`${DATA_DIR}/metadata.json`, 'utf8', (err, data) => {
      if (err) {
        throw err;
      }

      const metadata: Metadatum[] = JSON.parse(data);
      resolve(
        metadata.map((metadatum) => ({
          ...metadatum,
          average_rating: +metadatum.average_rating.toFixed(3),
        }))
      );
    });
  });

  await new Promise((resolve) =>
    fs.readFile(`${DATA_DIR}/projected_embedding.json`, (err, data) => {
      if (err) {
        throw err;
      }

      const raw: RawEmbedding = JSON.parse(data.toString());
      cachedEmbedding = Object.entries(raw.points).map(([index, point]) => {
        const i = +index;
        const metadatum = metadata[i];
        if (!metadatum) {
          throw new Error('Missing metadata for point ' + i);
        }

        return {
          vector: [point.x, point.y],
          metadata: metadatum,
        };
      });
      cachedEmbedding.sort((a, b) => b.metadata.rating_count - a.metadata.rating_count);

      resolve(undefined);
    })
  );
  return cachedEmbedding!;
};

export const get: RequestHandler<Record<string, never>, { embedding: Embedding }> = async () => {
  const embedding = await loadEmbedding();
  return { body: { embedding } };
};
