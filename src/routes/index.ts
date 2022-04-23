import type { RequestHandler } from '@sveltejs/kit';
import fs from 'fs';
import { parse } from 'csv-parse';

import { DATA_DIR } from '../conf';

interface RawEmbedding {
  points: { [index: string]: { x: number; y: number } };
  neighbors: { [index: string]: number[] };
  ids: number[];
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

// HEADERS: 'id', 'title', 'title_english', 'related_anime', 'recommendations', 'aired_from_year', 'rating_count', 'average_rating'
const METADATA_FILE_NAME = `${DATA_DIR}/processed-metadata.csv`;

const loadMetadata = async () => {
  const metadata = new Map<number, Metadatum>();
  await new Promise((resolve) =>
    fs
      .createReadStream(METADATA_FILE_NAME)
      .pipe(parse({ delimiter: ',' }))
      .on('data', (row) => {
        // Skip header row
        const id = +row[0];
        if (Number.isNaN(id)) {
          if (metadata.size === 0) {
            return;
          } else {
            console.error('Missing id for row ' + metadata.size);
          }
        }
        metadata.set(id, {
          id,
          title: row[1],
          title_english: row[2],
          rating_count: +row[6],
          average_rating: +row[7],
          aired_from_year: +row[5],
        });
      })
      .on('end', () => {
        resolve(metadata);
      })
  );
  return metadata;
};

const loadEmbedding = async (): Promise<Embedding> => {
  if (cachedEmbedding) {
    return cachedEmbedding;
  }

  const metadata = await loadMetadata();

  const embeddingFilename = 'projected_embedding.json';
  // const embeddingFilename = 'ggvec_projected_embedding.json';
  await new Promise((resolve) =>
    fs.readFile(`${DATA_DIR}/${embeddingFilename}`, (err, data) => {
      if (err) {
        throw err;
      }

      const raw: RawEmbedding = JSON.parse(data.toString());
      const entries = Object.entries(raw.points);
      if (raw.ids.length !== entries.length) {
        throw new Error(`Have ${entries.length} embedding entries, but ${raw.ids.length} ids`);
      }
      cachedEmbedding = entries.map(([index, point]) => {
        const i = +index;
        const id = +raw.ids[i];
        const metadatum = metadata.get(id);
        if (!metadatum) {
          throw new Error('Missing metadata for id ' + id);
        }

        return {
          vector: [point.x * 1.2, point.y * 1.2],
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
