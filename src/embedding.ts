import fs from 'fs';
import { parse } from 'csv-parse';

import { DATA_DIR } from './conf';
import type { Embedding } from './routes/index';

interface RawEmbedding {
  points: { [index: string]: { x: number; y: number } };
  neighbors: { neighbors: { [index: string]: number[] } };
  ids: number[];
}

export interface Metadatum {
  id: number;
  title: string;
  title_english: string;
  rating_count: number;
  average_rating: number;
  aired_from_year: number;
}

let CachedRawEmbedding: RawEmbedding | null = null;
let CachedEmbedding: Embedding | null = null;
let CachedNeighbors: number[][] | null = null;

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

const embeddingFilename = 'projected_embedding.json';
// const embeddingFilename = 'ggvec_projected_embedding.json';

const loadRawEmbedding = (): Promise<RawEmbedding> =>
  new Promise((resolve) =>
    fs.readFile(`${DATA_DIR}/${embeddingFilename}`, (err, data) => {
      if (err) {
        throw err;
      }

      CachedRawEmbedding = JSON.parse(data.toString());
      const entries = Object.entries(CachedRawEmbedding.points);
      if (CachedRawEmbedding.ids.length !== entries.length) {
        throw new Error(`Have ${entries.length} embedding entries, but ${CachedRawEmbedding.ids.length} ids`);
      }

      resolve(CachedRawEmbedding);
    })
  );

export const loadEmbedding = async (): Promise<Embedding> => {
  if (CachedEmbedding) {
    return CachedEmbedding;
  }

  const metadata = await loadMetadata();

  const rawEmbedding = await loadRawEmbedding();
  const entries = Object.entries(rawEmbedding.points);

  CachedEmbedding = entries.map(([index, point]) => {
    const i = +index;
    const id = +rawEmbedding.ids[i];
    const metadatum = metadata.get(id);
    if (!metadatum) {
      throw new Error('Missing metadata for id ' + id);
    }

    return {
      vector: [point.x * 1.85, point.y * 1.85],
      metadata: metadatum,
    };
  });
  CachedEmbedding.sort((a, b) => b.metadata.rating_count - a.metadata.rating_count);

  return CachedEmbedding!;
};

export const loadNeighbors = async (): Promise<number[][]> => {
  if (CachedNeighbors) {
    return CachedNeighbors;
  }

  const rawEmbedding = await loadRawEmbedding();
  console.log(rawEmbedding.neighbors);
  CachedNeighbors = Object.keys(rawEmbedding.points).map(
    (i) => rawEmbedding.neighbors.neighbors[+i].map((ix) => +rawEmbedding.ids[+ix]) ?? []
  );
  return CachedNeighbors!;
};
