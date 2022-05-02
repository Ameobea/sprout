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

// const embeddingFilename = 'projected_embedding.json';
// const embeddingFilename = 'ggvec_projected_embedding.json';
const embeddingFilename = 'projected_embedding_pymde.json';

const loadRawEmbedding = async (): Promise<RawEmbedding> => {
  if (CachedRawEmbedding) {
    return CachedRawEmbedding;
  }

  return new Promise((resolve) =>
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
};

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
      vector: { x: point.x * 2.5, y: point.y * 2.5 },
      metadata: metadatum,
    };
  });
  CachedEmbedding.sort((a, b) => {
    if (b.metadata.rating_count !== a.metadata.rating_count) {
      return b.metadata.rating_count - a.metadata.rating_count;
    }
    return b.metadata.id - a.metadata.id;
  });

  return CachedEmbedding!;
};

export const loadNeighbors = async (): Promise<number[][]> => {
  if (CachedNeighbors) {
    return CachedNeighbors;
  }

  const rawEmbedding = await loadRawEmbedding();
  const embedding = await loadEmbedding();

  const idByOriginalIndex = rawEmbedding.ids;
  const originalIndexByID = new Map<number, number>();
  for (let i = 0; i < idByOriginalIndex.length; i++) {
    originalIndexByID.set(+idByOriginalIndex[i], i);
  }

  CachedNeighbors = embedding.map(({ metadata: { id } }) => {
    const originalIndex = originalIndexByID.get(+id);
    if (!originalIndex) {
      console.error('Missing original index for id ' + id);
      return [];
    }
    const neighbors = rawEmbedding.neighbors.neighbors[originalIndex];
    if (!neighbors) {
      throw new Error('Missing neighbors for original index ' + originalIndex);
    }

    return neighbors.map((neighborIndex) => +idByOriginalIndex[neighborIndex]);
  });
  return CachedNeighbors;
};
