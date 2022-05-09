import fs from 'fs';
import { parse } from 'csv-parse';

import { DATA_DIR } from './conf';
import type { Embedding } from './routes/embedding';
import { EmbeddingName } from './types';

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

const CachedRawEmbeddings: Map<EmbeddingName, RawEmbedding> = new Map();
const CachedEmbeddings: Map<EmbeddingName, Embedding> = new Map();
const CachedNeighbors: Map<EmbeddingName, number[][]> = new Map();

// HEADERS: 'id', 'title', 'title_english', 'related_anime', 'recommendations', 'aired_from_year', 'rating_count', 'average_rating'
const METADATA_FILE_NAME = `${DATA_DIR}/processed-metadata.csv`;

export const loadMetadata = async () => {
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

const EmbeddingFilenameByName: { [K in EmbeddingName]: string } = {
  [EmbeddingName.PyMDE]: 'projected_embedding_pymde_3d_40n.json',
  [EmbeddingName.GGVec]: 'projected_embedding.json',
  [EmbeddingName.PyMDE_4D_40N]: 'projected_embedding_pymde_4d_40n.json',
  [EmbeddingName.PyMDE_4D_100N]: 'projected_embedding_pymde_4d_100n.json',
};

const AllValidEmbeddingNames = new Set(Object.keys(EmbeddingFilenameByName) as EmbeddingName[]);

export const validateEmbeddingName = (name: string | null | undefined): EmbeddingName | null =>
  AllValidEmbeddingNames.has(name as EmbeddingName) ? (name as EmbeddingName) : null;

const loadRawEmbedding = async (embeddingName: EmbeddingName): Promise<RawEmbedding> => {
  const cached = CachedRawEmbeddings.get(embeddingName);
  if (cached) {
    return cached;
  }

  const embeddingFilename = EmbeddingFilenameByName[embeddingName];
  return new Promise((resolve) =>
    fs.readFile(`${DATA_DIR}/${embeddingFilename}`, (err, data) => {
      if (err) {
        throw err;
      }

      const embedding = JSON.parse(data.toString());
      const entries = Object.entries(embedding.points);
      if (embedding.ids.length !== entries.length) {
        throw new Error(`Have ${entries.length} embedding entries, but ${embedding.ids.length} ids`);
      }

      CachedRawEmbeddings.set(embeddingName, embedding);

      resolve(embedding);
    })
  );
};

export const loadEmbedding = async (embeddingName: EmbeddingName): Promise<Embedding> => {
  const cached = CachedEmbeddings.get(embeddingName);
  if (cached) {
    return cached;
  }

  const metadata = await loadMetadata();

  const rawEmbedding = await loadRawEmbedding(embeddingName);
  const entries = Object.entries(rawEmbedding.points);

  const embedding = entries.map(([index, point]) => {
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
  embedding.sort((a, b) => {
    if (b.metadata.rating_count !== a.metadata.rating_count) {
      return b.metadata.rating_count - a.metadata.rating_count;
    }
    return b.metadata.id - a.metadata.id;
  });

  CachedEmbeddings.set(embeddingName, embedding);
  return embedding;
};

export const loadNeighbors = async (embeddingName: EmbeddingName): Promise<number[][]> => {
  const cached = CachedNeighbors.get(embeddingName);
  if (cached) {
    return cached;
  }

  const rawEmbedding = await loadRawEmbedding(embeddingName);
  const embedding = await loadEmbedding(embeddingName);

  const idByOriginalIndex = rawEmbedding.ids;
  const originalIndexByID = new Map<number, number>();
  for (let i = 0; i < idByOriginalIndex.length; i++) {
    originalIndexByID.set(+idByOriginalIndex[i], i);
  }

  const neighbors = embedding.map(({ metadata: { id } }) => {
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

  CachedNeighbors.set(embeddingName, neighbors);
  return neighbors;
};
