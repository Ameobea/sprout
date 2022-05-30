import type { RequestHandler } from '@sveltejs/kit';

import { RECOMMENDATION_MODEL_CORPUS_SIZE } from 'src/components/recommendation/conf';
import { loadEmbedding } from 'src/embedding';
import { EmbeddingName } from 'src/types';
import { getEmbeddingMetadata } from './recommendation/recommendation';

let CachedEmbeddingPartialMetadata:
  | {
      metadata: {
        id: number;
        title: string;
        title_english: string;
        imageSrc: string;
      };
    }[]
  | null = null;

const getEmbeddingPartialMetadata = async () => {
  if (CachedEmbeddingPartialMetadata) {
    return CachedEmbeddingPartialMetadata;
  }

  const embedding = (await loadEmbedding(EmbeddingName.PyMDE_4D_40N)).slice(0, RECOMMENDATION_MODEL_CORPUS_SIZE);
  const embeddingMetadata = await getEmbeddingMetadata(embedding);
  const trimmed = embeddingMetadata
    .filter((entry) => entry)
    .map((entry) => ({
      metadata: {
        id: entry!.id,
        title: entry!.title,
        title_english: entry!.alternative_titles?.en || '',
        imageSrc: entry?.main_picture?.medium ?? '',
      },
    }));
  CachedEmbeddingPartialMetadata = trimmed;
  return trimmed;
};

export const get: RequestHandler = async () => ({
  status: 200,
  body: {
    embeddingMetadata: await getEmbeddingPartialMetadata(),
  },
});
