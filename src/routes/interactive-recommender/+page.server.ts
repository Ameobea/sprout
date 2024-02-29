import { RECOMMENDATION_MODEL_CORPUS_SIZE } from 'src/components/recommendation/conf';
import { loadEmbedding } from 'src/embedding';
import { EmbeddingName } from 'src/types';
import { getEmbeddingMetadata } from '../recommendation/recommendation/recommendation';
import type { PageServerLoad } from './$types';

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

export const load: PageServerLoad = async () => ({
  embeddingMetadata: await getEmbeddingPartialMetadata(),
});
