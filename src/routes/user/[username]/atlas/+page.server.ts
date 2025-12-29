import { typify } from 'src/components/recommendation/utils';
import { loadEmbedding } from 'src/embedding';
import { EmbeddingName } from 'src/types';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async () => {
  const embeddingName = EmbeddingName.Model;
  const embedding = await loadEmbedding(embeddingName);

  return typify({ embeddingName, embedding });
};
