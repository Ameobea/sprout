import { EmbeddingName } from '../../types';
import { loadEmbedding } from '../../embedding';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async () => {
  const embeddingName = EmbeddingName.GGVec_10D_40N_Order2;
  const embedding = await loadEmbedding(embeddingName);
  return { embedding, embeddingName };
};
