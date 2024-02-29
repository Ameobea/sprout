import { EmbeddingName } from '../../types';
import { loadEmbedding } from '../../embedding';
import { typify } from 'src/components/recommendation/utils';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async () => {
  const embeddingName = EmbeddingName.PyMDE_3D_40N;
  const embedding = await loadEmbedding(embeddingName);
  return typify({ embedding, embeddingName });
};
