import type { RequestHandler } from '@sveltejs/kit';

import { loadNeighbors } from 'src/embedding';

export const get: RequestHandler = async () => {
  const neighbors = await loadNeighbors();
  return { body: { neighbors } };
};
