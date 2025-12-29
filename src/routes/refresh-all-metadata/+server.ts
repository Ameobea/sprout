import type { RequestHandler } from '@sveltejs/kit';

import { refreshAllMetadataInDB } from 'src/malAPI';

export const POST: RequestHandler = async () => {
  await refreshAllMetadataInDB();

  return new Response(undefined, { status: 204 });
};
