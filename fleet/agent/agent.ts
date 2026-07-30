import { getUserAnimeList, getUserMangaList, MALAPIError } from '../../src/malAPI';
import { delay } from '../../src/util';

const BASE = process.env.MAIN_SERVER_URL ?? 'https://anime.ameo.dev';
const TOKEN = process.env.ADMIN_API_TOKEN!;
const TYPE = process.env.COLLECTION_TYPE === 'manga' ? 'manga' : 'anime';
const PACE_MS = +(process.env.PACE_MS ?? 1200);
const IDLE_MS = 60_000;

const fetchList = TYPE === 'manga' ? getUserMangaList : getUserAnimeList;

const collectOne = async (): Promise<boolean> => {
  const nextRes = await fetch(`${BASE}/next-username-to-collect?token=${TOKEN}&type=${TYPE}`);
  if (nextRes.status === 204) {
    return false;
  }
  if (!nextRes.ok) {
    throw new Error(`next-username-to-collect: ${nextRes.status} ${await nextRes.text()}`);
  }
  const username = await nextRes.text();

  let status = 200;
  let list: unknown[] | null = null;
  try {
    list = await fetchList(username);
  } catch (err) {
    status = err instanceof MALAPIError ? err.statusCode : 500;
    console.error(`Failed fetching ${TYPE} list for ${username}: `, err);
  }

  const submitRes = await fetch(`${BASE}/submit-collected-list?token=${TOKEN}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, type: TYPE, status, list }),
  });
  if (!submitRes.ok) {
    throw new Error(`submit-collected-list: ${submitRes.status} ${await submitRes.text()}`);
  }
  console.log(`${username}: ${status}${status === 200 ? ` (${list!.length} entries)` : ''}`);
  return true;
};

const main = async () => {
  console.log(`mal-collector agent starting: type=${TYPE} pace=${PACE_MS}ms server=${BASE}`);
  for (;;) {
    let collected = false;
    try {
      collected = await collectOne();
      if (!collected) {
        console.log('Queue drained; idling');
      }
    } catch (err) {
      console.error('Collection iteration failed: ', err);
    }
    await delay(collected ? PACE_MS : IDLE_MS);
  }
};

main();
