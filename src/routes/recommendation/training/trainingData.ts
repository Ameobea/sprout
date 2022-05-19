import type { RequestHandler } from '@sveltejs/kit';

import { getLocalAnimelistsDB } from './localAnimelistsDB';
import { loadEmbedding } from 'src/embedding';
import { AnimeListStatusCode, type MALUserAnimeListItem } from 'src/malAPI';
import { EmbeddingName } from 'src/types';

export interface TrainingDatum {
  animeIx: number;
  rating: number;
}

let AnimeIxByID: Map<number, number> | null = null;

const getAnimeIxByID = async (): Promise<Map<number, number>> => {
  if (!AnimeIxByID) {
    const embedding = await loadEmbedding(EmbeddingName.PyMDE_3D_40N);
    AnimeIxByID = new Map<number, number>();
    embedding.forEach((datum, i) => AnimeIxByID!.set(datum.metadata.id, i));
  }
  return AnimeIxByID;
};

export const convertMALProfileToTrainingData = async (
  rawData: MALUserAnimeListItem[][]
): Promise<TrainingDatum[][]> => {
  const AnimeIxByID = await getAnimeIxByID();

  return rawData.map((userProfileAnime) => {
    const validRatings = userProfileAnime.filter((datum) =>
      [
        AnimeListStatusCode.Completed,
        AnimeListStatusCode.Watching,
        AnimeListStatusCode.OnHold,
        AnimeListStatusCode.Dropped,
      ].includes(datum.list_status.status)
    );
    let unratedCount = 0;
    for (const rating of validRatings) {
      if (!rating?.list_status?.score) {
        unratedCount += 1;
      }
    }
    const userIsNonRater = unratedCount / userProfileAnime.length > 0.5;
    if (userIsNonRater) {
      console.log({ userIsNonRater });
    }

    return validRatings
      .filter((datum) => userIsNonRater || datum.list_status?.score > 0)
      .map((datum) => ({
        animeIx: AnimeIxByID.get(datum.node.id)!,
        rating: userIsNonRater && datum.list_status.score === 0 ? 7 : datum.list_status.score,
      }));
  });
};

const getTrainingDataFromProcessedTable = async (
  usernames: string[]
): Promise<{ username: string; trainingData: TrainingDatum[] }[]> => {
  const stmt = getLocalAnimelistsDB().prepare(
    `SELECT username, processed_animelist FROM \`processed-training-data\` WHERE username IN (${usernames
      .map(() => '?')
      .join(',')})`
  );
  return new Promise<{ username: string; trainingData: TrainingDatum[] }[]>((resolve, reject) => {
    stmt.all(usernames, (err, rows) => {
      if (err) {
        reject(err);
      } else {
        resolve(rows.map((row) => ({ username: row.username, trainingData: JSON.parse(row.processed_animelist) })));
      }
    });
  });
};

const getTrainingDataFromRawTable = async (usernames: string[]): Promise<TrainingDatum[][]> => {
  const replacers = usernames.map(() => '?').join(', ');
  const stmt = getLocalAnimelistsDB().prepare(
    `SELECT username, animelist_json FROM \`mal-user-animelists\` WHERE username IN (${replacers})`
  );
  const animeLists = await new Promise<{ username: string; animelist: MALUserAnimeListItem[] }[]>((resolve, reject) => {
    stmt.all(usernames, (err, rows) => {
      if (err) {
        reject(err);
      } else {
        resolve(
          rows.map((row) => ({
            username: row.username,
            animelist: JSON.parse(row.animelist_json),
          }))
        );
      }
    });
  });

  const parsedAnimeLists = await convertMALProfileToTrainingData(animeLists.map((user) => user.animelist));

  // Insert into the processed table for faster access next time
  if (parsedAnimeLists.length > 0) {
    const stmt2 = getLocalAnimelistsDB().prepare(
      'INSERT OR IGNORE INTO `processed-training-data` (username, processed_animelist) VALUES ' +
        animeLists.map(() => '(?, ?)').join(', ')
    );
    const binds: string[] = [];
    for (let i = 0; i < animeLists.length; i++) {
      const username = animeLists[i].username;
      const animeList = JSON.stringify(parsedAnimeLists[i]);
      binds.push(username);
      binds.push(animeList);
    }
    await new Promise<void>((resolve, reject) =>
      stmt2.run(binds, (err) => {
        if (err) {
          reject(err);
        } else {
          resolve(undefined);
        }
      })
    );
  }

  return parsedAnimeLists;
};

export const post: RequestHandler = async ({ request }) => {
  const allUsernames = await request.json();
  if (!Array.isArray(allUsernames)) {
    return { status: 400, body: 'Invalid body; must be an array of usernames' };
  }

  // Try to get as much data as possible from the processed table
  const processedData = await getTrainingDataFromProcessedTable(allUsernames);
  const gotUsernames = new Set(processedData.map((datum) => datum.username));
  const missingUsernames = allUsernames.filter((username) => !gotUsernames.has(username));

  const trainingDataFromRaw = await getTrainingDataFromRawTable(missingUsernames);

  const dataByUsername = new Map<string, TrainingDatum[]>();
  processedData.forEach((datum) => dataByUsername.set(datum.username, datum.trainingData));
  trainingDataFromRaw.forEach((datum, i) => dataByUsername.set(missingUsernames[i], datum));

  const trainingData = allUsernames.map((username) => {
    const animeList = dataByUsername.get(username);
    if (!animeList) {
      throw new Error(`No training data found for username ${username}`);
    }
    return animeList;
  });

  return { status: 200, body: { trainingData: trainingData as any } };
};
