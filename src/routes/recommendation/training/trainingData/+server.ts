import { error, json, type RequestHandler } from '@sveltejs/kit';

import { getLocalAnimelistsDB } from '../localAnimelistsDB';
import { convertMALProfileToTrainingData, type TrainingDatum } from './trainingData';
import type { MALUserAnimeListItem } from 'src/malAPI';

const getTrainingDataFromProcessedTable = async (
  usernames: string[]
): Promise<{ username: string; trainingData: TrainingDatum[] }[]> => {
  const stmt = getLocalAnimelistsDB().prepare(
    `SELECT username, processed_animelist FROM \`processed-training-data\` WHERE username IN (${usernames
      .map(() => '?')
      .join(',')})`
  );
  return new Promise<{ username: string; trainingData: TrainingDatum[] }[]>((resolve, reject) => {
    stmt.all<{ username: string; processed_animelist: string }>(usernames, (err, rows: any[]) => {
      if (err) {
        reject(err);
      } else {
        resolve(
          rows
            .map((row): { username: string; trainingData: any } => {
              let trainingData;
              try {
                trainingData = JSON.parse(row.processed_animelist);
              } catch (e) {
                console.error(`Error parsing processed animelist for ${row.username}:`, e);
              }
              return { username: row.username, trainingData };
            })
            .filter((r) => !!r.trainingData)
        );
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
    stmt.all<{ username: string; animelist_json: string }>(usernames, (err, rows) => {
      if (err) {
        reject(err);
      } else {
        resolve(
          rows
            .map((row) => {
              let animelist;
              try {
                animelist = JSON.parse(row.animelist_json);
              } catch (e) {
                console.error(`Error parsing animelist for ${row.username}:`, e);
              }
              return { username: row.username, animelist };
            })
            .filter((r) => !!r.animelist)
        );
      }
    });
  });

  const parsedAnimeLists = (await convertMALProfileToTrainingData(animeLists.map((user) => user.animelist))).map(
    ({ ratings }) => ratings
  );

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

let didInit = false;

const initOnce = () => {
  getLocalAnimelistsDB().exec(
    'CREATE TABLE IF NOT EXISTS `processed-training-data` (username TEXT, processed_animelist TEXT)',
    (err) => {
      if (err) {
        console.error('Error creating processed-training-data table', err);
        throw err;
      }

      getLocalAnimelistsDB().exec(
        'CREATE INDEX IF NOT EXISTS `processed-training-data-username-index` ON `processed-training-data` (username)',
        (err) => {
          if (err) {
            console.error('Error creating processed-training-data-username-index index', err);
            throw err;
          }
        }
      );
    }
  );

  getLocalAnimelistsDB().exec(
    'CREATE INDEX IF NOT EXISTS `mal-user-animelists-username-index` ON `mal-user-animelists` (username)',
    (err) => {
      if (err) {
        console.error('Error creating mal-user-animelists-username-index index', err);
        throw err;
      }
    }
  );
};

export const POST: RequestHandler = async ({ request }) => {
  if (!didInit) {
    initOnce();
    didInit = true;
  }

  const allUsernames = await request.json();
  if (!Array.isArray(allUsernames)) {
    error(400, 'Invalid body; must be an array of usernames');
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

  return json({ trainingData: trainingData as any });
};
