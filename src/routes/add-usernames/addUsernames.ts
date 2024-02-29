import { error } from '@sveltejs/kit';
import { DbPool } from 'src/dbUtil';

export const addUsernames = async (usernames: string[]) => {
  let attempts = 0;
  for (;;) {
    try {
      const insertedRowCount = await new Promise((resolve, reject) =>
        DbPool.query(
          'INSERT IGNORE INTO `usernames-to-collect` (username) VALUES ?',
          [usernames.map((username) => [username])],
          (err, results) => {
            if (err) {
              reject(err);
            } else {
              resolve(results.affectedRows);
            }
          }
        )
      );
      return `Successfully inserted ${insertedRowCount} usernames`;
    } catch (err) {
      attempts += 1;
      console.error('Error inserting usernames: ', err);

      if (attempts >= 10) {
        console.error(`Failed to insert usernames after ${attempts} attempts; giving up`);
        error(500, 'Error inserting usernames');
      }
    }
  }
};
