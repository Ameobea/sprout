import mysql from 'mysql';

import { MYSQL_DATABASE, MYSQL_HOST, MYSQL_PASSWORD, MYSQL_USER } from './conf';

export const DbPool = mysql.createPool({
  connectionLimit: 4,
  host: MYSQL_HOST,
  user: MYSQL_USER,
  password: MYSQL_PASSWORD,
  database: MYSQL_DATABASE,
  charset: 'utf8mb4_unicode_ci',
});

export const dbQuery = <T = any>(sql: string, values?: unknown[]): Promise<T> =>
  new Promise((resolve, reject) =>
    DbPool.query(sql, values, (err, results) => (err ? reject(err) : resolve(results)))
  );
