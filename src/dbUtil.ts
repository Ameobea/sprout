import TimedCache from 'timed-cache';
import mysql from 'mysql';

import { MYSQL_DATABASE, MYSQL_HOST, MYSQL_PASSWORD, MYSQL_USER } from './conf';

const ConnCache = new TimedCache({ deafultTtl: 60 * 6 * 1000 });

export const getConn = () => {
  const cached = ConnCache.get('conn');
  if (cached) {
    return cached;
  }

  const conn = mysql.createConnection({
    host: MYSQL_HOST,
    user: MYSQL_USER,
    password: MYSQL_PASSWORD,
    database: MYSQL_DATABASE,
  });
  conn.connect();
  ConnCache.put('conn', conn);
  return conn;
};
