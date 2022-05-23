const loadEnv = (key: string, defaultValue?: string) => {
  const value = process.env[key];
  if (!value) {
    if (defaultValue) {
      return defaultValue;
    } else {
      throw new Error(`Missing environment variable ${key}`);
    }
  }
  return value;
};

export const IS_DOCKER = loadEnv('IS_DOCKER', 'false') === 'true';
export const DATA_DIR = loadEnv('DATA_DIR', '/opt/data');

export const MAL_CLIENT_ID = loadEnv('MAL_CLIENT_ID');
export const MAL_CLIENT_SECRET = loadEnv('MAL_CLIENT_SECRET');

export const MAL_API_BASE_URL = 'https://api.myanimelist.net/v2';

export const ADMIN_API_TOKEN = loadEnv('ADMIN_API_TOKEN');

export const MYSQL_HOST = loadEnv('MYSQL_HOST');
export const MYSQL_USER = loadEnv('MYSQL_USER');
export const MYSQL_PASSWORD = loadEnv('MYSQL_PASSWORD');
export const MYSQL_DATABASE = loadEnv('MYSQL_DATABASE');
