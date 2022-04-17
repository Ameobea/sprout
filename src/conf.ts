const loadEnv = (key: string) => {
  const value = process.env[key];
  if (!value) {
    throw new Error(`Missing environment variable ${key}`);
  }
  return value;
};

export const MAL_CLIENT_ID = loadEnv('MAL_CLIENT_ID');
export const MAL_CLIENT_SECRET = loadEnv('MAL_CLIENT_SECRET');

export const MAL_API_BASE_URL = 'https://api.myanimelist.net/v2';
