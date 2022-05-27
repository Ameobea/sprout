import * as fs from 'fs';
import { DATA_DIR } from './conf';

export interface RawOfflineMetadataDB {
  license: License;
  repository: string;
  data: OfflineAnimeMetadatum[];
}

export interface License {
  name: string;
  url: string;
}

export interface OfflineAnimeMetadatum {
  sources: string[];
  title: string;
  type: string;
  episodes: number;
  status: string;
  animeSeason: AnimeSeason;
  picture: string;
  thumbnail: string;
  synonyms: string[];
  relations: string[];
  tags: string[];
}

export interface AnimeSeason {
  season: string;
  year: number;
}

interface OfflineMetadataDB {
  byMALID: Map<number, OfflineAnimeMetadatum>;
}

const loadRawOfflineMetadataDB = async (): Promise<RawOfflineMetadataDB> => {
  // Try to load from file first
  const offlineMetadataDBFile = `${DATA_DIR}/anime-offline-database.json`;
  if (fs.existsSync(offlineMetadataDBFile)) {
    const rawOfflineMetadataDB: RawOfflineMetadataDB = JSON.parse(fs.readFileSync(offlineMetadataDBFile, 'utf8'));
    return rawOfflineMetadataDB;
  }

  // If not found, download from github
  console.log('Local offline anime database not found; downloading database from GH...');
  const url =
    'https://github.com/manami-project/anime-offline-database/raw/master/anime-offline-database-minified.json';
  const rawOfflineMetadataDB: RawOfflineMetadataDB = await fetch(url).then((res) => res.json());
  return rawOfflineMetadataDB;
};

const loadOfflineMetadataDB = async (): Promise<OfflineMetadataDB> => {
  console.log('Loading offline anime metadata database...');
  const rawOfflineMetadataDB = await loadRawOfflineMetadataDB();
  const offlineMetadataDB: OfflineMetadataDB = {
    byMALID: new Map(),
  };

  for (const datum of rawOfflineMetadataDB.data) {
    const malURL = datum.sources.find((url) => url.includes('myanimelist.net/anime/'));
    if (!malURL) {
      continue;
    }

    const rawID = malURL.split('/').pop() ?? '';
    const malID = parseInt(rawID);
    if (isNaN(malID)) {
      console.error(`Invalid MAL ID: ${rawID}`);
      continue;
    }

    offlineMetadataDB.byMALID.set(malID, datum);
  }

  console.log(`Successfully loaded offline anime metadata database with ${offlineMetadataDB.byMALID.size} entries`);

  return offlineMetadataDB;
};

let CachedOfflineMetadataDB: OfflineMetadataDB | null = null;

export const getOfflineMetadataDB = async (): Promise<OfflineMetadataDB> => {
  if (CachedOfflineMetadataDB) {
    return CachedOfflineMetadataDB;
  }

  CachedOfflineMetadataDB = await loadOfflineMetadataDB();
  return CachedOfflineMetadataDB;
};
