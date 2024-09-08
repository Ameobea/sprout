import * as sqlite3Mod from 'sqlite3';

const sqlite3 = sqlite3Mod.default.verbose();

let LocalAnimelistsDB: sqlite3Mod.default.Database | null = null;

export const getLocalAnimelistsDB = () => {
  if (LocalAnimelistsDB) {
    return LocalAnimelistsDB;
  }
  LocalAnimelistsDB = new sqlite3.Database('/home/casey/anime-atlas/data/raw_animelists.db');
  return LocalAnimelistsDB!;
};
