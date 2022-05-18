import sqlite3Mod, { Database } from 'sqlite3';

const sqlite3 = sqlite3Mod.verbose();

let LocalAnimelistsDB: Database | null = null;

export const getLocalAnimelistsDB = () => {
  if (LocalAnimelistsDB) {
    return LocalAnimelistsDB;
  }
  LocalAnimelistsDB = new sqlite3.Database('/home/casey/anime-atlas/data/raw_animelists.db');
  return LocalAnimelistsDB!;
};
