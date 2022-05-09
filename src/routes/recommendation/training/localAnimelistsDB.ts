import sqlite3Mod from 'sqlite3';

const sqlite3 = sqlite3Mod.verbose();

export const LocalAnimelistsDB = new sqlite3.Database('/home/casey/anime-atlas/data/raw_animelists.db');
