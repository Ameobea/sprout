import * as t from 'io-ts';

/**
 * io-ts `t.number` accepts NaN and Infinity (bare typeof check), and those reach SQL as bare
 * identifiers (`IN (NaN)`) which error out the query. Always validate IDs with these.
 */
export const isValidAnimeID = (id: unknown): id is number =>
  typeof id === 'number' && Number.isSafeInteger(id) && id > 0;

export const AnimeID = new t.Type<number, number, unknown>(
  'AnimeID',
  isValidAnimeID,
  (u, c) => (isValidAnimeID(u) ? t.success(u) : t.failure(u, c)),
  t.identity
);

export const boundedArray = <C extends t.Mixed>(codec: C, maxLength: number, name: string) =>
  new t.Type<t.TypeOf<C>[], t.OutputOf<C>[], unknown>(
    name,
    (u): u is t.TypeOf<C>[] => Array.isArray(u) && u.length <= maxLength && u.every(codec.is),
    (u, c) => {
      if (!Array.isArray(u)) {
        return t.failure(u, c, 'must be an array');
      }
      if (u.length > maxLength) {
        return t.failure(u, c, `must contain at most ${maxLength} entries (got ${u.length})`);
      }
      return t.array(codec).validate(u, c);
    },
    t.identity
  );

export const MAX_PROFILE_ENTRIES = 5000;
export const MAX_EXCLUDED_IDS = 2000;
