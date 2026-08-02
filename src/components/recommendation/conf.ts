export const RECOMMENDATION_MODEL_CORPUS_SIZE = 6000;

export enum ModelName {
  // logq lift model on Aug 2026 data; niche knob drives the (alpha, k) serving path server-side
  Model_2026_logq = '2026-logq',
  Model_2025_jax = '2025-jax',
  // Served by proxying to the still-running pre-2026 version of the app with its tf.js model
  Legacy_2023 = 'legacy',
}

export const getIsModelScoresWeighted = (modelName: ModelName): boolean => {
  switch (modelName) {
    case ModelName.Model_2026_logq:
    case ModelName.Model_2025_jax:
      return true;
    default:
      return false;
  }
};

export const validateModelName = (name: string): ModelName | null => {
  switch (name) {
    case ModelName.Model_2026_logq:
    case ModelName.Model_2025_jax:
    case ModelName.Legacy_2023:
      return name as ModelName;
    default:
      console.error('Invalid model name: ' + name);
      return null;
  }
};

export const DEFAULT_MODEL_NAME = ModelName.Model_2026_logq;

// The 2026 knob path deliberately doesn't default to 0; see logq-presence-prior.md
export const getDefaultNicheBoostFactor = (modelName: ModelName): number =>
  modelName === ModelName.Model_2026_logq ? 0.35 : 0;

// Only used by the legacy model
export enum PopularityAttenuationFactor {
  None = 0,
  VeryLow = 0.0001,
  Low = 0.0004,
  Medium = 0.0008,
  High = 0.004,
  VeryHigh = 0.01,
}
export const DEFAULT_POPULARITY_ATTENUATION_FACTOR: number = PopularityAttenuationFactor.Medium;

export enum ProfileSource {
  MyAnimeList = 'mal',
  AniList = 'anilist',
}
export const DEFAULT_PROFILE_SOURCE = ProfileSource.MyAnimeList;
