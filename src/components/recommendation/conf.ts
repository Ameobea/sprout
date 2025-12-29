export const RECOMMENDATION_MODEL_CORPUS_SIZE = 6000;

export enum ModelName {
  Model_2025_jax = '2025-jax',
}

export const getIsModelScoresWeighted = (modelName: ModelName): boolean => {
  switch (modelName) {
    case ModelName.Model_2025_jax:
      return true;
    default:
      return false;
  }
};

export const validateModelName = (name: string): ModelName | null => {
  switch (name) {
    case ModelName.Model_2025_jax:
      return name as ModelName;
    default:
      console.error('Invalid model name: ' + name);
      return null;
  }
};

export const DEFAULT_MODEL_NAME = ModelName.Model_2025_jax;

export enum ProfileSource {
  MyAnimeList = 'mal',
  AniList = 'anilist',
}
export const DEFAULT_PROFILE_SOURCE = ProfileSource.MyAnimeList;
