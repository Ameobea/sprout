/**
 * Take only this many items from the embedding, taking more popular items first
 */
export const RECOMMENDATION_MODEL_CORPUS_SIZE = 6500;

export enum ModelName {
  // Model_6K = 'model_6k',
  // Model_6K_TFLite = 'model_6k_tflite',
  // Model_6K_Smaller = 'model_6k_smaller',
  // Model_6K_Smaller_Weighted = 'model_6k_smaller_weighted',
  // Model_6_5K_New = 'model_6-5k_new',
  // Model_6_5K_Unweighted = 'model_6-5k_unweighted',
  Model_6_5k_New2 = 'model_6-5k_new2',
  Model_6_5k_New2_Alt = 'model_6-5k_new2-alt',
}

export const getIsModelScoresWeighted = (modelName: ModelName): boolean => {
  switch (modelName) {
    // case ModelName.Model_6K_Smaller_Weighted:
    // case ModelName.Model_6_5K_New:
    // case ModelName.Model_6_5K_Unweighted:
    case ModelName.Model_6_5k_New2:
    case ModelName.Model_6_5k_New2_Alt:
      return true;
    default:
      return false;
  }
};

export const validateModelName = (name: string): ModelName | null => {
  switch (name) {
    // case ModelName.Model_6K:
    // case ModelName.Model_6K_TFLite:
    // case ModelName.Model_6K_Smaller:
    // case ModelName.Model_6K_Smaller_Weighted:
    // case ModelName.Model_6_5K_New:
    // case ModelName.Model_6_5K_Unweighted:
    case ModelName.Model_6_5k_New2:
    case ModelName.Model_6_5k_New2_Alt:
      return name as ModelName;
    default:
      console.error('Invalid model name: ' + name);
      return null;
  }
};

export const DEFAULT_MODEL_NAME = ModelName.Model_6_5k_New2;

export enum PopularityAttenuationFactor {
  None = 0,
  VeryLow = 0.0001,
  Low = 0.0004,
  Medium = 0.0008,
  High = 0.004,
  VeryHigh = 0.01,
  Extreme = 0.1,
}
export const DEFAULT_POPULARITY_ATTENUATION_FACTOR: PopularityAttenuationFactor = PopularityAttenuationFactor.Medium;

export enum ProfileSource {
  MyAnimeList = 'mal',
  AniList = 'anilist',
}
export const DEFAULT_PROFILE_SOURCE = ProfileSource.MyAnimeList;
