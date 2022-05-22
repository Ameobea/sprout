/**
 * Take only this many items from the embedding, taking more popular items first
 */
export const RECOMMENDATION_MODEL_CORPUS_SIZE = 6000;

export enum ModelName {
  Model_4K = 'model_4k',
  Model_4K_V2 = 'model_4k_v2',
  Model_6K = 'model_6k',
}

export const validateModelName = (name: string): ModelName | null => {
  switch (name) {
    case ModelName.Model_4K:
    case ModelName.Model_4K_V2:
    case ModelName.Model_6K:
      return name as ModelName;
    default:
      console.error('Invalid model name: ' + name);
      return null;
  }
};

export const DEFAULT_MODEL_NAME = ModelName.Model_6K;

export enum PopularityAttenuationFactor {
  None = 0,
  VeryLow = 0.0001,
  Low = 0.0004,
  Medium = 0.0008,
  High = 0.004,
  VeryHigh = 0.01,
}
export const DEFAULT_POPULARITY_ATTENUATION_FACTOR: PopularityAttenuationFactor = PopularityAttenuationFactor.Low;
