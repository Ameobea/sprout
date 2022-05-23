/**
 * Take only this many items from the embedding, taking more popular items first
 */
export const RECOMMENDATION_MODEL_CORPUS_SIZE = 6000;

export enum ModelName {
  Model_6K = 'model_6k',
  Model_6K_TFLite = 'model_6k_tflite',
  Model_6K_Smaller = 'model_6k_smaller',
}

export const validateModelName = (name: string): ModelName | null => {
  switch (name) {
    case ModelName.Model_6K:
    case ModelName.Model_6K_TFLite:
    case ModelName.Model_6K_Smaller:
      return name as ModelName;
    default:
      console.error('Invalid model name: ' + name);
      return null;
  }
};

export const DEFAULT_MODEL_NAME = ModelName.Model_6K_Smaller;

export enum PopularityAttenuationFactor {
  None = 0,
  VeryLow = 0.0001,
  Low = 0.0004,
  Medium = 0.0008,
  High = 0.004,
  VeryHigh = 0.01,
}
export const DEFAULT_POPULARITY_ATTENUATION_FACTOR: PopularityAttenuationFactor = PopularityAttenuationFactor.Medium;
