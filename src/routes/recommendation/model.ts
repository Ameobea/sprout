import { DATA_DIR } from 'src/conf';

import * as tf from '@tensorflow/tfjs-node';
import { getRecommenderModelCompileParams } from 'src/training/trainRecommender';

let LoadedModel: tf.LayersModel | null = null;

/**
 * Take only this many items from the embedding, taking more popular items first
 */
export const RECOMMENDATION_MODEL_CORPUS_SIZE = 4000;

export const loadRecommendationModel = async (): Promise<tf.LayersModel> => {
  if (LoadedModel) {
    return LoadedModel;
  }

  const modelPath = `file://${DATA_DIR}/tfjs_models/model_4k/model_4k.json`;
  const model = await tf.loadLayersModel(modelPath);
  console.log('Loaded model; compiling...');
  model.compile(getRecommenderModelCompileParams());
  console.log('Compiled model');
  LoadedModel = model;
  return model;
};
