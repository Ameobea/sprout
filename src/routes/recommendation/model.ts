import * as tf from '@tensorflow/tfjs-node';

import { ModelName, RECOMMENDATION_MODEL_CORPUS_SIZE } from 'src/components/recommendation/conf';
import { DATA_DIR } from 'src/conf';
import { getRecommenderModelCompileParams } from 'src/training/trainRecommender';
import type { Embedding } from '../embedding';

const ModelsCache: Map<ModelName, tf.LayersModel> = new Map();

export const loadRecommendationModel = async (embedding: Embedding, modelName: ModelName): Promise<tf.LayersModel> => {
  const cached = ModelsCache.get(modelName);
  if (cached) {
    return cached;
  }

  const modelPath = `file://${DATA_DIR}/tfjs_models/model_4k/model_4k.json`;
  const model = await tf.loadLayersModel(modelPath);
  console.log('Loaded model; compiling...');
  model.compile(getRecommenderModelCompileParams(embedding.slice(0, RECOMMENDATION_MODEL_CORPUS_SIZE)));
  console.log('Compiled model');
  ModelsCache.set(modelName, model);
  return model;
};
