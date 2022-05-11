import * as tf from '@tensorflow/tfjs-node';

import { RECOMMENDATION_MODEL_CORPUS_SIZE } from 'src/components/recommendation/conf';
import { DATA_DIR } from 'src/conf';
import { getRecommenderModelCompileParams } from 'src/training/trainRecommender';
import type { Embedding } from '../embedding';

let LoadedModel: tf.LayersModel | null = null;

export const loadRecommendationModel = async (embedding: Embedding): Promise<tf.LayersModel> => {
  if (LoadedModel) {
    return LoadedModel;
  }

  const modelPath = `file://${DATA_DIR}/tfjs_models/model_4k_v2/model_4k_v2.json`;
  const model = await tf.loadLayersModel(modelPath);
  console.log('Loaded model; compiling...');
  model.compile(getRecommenderModelCompileParams(embedding.slice(0, RECOMMENDATION_MODEL_CORPUS_SIZE)));
  console.log('Compiled model');
  LoadedModel = model;
  return model;
};
