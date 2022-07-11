import * as tf from '@tensorflow/tfjs-node';
import { parentPort } from 'node:worker_threads';

const DATA_DIR = process.env['DATA_DIR'];
const RECOMMENDATION_MODEL_CORPUS_SIZE = 6500;

const ModelsCache = new Map();

export const getModelFilename = (modelName) => {
  switch (modelName) {
    case 'model_6k':
    case 'model_6k_smaller':
    case 'model_6k_smaller_weighted':
    case 'model_6-5k_new':
    case 'model_6-5k_unweighted':
      return `file://${DATA_DIR}/tfjs_models/${modelName}/model.json`;
    default:
      throw new Error(`Unimplemented model name: ${modelName}`);
  }
};

const loadTfJSModel = async (modelPath) => {
  const model = await tf.loadLayersModel(modelPath);
  console.log('Loaded model; compiling...');
  model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });
  console.log('Compiled model');
  return model;
};

export class TFJSInferrenceWrapper {
  constructor(model) {
    this.model = model;
  }

  async predict(input) {
    const batchSize = input.length / RECOMMENDATION_MODEL_CORPUS_SIZE;
    const inputTensor = tf.tensor(input, [batchSize, RECOMMENDATION_MODEL_CORPUS_SIZE]);
    const outputTensor = this.model.predict(inputTensor);
    const output = await outputTensor.data();
    tf.dispose(inputTensor);
    tf.dispose(outputTensor);
    return output;
  }
}

export const loadRecommendationModel = async (modelName) => {
  const cached = ModelsCache.get(modelName);
  if (cached) {
    return cached;
  }

  const modelPath = getModelFilename(modelName);
  const model = new TFJSInferrenceWrapper(await loadTfJSModel(modelPath));
  ModelsCache.set(modelName, model);
  return model;
};

parentPort.on('message', async (msg) => {
  const { token, modelName, inputs } = msg;
  const model = await loadRecommendationModel(modelName);

  const outputs = await model.predict(inputs);
  parentPort.postMessage({ token, outputs });
});
