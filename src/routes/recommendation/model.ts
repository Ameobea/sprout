import * as tf from '@tensorflow/tfjs-node';
// import { Interpreter } from 'node-tflite';

import { ModelName, RECOMMENDATION_MODEL_CORPUS_SIZE } from 'src/components/recommendation/conf';
import { DATA_DIR } from 'src/conf';
import { getRecommenderModelCompileParams } from 'src/training/trainRecommender';
import type { Embedding } from '../embedding/+server';

const ModelsCache: Map<ModelName, TFJSInferrenceWrapper /* | TFLiteInferrenceWrapper */> = new Map();

export const getModelFilename = (modelName: ModelName): string => {
  switch (modelName) {
    // case ModelName.Model_6K:
    // case ModelName.Model_6K_Smaller:
    // case ModelName.Model_6K_Smaller_Weighted:
    // case ModelName.Model_6_5K_New:
    // case ModelName.Model_6_5K_Unweighted:
    // return `file://${DATA_DIR}/tfjs_models/${modelName}/model.json`;
    // case ModelName.Model_6K_TFLite:
    //   return `${DATA_DIR}/tflite_models/model_6k/model_6k.tflite`;
    default:
      return `file://${DATA_DIR}/tfjs_models/${modelName}/${modelName}.json`;
  }
};

const loadTfJSModel = async (embedding: Embedding, modelPath: string): Promise<tf.LayersModel> => {
  const model = await tf.loadLayersModel(modelPath);
  console.log('Loaded model; compiling...');
  model.compile(getRecommenderModelCompileParams(embedding.slice(0, RECOMMENDATION_MODEL_CORPUS_SIZE)));
  console.log('Compiled model');
  return model;
};

// export class TFLiteInferrenceWrapper {
//   private interpreter: Interpreter;

//   constructor(interpreter: Interpreter) {
//     interpreter.allocateTensors();
//     this.interpreter = interpreter;
//   }

//   public predict(inputData: Float32Array) {
//     const batchSize = inputData.length / RECOMMENDATION_MODEL_CORPUS_SIZE;
//     if (this.interpreter.inputs[0].dims[0] !== batchSize) {
//       console.log('resize start');
//       this.interpreter.resizeInputTensor(0, [batchSize, RECOMMENDATION_MODEL_CORPUS_SIZE]);
//       this.interpreter.allocateTensors();
//       console.log('resize end');
//     }
//     this.interpreter.inputs[0].copyFrom(inputData);

//     const now = Date.now();
//     console.log('invoke start: ', batchSize);
//     this.interpreter.invoke();
//     console.log('invoke end: ', Date.now() - now);

//     const outputData = new Float32Array(this.interpreter.outputs[0].byteSize / 4);
//     this.interpreter.outputs[0].copyTo(outputData);
//     return outputData;
//   }
// }

export class TFJSInferrenceWrapper {
  private model: tf.LayersModel;

  constructor(model: tf.LayersModel) {
    this.model = model;
  }

  public async predict(input: Float32Array) {
    const batchSize = input.length / RECOMMENDATION_MODEL_CORPUS_SIZE;
    const inputTensor = tf.tensor(input, [batchSize, RECOMMENDATION_MODEL_CORPUS_SIZE]);
    const outputTensor = this.model.predict(inputTensor) as tf.Tensor;
    const output = (await outputTensor.data()) as Float32Array;
    tf.dispose(inputTensor);
    tf.dispose(outputTensor);
    return output;
  }
}

// const loadTfliteModel = async (modelPath: string): Promise<TFLiteInferrenceWrapper> => {
//   const modelData = await new Promise<Buffer>((resolve, reject) => {
//     fs.readFile(modelPath, (err, data) => {
//       if (err) {
//         reject(err);
//       } else {
//         resolve(data);
//       }
//     });
//   });

//   const interpreter = new Interpreter(modelData, { numThreads: 10 });
//   return new TFLiteInferrenceWrapper(interpreter);
// };

export const loadRecommendationModel = async (
  embedding: Embedding,
  modelName: ModelName
): Promise<TFJSInferrenceWrapper /* | TFLiteInferrenceWrapper */> => {
  const cached = ModelsCache.get(modelName);
  if (cached) {
    return cached;
  }

  const modelPath = getModelFilename(modelName);
  if (modelPath.endsWith('tflite')) {
    // const model = await loadTfliteModel(modelPath);
    // ModelsCache.set(modelName, model);
    // return model;
    throw new Error('disabled');
  }

  const model = new TFJSInferrenceWrapper(await loadTfJSModel(embedding, modelPath));
  ModelsCache.set(modelName, model);
  return model;
};
