import * as tf from '@tensorflow/tfjs';
import type { Embedding } from 'src/routes/embedding';

import { EmbeddingName } from '../types';
import { DataContainer, fetchAllUsernames, fetchTrainingData, loadAnimeMetadata } from './data';

const EMBEDDING_NAME = EmbeddingName.PyMDE_3D_40N;

export type Logger = ((msg: string) => void) & { error: (msg: string) => void; warn: (msg: string) => void };

const trainIter = async (data: DataContainer, model: tf.Sequential, dataCount: number): Promise<tf.History> => {
  const [xTrain, yTrain] = await data.getTrainingData(dataCount);
  const history = await model.fit(xTrain, yTrain, {
    batchSize: 32,
    epochs: 1,
    validationSplit: 0, // validation handled manually
  });
  tf.dispose([xTrain, yTrain]);
  return history;
};

export const getRecommenderModelCompileParams = (embedding: Embedding) => {
  const nClasses = embedding.length;
  const totalRatingCount = embedding.reduce((acc, datum) => acc + datum.metadata.rating_count, 0);
  const ratingCountsTensor = tf.tensor1d(embedding.map((datum) => datum.metadata.rating_count));

  const classWeights = tf.mul(tf.add(tf.div(totalRatingCount, tf.mul(nClasses, ratingCountsTensor)), 0.5), 0.8);
  classWeights.print();

  return {
    optimizer: tf.train.sgd(0.25),
    loss: (yTrue, yPred) => {
      // Construct a mask for the non-zero entries in the yTrue tensor
      const nonZeroMask = yTrue.notEqual(0).asType('float32');
      const lossWeights = tf.mul(nonZeroMask.mul(8).add(1), classWeights);
      return tf.losses.meanSquaredError(yTrue, yPred, lossWeights);
    },
    metrics: ['accuracy'],
  };
};

const validateModel = async (data: DataContainer, model: tf.Sequential, log: Logger) => {
  const dataForValidationUser = await fetchTrainingData(['ameo___']);
  const [xValidate] = data.buildTrainingDataTensors(dataForValidationUser, false);
  log(xValidate.dataSync().toString());
  const predsTensor = (await model.predict(xValidate)) as tf.Tensor;
  log(predsTensor.dataSync().toString());
  const preds = predsTensor.dataSync() as Float32Array;

  console.log(
    'Current animelist: ',
    dataForValidationUser[0].map(({ animeIx }) => data.animeMetadata[animeIx]?.metadata.title).filter((x) => x)
  );

  // Get sorted indices of predictions
  const sortedPreds = [...preds].map((p, i) => [p, i]).sort(([p1], [p2]) => p2 - p1);
  const sortedPredsIndices = sortedPreds.map(([, i]) => i);

  const predictedTopAnimeAll = sortedPredsIndices.slice(0, 50).map((i) => data.animeMetadata[i].metadata.title);
  console.log('Top 50 predictions:');
  console.log(predictedTopAnimeAll);

  const predictedTopNew = sortedPredsIndices
    .filter((i) => !dataForValidationUser[0].some((ranking) => ranking.animeIx === i))
    .slice(0, 50)
    .map((i) => data.animeMetadata[i].metadata.title);
  console.log('Top 50 new anime:');
  console.log(predictedTopNew);
};

export const trainRecommender = async (iters: number, log: Logger, recordLoss: (loss: number) => void) => {
  (window as any).tf = tf;
  const animeMetadata = await loadAnimeMetadata(EMBEDDING_NAME);
  log(`Loaded anime metadata totalling ${animeMetadata.length} items`);

  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: [animeMetadata.length],
      units: 1024 * 18,
      activation: 'tanh',
      useBias: true,
      kernelInitializer: 'glorotNormal',
    })
  );
  model.add(
    tf.layers.dense({
      units: 1024 * 10,
      activation: 'tanh',
      useBias: true,
      kernelInitializer: 'glorotNormal',
    })
  );
  model.add(
    tf.layers.dense({
      units: 1024 * 10,
      activation: 'tanh',
      useBias: true,
      kernelInitializer: 'glorotNormal',
    })
  );
  model.add(
    tf.layers.dense({
      units: animeMetadata.length,
      activation: 'tanh',
      useBias: true,
      kernelInitializer: 'glorotNormal',
    })
  );
  log('\nBuilt model:');
  model.summary(undefined, undefined, log);
  model.compile(getRecommenderModelCompileParams(animeMetadata));

  const allUsernames = await fetchAllUsernames();
  const data = new DataContainer(tf, allUsernames, animeMetadata);

  for (let i = 0; i < iters; i++) {
    const history = await trainIter(data, model, 5_000);
    log(`Training iter ${i + 1} complete`);
    log(`Train loss: ${history.history.loss[0]}`);
    recordLoss(history.history.loss[0] as number);
  }

  await validateModel(data, model, log);

  // serialize the model to disk
  await model.save('downloads://model');
};
