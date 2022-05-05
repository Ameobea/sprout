import * as tf from '@tensorflow/tfjs';

import { MyRNN, MySimpleRNNCell } from './customRNN';
import type { Embedding } from 'src/routes/embedding';
import { EmbeddingName } from '../types';

const EMBEDDING_NAME = EmbeddingName.PyMDE;
/**
 * Take only this many items from the embedding, taking more popular items first
 */
const RETAINED_CORPUS_SIZE = 5000;

export type Logger = ((msg: string) => void) & { error: (msg: string) => void; warn: (msg: string) => void };

const loadEmbedding = async (embeddingName: EmbeddingName): Promise<Embedding> => {
  const { embedding }: { embedding: Embedding } = await fetch(`/embedding?embedding=${embeddingName}`).then((res) =>
    res.json()
  );
  return embedding.slice(0, RETAINED_CORPUS_SIZE);
};

const fetchTrainingData = async (seqCount: number) => {
  // const examples;
};

export const trainRecommender = async (log: Logger) => {
  const embedding = await loadEmbedding(EMBEDDING_NAME);
  log(`Loaded embedding and metadata totalling ${embedding.length} items`);

  const maxSequenceLength = 100;

  const model = tf.sequential();
  model.add(tf.layers.masking({ maskValue: -1, inputShape: [maxSequenceLength, embedding.length] }));
  model.add(
    new MyRNN({
      cell: [
        new MySimpleRNNCell({
          stateSize: 512,
          outputDim: 2048,
          recurrentActivation: 'tanh',
          outputActivation: 'tanh',
          useOutputBias: true,
          useRecurrentBias: false,
          recurrentInitializer: 'glorotNormal',
        }),
        new MySimpleRNNCell({
          stateSize: 256,
          outputDim: embedding.length,
          recurrentActivation: 'tanh',
          outputActivation: 'sigmoid',
          useOutputBias: true,
          useRecurrentBias: false,
          recurrentInitializer: 'glorotNormal',
        }),
      ],
      returnSequences: false,
      trainableInitialState: true,
    })
  );
  log('\nBuilt model:');
  model.summary(undefined, undefined, log);
};
