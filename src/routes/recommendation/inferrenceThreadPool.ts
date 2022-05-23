import type { ModelName } from 'src/components/recommendation/conf';
import { IS_DOCKER } from 'src/conf';
import { Worker } from 'worker_threads';

let InferrenceWorker: Worker | null = null;

const InferrenceCBsByToken: Map<string, (outputs: Float32Array) => void> = new Map();

const WORKER_FILENAME = IS_DOCKER
  ? '/app/inferrenceWorker.mjs'
  : '/home/casey/anime-atlas/src/routes/recommendation/inferrenceWorker.js';

const getWorker = () => {
  if (InferrenceWorker) {
    return InferrenceWorker;
  }

  InferrenceWorker = new Worker(WORKER_FILENAME);
  InferrenceWorker.on('message', (msg) => {
    const cb = InferrenceCBsByToken.get(msg.token);
    if (cb) {
      cb(msg.outputs);
    } else {
      console.error('No callback for token: ' + msg.token);
    }
  });
  InferrenceWorker.on('error', (err) => {
    console.error('Error in worker: ' + err);
  });
  InferrenceWorker.on('exit', (code) => {
    console.error('Worker exited with code: ' + code);
  });

  return InferrenceWorker;
};

const genRandomToken = () => {
  return Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
};

export const performInferrence = async (modelName: ModelName, inputs: Float32Array): Promise<Float32Array> => {
  const prom = new Promise<Float32Array>((resolve) => {
    const token = genRandomToken();
    InferrenceCBsByToken.set(token, resolve);
    getWorker().postMessage(
      {
        token,
        modelName,
        inputs,
      },
      [inputs.buffer]
    );
  });
  return prom;
};
