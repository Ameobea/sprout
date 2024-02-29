import type * as tfWeb from '@tensorflow/tfjs';
import type * as tfNode from '@tensorflow/tfjs-node';

import { RECOMMENDATION_MODEL_CORPUS_SIZE } from 'src/components/recommendation/conf';
import type { Embedding } from 'src/routes/embedding/+server';
import type { TrainingDatum } from 'src/routes/recommendation/training/trainingData/trainingData';
import type { EmbeddingName } from 'src/types';

const getHoldoutCount = (validRatingCount: number): number => {
  const holdoutCount = Math.floor(validRatingCount * 0.4 * Math.random());
  return holdoutCount;
};

export const scoreRating = (rating: number, weightScores: boolean): number => {
  if (!weightScores) {
    return rating > 5 ? 1 : -1;
  }

  switch (rating) {
    case 10:
      return 1;
    case 9:
      return 0.95;
    case 8:
      return 0.85;
    case 7:
      return 0.6;
    case 6:
      return 0.3;
    case 5:
      return -0.3;
    case 4:
      return -0.8;
    case 3:
      return -0.95;
    case 2:
      return -1;
    case 1:
      return -1;
    default:
      return 0;
  }
};

export const loadAnimeMetadata = async (embeddingName: EmbeddingName): Promise<Embedding> => {
  const { embedding }: { embedding: Embedding } = await fetch(`/embedding?embedding=${embeddingName}`).then((res) =>
    res.json()
  );
  return embedding.slice(0, RECOMMENDATION_MODEL_CORPUS_SIZE);
};

export const fetchTrainingData = async (usernames: string[]): Promise<TrainingDatum[][]> =>
  fetch('/recommendation/training/trainingData', { method: 'POST', body: JSON.stringify(usernames) }).then(
    async (res) => (await res.json()).trainingData
  );

export const fetchAllUsernames = async (): Promise<string[]> =>
  fetch('/recommendation/training/allUsernames').then(async (res) => (await res.json()).usernames);

interface ModelInput {
  /**
   * Indices of ratings in the user's profile that were valid and included in the returned `input`.
   */
  validIndices: number[];
  /**
   * Per-anime scaled ratings based on ratings from the user's anime list.  Indices line up with those from
   * the embedding.
   */
  input: number[];
}

export class DataContainer {
  public animeMetadata: Embedding;
  private tf: typeof tfWeb | typeof tfNode;
  private readonly trainUsernames: string[];
  private testUsernames: string[] = [];
  private trainUsernameCounter = 0;
  private nextTrainingData: Promise<TrainingDatum[][]> | null = null;

  constructor(tf: typeof tfWeb | typeof tfNode, allUsernames: string[], animeMetadata: Embedding) {
    this.tf = tf;
    tf.util.shuffle(allUsernames);

    // // Partition username tensors into training and test sets
    // const cutoffIx = Math.floor(allUsernames.length * 0.8);
    // const [trainUsernames, _testUsernames] = [allUsernames.slice(0, cutoffIx), allUsernames.slice(cutoffIx)];
    const trainUsernames = allUsernames;

    this.animeMetadata = animeMetadata;
    this.trainUsernames = trainUsernames;
    // this.testUsernames = testUsernames;
  }

  public static buildModelInput = (
    userProfile: TrainingDatum[],
    corpusSize: number,
    weightScores: boolean
  ): ModelInput => {
    const scoreByAnimeIx = new Array<number>(corpusSize).fill(0);
    const validIndices: number[] = [];
    userProfile.forEach((datum, ratingIx) => {
      const { animeIx, rating } = datum;
      if (typeof animeIx !== 'number' || animeIx >= corpusSize) {
        return;
      }
      validIndices.push(ratingIx);
      const score = scoreRating(rating, weightScores);
      scoreByAnimeIx[animeIx] = score;
    });

    return { validIndices, input: scoreByAnimeIx };
  };

  public buildTrainingDataTensors = (
    trainingData: TrainingDatum[][],
    holdout = true,
    weightScores: boolean
  ): [tfWeb.Tensor2D, tfWeb.Tensor2D] => {
    const processedTrainingData = trainingData
      .map((userProfile) => {
        const xScoreByAnimeIx = new Array<number>(this.animeMetadata.length).fill(0);

        // for (let i = 0; i < animeMetadata.length; i++) {
        //   const v = Math.random() > 0.5;
        //   xScoreByAnimeIx[i] = v ? 1 : -1;
        //   yScoreByAnimeIx[i] = v ? 1 : -1;
        // }
        // return [xScoreByAnimeIx, yScoreByAnimeIx] as const;

        const { validIndices, input: yScoreByAnimeIx } = DataContainer.buildModelInput(
          userProfile,
          this.animeMetadata.length,
          weightScores
        );
        if (validIndices.length < 10) {
          return null;
        }

        const holdoutCount = holdout ? getHoldoutCount(validIndices.length) : 0;
        const retainedIndices = this.tf.util
          .createShuffledIndices(validIndices.length)
          .slice(0, validIndices.length - holdoutCount);
        retainedIndices.forEach((ixIx) => {
          const ix = validIndices[ixIx];
          const { animeIx, rating } = userProfile[ix];
          if (typeof animeIx !== 'number' || animeIx >= this.animeMetadata.length) {
            return;
          }
          const score = scoreRating(rating, weightScores);
          xScoreByAnimeIx[animeIx] = score;
        });

        return [xScoreByAnimeIx, yScoreByAnimeIx] as const;
      })
      .filter((x) => x) as (readonly [number[], number[]])[];

    const xs = processedTrainingData.map(([xs]) => xs);
    const ys = processedTrainingData.map(([, ys]) => ys);
    const xTrain = this.tf.tensor2d(xs, [xs.length, xs[0].length]);
    const yTrain = this.tf.tensor2d(ys, [ys.length, ys[0].length]);
    return [xTrain, yTrain];
  };

  /**
   * Retrieves usernames of profiles to fetch data for.
   */
  private getTrainUsernames = (count: number): string[] => {
    const usernames: string[] = [];
    for (let i = 0; i < count; i++) {
      usernames.push(this.trainUsernames[this.trainUsernameCounter]);
      this.trainUsernameCounter += 1;
      if (this.trainUsernameCounter >= this.trainUsernames.length) {
        this.trainUsernameCounter = 0;
        this.tf.util.shuffle(this.trainUsernames);
      }
    }
    return usernames;
  };

  public getTrainingData = async (count: number, weightScores: boolean): Promise<[tfWeb.Tensor2D, tfWeb.Tensor2D]> => {
    if (this.nextTrainingData) {
      const toReturn = this.nextTrainingData;
      const nextTrainUsernames = this.getTrainUsernames(count);
      this.nextTrainingData = fetchTrainingData(nextTrainUsernames);
      return toReturn.then((trainingData) => this.buildTrainingDataTensors(trainingData, undefined, weightScores));
    }

    const trainUsernames = this.getTrainUsernames(count);
    const toReturn = fetchTrainingData(trainUsernames).then((trainingData) =>
      this.buildTrainingDataTensors(trainingData, undefined, weightScores)
    );
    const nextTrainUsernames = this.getTrainUsernames(count);
    this.nextTrainingData = fetchTrainingData(nextTrainUsernames);
    return toReturn;
  };

  public getTestData = async (count: number, weightScores: boolean): Promise<[tfWeb.Tensor2D, tfWeb.Tensor2D]> => {
    this.tf.util.shuffle(this.testUsernames);
    const testUsernames = this.testUsernames.slice(0, count);
    const testData = await fetchTrainingData(testUsernames);
    return this.buildTrainingDataTensors(testData, false, weightScores);
  };
}
