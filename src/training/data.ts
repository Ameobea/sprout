import type * as tfWeb from '@tensorflow/tfjs';
import type * as tfNode from '@tensorflow/tfjs-node';

import type { Embedding } from 'src/routes/embedding';
import type { TrainingDatum } from 'src/routes/recommendation/training/trainingData';
import type { EmbeddingName } from 'src/types';
import { RECOMMENDATION_MODEL_CORPUS_SIZE } from 'src/routes/recommendation/model';

const getHoldoutCount = (validRatingCount: number): number => {
  // return 0;
  const holdoutCount = Math.floor(validRatingCount * 0.6 * Math.random());
  return holdoutCount;
};

const scoreRating = (rating: number): number => {
  // switch (rating) {
  //   case 10:
  //     return 1;
  //   case 9:
  //     return 0.95;
  //   case 8:
  //     return 0.75;
  //   case 7:
  //     return 0.3;
  //   case 6:
  //     return 0.1;
  //   default:
  //     return 0;
  // }
  return rating > 5 ? 1 : -1;
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
  private trainUsernameCounter = 0;
  private nextTrainingData: Promise<TrainingDatum[][]> | null = null;

  constructor(tf: typeof tfWeb | typeof tfNode, allUsernames: string[], animeMetadata: Embedding) {
    this.tf = tf;
    tf.util.shuffle(allUsernames);

    // Partition username tensors into training and test sets
    const cutoffIx = Math.floor(allUsernames.length * 0.8);
    const [trainUsernames, testUsernames] = [allUsernames.slice(0, cutoffIx), allUsernames.slice(cutoffIx)];

    this.animeMetadata = animeMetadata;
    this.trainUsernames = trainUsernames;
  }

  public static buildModelInput = (userProfile: TrainingDatum[], corpusSize: number): ModelInput => {
    const scoreByAnimeIx = new Array<number>(corpusSize).fill(0);
    const validIndices: number[] = [];
    userProfile.forEach((datum, i) => {
      const { animeIx, rating } = datum;
      if (typeof animeIx !== 'number' || animeIx >= corpusSize) {
        return;
      }
      validIndices.push(i);
      const score = scoreRating(rating);
      scoreByAnimeIx[animeIx] = score;
    });

    return { validIndices, input: scoreByAnimeIx };
  };

  public buildTrainingDataTensors = (
    trainingData: TrainingDatum[][],
    holdout = true
  ): [tfWeb.Tensor2D, tfWeb.Tensor2D] => {
    const processedTrainingData = trainingData
      .map((userProfile) => {
        const yScoreByAnimeIx = new Array<number>(this.animeMetadata.length).fill(0);

        // for (let i = 0; i < animeMetadata.length; i++) {
        //   const v = Math.random() > 0.5;
        //   xScoreByAnimeIx[i] = v ? 1 : -1;
        //   yScoreByAnimeIx[i] = v ? 1 : -1;
        // }
        // return [xScoreByAnimeIx, yScoreByAnimeIx] as const;

        const { validIndices, input: xScoreByAnimeIx } = DataContainer.buildModelInput(
          userProfile,
          this.animeMetadata.length
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
          const score = scoreRating(rating);
          xScoreByAnimeIx[animeIx] = score;
        });

        return [xScoreByAnimeIx, yScoreByAnimeIx] as const;
      })
      .filter((x) => x);

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
    const usernames = [];
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

  public getTrainingData = async (count: number): Promise<[tfWeb.Tensor2D, tfWeb.Tensor2D]> => {
    if (this.nextTrainingData) {
      const toReturn = this.nextTrainingData;
      const nextTrainUsernames = this.getTrainUsernames(count);
      this.nextTrainingData = fetchTrainingData(nextTrainUsernames);
      return toReturn.then((trainingData) => this.buildTrainingDataTensors(trainingData));
    }

    const trainUsernames = this.getTrainUsernames(count);
    const toReturn = fetchTrainingData(trainUsernames).then((trainingData) =>
      this.buildTrainingDataTensors(trainingData)
    );
    const nextTrainUsernames = this.getTrainUsernames(count);
    this.nextTrainingData = fetchTrainingData(nextTrainUsernames);
    return toReturn;
  };
}
