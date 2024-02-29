import { loadEmbedding } from 'src/embedding';
import { AnimeListStatusCode } from 'src/malAPI';
import { EmbeddingName } from 'src/types';
import type { CompatAnimeListEntry } from 'src/anilistAPI';

export interface TrainingDatum {
  animeIx: number;
  rating: number;
}

let AnimeIxByID: Map<number, number> | null = null;

const getAnimeIxByID = async (): Promise<Map<number, number>> => {
  if (!AnimeIxByID) {
    const embedding = await loadEmbedding(EmbeddingName.PyMDE_3D_40N);
    AnimeIxByID = new Map<number, number>();
    embedding.forEach((datum, i) => AnimeIxByID!.set(datum.metadata.id, i));
  }
  return AnimeIxByID;
};

export const convertMALProfileToTrainingData = async (
  rawData: CompatAnimeListEntry[][]
): Promise<{ ratings: TrainingDatum[]; userIsNonRater: boolean }[]> => {
  const AnimeIxByID = await getAnimeIxByID();

  return rawData.map((userProfileAnime) => {
    const validRatings = userProfileAnime.filter((datum) =>
      [
        AnimeListStatusCode.Completed,
        AnimeListStatusCode.Watching,
        AnimeListStatusCode.OnHold,
        AnimeListStatusCode.Dropped,
      ].includes(datum.list_status.status)
    );
    let unratedCount = 0;
    for (const rating of validRatings) {
      if (
        !rating?.list_status ||
        (!rating.list_status.score && rating.list_status.status !== AnimeListStatusCode.Dropped)
      ) {
        unratedCount += 1;
      }
    }
    const userIsNonRater = unratedCount / validRatings.length > 0.2;

    const ratings = validRatings
      .filter(
        (datum) =>
          userIsNonRater || datum.list_status?.score > 0 || datum.list_status.status === AnimeListStatusCode.Dropped
      )
      .map((datum) => {
        let rating = datum.list_status.score;
        if (rating === 0 && datum.list_status.status === AnimeListStatusCode.Dropped) {
          rating = 4;
        } else if (userIsNonRater && rating === 0) {
          rating = 7;
        }

        return {
          animeIx: AnimeIxByID.get(datum.node.id)!,
          rating,
        };
      });
    return { ratings, userIsNonRater };
  });
};
