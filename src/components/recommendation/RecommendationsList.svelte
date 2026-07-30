<script lang="ts">
  import { fade } from 'svelte/transition';
  import { flip } from 'svelte/animate';

  import { getSurface, submitAnalyticsEvent } from 'src/analytics';
  import type { AnimeDetails } from 'src/malAPI';
  import type { Recommendation, UserRatingStats } from '../../routes/recommendation/recommendation/recommendation';
  import { getRatingTier, type RatingTier } from 'src/util/ratingTiers';
  import RecommendationListItem from './RecommendationListItem.svelte';

  export let recommendations: Recommendation[];
  export let animeMetadataDatabase: { [animeID: number]: AnimeDetails };
  export let excludeRanking: ((animeID: number) => void) | undefined = undefined;
  export let excludeGenre: ((genreID: number, genreName: string) => void) | undefined = undefined;
  export let addRanking: ((animeID: number) => void) | undefined = undefined;
  export let contributorsLoading: boolean;
  export let userRatingStats: UserRatingStats | null = null;

  let displayRecommendations: (Recommendation & {
    shownPredictedRating: number | null;
    ratingTier: RatingTier | null;
  })[] = [];
  $: displayRecommendations = recommendations.map((reco) => {
    const showRating = !!userRatingStats && !userRatingStats.isNonRater && typeof reco.predictedRating === 'number';
    return {
      ...reco,
      shownPredictedRating: showRating ? reco.predictedRating! : null,
      ratingTier: showRating ? getRatingTier(reco.predictedRating!, userRatingStats!) : null,
    };
  });

  let expandedAnimeID: number | null = null;
  $: if (expandedAnimeID !== null && !recommendations.some((reco) => reco.id === expandedAnimeID)) {
    expandedAnimeID = null;
  }
</script>

<div class="recommendations">
  {#each displayRecommendations as { id, topRatingContributorsIds, planToWatch, shownPredictedRating, ratingTier }, rank (id)}
    {@const animeMetadata = animeMetadataDatabase[id]}
    <div in:fade animate:flip={{ duration: (d) => 39 * Math.sqrt(d) }}>
      <RecommendationListItem
        {animeMetadata}
        {rank}
        expanded={expandedAnimeID === animeMetadata.id}
        toggleExpanded={() => {
          const expanded = expandedAnimeID !== animeMetadata.id;
          submitAnalyticsEvent({
            category: 'recommendations',
            subcategory: 'item_expand',
            payload: { anime_id: animeMetadata.id, rank, expanded, surface: getSurface() },
          });
          expandedAnimeID = expanded ? animeMetadata.id : null;
        }}
        topRatingContributors={topRatingContributorsIds?.map((id) => ({
          datum: animeMetadataDatabase[Math.abs(id)],
          positiveRating: id > 0,
        }))}
        planToWatch={planToWatch ?? false}
        predictedRating={shownPredictedRating}
        {ratingTier}
        {excludeRanking}
        {excludeGenre}
        {addRanking}
        {contributorsLoading}
      />
    </div>
  {/each}
</div>

<style lang="css">
  .recommendations {
    display: flex;
    flex-direction: column;
  }
</style>
