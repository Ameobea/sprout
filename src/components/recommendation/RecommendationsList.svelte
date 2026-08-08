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
  export let contributionBaseline: number | undefined = undefined;

  // Drop contributors dwarfed by the rec's top one or below the profile-wide
  // significance scale; always keep at least the top contributor.
  const REL_CUTOFF = 0.15;
  const SIG_MULT = 3;
  const visibleContributors = (contributors: Recommendation['topContributors']) => {
    if (!contributors?.length) {
      return contributors;
    }
    const sigFloor = contributionBaseline !== undefined ? SIG_MULT * contributionBaseline : 0;
    const cutoff = Math.max(REL_CUTOFF * contributors[0].strength, sigFloor);
    const kept = contributors.filter((c) => c.strength >= cutoff);
    return kept.length > 0 ? kept : contributors.slice(0, 1);
  };

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
  {#each displayRecommendations as { id, topContributors, planToWatch, shownPredictedRating, ratingTier }, rank (id)}
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
        topContributors={visibleContributors(topContributors)?.map((c) => ({
          ...c,
          datum: animeMetadataDatabase[c.animeId],
        }))}
        {contributionBaseline}
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
