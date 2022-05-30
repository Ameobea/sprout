<script lang="ts">
  import { fade } from 'svelte/transition';
  import { flip } from 'svelte/animate';

  import type { AnimeDetails } from 'src/malAPI';
  import type { Recommendation } from 'src/routes/recommendation/recommendation';
  import RecommendationListItem from './RecommendationListItem.svelte';

  export let recommendations: Recommendation[];
  export let animeMetadataDatabase: { [animeID: number]: AnimeDetails };
  export let excludeRanking: ((animeID: number) => void) | undefined = undefined;
  export let excludeGenre: ((genreID: number, genreName: string) => void) | undefined = undefined;
  export let addRanking: ((animeID: number) => void) | undefined = undefined;
  export let contributorsLoading: boolean;

  let expandedAnimeID: number | null = null;
  $: if (expandedAnimeID !== null && !recommendations.some((reco) => reco.id === expandedAnimeID)) {
    expandedAnimeID = null;
  }
</script>

<div class="recommendations">
  {#each recommendations as { id, topRatingContributorsIds, planToWatch } (id)}
    {@const animeMetadata = animeMetadataDatabase[id]}
    <div in:fade animate:flip={{ duration: (d) => 39 * Math.sqrt(d) }}>
      <RecommendationListItem
        {animeMetadata}
        expanded={expandedAnimeID === animeMetadata.id}
        toggleExpanded={() => {
          if (expandedAnimeID === animeMetadata.id) {
            expandedAnimeID = null;
          } else {
            expandedAnimeID = animeMetadata.id;
          }
        }}
        topRatingContributors={topRatingContributorsIds?.map((id) => ({
          datum: animeMetadataDatabase[Math.abs(id)],
          positiveRating: id > 0,
        }))}
        planToWatch={planToWatch ?? false}
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
