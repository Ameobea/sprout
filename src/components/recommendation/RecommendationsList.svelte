<script lang="ts">
  import type { AnimeDetails } from 'src/malAPI';
  import type { Recommendation } from 'src/routes/recommendation/recommendation';
  import RecommendationListItem from './RecommendationListItem.svelte';

  export let recommendations: Recommendation[];
  export let getAnimeMetadata: (animeId: number) => AnimeDetails;
  export let excludeRanking: (animeID: number) => void;

  let expandedAnimeID: number | null = null;
</script>

<div class="recommendations">
  {#each recommendations as { id, topRatingContributorsIds } (id)}
    {@const animeMetadata = getAnimeMetadata(id)}
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
      topRatingContributors={topRatingContributorsIds.map(getAnimeMetadata)}
      {excludeRanking}
    />
  {/each}
</div>

<style lang="css">
  .recommendations {
    display: flex;
    flex-direction: column;
  }
</style>
