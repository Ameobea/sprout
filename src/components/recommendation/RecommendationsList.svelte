<script lang="ts">
  import { slide } from 'svelte/transition';
  import { flip } from 'svelte/animate';

  import type { AnimeDetails } from 'src/malAPI';
  import type { Recommendation } from 'src/routes/recommendation/recommendation';
  import RecommendationListItem from './RecommendationListItem.svelte';

  export let recommendations: Recommendation[];
  export let animeMetadataDatabase: { [animeID: number]: AnimeDetails };
  export let excludeRanking: (animeID: number) => void;

  let expandedAnimeID: number | null = null;
  $: if (expandedAnimeID !== null && !recommendations.some((reco) => reco.id === expandedAnimeID)) {
    expandedAnimeID = null;
  }
</script>

<div class="recommendations">
  {#each recommendations as { id, topRatingContributorsIds } (id)}
    {@const animeMetadata = animeMetadataDatabase[id]}
    <div animate:flip={{ duration: (d) => 50 * Math.sqrt(d) }} in:slide>
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
        topRatingContributors={topRatingContributorsIds.map((id) => animeMetadataDatabase[id])}
        {excludeRanking}
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
