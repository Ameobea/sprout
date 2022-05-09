<script lang="ts">
  import { page } from '$app/stores';

  import { HeaderTitle } from 'src/components/recommendation/HeaderTitle';
  import RecommendationListItem from 'src/components/recommendation/RecommendationListItem.svelte';
  import type { AnimeDetails } from 'src/malAPI';
  import type { Recommendation } from './recommendation';

  export let recommendations:
    | {
        type: 'ok';
        recommendations: Recommendation[];
        animeData: { [id: number]: AnimeDetails };
      }
    | { type: 'error'; error: string };

  $: username = $page.params.username;
  $: {
    HeaderTitle.set(`Recommendations for ${username}`);
  }

  let expandedAnimeID: number | null = null;

  const getAnimeMetadata = (id: number): AnimeDetails => {
    if (recommendations.type !== 'ok') {
      throw new Error('Unreachable');
    }
    const { animeData } = recommendations;
    return animeData[id];
  };
</script>

<div class="root">
  {#if recommendations.type === 'error'}
    <b>Error fetching recommendations: {recommendations.error}</b>
  {:else}
    <div class="recommendations">
      {#each recommendations.recommendations as { id, topRatingContributorsIds }}
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
        />
      {/each}
    </div>
  {/if}
</div>

<style lang="css">
  .root {
    display: flex;
    flex-direction: column;
  }

  .recommendations {
    display: flex;
    flex-direction: column;
  }
</style>
