<script context="module" lang="ts">
  import { ModelName } from './conf';

  export interface RecommendationControlParams {
    modelName: ModelName;
    excludedRankingAnimeIDs: number[];
  }

  const DefaultRecommendationControlParams: RecommendationControlParams = {
    modelName: ModelName.Model_4K_V2,
    excludedRankingAnimeIDs: [],
  };

  const updateQueryParams = (params: RecommendationControlParams) => {
    console.log(params);
    // TODO
  };
</script>

<script lang="ts">
  import { writable } from 'svelte/store';
  import { useQuery } from '@sveltestack/svelte-query';

  import RecommendationsList from 'src/components/recommendation/RecommendationsList.svelte';
  import type { AnimeDetails } from 'src/malAPI';
  import type { RecommendationsResponse } from 'src/routes/recommendation/[username]';
  import RecommendationControls from './RecommendationControls.svelte';

  export let initialRecommendations: RecommendationsResponse;
  export let username: string;
  $: recommendations = initialRecommendations;

  const params = writable({ ...DefaultRecommendationControlParams });
  $: updateQueryParams($params);

  const queryResult = useQuery(
    ['recommendations', username, $params] as const,
    async () =>
      fetch('/recommendation/recommendation', {
        method: 'POST',
        body: JSON.stringify({ username, params: $params }),
      }).then(async (res) => {
        if (!res.ok) {
          throw await res.text();
        }
        return res.json();
      }),
    { cacheTime: 60 * 1000, refetchOnMount: false }
  );
  $: console.log($queryResult);

  const getAnimeMetadata = (id: number): AnimeDetails => {
    if (recommendations.type !== 'ok') {
      throw new Error('Unreachable');
    }
    const { animeData } = recommendations;
    return animeData[id];
  };

  const excludedRankingAnimeIDs = (animeID: number) =>
    params.update((state) => {
      if (state.excludedRankingAnimeIDs.includes(animeID)) {
        return;
      }

      state.excludedRankingAnimeIDs.push(animeID);
      return state;
    });
</script>

<div class="root">
  {#if recommendations.type === 'error'}
    <b>Error fetching recommendations: {recommendations.error}</b>
  {:else}
    <RecommendationControls state={params} {getAnimeMetadata} />
    <RecommendationsList
      recommendations={recommendations.recommendations}
      {getAnimeMetadata}
      excludeRanking={excludedRankingAnimeIDs}
    />
  {/if}
</div>

<style lang="css">
  .root {
    display: flex;
    flex-direction: column;
  }
</style>
