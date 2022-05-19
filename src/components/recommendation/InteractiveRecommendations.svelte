<script context="module" lang="ts">
  import { DEFAULT_MODEL_NAME, ModelName } from './conf';

  export interface RecommendationControlParams {
    modelName: ModelName;
    excludedRankingAnimeIDs: number[];
    includeONAsOVAsSpecials: boolean;
    includeMovies: boolean;
    includeMusic: boolean;
  }

  const getDefaultRecommendationControlParams = (): RecommendationControlParams => {
    const queryParams = new URLSearchParams(browser ? window.location.search : '');
    return {
      modelName: (queryParams.get('model') as any) ?? DEFAULT_MODEL_NAME,
      excludedRankingAnimeIDs: queryParams.getAll('eid').map((eid) => +eid),
      includeONAsOVAsSpecials: queryParams.get('specials') !== 'false',
      includeMovies: queryParams.get('movies') !== 'false',
      includeMusic: queryParams.get('music') === 'true',
    };
  };

  const updateQueryParams = (params: RecommendationControlParams) => {
    if (!browser) {
      return;
    }

    const url = new URL(window.location.toString());
    url.searchParams.forEach((_val, key) => url.searchParams.delete(key));
    const oldSearchParams = new URLSearchParams(window.location.search).toString();

    for (const animeID of params.excludedRankingAnimeIDs) {
      url.searchParams.append('eid', animeID.toString());
    }
    if (params.modelName !== DEFAULT_MODEL_NAME) {
      url.searchParams.set('model', params.modelName);
    }
    if (!params.includeONAsOVAsSpecials) {
      url.searchParams.set('specials', 'false');
    }
    if (!params.includeMovies) {
      url.searchParams.set('movies', 'false');
    }
    if (params.includeMusic) {
      url.searchParams.set('music', 'true');
    }

    const newSearchParams = url.searchParams.toString();
    if (newSearchParams !== oldSearchParams) {
      history.replaceState({}, '', url);
    }
  };

  const fetchRecommendations = (
    username: string,
    params: RecommendationControlParams,
    availableAnimeMetadataIDs: number[],
    includeContributors: boolean
  ): Promise<{ recommendations: Recommendation[]; animeData: { [animeID: number]: AnimeDetails } }> =>
    fetch('/recommendation/recommendation', {
      method: 'POST',
      body: JSON.stringify({ username, availableAnimeMetadataIDs, includeContributors, ...params }),
    }).then(async (res) => {
      if (!res.ok) {
        throw await res.text();
      }
      return res.json();
    });
</script>

<script lang="ts">
  import { writable } from 'svelte/store';
  import { useQuery } from '@sveltestack/svelte-query';
  import { browser } from '$app/env';

  import RecommendationsList from 'src/components/recommendation/RecommendationsList.svelte';
  import type { AnimeDetails } from 'src/malAPI';
  import RecommendationControls from './RecommendationControls.svelte';
  import type { Recommendation } from 'src/routes/recommendation/recommendation';
  import type { RecommendationsResponse } from 'src/routes/user/[username]/recommendations';

  export let initialRecommendations: RecommendationsResponse;
  export let username: string;

  const animeMetadataDatabase = writable(initialRecommendations.type === 'ok' ? initialRecommendations.animeData : {});

  const params = writable(getDefaultRecommendationControlParams());
  $: updateQueryParams($params);

  let usedInitialData = false;
  const initialData = initialRecommendations.type === 'ok' ? initialRecommendations : undefined;

  const recosRes = useQuery(
    ['recommendations', username, $params] as const,
    () =>
      fetchRecommendations(
        username,
        $params,
        Object.keys($animeMetadataDatabase).map((x) => +x),
        false
      ),
    {
      refetchOnMount: false,
      refetchOnWindowFocus: false,
      initialData,
      keepPreviousData: true,
    }
  );
  const recoContributorsRes = useQuery(
    ['recommendations_contributors', username, $params] as const,
    () => {
      const availableMetadataAnimeIDs = Object.keys($animeMetadataDatabase).map((x) => +x);
      return fetchRecommendations(
        username,
        $params,
        availableMetadataAnimeIDs.length === 0 && initialRecommendations.type === 'ok'
          ? Object.keys(initialRecommendations.animeData).map((n) => +n)
          : availableMetadataAnimeIDs,
        true
      );
    },
    { refetchOnMount: false, refetchOnWindowFocus: false, keepPreviousData: false }
  );

  $: {
    recosRes.updateOptions({
      queryKey: ['recommendations', username, $params] as const,
      initialData: usedInitialData ? undefined : initialData,
    });
    recoContributorsRes.updateOptions({
      queryKey: ['recommendations_contributors', username, $params] as const,
    });
    usedInitialData = true;
  }

  $: recommendations = (() => {
    if ($recoContributorsRes.data) {
      return $recoContributorsRes.data;
    }
    return $recosRes.data ? $recosRes.data : null;
  })();

  const updateAnimeDB = (animeData: { [animeID: number]: AnimeDetails }) =>
    animeMetadataDatabase.update((db) => {
      Object.entries(animeData).forEach(([animeID, metadata]) => {
        db[+animeID] = metadata;
      });
      return db;
    });
  $: {
    if ($recosRes.data) {
      updateAnimeDB($recosRes.data.animeData);
    }
    if ($recoContributorsRes.data) {
      updateAnimeDB($recoContributorsRes.data.animeData);
    }
  }

  const excludedRankingAnimeIDs = (animeID: number) =>
    params.update((state) => {
      if (state.excludedRankingAnimeIDs.includes(animeID)) {
        return state;
      }

      state.excludedRankingAnimeIDs.push(animeID);
      return state;
    });
</script>

<div class="root">
  {#if $recosRes.isError}
    <b>Error fetching recommendations: {$recosRes.error}</b>
  {:else}
    <RecommendationControls
      {params}
      animeMetadataDatabase={$animeMetadataDatabase}
      isLoading={$recosRes.isLoading || $recosRes.isRefetching}
    />
    <RecommendationsList
      recommendations={recommendations?.recommendations ?? []}
      animeMetadataDatabase={$animeMetadataDatabase}
      excludeRanking={excludedRankingAnimeIDs}
      contributorsLoading={$recosRes.isLoading ||
        $recosRes.isRefetching ||
        $recoContributorsRes.isLoading ||
        $recoContributorsRes.isRefetching}
    />
  {/if}
</div>

<style lang="css">
  .root {
    display: flex;
    flex-direction: column;
  }
</style>
