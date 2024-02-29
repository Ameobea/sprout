<script context="module" lang="ts">
  const fetchRecommendations = (
    username: string,
    params: RecommendationControlParams,
    availableAnimeMetadataIDs: number[],
    includeContributors: boolean
  ): Promise<{ recommendations: Recommendation[]; animeData: { [animeID: number]: AnimeDetails } }> =>
    fetch('/recommendation/recommendation', {
      method: 'POST',
      body: JSON.stringify({
        dataSource: { type: 'username', username },
        availableAnimeMetadataIDs,
        includeContributors,
        ...params,
      }),
    }).then(async (res) => {
      if (!res.ok) {
        throw await res.text();
      }
      return res.json();
    });
</script>

<script lang="ts">
  import { writable, type Writable } from 'svelte/store';
  import { createQuery, QueryClient } from '@tanstack/svelte-query';

  import RecommendationsList from 'src/components/recommendation/RecommendationsList.svelte';
  import type { AnimeDetails } from 'src/malAPI';
  import type { Recommendation } from 'src/routes/recommendation/recommendation/recommendation';
  import type { RecommendationsResponse } from 'src/routes/user/[username]/recommendations/+page.server';
  import RecommendationControls from './RecommendationControls.svelte';
  import { captureMessage, getSentry } from 'src/sentry';
  import { getDefaultRecommendationControlParams, updateQueryParams, type RecommendationControlParams } from './utils';
  import { browser } from '$app/environment';

  export let initialRecommendations: RecommendationsResponse;
  export let username: string;
  export let genreNames: { [genreID: number]: string } | undefined;

  const genresDB: Writable<Map<number, string>> = writable(new Map());

  $: if (genreNames) {
    genresDB.update((db) => {
      for (const [genreID, genreName] of Object.entries(genreNames ?? {})) {
        db.set(+genreID, genreName);
      }
      return db;
    });
  }

  const animeMetadataDatabase = writable(initialRecommendations.type === 'ok' ? initialRecommendations.animeData : {});

  const params = writable(getDefaultRecommendationControlParams());
  params.subscribe(updateQueryParams);

  let usedInitialData = false;
  const initialData = initialRecommendations.type === 'ok' ? initialRecommendations : undefined;

  const queryClient = new QueryClient({});

  let lastRecosRes:
    | {
        recommendations: {
          id: number;
          score: number;
        }[];
        animeData: {
          [animeID: number]: AnimeDetails;
        };
      }
    | undefined = undefined;
  $: recosRes = createQuery<
    { recommendations: Recommendation[]; animeData: { [animeID: number]: AnimeDetails } } | undefined
  >(
    {
      queryKey: ['recommendations', username, $params] as const,
      queryFn: async () => {
        if (!usedInitialData) {
          usedInitialData = true;
          return initialData;
        }
        return fetchRecommendations(
          username,
          $params,
          Object.keys($animeMetadataDatabase).map((x) => +x),
          false
        );
      },
      refetchOnMount: false,
      refetchOnWindowFocus: false,
    },
    queryClient
  );
  $: if (!$recosRes.isLoading && $recosRes.data) {
    lastRecosRes = $recosRes.data;
  }

  $: recoContributorsRes = createQuery(
    {
      queryKey: ['recommendations_contributors', username, $params] as const,
      queryFn: () => {
        if (!browser) {
          return null;
        }
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
      refetchOnMount: false,
      refetchOnWindowFocus: false,
    },
    queryClient
  );

  $: recosData = $recosRes.data ?? lastRecosRes;
  $: if (recosData) {
    updateAnimeDB(recosData.animeData);
  }
  $: if ($recoContributorsRes.data) {
    updateAnimeDB($recoContributorsRes.data.animeData);
  }

  $: if ($recosRes.isError) {
    getSentry()?.captureException('Error fetching recommendations', { extra: { params: $params } });
  }

  $: recommendations = (() => {
    if ($recoContributorsRes.data) {
      return $recoContributorsRes.data;
    }
    return recosData ? recosData : null;
  })();

  const updateAnimeDB = (animeData: { [animeID: number]: AnimeDetails }) =>
    animeMetadataDatabase.update((db) => {
      Object.entries(animeData).forEach(([animeID, metadata]) => {
        db[+animeID] = metadata;
      });
      return db;
    });

  const excludedRankingAnimeIDs = (animeID: number) =>
    params.update((state) => {
      if (state.excludedRankingAnimeIDs.includes(animeID)) {
        return state;
      }

      captureMessage('Add excluded anime', { id: animeID, title: $animeMetadataDatabase[animeID]?.title });
      state.excludedRankingAnimeIDs.push(animeID);
      return state;
    });

  const excludeGenreID = (genreID: number, genreName: string) => {
    genresDB.update((db) => db.set(genreID, genreName));

    params.update((state) => {
      if (state.excludedGenreIDs.includes(genreID)) {
        return state;
      }

      captureMessage('Add excluded genre', { id: genreID, name: genreName });
      state.excludedGenreIDs.push(genreID);
      return state;
    });
  };
</script>

<div class="root">
  {#if $recosRes.isError}
    <b>Error fetching recommendations: {$recosRes.error}</b>
  {:else}
    <RecommendationControls
      {params}
      animeMetadataDatabase={$animeMetadataDatabase}
      isLoading={$recosRes.isLoading || $recosRes.isRefetching}
      {genresDB}
    />
    <RecommendationsList
      recommendations={recommendations?.recommendations ?? []}
      animeMetadataDatabase={$animeMetadataDatabase}
      excludeRanking={excludedRankingAnimeIDs}
      excludeGenre={excludeGenreID}
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
