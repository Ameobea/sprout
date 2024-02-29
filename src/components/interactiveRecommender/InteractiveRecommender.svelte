<script context="module" lang="ts">
  const MIN_REQUIRED_RATING_COUNT = 4;

  const fetchRecommendations = async (
    profile: { animeID: number; score: number }[],
    params: RecommendationControlParams,
    availableAnimeMetadataIDs: number[],
    includeContributors: boolean
  ): Promise<{ recommendations: { id: number; score: number }[]; animeData: { [animeID: number]: AnimeDetails } }> => {
    if (profile.length === 0) {
      return { recommendations: [], animeData: {} };
    }

    return fetch('/recommendation/recommendation', {
      method: 'POST',
      body: JSON.stringify({
        dataSource: { type: 'rawProfile', profile },
        availableAnimeMetadataIDs,
        includeContributors,
        ...params,
        popularityAttenuationFactor:
          profile.length < 5 ? PopularityAttenuationFactor.Extreme : PopularityAttenuationFactor.VeryHigh,
      }),
    }).then(async (res) => {
      if (!res.ok) {
        throw await res.text();
      }
      return res.json();
    });
  };
</script>

<script lang="ts">
  import { createQuery, QueryClient } from '@tanstack/svelte-query';
  import { writable, type Writable } from 'svelte/store';

  import type { AnimeDetails } from 'src/malAPI';
  import RecommendationControls from '../recommendation/RecommendationControls.svelte';
  import { getDefaultRecommendationControlParams, type RecommendationControlParams } from '../recommendation/utils';
  import RecommendationsList from '../recommendation/RecommendationsList.svelte';
  import { PopularityAttenuationFactor } from '../recommendation/conf';

  export let profile: {
    animeID: number;
    score: number;
    title: string;
    titleEnglish: string;
  }[];
  export let addRanking: (animeID: number) => void;

  const params = writable(getDefaultRecommendationControlParams());

  const animeMetadataDatabase = writable<{ [animeID: number]: AnimeDetails }>({});
  const genresDB: Writable<Map<number, string>> = writable(new Map());

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
  $: recosRes = createQuery(
    {
      queryKey: ['interactiveRecommendations', profile, $params] as const,
      queryFn: async () => {
        return fetchRecommendations(
          profile,
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
      queryKey: ['interactiveRecommendations_contributors', profile, $params] as const,
      queryFn: () => {
        const availableMetadataAnimeIDs = Object.keys($animeMetadataDatabase).map((x) => +x);
        return fetchRecommendations(profile, $params, availableMetadataAnimeIDs, true);
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
</script>

<div class="root">
  {#if profile.length >= MIN_REQUIRED_RATING_COUNT}
    <RecommendationControls
      {params}
      animeMetadataDatabase={$animeMetadataDatabase}
      isLoading={$recosRes.isLoading || $recosRes.isRefetching}
      {genresDB}
      forceHideTopBar
    />
    <RecommendationsList
      recommendations={recommendations?.recommendations ?? []}
      animeMetadataDatabase={$animeMetadataDatabase}
      {addRanking}
      contributorsLoading={$recosRes.isLoading ||
        $recosRes.isRefetching ||
        $recoContributorsRes.isLoading ||
        $recoContributorsRes.isRefetching}
    />
  {:else if profile.length === 0}
    <p>Rate some anime you've seen to to get recommendations</p>
    <p>The more you rate, the better the recommendations will be!</p>
  {:else}
    <p>Rate {MIN_REQUIRED_RATING_COUNT - profile.length} more anime to see recommendations</p>
  {/if}
</div>

<style lang="css">
  .root {
    display: flex;
    flex: 1;
    flex-direction: column;
    background-color: #0a0a0a;
    padding: 5px 4px;
  }

  p {
    text-align: center;
    margin-top: 10px;
  }
</style>
