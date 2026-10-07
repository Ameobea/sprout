<script context="module" lang="ts">
  const fetchRecommendations = (
    username: string,
    params: RecommendationControlParams,
    availableAnimeMetadataIDs: number[],
    includeContributors: boolean,
    forceProfileRefresh = false
  ): Promise<{
    recommendations: Recommendation[];
    animeData: { [animeID: number]: AnimeDetails };
    userRatingStats: UserRatingStats | null;
    contributionBaseline?: number;
  }> =>
    fetch('/recommendation/recommendation', {
      method: 'POST',
      body: JSON.stringify({
        dataSource: { type: 'username', username },
        availableAnimeMetadataIDs,
        includeContributors,
        forceProfileRefresh,
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
  import { onMount, tick } from 'svelte';
  import { writable, type Writable } from 'svelte/store';
  import { createQuery, QueryClient } from '@tanstack/svelte-query';

  import RecommendationsList from 'src/components/recommendation/RecommendationsList.svelte';
  import type { AnimeDetails } from 'src/malAPI';
  import type { Recommendation, UserRatingStats } from 'src/routes/recommendation/recommendation/recommendation';
  import type { RecommendationsResponse } from 'src/routes/user/[username]/recommendations/+page.server';
  import RecommendationControls from './RecommendationControls.svelte';
  import { getSurface, submitAnalyticsEvent } from 'src/analytics';
  import { getSentry } from 'src/sentry';
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

  // Keep the URL query string in sync with the control params, but only after the SvelteKit
  // client router has initialized. `updateQueryParams` calls `replaceState`, which throws if
  // invoked during the initial mount flush (the router root isn't assigned yet). Waiting for a
  // tick guarantees hydration has completed before the first sync.
  let canSyncQueryParams = false;
  onMount(async () => {
    await tick();
    canSyncQueryParams = true;
  });
  $: if (canSyncQueryParams) {
    updateQueryParams($params);
  }

  let usedInitialData = false;
  const initialData = initialRecommendations.type === 'ok' ? initialRecommendations : undefined;

  const queryClient = new QueryClient({});
  let isProfileRefreshing = false;
  let profileRefreshError = '';

  const forceProfileRefresh = async () => {
    if (isProfileRefreshing) {
      return;
    }

    isProfileRefreshing = true;
    profileRefreshError = '';
    try {
      // Controls can change while MAL is loading. Keep the response associated
      // with the exact settings that requested it, and cancel older cache work.
      const refreshParams = structuredClone($params);
      const recommendationsKey = ['recommendations', username, refreshParams];
      const contributorsKey = ['recommendations_contributors', username, refreshParams];
      await Promise.all([
        queryClient.cancelQueries({ queryKey: recommendationsKey, exact: true }),
        queryClient.cancelQueries({ queryKey: contributorsKey, exact: true }),
      ]);
      const availableAnimeMetadataIDs = Object.keys($animeMetadataDatabase).map((x) => +x);
      // A single contributor-enabled request returns the complete recommendation payload while
      // forcing the MAL profile fetch. Reuse that result for both query caches so the refresh is
      // atomic and cannot race a second request that still has the old profile cached.
      const freshRecommendations = await fetchRecommendations(
        username,
        refreshParams,
        availableAnimeMetadataIDs,
        true,
        true
      );
      updateAnimeDB(freshRecommendations.animeData);
      queryClient.setQueryData(recommendationsKey, freshRecommendations);
      queryClient.setQueryData(contributorsKey, freshRecommendations);

      submitAnalyticsEvent({
        category: 'recommendations',
        subcategory: 'profile_force_refresh',
        payload: { source: refreshParams.profileSource, model: refreshParams.modelName, surface: getSurface() },
      });
    } catch (err) {
      profileRefreshError = typeof err === 'string' ? err : err instanceof Error ? err.message : 'Refresh failed';
      getSentry()?.captureException('Error force-refreshing recommendation profile', {
        extra: { params: $params, username },
      });
    } finally {
      isProfileRefreshing = false;
    }
  };

  let lastRecosRes:
    | {
        recommendations: Recommendation[];
        animeData: {
          [animeID: number]: AnimeDetails;
        };
        userRatingStats?: UserRatingStats | null;
        contributionBaseline?: number;
      }
    | undefined = undefined;
  $: recosRes = createQuery<
    | {
        recommendations: Recommendation[];
        animeData: { [animeID: number]: AnimeDetails };
        userRatingStats?: UserRatingStats | null;
        contributionBaseline?: number;
      }
    | undefined
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
    submitAnalyticsEvent(
      {
        category: 'recommendations',
        subcategory: 'refetch_error',
        payload: { surface: getSurface(), model: $params.modelName, source: $params.profileSource },
      },
      true
    );
  }

  $: recommendations = (() => {
    if ($recoContributorsRes.data) {
      return $recoContributorsRes.data;
    }
    return recosData ? recosData : null;
  })();

  $: userRatingStats = recommendations?.userRatingStats ?? null;

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

      submitAnalyticsEvent({
        category: 'recommendations',
        subcategory: 'exclude_ranking_add',
        payload: { anime_id: animeID, surface: getSurface() },
      });
      state.excludedRankingAnimeIDs.push(animeID);
      return state;
    });

  const excludeGenreID = (genreID: number, genreName: string) => {
    genresDB.update((db) => db.set(genreID, genreName));

    params.update((state) => {
      if (state.excludedGenreIDs.includes(genreID)) {
        return state;
      }

      submitAnalyticsEvent({
        category: 'recommendations',
        subcategory: 'exclude_genre_add',
        payload: { genre_id: genreID, genre_name: genreName, surface: getSurface() },
      });
      state.excludedGenreIDs.push(genreID);
      return state;
    });
  };

  const includeGenreID = (genreID: number) => {
    submitAnalyticsEvent({ category: 'recommendations', subcategory: 'exclude_genre_remove', payload: { genre_id: genreID, surface: getSurface() } });
    $params.excludedGenreIDs = $params.excludedGenreIDs.filter((id) => id !== genreID);
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
      hideLogitWeight={userRatingStats?.isNonRater ?? false}
      onForceProfileRefresh={forceProfileRefresh}
      {isProfileRefreshing}
      {profileRefreshError}
    />
    <RecommendationsList
      recommendations={recommendations?.recommendations ?? []}
      contributionBaseline={recommendations?.contributionBaseline}
      animeMetadataDatabase={$animeMetadataDatabase}
      {userRatingStats}
      excludeRanking={excludedRankingAnimeIDs}
      excludeGenre={excludeGenreID}
      includeGenre={includeGenreID}
      excludedGenreIDs={$params.excludedGenreIDs}
      genreNames={$genresDB}
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
