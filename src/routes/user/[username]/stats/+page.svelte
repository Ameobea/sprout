<script context="module" lang="ts">
  const DESCRIPTION = 'AI-powered anime recommendations, visualizations, and tools';
</script>

<script lang="ts">
  import { page } from '$app/stores';
  import { onMount } from 'svelte';
  import SvelteSeo from 'svelte-seo';

  import { submitAnalyticsEvent, wasInAppNav } from 'src/analytics';
  import { DEFAULT_PROFILE_SOURCE } from 'src/components/recommendation/conf';
  import GenresBarChart from 'src/components/profileStats/GenresBarChart.svelte';
  import ProfileAnalysisLists from 'src/components/profileStats/ProfileAnalysisLists.svelte';
  import RatingDistributionChart from 'src/components/profileStats/RatingDistributionChart.svelte';
  import type { PageData } from './$types';

  $: username = $page.params.username;
  $: title = `Anime Profile Stats for ${username}`;

  export let data: PageData;

  onMount(() => {
    if (!wasInAppNav()) {
      return;
    }
    const source = $page.url.searchParams.get('source') ?? DEFAULT_PROFILE_SOURCE;
    if (data.profileRes.type === 'ok') {
      submitAnalyticsEvent({
        category: 'profile_stats',
        subcategory: 'shown',
        payload: { username, source, profile_size: data.profileRes.profile.length, has_analysis: !!data.profileAnalysis },
      });
    } else {
      submitAnalyticsEvent({
        category: 'profile_stats',
        subcategory: 'load_error',
        payload: { username, source, kind: data.profileRes.kind ?? 'unknown' },
      });
    }
  });
</script>

<SvelteSeo
  {title}
  description={DESCRIPTION}
  openGraph={{
    title,
    description: DESCRIPTION,
  }}
  twitter={{
    card: 'summary',
    title,
    description: DESCRIPTION,
  }}
/>

<div class="root">
  {#if data.profileRes.type === 'error'}
    <div class="error">
      <h2>Error fetching profile</h2>
      <p>{data.profileRes.error}</p>
      <a href="/">Back to Homepage</a>
    </div>
  {:else}
    {#if data.ratedStats && data.ratedStats.consideredCount > 0}
      <div class="rated-stat">
        <b>{Math.round(data.ratedStats.ratedFraction * 100)}%</b> of your list is rated ({data.ratedStats.ratedCount} of
        {data.ratedStats.consideredCount} watched entries)
        {#if data.ratedStats.isNonRater}
          — treated as a presence-only profile, so predicted ratings are hidden in your recommendations
        {/if}
      </div>
    {/if}
    {#if data.profileAnalysis}
      <ProfileAnalysisLists
        mostImpactfulRatings={data.profileAnalysis.mostImpactfulRatings}
        mostSurprisingItems={data.profileAnalysis.mostSurprisingItems}
        normalizationStats={data.profileAnalysis.normalizationStats}
        animeData={data.animeData}
      />
    {/if}

    <RatingDistributionChart profile={data.profileRes.profile} />
    <GenresBarChart profile={data.profileRes.profile} animeData={data.animeData} />

    <i style="margin-top: 20px;">More stats + charts will be added soon!</i>
  {/if}
</div>

<style lang="css">
  .root {
    display: flex;
    flex-direction: column;
    padding: 4px;
    text-align: center;
  }

  .rated-stat {
    font-size: 0.9rem;
    color: #aaa;
    text-align: left;
    margin-bottom: 14px;
  }

  .rated-stat b {
    color: #ddd;
  }

  .error h2 {
    margin-bottom: 10px;
  }

  .error p {
    margin-bottom: 20px;
  }

  .error a {
    font-size: 18px;
  }
</style>
