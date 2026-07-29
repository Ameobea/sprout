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
  import RatingDistributionChart from 'src/components/profileStats/RatingDistributionChart.svelte';
  import type { UserStatsLoadProps } from './stats';

  $: username = $page.params.username;
  $: title = `Anime Profile Stats for ${username}`;

  export let animeData: UserStatsLoadProps['animeData'];
  export let profileRes: UserStatsLoadProps['profileRes'];

  onMount(() => {
    if (!wasInAppNav()) {
      return;
    }
    const source = $page.url.searchParams.get('source') ?? DEFAULT_PROFILE_SOURCE;
    if (profileRes.type === 'ok') {
      submitAnalyticsEvent({
        category: 'profile_stats',
        subcategory: 'shown',
        payload: { username, source, profile_size: profileRes.profile.length },
      });
    } else {
      submitAnalyticsEvent({
        category: 'profile_stats',
        subcategory: 'load_error',
        payload: { username, source, kind: 'unknown' },
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
  {#if profileRes.type === 'error'}
    <div class="error">
      <h2>Error fetching profile</h2>
      <p>{profileRes.error}</p>
      <a href="/">Back to Homepage</a>
    </div>
  {:else}
    <RatingDistributionChart profile={profileRes.profile} />
    <GenresBarChart profile={profileRes.profile} {animeData} />

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
