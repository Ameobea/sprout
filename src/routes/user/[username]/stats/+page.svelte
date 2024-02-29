<script context="module" lang="ts">
  const DESCRIPTION = 'AI-powered anime recommendations, visualizations, and tools';
</script>

<script lang="ts">
  import { page } from '$app/stores';
  import SvelteSeo from 'svelte-seo';

  import GenresBarChart from 'src/components/profileStats/GenresBarChart.svelte';
  import RatingDistributionChart from 'src/components/profileStats/RatingDistributionChart.svelte';
  import type { PageData } from './$types';

  $: username = $page.params.username;
  $: title = `Anime Profile Stats for ${username}`;

  export let data: PageData;
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
