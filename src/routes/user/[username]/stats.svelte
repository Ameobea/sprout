<script lang="ts">
  import GenresBarChart from 'src/components/profileStats/GenresBarChart.svelte';
  import RatingDistributionChart from 'src/components/profileStats/RatingDistributionChart.svelte';
  import type { UserStatsLoadProps } from './stats';

  export let animeData: UserStatsLoadProps['animeData'];
  export let profileRes: UserStatsLoadProps['profileRes'];
</script>

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
