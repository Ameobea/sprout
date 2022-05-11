<script lang="ts" context="module">
  const buildOpengraphDescription = (username: string, recommendations: RecommendationsResponse) => {
    if (recommendations.type === 'error') {
      return '';
    }

    const recommendationsString = recommendations.recommendations
      .slice(0, 4)
      .map((recommendation) => recommendations.animeData[recommendation.id].title)
      .join(', ');
    return `Top anime recommendations for ${username}:\n\n${recommendationsString}...`;
  };
</script>

<script lang="ts">
  import { page } from '$app/stores';
  import SvelteSeo from 'svelte-seo';

  import { HeaderTitle } from 'src/components/recommendation/HeaderTitle';
  import RecommendationListItem from 'src/components/recommendation/RecommendationListItem.svelte';
  import type { AnimeDetails } from 'src/malAPI';
  import type { RecommendationsResponse } from './[username]';

  export let recommendations: RecommendationsResponse;

  $: animeData =
    recommendations.type === 'ok'
      ? recommendations.animeData
      : ({} as {
          [id: number]: AnimeDetails;
        });
  $: recommendationsList = recommendations.type === 'ok' ? recommendations.recommendations : [];
  $: username = $page.params.username;
  $: title = `Anime Recommendations for ${username}`;
  $: {
    HeaderTitle.set(`Anime Recommendations for ${username}`);
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

<SvelteSeo
  {title}
  description={'AI-powered personalized anime recommendations, visualizations, and resources'}
  openGraph={{
    title: `Anime Recommendations for ${username}`,
    description: buildOpengraphDescription(username, recommendations),
    images:
      recommendations.type === 'ok'
        ? recommendationsList.slice(0, 2).map(({ id }) => {
            const datum = animeData[id];
            return {
              url: datum.main_picture.large ?? datum.main_picture.medium,
              alt: datum.title,
            };
          })
        : undefined,
  }}
  twitter={{
    card: 'summary',
    title,
    image: animeData[recommendationsList[0]?.id]?.main_picture.large,
    imageAlt: animeData[recommendationsList[0]?.id]?.title,
    description: buildOpengraphDescription(username, recommendations),
  }}
/>

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
