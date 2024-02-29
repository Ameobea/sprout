<script lang="ts" context="module">
  const buildOpengraphDescription = (username: string, recommendations: RecommendationsResponse) => {
    if (recommendations.type === 'error') {
      return '';
    }

    const recommendationsString = recommendations.recommendations
      .slice(0, 4)
      .map(
        (recommendation) =>
          recommendations.animeData[recommendation.id].alternative_titles?.en ||
          recommendations.animeData[recommendation.id].title
      )
      .join(', ');
    return `Top anime recommendations for ${username}:\n\n${recommendationsString}...`;
  };
</script>

<script lang="ts">
  import { page } from '$app/stores';
  import SvelteSeo from 'svelte-seo';

  import type { AnimeDetails } from 'src/malAPI';
  import type { RecommendationsResponse } from 'src/routes/user/[username]/recommendations/+page.server';
  import InteractiveRecommendations from 'src/components/recommendation/InteractiveRecommendations.svelte';
  import type { PageData } from './$types';

  export let data: PageData;
  $: ({ initialRecommendations: rawInitialRecommendations, genreNames } = data);
  $: initialRecommendations = rawInitialRecommendations as RecommendationsResponse;

  $: animeData =
    initialRecommendations.type === 'ok'
      ? initialRecommendations.animeData
      : ({} as {
          [id: number]: AnimeDetails;
        });
  $: recommendationsList = initialRecommendations.type === 'ok' ? initialRecommendations.recommendations : [];
  $: username = $page.params.username;
  $: title = `Anime Recommendations for ${username}`;
</script>

{#if initialRecommendations.type === 'ok'}
  <SvelteSeo
    {title}
    description="AI-powered anime recommendations, visualizations, and tools"
    openGraph={{
      title: `Anime Recommendations for ${username}`,
      description: buildOpengraphDescription(username, initialRecommendations),
      images:
        initialRecommendations.type === 'ok'
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
      imageAlt:
        animeData[recommendationsList[0]?.id]?.alternative_titles?.en || animeData[recommendationsList[0]?.id]?.title,
      description: buildOpengraphDescription(username, initialRecommendations),
    }}
  />

  <InteractiveRecommendations {username} {initialRecommendations} {genreNames} />
{:else}
  <div style="text-align: center; padding-top: 10px;">
    <h2 style="margin-bottom: 20px;">Error Loading Recommendations</h2>
    <p>{initialRecommendations.error}</p>
    <p style="margin-top: 8px"><a data-sveltekit-preload-data="hover" href="/">Back to Homepage</a></p>
  </div>
{/if}
