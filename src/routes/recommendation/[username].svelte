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
  import { QueryClient, QueryClientProvider } from '@sveltestack/svelte-query';

  import { HeaderTitle } from 'src/components/recommendation/HeaderTitle';
  import type { AnimeDetails } from 'src/malAPI';
  import type { RecommendationsResponse } from './[username]';
  import InteractiveRecommendations from 'src/components/recommendation/InteractiveRecommendations.svelte';

  export let initialRecommendations: RecommendationsResponse;

  $: animeData =
    initialRecommendations.type === 'ok'
      ? initialRecommendations.animeData
      : ({} as {
          [id: number]: AnimeDetails;
        });
  $: recommendationsList = initialRecommendations.type === 'ok' ? initialRecommendations.recommendations : [];
  $: username = $page.params.username;
  $: title = `Anime Recommendations for ${username}`;
  $: {
    HeaderTitle.set(`Anime Recommendations for ${username}`);
  }

  const queryClient = new QueryClient();
</script>

<SvelteSeo
  {title}
  description={'AI-powered personalized anime recommendations, visualizations, and resources'}
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
    imageAlt: animeData[recommendationsList[0]?.id]?.title,
    description: buildOpengraphDescription(username, initialRecommendations),
  }}
/>

<QueryClientProvider client={queryClient}>
  <InteractiveRecommendations {username} {initialRecommendations} />
</QueryClientProvider>
