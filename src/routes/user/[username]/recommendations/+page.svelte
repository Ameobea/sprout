<script lang="ts" context="module">
  // Inert plain-JSON copy of the recommendations, rendered only for non-browser clients.  The
  // AnymeX app scrapes this page's HTML for a <script> containing the quoted key
  // "initialRecommendations" and JSON-parses it; SvelteKit 2's own data embedding is a JS object
  // literal which their parser can't read.
  const buildAnymexShimJSON = (recommendations: Extract<RecommendationsResponse, { type: 'ok' }>): string => {
    const animeData: { [id: number]: unknown } = {};
    for (const rec of recommendations.recommendations) {
      const datum = recommendations.animeData[rec.id];
      if (!datum) {
        continue;
      }
      animeData[rec.id] = {
        id: datum.id,
        title: datum.title,
        alternative_titles: { en: datum.alternative_titles?.en },
        main_picture: { large: datum.main_picture?.large },
        synopsis: datum.synopsis,
        genres: datum.genres?.map((genre) => ({ name: genre.name })),
      };
    }

    return JSON.stringify({
      initialRecommendations: {
        type: 'ok',
        recommendations: recommendations.recommendations.map(({ id, score, planToWatch }) => ({
          id,
          score,
          planToWatch,
        })),
        animeData,
      },
    }).replace(/</g, '\\u003c');
  };

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
  import { onMount } from 'svelte';
  import SvelteSeo from 'svelte-seo';

  import { submitAnalyticsEvent, wasInAppNav } from 'src/analytics';
  import { DEFAULT_MODEL_NAME, DEFAULT_PROFILE_SOURCE } from 'src/components/recommendation/conf';
  import type { AnimeDetails } from 'src/malAPI';
  import type { RecommendationsResponse } from 'src/routes/user/[username]/recommendations/+page.server';
  import InteractiveRecommendations from 'src/components/recommendation/InteractiveRecommendations.svelte';
  import type { PageData } from './$types';

  export let data: PageData;
  $: ({ initialRecommendations: rawInitialRecommendations, genreNames } = data);
  $: initialRecommendations = rawInitialRecommendations as RecommendationsResponse;

  $: anymexShimHtml =
    data.nonBrowserClient && initialRecommendations.type === 'ok'
      ? `<script type="application/json" id="anymex-compat">${buildAnymexShimJSON(initialRecommendations)}</${'script'}>`
      : null;

  onMount(() => {
    if (!wasInAppNav()) {
      return;
    }
    const source = $page.url.searchParams.get('source') ?? DEFAULT_PROFILE_SOURCE;
    const model = $page.url.searchParams.get('model') ?? DEFAULT_MODEL_NAME;
    if (initialRecommendations.type === 'ok') {
      submitAnalyticsEvent({
        category: 'recommendations',
        subcategory: 'results_shown',
        payload: { username, source, model, count: initialRecommendations.recommendations.length },
      });
    } else {
      submitAnalyticsEvent({
        category: 'recommendations',
        subcategory: 'load_error',
        payload: { username, source, kind: initialRecommendations.kind ?? 'unknown' },
      });
    }
  });

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

  {#if anymexShimHtml}
    {@html anymexShimHtml}
  {/if}
{:else}
  <div style="text-align: center; padding-top: 10px;">
    <h2 style="margin-bottom: 20px;">Error Loading Recommendations</h2>
    <p>{initialRecommendations.error}</p>
    <p style="margin-top: 8px"><a data-sveltekit-preload-data="hover" href="/">Back to Homepage</a></p>
  </div>
{/if}
