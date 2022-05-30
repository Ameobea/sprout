<script context="module" lang="ts">
  const DESCRIPTION =
    'Interactive 2D visualization of over 10,000 anime related media - personalized with your ratings from MyAnimeList or AniList';
</script>

<script lang="ts">
  import { browser } from '$app/env';
  import { onDestroy, onMount } from 'svelte';
  import { page } from '$app/stores';
  import { Maximize } from 'carbon-icons-svelte';
  import SvelteSeo from 'svelte-seo';

  import Atlas from 'src/components/Atlas.svelte';
  import type { Embedding } from 'src/routes/embedding';
  import type { EmbeddingName } from 'src/types';
  import { captureMessage } from 'src/sentry';
  import { DEFAULT_PROFILE_SOURCE, type ProfileSource } from 'src/components/recommendation/conf';

  export let embeddingName: EmbeddingName;
  export let embedding: Embedding;

  let maxWidth: number | undefined = 950;
  let atlasWrapperElem: HTMLDivElement | null = null;
  $: username = $page.params.username;
  $: profileSource = ($page.url.searchParams.get('source') as ProfileSource) ?? DEFAULT_PROFILE_SOURCE;
  $: title = `Sprout Atlas Viz for ${username}`;

  onMount(() => {
    document.getElementsByTagName('body')[0]!.style.overflowY = 'hidden';
  });

  onDestroy(() => {
    if (browser) {
      document.getElementsByTagName('body')[0]!.style.overflowY = 'auto';
    }
  });

  const enableFullscreen = () => {
    if (!atlasWrapperElem) {
      return;
    }

    captureMessage('Maximizing Atlas on user profile');
    atlasWrapperElem.requestFullscreen();
  };
</script>

<SvelteSeo
  {title}
  description={DESCRIPTION}
  openGraph={{
    title,
    description: DESCRIPTION,
    images: [
      {
        url: 'https://anime.ameo.dev/atlas-viz.jpg',
        alt: "A screenshot of the Sprout atlas visualization showing many anime and connections indicating the relationships between anime in a user's list",
        width: 1214,
        height: 822,
      },
    ],
  }}
  twitter={{
    card: 'summary',
    title,
    description: DESCRIPTION,
  }}
/>

<div class="root">
  <div
    role="button"
    tabindex="0"
    aria-label="full-screen mode"
    class="maximize-icon-container"
    on:click={enableFullscreen}
    on:keydown={(event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        enableFullscreen();
      }
    }}
  >
    <Maximize size={32} />
  </div>
  <div
    style="position: absolute;"
    bind:this={atlasWrapperElem}
    on:fullscreenchange={() => {
      const isFullScreen = document.fullscreenElement === atlasWrapperElem;
      maxWidth = isFullScreen ? undefined : 950;
    }}
  >
    <Atlas
      {embeddingName}
      {embedding}
      {username}
      {profileSource}
      {maxWidth}
      disableEmbeddingSelection
      disableUsernameSearch
    />
  </div>
</div>

<style lang="css">
  .root {
    display: flex;
    flex-direction: column;
  }

  .maximize-icon-container {
    margin-left: auto;
    margin-top: 4px;
    margin-right: 4px;
    z-index: 2;
    cursor: pointer;
    transform: scale(0.8);
  }

  .maximize-icon-container:hover {
    transform: scale(0.9);
  }
</style>
