<script lang="ts">
  import { onMount } from 'svelte';

  import type { Embedding } from '../routes/index';
  import AnimeDetails from './AnimeDetails.svelte';
  import { AtlasViz } from './AtlasViz';
  import Search from './Search.svelte';

  export let embedding: Embedding;

  let viz: AtlasViz | null = null;
  let selectedAnimeID: number | null = null;

  onMount(() => {
    viz = new AtlasViz('viz', embedding, (newSelectedAnimeID: number | null) => {
      selectedAnimeID = newSelectedAnimeID;
    });
    // TODO: Remove
    fetch('/mal-profile?username=Expochant')
      .then((res) => res.json())
      .then((profile) => {
        viz.displayMALUser(profile);
      });
  });

  const handleSVGClick = (evt) => {
    if ((evt.target as SVGSVGElement).tagName === 'svg') {
      selectedAnimeID = null;
    }
  };
</script>

<div class="root">
  <svg id="viz" width="100%" height="100%" on:click={handleSVGClick}>
    <defs>
      <filter x="0" y="0" width="1" height="1" id="solid">
        <feFlood flood-color="#111111aa" result="bg" />
        <feMerge>
          <feMergeNode in="bg" />
          <feMergeNode in="SourceGraphic" />
        </feMerge>
      </filter>
      <filter id="blur">
        <feGaussianBlur stdDeviation="50" />
      </filter>
      <radialGradient id="red-transparent-gradient">
        <stop offset="10%" stop-color="#ff000022" />
        <stop offset="95%" stop-color="#ff000001" />
      </radialGradient>
    </defs>
    <g id="root-container">
      <g id="decorations-container" />
      <g id="points-container" />
      <g id="labels-container" />
    </g></svg
  >
</div>
{#if viz}
  <Search {embedding} onSubmit={(id) => viz.flyTo(id)} />
{/if}
<div id="atlas-viz-legend" />
{#if selectedAnimeID !== null}
  <AnimeDetails id={selectedAnimeID} />
{/if}

<style lang="css">
  .root {
    display: flex;
    flex-direction: column;
    height: 100vh;
    width: 100vw;
  }

  #atlas-viz-legend {
    position: absolute;
    top: 0;
    right: 16px;
    z-index: 1;
    background-color: #11111188;
  }

  :global(g#points-container circle) {
    cursor: pointer;
  }

  :global(text.hover-label) {
    font-weight: bold;
    fill: #ccc;
    pointer-events: none;
    background-color: #11111188;
  }

  :global(circle.mal-user-point) {
    fill: red !important;
  }

  :global(circle.mal-user-point-background) {
    /* fill: #ff000022; */
    /* z-index: -1; */
  }
</style>
