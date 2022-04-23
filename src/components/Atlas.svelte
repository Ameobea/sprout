<script lang="ts">
  import { onMount } from 'svelte';

  import type { Embedding } from '../routes/index';
  import AnimeDetails from './AnimeDetails.svelte';
  import { AtlasViz } from './AtlasViz';
  import Search from './Search.svelte';

  export let embedding: Embedding;

  let viz: AtlasViz | null = null;
  let selectedAnimeID: number | null = null;
  let mouseDown = false;
  let svgPointerDownPos: { trigger: 'circle' | 'svg'; pos: [number, number] } | null = null;

  onMount(() => {
    viz = new AtlasViz(
      'viz',
      embedding,
      (newSelectedAnimeID: number | null) => {
        selectedAnimeID = newSelectedAnimeID;
      },
      (evt: MouseEvent) => {
        mouseDown = true;
        svgPointerDownPos = { trigger: 'circle', pos: [evt.clientX, evt.clientY] };
      }
    );
    const username = new URLSearchParams(window.location.search).get('username') ?? 'Expochant';
    fetch(`/mal-profile?username=${username}`)
      .then((res) => res.json())
      .then((profile) => {
        viz.displayMALUser(profile);
      });
  });

  const handleSVGPointerDown = (evt) => {
    mouseDown = true;
    if (svgPointerDownPos == null) {
      svgPointerDownPos = { trigger: 'svg', pos: [evt.clientX, evt.clientY] };
    }
  };
</script>

<div class="root">
  <svg
    xmlns="http://www.w3.org/2000/svg"
    id="viz"
    width="100%"
    height="100%"
    on:pointerdown={handleSVGPointerDown}
    on:pointerup={(evt) => {
      mouseDown = false;
      if (
        svgPointerDownPos.pos[0] === evt.clientX &&
        svgPointerDownPos.pos[1] === evt.clientY &&
        svgPointerDownPos.trigger === 'svg'
      ) {
        selectedAnimeID = null;
      }
      svgPointerDownPos = null;
    }}
  >
    <defs>
      <filter x="0" y="0" width="1" height="1" id="solid">
        <feFlood flood-color="#111111aa" result="bg" />
        <feMerge>
          <feMergeNode in="bg" />
          <feMergeNode in="SourceGraphic" />
        </feMerge>
      </filter>
      <!-- <radialGradient id="red-transparent-gradient">
        <stop offset="1%" stop-color="#ff000024" />
        <stop offset="25%" stop-color="#ff000015" />
        <stop offset="65%" stop-color="#ff000006" />
        <stop offset="95%" stop-color="#ff000000" />
      </radialGradient> -->
      <radialGradient id="red-transparent-gradient">
        <stop offset="1%" stop-color="#dddddd24" />
        <stop offset="25%" stop-color="#dddddd15" />
        <stop offset="65%" stop-color="#dddddd06" />
        <stop offset="95%" stop-color="#dddddd00" />
      </radialGradient>
    </defs>
    <g id="root-container">
      <g id="decorations-container" />
      <g id="points-container" />
      <g id="labels-container" />
    </g>
  </svg>
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

  /* Setting `point-events` to `none` has the effect of speeding up rendering significantly since
     the browser doesn't have to perform hit tests. */
  :global(g#points-container.disable-hittest) {
    pointer-events: none !important;
  }
  :global(g#points-container.disable-hittest circle) {
    pointer-events: none !important;
  }

  :global(text.hover-label) {
    font-weight: bold;
    fill: #ccc;
    pointer-events: none;
    background-color: #11111188;
  }

  :global(circle.mal-user-point) {
    fill: red !important;
    stroke: hotpink;
    stroke-width: 30px;
  }

  :global(circle.mal-user-point-background) {
    pointer-events: none;
  }
</style>
