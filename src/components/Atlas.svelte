<script lang="ts">
  import { browser } from '$app/env';

  import { onDestroy, onMount } from 'svelte';

  import type { Embedding } from '../routes/embedding';
  import AnimeDetails from './AnimeDetails.svelte';
  import { AtlasViz, ColorBy, getDefaultColorBy } from './AtlasViz';
  import Search from './Search.svelte';
  import VizControls from './VizControls.svelte';

  export let embedding: Embedding;

  let viz: AtlasViz | null = null;
  let selectedAnimeID: number | null = null;

  let colorBy = browser ? getDefaultColorBy() : ColorBy.AiredFromYear;
  const setColorBy = (newColorBy: ColorBy) => {
    colorBy = newColorBy;
    viz?.setColorBy(colorBy);
  };

  const loadMALProfile = (username: string) => {
    if (!viz) {
      console.error('Tried to load MAL profile before Atlas viz was loaded.');
      return;
    }
    fetch(`/mal-profile?username=${username}`)
      .then((res) => res.json())
      .then((profile) => {
        viz.displayMALUser(profile);
      });
  };

  onMount(() => {
    const username = new URLSearchParams(window.location.search).get('username') ?? 'Expochant';
    const userProfilePromise = fetch(`/mal-profile?username=${username}`).then((res) => res.json());
    const neighborsPromise: Promise<{ neighbors: number[][] }> = fetch('/neighbors').then((res) => res.json());

    import('../pixi').then((mod) => {
      viz = new AtlasViz(mod, 'viz', embedding, (newSelectedAnimeID: number | null) => {
        selectedAnimeID = newSelectedAnimeID;
      });
      viz.setColorBy(colorBy);

      neighborsPromise.then(({ neighbors }) => {
        viz.setNeighbors(neighbors);
        userProfilePromise.then((profile) => {
          viz.displayMALUser(profile);
        });
      });
    });
  });

  onDestroy(() => {
    if (viz) {
      viz.dispose();
    }
  });
</script>

<div class="root">
  <canvas id="viz" />
</div>
{#if viz}
  <Search {embedding} onSubmit={(id) => viz.flyTo(id)} />
  <VizControls {colorBy} {setColorBy} {loadMALProfile} />
{/if}
<div id="atlas-viz-legend" />
{#if selectedAnimeID !== null}
  <AnimeDetails id={selectedAnimeID} datum={viz.embeddedPointByID.get(selectedAnimeID)} />
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

  @media (max-width: 800px) {
    #atlas-viz-legend {
      /* Scale to 83% but keep it aligned to the right of the screen */
      transform: scale(0.83);
      transform-origin: right top;
      right: 8px;
    }
  }

  @media (max-width: 600px) {
    #atlas-viz-legend {
      top: 106px;
      right: 0;
      background: rgba(0, 0, 0, 0.8);
      padding: 4px;
    }
  }
</style>
