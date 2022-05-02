<script lang="ts">
  import { onDestroy, onMount } from 'svelte';

  import type { Embedding } from '../routes/index';
  import AnimeDetails from './AnimeDetails.svelte';
  import { AtlasViz } from './AtlasViz';
  import Search from './Search.svelte';

  export let embedding: Embedding;

  let viz: AtlasViz | null = null;
  let selectedAnimeID: number | null = null;

  onMount(() => {
    const username = new URLSearchParams(window.location.search).get('username') ?? 'Expochant';
    const userProfilePromise = fetch(`/mal-profile?username=${username}`).then((res) => res.json());
    const neighborsPromise: Promise<{ neighbors: number[][] }> = fetch('/neighbors').then((res) => res.json());

    import('../pixi').then((mod) => {
      viz = new AtlasViz(mod, 'viz', embedding, (newSelectedAnimeID: number | null) => {
        selectedAnimeID = newSelectedAnimeID;
      });

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
</style>
