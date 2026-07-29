<script lang="ts">
  import { browser } from '$app/env';

  import { onMount } from 'svelte';

  import { getSurface, submitAnalyticsEvent } from 'src/analytics';
  import type { EmbeddingName } from 'src/types';
  import type { Embedding } from '../routes/embedding';
  import AnimeDetails from './AnimeDetails.svelte';
  import { AtlasViz, ColorBy, getDefaultColorBy } from './AtlasViz';
  import { DEFAULT_PROFILE_SOURCE, ProfileSource } from './recommendation/conf';
  import Search from './Search.svelte';
  import VizControls from './VizControls.svelte';

  export let embeddingName: EmbeddingName;
  export let embedding: Embedding;
  export let username: string | undefined = undefined;
  export let maxWidth: number | undefined = undefined;
  export let disableEmbeddingSelection: boolean = false;
  export let disableUsernameSearch: boolean = false;
  export let profileSource: ProfileSource = DEFAULT_PROFILE_SOURCE;

  let viz: AtlasViz | null = null;
  let selectedAnimeID: number | null = null;
  $: selectedDatum = selectedAnimeID === null || !viz ? null : viz.embeddedPointByID.get(selectedAnimeID)!;

  $: if (viz) {
    viz.setMaxWidth(maxWidth);
  }

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
    const startTime = performance.now();
    fetch(`/${profileSource}-profile?username=${username}`)
      .then((res) => res.json())
      .then((profile) => {
        const overlayStats = viz?.displayMALUser(profile);
        submitAnalyticsEvent({
          category: 'atlas',
          subcategory: 'load_profile_result',
          payload: {
            ok: true,
            entry_count: overlayStats?.entryCount,
            rendered_count: overlayStats?.renderedCount,
            missing_count: overlayStats?.missingCount,
            duration_ms: Math.round(performance.now() - startTime),
            surface: getSurface(),
          },
        });
      })
      .catch((err) => {
        console.error('Failed to load user profile for atlas', err);
        submitAnalyticsEvent({
          category: 'atlas',
          subcategory: 'load_profile_result',
          payload: { ok: false, surface: getSurface() },
        });
      });
  };

  onMount(() => {
    const usernameToLoad = username ?? new URLSearchParams(window.location.search).get('username');
    const userProfilePromise =
      usernameToLoad &&
      fetch(`/${profileSource}-profile?username=${usernameToLoad}`)
        .then((res) => res.json())
        .catch((err) => {
          console.error('Failed to load user profile for atlas', err);
          return null;
        });
    const neighborsPromise: Promise<{ neighbors: number[][] }> = fetch(`/neighbors?embedding=${embeddingName}`).then(
      (res) => res.json()
    );

    import('../pixi').then((mod) => {
      const setSelectedAnimeID = (newSelectedAnimeID: number | null) => {
        selectedAnimeID = newSelectedAnimeID;
      };
      viz = new AtlasViz(mod, 'viz', embedding, setSelectedAnimeID, maxWidth);
      viz.setColorBy(colorBy);

      neighborsPromise.then(({ neighbors }) => {
        viz?.setNeighbors(neighbors);
        if (userProfilePromise) {
          userProfilePromise.then((profile) => {
            if (profile) {
              viz?.displayMALUser(profile);
            }
          });
        }
      });
    });

    return () => {
      viz?.dispose();
      viz = null;
    };
  });
</script>

<svelte:head>
  <link href="https://fonts.googleapis.com/css2?family=PT+Sans" rel="stylesheet" />
</svelte:head>

<div class="root">
  <canvas id="viz" />
</div>
{#if viz}
  <Search
    {embedding}
    onSubmit={(id) => {
      submitAnalyticsEvent({ category: 'atlas', subcategory: 'search_select', payload: { anime_id: id, surface: getSurface() } });
      viz?.flyTo(id);
    }}
    suggestionsStyle="top: 30px;"
  />
  <VizControls {colorBy} {setColorBy} {loadMALProfile} {disableEmbeddingSelection} {disableUsernameSearch} />
{/if}
<div id="atlas-viz-legend" />
{#if selectedDatum !== null && viz}
  <AnimeDetails id={selectedDatum.metadata.id} datum={selectedDatum} />
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
