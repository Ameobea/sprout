<script lang="ts">
  import { browser } from '$app/environment';

  import { onMount } from 'svelte';
  import { ToastNotification } from 'carbon-components-svelte';

  import { getSurface, submitAnalyticsEvent } from 'src/analytics';
  import type { CompatAnimeListEntry } from 'src/anilistAPI';
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

  let profileError: string | null = null;
  let profileErrorSeq = 0;
  const showProfileError = (err: unknown) => {
    profileError = err instanceof Error ? err.message : String(err);
    profileErrorSeq += 1;
  };

  const parseErrorMessage = (body: string, status: number): string => {
    try {
      return JSON.parse(body).message || `Request failed with status ${status}`;
    } catch {
      return body.slice(0, 300) || `Request failed with status ${status}`;
    }
  };

  const fetchProfile = async (username: string): Promise<CompatAnimeListEntry[]> => {
    const res = await fetch(`/${profileSource}-profile?username=${encodeURIComponent(username)}`);
    if (!res.ok) {
      throw new Error(parseErrorMessage(await res.text(), res.status));
    }
    return res.json();
  };

  const loadMALProfile = (username: string) => {
    if (!viz) {
      console.error('Tried to load MAL profile before Atlas viz was loaded.');
      return;
    }
    const startTime = performance.now();
    fetchProfile(username)
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
        showProfileError(err);
        submitAnalyticsEvent({
          category: 'atlas',
          subcategory: 'load_profile_result',
          payload: { ok: false, surface: getSurface() },
        });
      });
  };

  onMount(() => {
    const usernameToLoad = username ?? new URLSearchParams(window.location.search).get('username');
    const userProfilePromise = usernameToLoad
      ? fetchProfile(usernameToLoad).catch((err) => {
          console.error('Failed to load user profile for atlas', err);
          showProfileError(err);
          return null;
        })
      : null;
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
        userProfilePromise?.then((profile) => {
          if (profile) {
            viz?.displayMALUser(profile);
          }
        });
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
  <VizControls {colorBy} {setColorBy} {loadMALProfile} {disableUsernameSearch} />
{/if}
<div id="atlas-viz-legend" />
{#if selectedDatum !== null && viz}
  <AnimeDetails id={selectedDatum.metadata.id} datum={selectedDatum} />
{/if}
{#if profileError}
  {#key profileErrorSeq}
    <div class="profile-error-toast">
      <ToastNotification
        lowContrast
        kind="error"
        title="Failed to load profile"
        subtitle={profileError}
        timeout={10000}
        on:close={() => {
          profileError = null;
        }}
      />
    </div>
  {/key}
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

  .profile-error-toast {
    position: fixed;
    z-index: 2;
    bottom: 12px;
    right: 12px;
  }

  .profile-error-toast :global(.bx--toast-notification) {
    margin: 0;
    width: min(360px, calc(100vw - 24px));
  }

  .profile-error-toast :global(.bx--toast-notification__subtitle) {
    overflow-wrap: anywhere;
  }

  @media (max-width: 600px) {
    .profile-error-toast {
      left: 12px;
      right: 12px;
    }

    .profile-error-toast :global(.bx--toast-notification) {
      width: 100%;
    }
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
