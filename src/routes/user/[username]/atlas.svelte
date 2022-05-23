<script lang="ts">
  import { browser } from '$app/env';

  import { page } from '$app/stores';
  import { FitToScreen, Maximize } from 'carbon-icons-svelte';
  import Atlas from 'src/components/Atlas.svelte';
  import type { Embedding } from 'src/routes/embedding';
  import type { EmbeddingName } from 'src/types';
  import { onDestroy, onMount } from 'svelte';

  export let embeddingName: EmbeddingName;
  export let embedding: Embedding;

  let maxWidth: number | undefined = 950;
  let atlasWrapperElem: HTMLDivElement | null = null;
  $: username = $page.params.username;

  onMount(() => {
    document.getElementsByTagName('body')[0]!.style.overflowY = 'hidden';
  });

  onDestroy(() => {
    if (browser) {
      document.getElementsByTagName('body')[0]!.style.overflowY = 'auto';
    }
  });
</script>

<div class="root">
  <div
    class="maximize-icon-container"
    on:click={() => {
      if (!atlasWrapperElem) {
        return;
      }

      atlasWrapperElem.requestFullscreen();
    }}
  >
    <Maximize size={32} />
  </div>
  <div
    style="position: absolute;"
    bind:this={atlasWrapperElem}
    on:fullscreenchange={(evt) => {
      const isFullScreen = document.fullscreenElement === atlasWrapperElem;
      maxWidth = isFullScreen ? undefined : 950;
    }}
  >
    <Atlas {embeddingName} {embedding} {username} {maxWidth} disableEmbeddingSelection disableUsernameSearch />
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
