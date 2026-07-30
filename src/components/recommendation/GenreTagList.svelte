<script lang="ts">
  import { onMount, tick } from 'svelte';
  import { Tag } from 'carbon-components-svelte';

  import type { Genre } from 'src/malAPI';

  export let genres: Genre[];
  export let excludeGenre: ((genreID: number, genreName: string) => void) | undefined;

  const TAG_GAP = 4;

  let containerElem: HTMLDivElement | null = null;
  let measureElem: HTMLDivElement | null = null;
  let chipElem: HTMLButtonElement | null = null;
  let visibleCount = genres.length;
  let popoverOpen = false;
  let popoverStyle = '';

  $: hiddenCount = genres.length - visibleCount;

  // Measures against a hidden copy of the full tag list to determine how many tags fit in the
  // cell, reserving room for the "+N" chip on the last visible row when some must be hidden.
  const measure = () => {
    if (!containerElem || !measureElem) {
      return;
    }

    const availHeight = containerElem.clientHeight;
    const children = Array.from(measureElem.children) as HTMLElement[];
    const chipCopy = children[children.length - 1];
    const tagEls = children.slice(0, -1);

    let fits = 0;
    for (const el of tagEls) {
      if (el.offsetTop + el.offsetHeight > availHeight) {
        break;
      }
      fits += 1;
    }

    if (fits >= genres.length) {
      visibleCount = genres.length;
      return;
    }
    if (fits === 0) {
      visibleCount = 0;
      return;
    }

    const lastVisible = tagEls[fits - 1];
    const chipFitsOnRow =
      lastVisible.offsetLeft + lastVisible.offsetWidth + TAG_GAP + chipCopy.offsetWidth <= measureElem.clientWidth;
    visibleCount = chipFitsOnRow ? fits : fits - 1;
  };

  onMount(() => {
    const observer = new ResizeObserver(() => measure());
    if (containerElem) {
      observer.observe(containerElem);
    }
    measure();
    return () => observer.disconnect();
  });

  $: if (genres && measureElem) {
    void tick().then(measure);
  }

  const togglePopover = () => {
    if (!popoverOpen && chipElem) {
      const rect = chipElem.getBoundingClientRect();
      const width = Math.min(280, window.innerWidth - 16);
      const left = Math.min(rect.left, window.innerWidth - width - 8);
      popoverStyle = `left: ${left}px; top: ${rect.bottom + 4}px; width: ${width}px;`;
    }
    popoverOpen = !popoverOpen;
  };

  const onWindowClick = (evt: MouseEvent) => {
    if (popoverOpen && !(evt.target instanceof Node && chipElem?.contains(evt.target))) {
      popoverOpen = false;
    }
  };

  const onWindowKeydown = (evt: KeyboardEvent) => {
    if (evt.key === 'Escape') {
      popoverOpen = false;
    }
  };
</script>

<svelte:window on:click={onWindowClick} on:keydown={onWindowKeydown} on:scroll={() => (popoverOpen = false)} />

<div class="genre-tags" bind:this={containerElem}>
  <div class="tags-flow">
    {#each genres.slice(0, visibleCount) as genre (genre.id)}
      <Tag size="sm" type="cool-gray" filter={!!excludeGenre} on:close={() => excludeGenre?.(genre.id, genre.name)}>
        {genre.name}
      </Tag>
    {/each}
    {#if hiddenCount > 0}
      <button
        class="overflow-chip"
        bind:this={chipElem}
        aria-expanded={popoverOpen}
        aria-label={`Show ${hiddenCount} more genres`}
        on:click|stopPropagation={togglePopover}
      >
        +{hiddenCount}
      </button>
    {/if}
  </div>
  <div class="tags-flow measure" bind:this={measureElem} aria-hidden="true">
    {#each genres as genre (genre.id)}
      <Tag size="sm" type="cool-gray" filter={!!excludeGenre}>{genre.name}</Tag>
    {/each}
    <button class="overflow-chip" tabindex="-1">+{genres.length}</button>
  </div>
</div>

{#if popoverOpen}
  <!-- svelte-ignore a11y-click-events-have-key-events -->
  <!-- svelte-ignore a11y-no-static-element-interactions -->
  <div class="popover" style={popoverStyle} on:click|stopPropagation>
    {#each genres as genre (genre.id)}
      <Tag size="sm" type="cool-gray" filter={!!excludeGenre} on:close={() => excludeGenre?.(genre.id, genre.name)}>
        {genre.name}
      </Tag>
    {/each}
  </div>
{/if}

<style lang="css">
  .genre-tags {
    position: relative;
    height: 100%;
    min-width: 0;
  }

  .tags-flow {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    align-content: flex-start;
    padding: 4px;
    height: 100%;
    overflow: hidden;
    box-sizing: border-box;
  }

  .tags-flow :global(.bx--tag) {
    margin: 0;
  }

  .tags-flow.measure {
    position: absolute;
    inset: 0;
    visibility: hidden;
    pointer-events: none;
    overflow: visible;
  }

  .overflow-chip {
    appearance: none;
    background: transparent;
    border: 1px dashed #6b7176;
    border-radius: 999px;
    color: #a8b0b8;
    font-size: 12px;
    line-height: 1;
    height: 18px;
    padding: 0 8px;
    cursor: pointer;
    align-self: center;
  }

  .overflow-chip:hover {
    border-color: #a8b0b8;
    color: #d0d5da;
  }

  .overflow-chip:focus-visible {
    outline: 2px solid #78a9ff;
    outline-offset: 1px;
  }

  .popover {
    position: fixed;
    z-index: 100;
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    background: #26292d;
    border: 1px solid #4d5358;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.55);
    padding: 8px;
  }

  .popover :global(.bx--tag) {
    margin: 0;
  }
</style>
