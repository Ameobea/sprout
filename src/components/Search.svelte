<script lang="ts">
  import Fuse from 'fuse.js';

  import type { Embedding } from '../routes/embedding';

  export let embedding: Embedding;
  export let onSubmit: (id: number) => void;

  let value = '';
  const fuse = new Fuse(embedding, {
    keys: ['metadata.title', 'metadata.title_english'],
  });

  let suggestionsVisible = true;
  $: suggestions = value && suggestionsVisible ? fuse.search(value, { limit: 8 }) : [];
</script>

<div class="root">
  <input
    type="text"
    placeholder="Search"
    bind:value
    on:blur={() => {
      suggestionsVisible = false;
    }}
    on:focus={() => {
      suggestionsVisible = true;
    }}
  />

  {#each suggestions as suggestion (suggestion.item.metadata.id)}
    <div role="button" tabindex={0} class="suggestion" on:mousedown={() => onSubmit(suggestion.item.metadata.id)}>
      {suggestion.item.metadata.title}
    </div>
  {/each}
</div>

<style lang="css">
  .root {
    box-sizing: border-box;
    width: calc(min(100vw, 320px));
    position: absolute;
    top: 0;
    left: 0;
    display: flex;
    flex-direction: column;
  }

  @media (max-width: 600px) {
    .root {
      width: 100vw;
    }
  }

  input {
    font-size: 16px;
    padding: 2px 5px;
    box-sizing: border-box;
    width: 100%;
    height: 30px;
  }

  .suggestion {
    z-index: 2;
    cursor: pointer;
    box-sizing: border-box;
    background: #121212ee;
    display: flex;
    align-items: center;
    font-size: 16px;
    padding: 4px 6px;
    border-bottom: 1px solid #888;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
  }
</style>
