<script lang="ts">
  import Fuse from 'fuse.js';

  export let embedding: { metadata: { id: number; title: string; title_english: string } }[];
  export let onSubmit: (id: number, title: string, titleEnglish: string) => void;
  export let style: string | undefined = undefined;
  export let inputStyle: string | undefined = undefined;
  export let suggestionsStyle: string | undefined = undefined;
  export let blurredValue: string | undefined = undefined;

  let value = '';
  const fuse = new Fuse(embedding, {
    keys: ['metadata.title', 'metadata.title_english'],
  });

  let isFocused = false;
  $: suggestions = value && isFocused ? fuse.search(value, { limit: 8 }) : [];

  const handleInputChange = (evt: any) => {
    value = evt.target.value;
  };
</script>

<div class="root" {style}>
  <input
    style={inputStyle}
    type="text"
    placeholder={isFocused ? undefined : 'Search for Anime'}
    value={isFocused ? value : blurredValue || value}
    on:input={handleInputChange}
    on:blur={() => {
      isFocused = false;
    }}
    on:focus={() => {
      isFocused = true;
    }}
  />

  {#if suggestions.length > 0}
    <div class="suggestions-container" style={suggestionsStyle}>
      {#each suggestions as suggestion (suggestion.item.metadata.id)}
        <div
          role="button"
          tabindex={0}
          class="suggestion"
          on:mousedown={() => {
            onSubmit(
              suggestion.item.metadata.id,
              suggestion.item.metadata.title,
              suggestion.item.metadata.title_english
            );
            value = '';
          }}
        >
          {suggestion.item.metadata.title_english || suggestion.item.metadata.title}
        </div>
      {/each}
    </div>
  {/if}
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

  .suggestions-container {
    position: absolute;
    top: 36px;
    width: 100%;
    z-index: 2;
  }

  .suggestion {
    z-index: 2;
    cursor: pointer;
    font-size: 17px;
    padding: 6px 8px;
    border-bottom: 1px solid #888;
    background-color: #080808;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
  }
</style>
