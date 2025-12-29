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
  let selectedIndex = -1;
  $: suggestions = value && isFocused ? fuse.search(value, { limit: 8 }) : [];

  // Reset selected index when suggestions change
  $: if (suggestions) {
    selectedIndex = -1;
  }

  const handleInputChange = (evt: any) => {
    value = evt.target.value;
  };

  const handleKeyDown = (evt: KeyboardEvent) => {
    if (!suggestions.length) return;

    if (evt.key === 'ArrowDown') {
      evt.preventDefault();
      selectedIndex = Math.min(selectedIndex + 1, suggestions.length - 1);
      scrollToSelected();
    } else if (evt.key === 'ArrowUp') {
      evt.preventDefault();
      selectedIndex = Math.max(selectedIndex - 1, -1);
      scrollToSelected();
    } else if (evt.key === 'Enter') {
      evt.preventDefault();
      if (selectedIndex >= 0 && selectedIndex < suggestions.length) {
        const suggestion = suggestions[selectedIndex];
        onSubmit(suggestion.item.metadata.id, suggestion.item.metadata.title, suggestion.item.metadata.title_english);
        value = '';
        selectedIndex = -1;
      }
    } else if (evt.key === 'Escape') {
      evt.preventDefault();
      isFocused = false;
      selectedIndex = -1;
    }
  };

  const scrollToSelected = () => {
    if (selectedIndex >= 0) {
      const element = document.querySelector(`.suggestion[data-index="${selectedIndex}"]`);
      if (element) {
        element.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      }
    }
  };

  const selectSuggestion = (index: number) => {
    const suggestion = suggestions[index];
    onSubmit(suggestion.item.metadata.id, suggestion.item.metadata.title, suggestion.item.metadata.title_english);
    value = '';
    selectedIndex = -1;
  };
</script>

<div class="root" {style}>
  <input
    style={inputStyle}
    type="text"
    placeholder={isFocused ? undefined : 'Search for Anime'}
    value={isFocused ? value : blurredValue || value}
    on:input={handleInputChange}
    on:keydown={handleKeyDown}
    on:blur={() => {
      isFocused = false;
    }}
    on:focus={() => {
      isFocused = true;
    }}
  />

  {#if suggestions.length > 0}
    <div class="suggestions-container" style={suggestionsStyle}>
      {#each suggestions as suggestion, index (suggestion.item.metadata.id)}
        <div
          role="button"
          tabindex={0}
          class="suggestion"
          class:selected={index === selectedIndex}
          data-index={index}
          on:mousedown={() => selectSuggestion(index)}
          on:mouseenter={() => {
            selectedIndex = index;
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

  .suggestion.selected {
    background-color: #1a1a1a;
    outline: 2px solid #4a9eff;
    outline-offset: -2px;
  }

  .suggestion:hover {
    background-color: #1a1a1a;
  }
</style>
