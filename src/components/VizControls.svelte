<script context="module" lang="ts">
  import { EmbeddingName } from '../types';

  const getCurrentEmbeddingName = (): EmbeddingName => {
    switch (window.location.pathname) {
      case '/':
        return EmbeddingName.PyMDE;
      case '/ggvec':
        return EmbeddingName.GGVec;
      default:
        console.error('Unknown embedding name:', window.location.pathname);
        return EmbeddingName.PyMDE;
    }
  };

  const AllColorBys = [
    { label: 'Release Year', value: ColorBy.AiredFromYear },
    { label: 'Average Rating', value: ColorBy.AverageRating },
  ];

  const AllEmbeddingNames = [
    { label: 'PyMDE + t-SNE', value: EmbeddingName.PyMDE },
    { label: 'GGVec + t-SNE', value: EmbeddingName.GGVec },
  ];
</script>

<script lang="ts">
  import { goto } from '$app/navigation';

  import { ColorBy } from './AtlasViz';

  export let colorBy: ColorBy;
  export let setColorBy: (newColorBy: ColorBy) => void;
  export let loadMALProfile: (username: string) => void;

  const handleColorByChange = (newColorBy: ColorBy) => {
    setColorBy(newColorBy);

    // Set query param
    const queryParams = new URLSearchParams(window.location.search);
    queryParams.set('colorBy', newColorBy);
    window.history.pushState({}, '', `?${queryParams.toString()}`);
  };

  let embeddingName = getCurrentEmbeddingName();
  const handleEmbeddingNameChange = (newEmbeddingName: EmbeddingName) => {
    embeddingName = newEmbeddingName;

    const embeddingPathname = (
      {
        [EmbeddingName.PyMDE]: '/',
        [EmbeddingName.GGVec]: '/ggvec',
      } as const
    )[embeddingName];
    goto(`${embeddingPathname}${window.location.search}`);
  };

  const handleLoadMALProfileButtonClick = () => {
    loadMALProfile(malUsername);

    // Set search params
    const queryParams = new URLSearchParams(window.location.search);
    queryParams.set('username', malUsername);
    window.history.pushState({}, '', `?${queryParams.toString()}`);
  };

  let malUsername = '';
</script>

<div class="root">
  <div class="row">
    <div class="label">Color By</div>
    <div class="tabs">
      {#each AllColorBys as { label, value }}
        <div class="tab" role="button" data-selected={colorBy == value} on:click={() => handleColorByChange(value)}>
          {label}
        </div>
      {/each}
    </div>
  </div>
  <div class="row">
    <div class="label">Embedding</div>
    <div class="tabs">
      {#each AllEmbeddingNames as { label, value }}
        <div
          class="tab"
          role="button"
          data-selected={embeddingName == value}
          on:click={() => handleEmbeddingNameChange(value)}
        >
          {label}
        </div>
      {/each}
    </div>
  </div>
  <div class="row">
    <div class="label">MAL Username</div>
    <div class="input">
      <input
        type="text"
        bind:value={malUsername}
        on:keydown={(evt) => {
          if (evt.key === 'Enter') {
            handleLoadMALProfileButtonClick();
          }
        }}
        placeholder="Enter MyAnimeList Username"
      />
      {#if malUsername.length > 0}
        <button class="load-mal-profile-button" on:click={handleLoadMALProfileButtonClick}>Go</button>
      {/if}
    </div>
  </div>
</div>

<style lang="css">
  .root {
    position: absolute;
    left: 0;
    background-color: #050505;
    box-sizing: border-box;
    font-size: 12.5px;
    width: 320px;
    border: 1px solid #444;
    top: 30px;
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 1px 0;
  }

  @media (max-width: 600px) {
    .root {
      width: 100vw;
    }
  }

  .row {
    display: flex;
    flex-direction: row;
  }

  .row .label {
    display: flex;
    flex: 0;
    flex-basis: 90px;
    align-items: center;
    padding: 0 2px;
  }

  .row .tabs {
    display: flex;
    gap: 1px;
    justify-content: flex-end;
    flex: 1;
  }

  .row .tabs .tab {
    box-sizing: border-box;
    padding: 3px 3px;
    cursor: pointer;
    border-radius: 1px;
    border: 1px solid transparent;
  }

  .row .tabs .tab:hover {
    border: 1px solid #333;
  }

  .row .tabs .tab[data-selected='true'] {
    border: 1px solid #4a4a4a;
    color: #fff;
    cursor: default;
  }

  .row .input {
    display: flex;
    flex: 1;
    justify-content: flex-end;
    align-items: center;
    padding: 1px;
  }

  .row .input input[type='text'] {
    height: 18px;
    padding-right: 20px;
    box-sizing: border-box;
  }

  .row .input button.load-mal-profile-button {
    width: 20px;
    height: 18px;
    position: absolute;
    right: 1px;
    font-size: 10px;
    padding: 0;
    background: #000;
    border-radius: 0;
    border: 1px solid #777;
    color: #ccc;
    cursor: pointer;
  }
</style>
