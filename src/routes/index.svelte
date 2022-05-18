<script lang="ts" context="module">
  export const prerender = true;
</script>

<script lang="ts">
  import { goto, prefetch } from '$app/navigation';
  import { onMount } from 'svelte';

  let searchValue = '';
  let isLoading = false;

  onMount(() => prefetch('/user/_/recommendations'));

  const buildRecommendationsURL = (username: string) => `/user/${searchValue}/recommendations`;

  const handleRecommendationsButtonClick = () => {
    if (!searchValue) {
      return;
    }

    isLoading = true;
    goto(buildRecommendationsURL(searchValue)).then(() => {
      isLoading = false;
    });
  };

  const handleGalaxyVizButtonClick = () => {
    if (!searchValue) {
      return;
    }

    isLoading = true;
    goto('/pymde_4d_40n?username=' + searchValue).then(() => {
      isLoading = false;
    });
  };

  const handleSearchKeyDown = (event: KeyboardEvent) => {
    if (event.key === 'Enter' && searchValue) {
      handleRecommendationsButtonClick();
    }
  };

  const prefetchRecommendations = () => prefetch(buildRecommendationsURL(searchValue));
</script>

<div class="root">
  <h1>Ameo's To-Be-Named Anime Site</h1>
  {#if isLoading}
    LOADING TODO
  {/if}

  <input
    type="text"
    class="main-search"
    bind:value={searchValue}
    placeholder="Enter MyAnimeList Username"
    on:keydown={handleSearchKeyDown}
  />
  <div class="buttons-container">
    <button on:mouseenter={prefetchRecommendations} on:click={handleRecommendationsButtonClick} disabled={!searchValue}
      >Recommendations</button
    >
    <button on:click={handleGalaxyVizButtonClick} disabled={!searchValue}>Galaxy Visualization</button>
  </div>

  <footer>Created by <a href="https://cprimozic.net">Casey Primozic / ameo</a></footer>
</div>

<style lang="css">
  .root {
    display: flex;
    flex-direction: column;
    text-align: center;
    height: 100vh;
    align-items: center;
    justify-content: center;
  }

  .main-search {
    box-sizing: border-box;
    width: 100%;
    height: 40px;
    margin-left: 20px;
    margin-right: 20px;
    max-width: calc(min(800px, 100vw - 40px));
    border-radius: 5px;
    margin-left: auto;
    margin-right: auto;
    text-align: center;
    font-size: 22px;
    margin-top: 40px;
  }

  .buttons-container {
    margin-top: 30px;
    display: flex;
    flex-direction: row;
    justify-content: center;
    gap: 20px;
    margin-bottom: 25vh;
  }

  .buttons-container button {
    height: 40px;
    border-radius: 5px;
    border: 1px solid #444;
    background-color: #050505;
    color: #ddd;
    font-size: 24px;
    cursor: pointer;
  }

  .buttons-container button:disabled {
    opacity: 0.5;
    cursor: default;
  }

  .buttons-container button:hover:not(:disabled) {
    background-color: #111;
  }

  footer {
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100vw;
    padding-bottom: 4px;
  }
</style>
