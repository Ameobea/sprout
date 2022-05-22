<script lang="ts" context="module">
  export const prerender = true;

  const SITE_TITLE = 'Anime Recommendations, Stats, MAL Profile Analysis';
  const SITE_DESCRIPTION =
    'Personalized anime recommendations based your MyAnimeList profile. Find new shows, visualize your preferences, and explore the world of anime!';

  const buildRecommendationsURL = (username: string) => `/user/${username}/recommendations`;
</script>

<script lang="ts">
  import { goto, prefetch } from '$app/navigation';
  import { InlineLoading } from 'carbon-components-svelte';
  import { Search } from 'carbon-icons-svelte';
  import { onMount } from 'svelte';
  import SvelteSeo from 'svelte-seo';

  let searchValue = '';
  let searchFocused = false;
  let isLoading = false;

  onMount(() => prefetch('/user/_/recommendations'));

  const handleRecommendationsButtonClick = () => {
    if (!searchValue || isLoading) {
      return;
    }

    isLoading = true;
    goto(buildRecommendationsURL(searchValue))
      .then(() => {
        isLoading = false;
      })
      .catch(() => {
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

<SvelteSeo
  title={SITE_TITLE}
  description={SITE_DESCRIPTION}
  openGraph={{
    title: SITE_TITLE,
    description: SITE_DESCRIPTION,
    images: [
      // TODO
    ],
  }}
  twitter={{
    card: 'summary',
    title: SITE_TITLE,
    // TODO: image
    description: SITE_DESCRIPTION,
  }}
/>

<div class="root">
  <div class="top">
    <h1>Ameo's To-Be-Named Anime Site</h1>

    <div class="search-container">
      <input
        type="text"
        class="main-search"
        bind:value={searchValue}
        on:focus={() => {
          searchFocused = true;
        }}
        on:blur={() => {
          searchFocused = false;
        }}
        placeholder={searchFocused ? undefined : 'Enter MyAnimeList Username'}
        on:keydown={handleSearchKeyDown}
      />
      <button
        on:mouseenter={prefetchRecommendations}
        on:click={handleRecommendationsButtonClick}
        disabled={!searchValue}
        style={isLoading ? 'background-color: #000' : undefined}
      >
        {#if isLoading}
          <InlineLoading style="transform: scale(1.7) translate(9px, 0px); transform-origin: center center" />
        {:else}
          <Search size={32} style="margin-top: 10px" />
        {/if}
      </button>
    </div>
  </div>

  <footer>Created by <a style="margin-left: 4px;" href="https://cprimozic.net">Casey Primozic / ameo</a></footer>
</div>

<style lang="css">
  .root {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
    justify-content: space-between;
  }

  .top {
    display: flex;
    flex-direction: column;
    text-align: center;
    height: calc(min(100vh, 800px));
    align-items: center;
    justify-content: center;
    margin-left: 20px;
    margin-right: 20px;
  }

  .search-container {
    display: flex;
    flex-direction: row;
    margin-top: 40px;
    max-width: calc(min(800px, 100vw - 40px));
    width: 100%;
    margin-bottom: 21.2vh;
  }

  .main-search {
    background-color: #000;
    box-sizing: border-box;
    width: 100%;
    height: 50px;
    border: 1px solid #141414;
    border-radius: 6px 0px 0px 6px;
    margin-left: auto;
    margin-right: auto;
    text-align: center;
    font-size: 28px;
  }

  @media (max-width: 800px) {
    .main-search {
      font-size: 20px;
    }
  }

  /* placeholder color */
  .main-search::-webkit-input-placeholder,
  .main-search::placeholder {
    color: #aaa;
  }

  .search-container button {
    height: 50px;
    width: 58px;
    border-radius: 0px 6px 6px 0px;
    border: 1px solid #141414;
    border-left: none;
    padding: 0 10px;
    background-color: #050505;
    color: #fff;
    cursor: pointer;
  }

  .search-container button:disabled {
    background-color: #000;
    color: #ffffff66;
    cursor: default;
  }

  .search-container button:not(:disabled) {
    background-color: #0f62fe;
    transition: background-color 0.15s ease-in-out;
  }

  .search-container button:not(:disabled):active {
    background-color: #002d9c !important;
  }

  .search-container button:not(:disabled):hover {
    background-color: #0353e9;
    transition: background-color 0.1s ease-in-out;
  }

  footer {
    display: flex;
    flex-direction: row;
    justify-content: center;
    width: 100vw;
    max-width: 100%;
    padding-bottom: 6px;
  }
</style>
