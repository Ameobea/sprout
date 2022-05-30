<script lang="ts" context="module">
  import { DEFAULT_PROFILE_SOURCE, ProfileSource } from 'src/components/recommendation/conf';

  export const prerender = true;

  const SITE_TITLE = 'Sprout Anime Recommender, Stats, and Visualizations';
  const SITE_DESCRIPTION =
    'Personalized AI-powered anime recommendations based your MyAnimeList or AniList profile. Find new shows, visualize your watch history, and find your next favorite show!';

  const buildRecommendationsURL = (username: string, profileSource: ProfileSource) =>
    `/user/${username}/recommendations${profileSource !== DEFAULT_PROFILE_SOURCE ? `?source=${profileSource}` : ''}`;
</script>

<script lang="ts">
  import { goto, prefetch } from '$app/navigation';
  import { InlineLoading } from 'carbon-components-svelte';
  import { Search } from 'carbon-icons-svelte';
  import { onMount } from 'svelte';
  import SvelteSeo from 'svelte-seo';

  import SproutLogo from 'src/components/SproutLogo.svelte';

  let selectedProfileSource = DEFAULT_PROFILE_SOURCE;
  let searchValue = '';
  let searchFocused = false;
  let isLoading = false;

  onMount(() => prefetch('/user/_/recommendations'));

  const handleRecommendationsButtonClick = async () => {
    if (!searchValue || isLoading) {
      return;
    }

    isLoading = true;
    try {
      await goto(buildRecommendationsURL(searchValue, selectedProfileSource));
    } finally {
      isLoading = false;
    }
  };

  const handleSearchKeyDown = (event: KeyboardEvent) => {
    if (event.key === 'Enter' && searchValue) {
      handleRecommendationsButtonClick();
    }
  };

  const prefetchRecommendations = () => prefetch(buildRecommendationsURL(searchValue, selectedProfileSource));
</script>

<SvelteSeo
  title={SITE_TITLE}
  description={SITE_DESCRIPTION}
  openGraph={{
    title: SITE_TITLE,
    description: SITE_DESCRIPTION,
    images: [
      {
        url: 'https://anime.ameo.dev/sprout-big.png',
        alt: 'Sprout logo, a small green leafy pixel art plant sprouting out of a pot',
        height: 128,
        width: 128,
      },
    ],
  }}
  twitter={{
    card: 'summary',
    title: SITE_TITLE,
    image: 'https://anime.ameo.dev/sprout-big.png',
    imageAlt: 'Sprout logo, a small green leafy pixel art plant sprouting out of a pot',
    description: SITE_DESCRIPTION,
  }}
/>

<div class="root">
  <div class="top">
    <h1 style="display: inline; justify-content: center; align-items: center">
      <SproutLogo />
      Sprout Anime Recommender
    </h1>

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
        placeholder={searchFocused
          ? undefined
          : `Enter ${selectedProfileSource === ProfileSource.MyAnimeList ? 'MyAnimeList' : 'AniList'} Username`}
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

    <div class="profile-source-switcher">
      <div
        aria-selected={selectedProfileSource === ProfileSource.MyAnimeList}
        role="option"
        class="profile-switcher-option"
        on:click={() => {
          selectedProfileSource = ProfileSource.MyAnimeList;
        }}
        on:keydown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            selectedProfileSource = ProfileSource.MyAnimeList;
            event.preventDefault();
          }
        }}
        tabindex="0"
      >
        MyAnimeList
      </div>
      <div
        aria-selected={selectedProfileSource === ProfileSource.AniList}
        role="option"
        class="profile-switcher-option"
        on:click={() => {
          selectedProfileSource = ProfileSource.AniList;
        }}
        on:keydown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            selectedProfileSource = ProfileSource.AniList;
            event.preventDefault();
          }
        }}
        tabindex="0"
      >
        AniList
      </div>
    </div>

    <i style="color: #b6b6b6; margin-top: 24px; font-size: 12.5px;">Don't have an account to load from?</i>
    <a style="margin-top: 5px; font-size: 16px" sveltekit:prefetch href="/interactive-recommender">
      Try the Interactive Recommender
    </a>
  </div>

  <div class="bottom">
    <div class="about">
      <div>
        <h2>About Sprout</h2>
        <p>
          Sprout is an interactive anime recommendation engine. It uses a machine learning model to generate tailored
          recommendations based on your watch history and ratings from MyAnimeList or AniList. It also provides a
          variety of filters and options for narrowing the results.
        </p>
        <p>
          In addition to recommendation, Sprout also provides a growing variety of stats and visualizations of both your
          personal anime watch history and preferences as well as the entire world of anime via the <a
            sveltekit:prefetch
            href="/pymde_4d_40n"
          >
            Atlas Visualization
          </a>.
        </p>
        <p>
          <i>This site is still in active development</i>. This is a beta version, but all currently existing features
          are expected to work.
        </p>
        <p>
          If you encounter any issues or have any other feedback, please feel free to <a
            href="mailto:casey@cprimozic.net"
          >
            Email Me
          </a>
          or DM me on Twitter <a href="https://twitter.com/ameobea10">@ameobea10</a>. I'm particularly interested in
          feedback about the recommendations - please do let me know what you think. I'm interested in making Sprout as
          good + useful as it can be!
        </p>
      </div>
      <div>
        <h2>About the Recommendations</h2>
        <p>
          Sprout loads your anime list from your MyAnimeList or AniList profile and uses it to generate a list of
          recommendations of animes that you are likely to also like. It takes your ratings into account as well — low
          ratings are treated as negatives for the model.
        </p>
        <p>
          In order to do this, it uses a neural network that is trained using data from other MyAnimeList users. Unlike
          some other recommendation strategies that generate recommendations on a rating-by-rating basis, this model has
          the advantage of being able to consider your entire anime list at once. This allows it to make use of complex
          relationships between anime + ratings to produce higher-quality recommendations.
        </p>
        <p>
          If you don't have a MyAnimeList or AniList account, you can use the <a
            sveltekit:prefetch
            href="/interactive-recommender"
          >
            interactive recommender
          </a>
          or check out some <a sveltekit:prefetch href="/user/ameo___/recommendations">sample recommendations</a>.
        </p>
      </div>
      <div>
        <h2>About the Atlas Visualization</h2>
        <p>
          The <a sveltekit:prefetch href="/pymde_4d_40n">Atlas Visualization</a> is an interactive map of the world of anime.
          It is built by using data about relationships between different anime derived from both user ratings as well as
          anime metadata to place more closely related anime near each other.
        </p>
        <p>
          If you load in your anime list to the visualization or view it from your user profile page, it will highlight
          anime that you have watched with a blue glow. This is useful as context when exploring the atlas - I highly
          suggest doing so!
        </p>
        <p>
          The atlas visualization was built by constructing a high-dimensional <b>Graph Embedding</b> which encodes the
          relationship data between all the animes and then projecting that down into 2D using <b>t-SNE</b>.
        </p>
      </div>
    </div>

    <footer>Created by <a style="margin-left: 4px;" href="https://cprimozic.net">Casey Primozic / ameo</a></footer>
  </div>
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
    margin-top: 30px;
    max-width: calc(min(800px, 100vw - 40px));
    width: 100%;
  }

  .main-search {
    background-color: #000;
    box-sizing: border-box;
    width: 100%;
    height: 50px;
    border: 1px solid #323232;
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

    h1 {
      font-size: 30px;
    }

    .top {
      margin-top: -40px;
    }
  }

  .main-search::-webkit-input-placeholder,
  .main-search::placeholder {
    color: #aaa;
  }

  .search-container button {
    height: 50px;
    width: 58px;
    border-radius: 0px 6px 6px 0px;
    border: 1px solid #323232;
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
    background-color: #488d19;
    transition: background-color 0.15s ease-in-out;
  }

  .search-container button:not(:disabled):active {
    background-color: #23420e !important;
  }

  .search-container button:not(:disabled):hover {
    background-color: #346413;
    transition: background-color 0.1s ease-in-out;
  }

  .profile-source-switcher {
    width: 300px;
    display: flex;
    flex-direction: row;
    margin-top: 10px;
    height: 35px;
  }

  .profile-source-switcher .profile-switcher-option {
    display: flex;
    flex: 1;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    border-top: 1px solid #323232;
    border-bottom: 1px solid #323232;
    border-left: 1px solid #323232;
    user-select: none;
  }

  .profile-source-switcher .profile-switcher-option:first-of-type {
    border-radius: 5px 0px 0px 5px;
  }

  .profile-source-switcher .profile-switcher-option:last-of-type {
    border-radius: 0px 5px 5px 0px;
    border-right: 1px solid #323232;
  }

  .profile-source-switcher .profile-switcher-option[aria-selected='false']:hover {
    background-color: #8de53d20;
  }

  .profile-source-switcher .profile-switcher-option[aria-selected='true'] {
    background-color: #8de53d59;
  }

  p {
    text-align: left;
  }

  p:not(:first-of-type) {
    margin-top: 7px;
  }

  .bottom {
    display: flex;
    flex: 1;
    flex-direction: column;
    text-align: center;
    align-items: center;
    justify-content: center;
    margin-left: 20px;
    margin-right: 20px;
  }

  .about {
    display: flex;
    flex-direction: row;
    flex: 1;
    gap: 40px;
    justify-content: space-around;
    width: 100%;
    flex-wrap: wrap;
  }

  .about > div {
    padding-left: 8px;
    padding-right: 8px;
    display: flex;
    flex: 1;
    flex-direction: column;
    min-width: calc(min(100vw - 10px, 400px));
    max-width: 750px;
  }

  .about > div h2 {
    padding-bottom: 8px;
    margin-top: 0;
    margin-bottom: 20px;
    border-bottom: 1px solid #181818;
  }

  footer {
    margin-top: 40px;
    display: flex;
    flex-direction: row;
    justify-content: center;
    width: 100vw;
    max-width: 100%;
    padding-bottom: 6px;
  }
</style>
