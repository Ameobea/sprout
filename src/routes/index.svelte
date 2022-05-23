<script lang="ts" context="module">
  export const prerender = true;

  const SITE_TITLE = 'Anime Recommendations, Stats, MAL Profile Analysis';
  const SITE_DESCRIPTION =
    'Personalized AI-powered anime recommendations based your MyAnimeList profile. Find new shows, visualize your watch history, and explore the world of anime!';

  const buildRecommendationsURL = (username: string) => `/user/${username}/recommendations`;
</script>

<script lang="ts">
  import { goto, prefetch } from '$app/navigation';
  import { InlineLoading, Tag } from 'carbon-components-svelte';
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
    <h1>Ameo's To-Be-Named Anime Recommendation Site</h1>

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

  <div class="bottom">
    <div class="about">
      <div>
        <h2>About This Site</h2>
        <p>
          This site's main function is an interactive anime recommendation engine. It uses a machine learning model to
          generate tailored recommendations based on your watch history and ratings. It also provides a variety of
          filters and options for narrowing the results.
        </p>
        <p>
          In addition to recommendation, this site also provides a growing variety of stats and visualizations of both
          your personal anime watch history and preferences as well as the entire world of anime via the <a
            sveltekit:prefetch
            href="/pymde_4d_40n"
          >
            Atlas Visualization
          </a>.
        </p>
        <p>
          <i>This site is still in active development</i>. This is an alpha version, but all currently existing features
          are expected to work.
        </p>
        <p>
          If you encounter any issues or have any other feedback, please feel free to <a
            href="mailto:casey@cprimozic.net"
          >
            Email Me
          </a>
          or DM me on Twitter <a href="https://twitter.com/ameobea10">@ameobea10</a>. I'm particularly interested in
          feedback about the recommendations - please do let me know what you think. I'm interested in making this site
          as good + useful as it can be!
        </p>
      </div>
      <div>
        <h2>About the Recommendations</h2>
        <p>
          This site loads your anime list from your MyAnimeList profile <span style="color: #bbb; font-size: 13px;">
            (Anilist support coming soon!)
          </span> and uses it to generate a list of recommendations of animes that you are likely to also like. It takes
          your ratings into account as well — low ratings are treated as negatives for the model.
        </p>
        <p>
          In order to do this, it uses a neural network that is trained using data from other MyAnimeList users. Unlike
          some other recommendation strategies that generate recommendations on a rating-by-rating basis, this model has
          the advantage of being able to consider your entire anime list at once. This allows it to make use of complex
          relationships between anime + ratings to produce higher-quality recommendations.
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
    padding-left: 6px;
    padding-right: 6px;
    display: flex;
    flex: 1;
    flex-direction: column;
    min-width: 400px;
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
