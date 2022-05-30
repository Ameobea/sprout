<script context="module" lang="ts">
  const TITLE = 'Sprout Interactive Anime Recommender';
  const DESCRIPTION = 'Enter anime that you like and get recommendations instantly via a neural network';

  interface EmbeddingMetadatum {
    metadata: { id: number; title: string; title_english: string; imageSrc: string };
  }

  interface ProfileEntry {
    animeID: number;
    score: number;
    title: string;
    titleEnglish: string;
  }

  const tryParseRating = (embeddingMetadata: EmbeddingMetadatum[], rawRating: string): ProfileEntry | null => {
    const [animeID, score] = rawRating.split('-');
    if (Number.isNaN(+animeID) || Number.isNaN(+score)) {
      return null;
    }

    const metadatum = embeddingMetadata.find(({ metadata }) => metadata.id === +animeID);
    if (!metadatum) {
      return null;
    }

    return {
      animeID: +animeID,
      score: +score,
      title: metadatum.metadata.title,
      titleEnglish: metadatum.metadata.title_english,
    };
  };

  const updateQueryParams = (profile: ProfileEntry[]) => {
    if (!browser) {
      return;
    }

    const url = new URL(window.location.toString());
    url.search = '';
    const oldSearchParams = new URLSearchParams(window.location.search).toString();

    for (const entry of profile) {
      url.searchParams.append('r', `${entry.animeID}-${entry.score}`);
    }

    const newSearchParams = url.searchParams.toString();
    if (newSearchParams !== oldSearchParams) {
      history.replaceState({}, '', url);
    }
  };

  const getDefaultProfile = (embeddingMetadata: EmbeddingMetadatum[], url: URL): ProfileEntry[] => {
    const ratings = url.searchParams.getAll('r');
    return filterNils(ratings.map((rating) => tryParseRating(embeddingMetadata, rating)));
  };
</script>

<script lang="ts">
  import SvelteSeo from 'svelte-seo';
  import { Tag } from 'carbon-components-svelte';
  import { QueryClient, QueryClientProvider } from '@sveltestack/svelte-query';
  import { page } from '$app/stores';

  import Header from 'src/components/recommendation/Header.svelte';
  import Search from 'src/components/Search.svelte';
  import ScoreSelector, { Score } from 'src/components/interactiveRecommender/ScoreSelector.svelte';
  import InteractiveRecommender from 'src/components/interactiveRecommender/InteractiveRecommender.svelte';
  import { filterNils } from 'src/components/recommendation/utils';
  import { browser } from '$app/env';

  export let embeddingMetadata: EmbeddingMetadatum[];

  let selectedAnime: { id: number; score: number; title: string; titleEnglish: string; imageSrc: string } | null = null;
  let profile: ProfileEntry[] = getDefaultProfile(embeddingMetadata, $page.url);

  let rootContainerNode: HTMLElement | null = null;

  $: updateQueryParams(profile);

  $: selectedAnimeMetadata = selectedAnime
    ? embeddingMetadata.find((anime) => anime.metadata.id === selectedAnime!.id)!
    : null;

  const handleSearchSubmit = (id: number, title: string, titleEnglish: string) => {
    const entry = embeddingMetadata.find((anime) => anime.metadata.id === id)!;
    selectedAnime = { id, title, titleEnglish, score: Score.Good, imageSrc: entry.metadata.imageSrc };
  };

  const addSelectedAnimeToProfile = () => {
    if (!selectedAnime) {
      return;
    }

    profile = profile.filter((anime) => anime.animeID !== selectedAnime!.id);
    profile.push({
      animeID: selectedAnime.id,
      score: selectedAnime.score,
      title: selectedAnime.title,
      titleEnglish: selectedAnime.titleEnglish,
    });
    selectedAnime = null;
  };

  const queryClient = new QueryClient();

  const addRanking = (animeID: number) => {
    const metadata = embeddingMetadata.find((anime) => anime.metadata.id === animeID);
    if (!metadata) {
      return;
    }

    selectedAnime = {
      id: metadata.metadata.id,
      score: Score.Good,
      title: metadata.metadata.title,
      titleEnglish: metadata.metadata.title_english,
      imageSrc: metadata.metadata.imageSrc,
    };

    const searchDOMNode = document.querySelector('body input[type="text"]');
    const searchInView = searchDOMNode && searchDOMNode?.getBoundingClientRect().bottom > 0;
    if (!searchInView) {
      rootContainerNode?.scrollIntoView({ behavior: 'smooth' });
    }
  };
</script>

<SvelteSeo
  title={TITLE}
  description={DESCRIPTION}
  openGraph={{
    title: TITLE,
    description: DESCRIPTION,
  }}
  twitter={{
    card: 'summary',
    title: TITLE,
    description: DESCRIPTION,
  }}
/>

<Header content="Sprout Anime Recommender" />
<div class="root" bind:this={rootContainerNode}>
  <Search
    style={`position: relative;
    width: calc(min(100%, 550px));
    margin-left: auto;
    margin-right: auto;
    margin-top: 2px;
    margin-bottom: ${profile.length === 0 && !selectedAnime ? 4 : 0}px;`}
    inputStyle="height: 38px; font-size: 22px; text-align: center"
    embedding={embeddingMetadata}
    onSubmit={handleSearchSubmit}
    blurredValue={selectedAnime?.titleEnglish || selectedAnime?.title || undefined}
    suggestionsStyle="z-index: 4;"
  />

  {#if selectedAnime && selectedAnimeMetadata}
    <div class="selected-anime">
      <div class="selected-anime-content">
        <img alt={selectedAnimeMetadata.metadata.title_english} src={selectedAnimeMetadata.metadata.imageSrc} />
        <div class="score-selector-wrapper">
          <ScoreSelector bind:score={selectedAnime.score} />
        </div>
      </div>
      <button class="add-anime-button" on:click={addSelectedAnimeToProfile}>Submit</button>
    </div>
  {/if}

  {#if profile.length > 0}
    <div class="profile">
      {#each profile as entry (entry.animeID)}
        <Tag
          type="green"
          size="default"
          filter
          on:close={() => {
            profile = profile.filter((anime) => anime.animeID !== entry.animeID);
          }}
        >
          {entry.titleEnglish || entry.title}
        </Tag>
      {/each}
    </div>
  {/if}
  <QueryClientProvider client={queryClient}>
    <InteractiveRecommender {profile} {addRanking} />
  </QueryClientProvider>
</div>

<style lang="css">
  .root {
    display: flex;
    flex-direction: column;
    max-width: 750px;
    margin: 0 auto;
    background: #131313;
    min-height: calc(100vh - 38px);
    padding: 2px 8px;
  }

  .selected-anime {
    background-color: #0a0a0a;
    display: flex;
    flex-direction: column;
    padding: 4px;
    margin-top: 4px;
    height: 276px;
  }

  .selected-anime-content {
    display: flex;
    flex-direction: row;
    gap: 30px;
    align-items: flex-start;
    justify-content: center;
    align-items: center;
  }

  .score-selector-wrapper {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 100%;
  }

  .selected-anime img {
    background-color: #000;
    object-fit: cover;
    grid-area: thumbnail;
    max-height: 800px;
    max-width: 225px;
    width: 150px;
    height: 214px;
  }

  .selected-anime .add-anime-button {
    margin-top: 10px;
    background-color: #000;
    color: #fff;
    border: none;
    font-size: 20px;
    padding: 3px;
    border-radius: 4px;
    cursor: pointer;
    width: calc(min(400px, 100% - 40px));
    border: 1px solid #333;
    margin-left: auto;
    margin-right: auto;
  }

  .selected-anime .add-anime-button:hover {
    background-color: #080808;
  }

  .profile {
    background-color: #080808;
    margin-top: 4px;
    margin-bottom: 4px;
    padding: 4px 2px;
    border: 1px solid #333;
    max-height: 108px;
    overflow-x: hidden;
    overflow-y: auto;
  }
</style>
