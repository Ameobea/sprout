<script lang="ts">
  import { slide } from 'svelte/transition';
  import ChevronDown from 'carbon-icons-svelte/lib/ChevronDown.svelte';
  import ChevronUp from 'carbon-icons-svelte/lib/ChevronUp.svelte';
  import { Tag, Loading } from 'carbon-components-svelte';

  import type { AnimeDetails } from 'src/malAPI';

  export let animeMetadata: AnimeDetails;
  export let expanded: boolean;
  export let toggleExpanded: () => void;
  export let excludeRanking: ((animeID: number) => void) | undefined;
  export let excludeGenre: ((genreID: number, genreName: string) => void) | undefined;
  export let addRanking: ((animeID: number) => void) | undefined;
  export let topRatingContributors: { datum: AnimeDetails; positiveRating: boolean }[] | undefined;
  export let planToWatch: boolean;
  export let contributorsLoading: boolean;

  let synopsisElem: HTMLDivElement | null = null;
  $: {
    if (synopsisElem && !expanded) {
      synopsisElem.scrollTop = 0;
    }
  }
</script>

<!-- svelte-ignore a11y-click-events-have-key-events -->
<!-- svelte-ignore a11y-no-static-element-interactions -->
<div class="recommendation" data-plan-to-watch={planToWatch.toString()} data-expanded={expanded.toString()} in:slide>
  <!-- svelte-ignore a11y-click-events-have-key-events -->
  <!-- svelte-ignore a11y-no-noninteractive-element-interactions -->
  <img
    on:click={expanded ? undefined : toggleExpanded}
    src={animeMetadata.main_picture.medium}
    alt={animeMetadata.alternative_titles.en || animeMetadata.title}
    loading="lazy"
  />
  <!-- svelte-ignore a11y-click-events-have-key-events -->
  <!-- svelte-ignore a11y-no-static-element-interactions -->
  <div on:click={toggleExpanded} class="title">
    <div class="title-text">
      {#if expanded}
        <a target="_blank" href={`https://myanimelist.net/anime/${animeMetadata.id}`}>
          {animeMetadata.alternative_titles.en || animeMetadata.title}
        </a>
      {:else}
        {animeMetadata.alternative_titles.en || animeMetadata.title}
      {/if}
    </div>
    {#if planToWatch && !expanded}
      <div class="tag"><Tag style="color: white" type="green">Plan To Watch</Tag></div>
    {:else if addRanking && !expanded}
      <div class="tag">
        <Tag
          style="color: white"
          type="outline"
          interactive
          on:click={(evt) => {
            evt.stopPropagation();
            addRanking?.(animeMetadata.id);
          }}
        >
          Already Watched
        </Tag>
      </div>
    {/if}
  </div>
  <!-- svelte-ignore a11y-click-events-have-key-events -->
  <!-- svelte-ignore a11y-no-static-element-interactions -->
  <div class="expander" on:click={toggleExpanded}>
    {#if expanded}
      <ChevronUp size={24} aria-label="Expand anime details" />
    {:else}
      <ChevronDown size={24} aria-label="Collapse anime details" />
    {/if}
  </div>
  {#if !expanded}
    <div class="genres">
      {#each animeMetadata.genres ?? [] as genre, i (genre)}
        <Tag
          style={i === 0 ? 'margin-left: 0' : undefined}
          size="sm"
          type="cool-gray"
          filter={!!excludeGenre}
          on:close={() => excludeGenre?.(genre.id, genre.name)}
        >
          {genre.name}
        </Tag>
      {/each}
    </div>
  {/if}
  <!-- svelte-ignore a11y-no-static-element-interactions -->
  <div on:click={expanded ? undefined : toggleExpanded} class="synopsis" bind:this={synopsisElem}>
    {animeMetadata.synopsis}
  </div>
  {#if expanded}
    <div class="details">
      <div class="top-influences">
        <h3>Recommended Because:</h3>
        {#if topRatingContributors && topRatingContributors.length > 0}
          {#each topRatingContributors as { datum, positiveRating } (datum.id)}
            <Tag
              style="color: white;"
              filter={!contributorsLoading && !!excludeRanking}
              skeleton={contributorsLoading}
              on:close={() => excludeRanking?.(datum.id)}
              type={positiveRating ? 'green' : 'red'}
            >
              You {positiveRating ? 'liked' : 'disliked'}:
              {datum.alternative_titles.en || datum.title}
            </Tag>
          {/each}
        {:else}
          <Tag skeleton /><Tag skeleton /><Tag skeleton />
        {/if}
      </div>
    </div>
  {:else}
    <Loading withOverlay={false} />
  {/if}
</div>

<style lang="css">
  .recommendation {
    display: grid;
    grid-gap: 0;
    border-bottom: 1px solid #ccc;
    max-height: 120px;

    overflow: hidden;
    align-items: center;
    grid-template-areas: 'thumbnail title genres synopsis expander';
  }

  .recommendation[data-plan-to-watch='true'] {
    background-color: #55d95f19;
  }

  .recommendation[data-expanded='false'] {
    height: 120px;
    grid-template-columns: 87px 140px 190px 1fr 60px;
  }

  .recommendation[data-expanded='true'] {
    max-height: 800px;
    transition: max-height 0.3s ease-in-out;
    grid-template-areas:
      'thumbnail title expander'
      'thumbnail synopsis synopsis'
      'details details details';
    grid-template-columns: 225px 1fr 60px;
    grid-template-rows: 28px auto auto;
  }

  @media (max-width: 768px) {
    .recommendation[data-expanded='false'] {
      grid-template-areas: 'thumbnail title genres expander';
      grid-template-columns: 90px 100px 1fr 45px;
    }

    .recommendation[data-expanded='false'] .synopsis {
      display: none;
    }

    .recommendation[data-expanded='true'] {
      grid-template-areas:
        'title title expander'
        'thumbnail synopsis expander'
        'details details details';
      grid-template-columns: 150px 1fr 45px;
      grid-template-rows: 30px auto auto;
    }
  }

  .recommendation .title {
    display: flex;
    cursor: pointer;
    grid-area: title;
    height: 100%;
    justify-content: center;
    text-align: center;
    padding: 0 5px;
  }

  .recommendation[data-expanded='false'] .title {
    flex-direction: column;
    line-height: 1.15rem;
    font-size: 15px;
    font-weight: 500;
    border-right: 1px solid #cccccc55;
    max-height: 120px;
  }

  .recommendation[data-expanded='false'] .title .title-text {
    display: flex;
    flex: 1;
    justify-content: center;
    align-items: center;
  }

  .recommendation[data-expanded='false'] .title .tag {
    display: flex;
    justify-content: center;
    flex: 0;
    padding: 4px 4px;
  }

  .recommendation[data-expanded='true'] .title {
    padding: 2px 6px;
    font-weight: bold;
    text-align: center;
    font-size: 20px;
    border-bottom: 1px solid #cccccc44;
  }

  .recommendation .expander {
    grid-area: expander;
    display: flex;
    justify-content: center;
    align-items: flex-end;
    height: 100%;
    padding: 5px 0;
    cursor: pointer;
    background-color: #202428;
  }

  .recommendation .expander:hover {
    background-color: #24282b;
  }

  .recommendation[data-expanded='true'] .expander {
    border-bottom: 1px solid #cccccc44;
    align-items: center;
  }

  .recommendation img {
    /* preserve aspect ratio */
    object-fit: cover;
    grid-area: thumbnail;
  }

  .recommendation[data-expanded='false'] img {
    min-height: 120px;
    max-height: 120px;
    min-width: 87px;
    cursor: pointer;
    height: 120px;
    width: 87px;
  }

  .recommendation[data-expanded='true'] img {
    max-height: 800px;
    max-width: 225px;
    transition: max-height 0.3s ease-in-out;
  }

  @media (max-width: 768px) {
    .recommendation[data-expanded='true'] img {
      max-width: 150px;
    }

    .recommendation[data-expanded='true'] .title {
      font-size: 18px;
      text-align: center;
      width: 100%;
      justify-content: center;
      white-space: pre-wrap;
      padding: 2px 2px !important;
    }

    .recommendation[data-expanded='true'] .synopsis {
      max-height: 212px !important;
    }
  }

  .recommendation .genres {
    overflow-y: hidden;
    height: 120px;
    max-height: 120px;
    padding: 4px 0px 0px 4px;
    justify-content: flex-start;
    align-items: flex-start;
    border-right: 1px solid #cccccc55;
  }

  .recommendation .synopsis {
    display: -webkit-box;
    -webkit-box-orient: vertical;
    font-size: 15px;
    line-height: 1.15rem;
    padding: 2px 0px;
    grid-area: synopsis;
    white-space: pre-line;
  }

  .recommendation[data-expanded='false'] .synopsis {
    cursor: pointer;
    overflow: hidden;
    text-overflow: ellipsis;
    -webkit-line-clamp: 6;
    line-clamp: 6;
    padding: 4px 6px;
  }

  .recommendation[data-expanded='true'] .synopsis {
    justify-content: flex-start;
    align-items: flex-start;
    height: 100%;
    padding: 4px 6px;
    max-height: 280px;
    overflow-y: auto;
  }

  .recommendation .details {
    display: flex;
    flex-direction: row;
    grid-area: details;
    border-top: 1px solid #cccccc44;
    height: 100%;
    gap: 10px;
    padding: 4px;
    box-sizing: border-box;
  }

  .top-influences {
    display: flex;
    flex-direction: row;
    min-height: 30px;
    align-items: center;
    flex-wrap: wrap;
  }

  .top-influences h3 {
    display: inline-flex;
    font-size: 18px;
    font-weight: 500;
    margin-right: 6px;
  }
</style>
