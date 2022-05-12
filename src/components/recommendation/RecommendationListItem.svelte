<script lang="ts">
  import ChevronDown from 'carbon-icons-svelte/lib/ChevronDown.svelte';
  import ChevronUp from 'carbon-icons-svelte/lib/ChevronUp.svelte';

  import type { AnimeDetails } from 'src/malAPI';
  import TopInfluence from './TopInfluenceCard.svelte';

  export let animeMetadata: AnimeDetails;
  export let expanded: boolean;
  export let toggleExpanded: () => void;
  export let excludeRanking: (animeID: number) => void;
  export let topRatingContributors: AnimeDetails[];
</script>

<div class="recommendation" data-expanded={expanded.toString()}>
  <img
    on:click={expanded ? undefined : toggleExpanded}
    src={animeMetadata.main_picture.medium}
    alt={animeMetadata.title}
  />
  <div on:click={toggleExpanded} class="title">
    {#if expanded}
      <a target="_blank" href={`https://myanimelist.net/anime/${animeMetadata.id}`}>{animeMetadata.title}</a>
    {:else}
      {animeMetadata.title}
    {/if}
  </div>
  <div class="expander" on:click={toggleExpanded}>
    {#if expanded}
      <ChevronUp size={24} aria-label="Expand anime details" />
    {:else}
      <ChevronDown size={24} aria-label="Collapse anime details" />
    {/if}
  </div>
  <div on:click={expanded ? undefined : toggleExpanded} class="synopsis">{animeMetadata.synopsis}</div>
  {#if expanded && topRatingContributors.length > 0}
    <div class="details">
      {#each topRatingContributors as contributor (contributor.id)}
        <TopInfluence onDelete={() => excludeRanking(contributor.id)} name={contributor.title} />
      {/each}
    </div>
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
    grid-template-areas: 'thumbnail title synopsis expander';
  }

  .recommendation[data-expanded='false'] {
    grid-template-columns: 100px 140px 1fr 60px;
  }

  .recommendation[data-expanded='true'] {
    max-height: 800px;
    transition: max-height 0.3s ease-in-out;
    grid-template-areas:
      'thumbnail title expander'
      'thumbnail synopsis synopsis'
      'details details details';
    grid-template-columns: 225px 1fr 60px;
    grid-template-rows: 40px auto 250px;
  }

  @media (max-width: 768px) {
    .recommendation[data-expanded='false'] {
      grid-template-columns: 90px 100px 1fr 45px;
    }

    .recommendation[data-expanded='true'] {
      grid-template-areas:
        'title title expander'
        'thumbnail synopsis expander'
        'details details details';
      grid-template-columns: 150px 1fr 45px;
      grid-template-rows: 30px auto 250px;
    }
  }

  .recommendation .title {
    cursor: pointer;
    grid-area: title;
    height: 100%;
    display: flex;
    align-items: center;
    padding: 2px 6px;
  }

  .recommendation[data-expanded='false'] .title {
    line-height: 1.15rem;
    font-size: 15px;
    font-weight: 500;
  }

  .recommendation[data-expanded='true'] .title {
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
    max-height: 120px;
    min-height: 120px;
    min-width: 87px;
    /* preserve aspect ratio */
    object-fit: cover;
    grid-area: thumbnail;
  }

  .recommendation[data-expanded='false'] img {
    cursor: pointer;
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
</style>