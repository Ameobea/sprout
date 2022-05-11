<script lang="ts">
  import type { AnimeDetails } from 'src/malAPI';
  import TopInfluence from './TopInfluenceCard.svelte';

  export let animeMetadata: AnimeDetails;
  export let expanded: boolean;
  export let toggleExpanded: () => void;
  export let topRatingContributors: AnimeDetails[];
</script>

<div class="recommendation" data-expanded={expanded.toString()}>
  <img on:click={toggleExpanded} src={animeMetadata.main_picture.medium} alt={animeMetadata.title} />
  <div on:click={toggleExpanded} class="title">{animeMetadata.title}</div>
  <div on:click={toggleExpanded} class="synopsis">{animeMetadata.synopsis}</div>
  {#if expanded && topRatingContributors.length > 0}
    <div class="details">
      {#each topRatingContributors as contributor}
        <TopInfluence
          onDelete={() => {
            // TODO
          }}
          name={contributor.title}
        />
      {/each}
    </div>
  {/if}
</div>

<style lang="css">
  .recommendation {
    display: grid;
    grid-template-columns: 100px 140px 1fr;
    grid-gap: 4px;
    border-bottom: 1px solid #ccc;
    max-height: 120px;
    overflow: hidden;
    align-items: center;
    grid-template-areas: 'thumbnail title synopsis';
  }

  .recommendation[data-expanded='true'] {
    max-height: 800px;
    transition: max-height 0.3s ease-in-out;
    grid-template-areas:
      'thumbnail title title'
      'thumbnail synopsis synopsis'
      'details details details';
    grid-template-columns: 225px 1fr;
    grid-template-rows: 25px auto 100px;
  }

  @media (max-width: 768px) {
    .recommendation[data-expanded='false'] {
      grid-template-columns: 90px 100px 1fr;
    }

    .recommendation[data-expanded='true'] {
      grid-template-areas:
        'title title title'
        'thumbnail synopsis synopsis'
        'details details details';
      grid-template-columns: 150px 1fr;
    }
  }

  .recommendation .title {
    cursor: pointer;
    grid-area: title;
    height: 100%;
    display: flex;
    align-items: center;
  }

  .recommendation[data-expanded='true'] .title {
    font-weight: bold;
    text-align: center;
    font-size: 20px;
    border-bottom: 1px solid #cccccc44;
  }

  .recommendation img {
    cursor: pointer;
    max-height: 120px;
    min-height: 120px;
    min-width: 87px;
    /* preserve aspect ratio */
    object-fit: cover;
    grid-area: thumbnail;
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
      font-size: 16px;
      text-align: center;
      width: 100%;
      justify-content: center;
    }
  }

  .recommendation .synopsis {
    cursor: pointer;
    display: -webkit-box;
    -webkit-box-orient: vertical;
    font-size: 15px;
    padding: 2px 0px;
    grid-area: synopsis;
    white-space: pre-line;
  }

  .recommendation[data-expanded='false'] .synopsis {
    overflow: hidden;
    text-overflow: ellipsis;
    -webkit-line-clamp: 6;
    line-clamp: 6;
  }

  .recommendation[data-expanded='true'] .synopsis {
    justify-content: flex-start;
    align-items: flex-start;
    height: 100%;
    padding: 4px 0;
    max-height: 310px;
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
