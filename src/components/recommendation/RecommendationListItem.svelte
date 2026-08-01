<script lang="ts">
  import { slide } from 'svelte/transition';
  import ChevronDown from 'carbon-icons-svelte/lib/ChevronDown.svelte';
  import ChevronUp from 'carbon-icons-svelte/lib/ChevronUp.svelte';
  import { Tag } from 'carbon-components-svelte';

  import { getSurface, submitAnalyticsEvent } from 'src/analytics';
  import type { AnimeDetails } from 'src/malAPI';
  import type { RatingTier } from 'src/util/ratingTiers';
  import GenreTagList from './GenreTagList.svelte';

  export let animeMetadata: AnimeDetails;
  export let rank: number;
  export let expanded: boolean;
  export let toggleExpanded: () => void;
  export let excludeRanking: ((animeID: number) => void) | undefined;
  export let excludeGenre: ((genreID: number, genreName: string) => void) | undefined;
  export let addRanking: ((animeID: number) => void) | undefined;
  export let topRatingContributors: { datum: AnimeDetails; positiveRating: boolean }[] | undefined;
  export let planToWatch: boolean;
  export let contributorsLoading: boolean;
  export let predictedRating: number | null = null;
  export let ratingTier: RatingTier | null = null;

  const MEDIA_TYPE_NAMES: { [mediaType: string]: string } = {
    tv: 'TV',
    tv_special: 'TV Special',
    ova: 'OVA',
    ona: 'ONA',
    movie: 'Movie',
    special: 'Special',
    music: 'Music',
    cm: 'Commercial',
    pv: 'PV',
  };

  $: metaLine = [
    animeMetadata.start_date?.slice(0, 4),
    MEDIA_TYPE_NAMES[animeMetadata.media_type],
    animeMetadata.num_episodes ? `${animeMetadata.num_episodes} ${animeMetadata.num_episodes === 1 ? 'ep' : 'eps'}` : null,
  ]
    .filter(Boolean)
    .join(' · ');

  let synopsisElem: HTMLDivElement | null = null;
  $: {
    if (synopsisElem && !expanded) {
      synopsisElem.scrollTop = 0;
    }
  }
</script>

<!-- svelte-ignore a11y-click-events-have-key-events -->
<!-- svelte-ignore a11y-no-static-element-interactions -->
<div
  class="recommendation"
  data-plan-to-watch={planToWatch.toString()}
  data-expanded={expanded.toString()}
  data-show-rating={(predictedRating !== null).toString()}
  in:slide
>
  <!-- svelte-ignore a11y-click-events-have-key-events -->
  <!-- svelte-ignore a11y-no-noninteractive-element-interactions -->
  <img
    on:click={expanded ? undefined : toggleExpanded}
    src={animeMetadata.main_picture.medium}
    alt={animeMetadata.alternative_titles.en || animeMetadata.title}
    loading="lazy"
  />
  {#if !expanded && predictedRating !== null}
    <!-- svelte-ignore a11y-click-events-have-key-events -->
    <!-- svelte-ignore a11y-no-static-element-interactions -->
    <div class="predicted-rating" data-tier={ratingTier ?? 'neutral'} on:click={toggleExpanded}>
      <span class="rating-num">{predictedRating.toFixed(1)}</span>
      <span class="rating-label">predicted</span>
    </div>
  {/if}
  <!-- svelte-ignore a11y-click-events-have-key-events -->
  <!-- svelte-ignore a11y-no-static-element-interactions -->
  <div on:click={toggleExpanded} class="title">
    <div class="title-text">
      {#if expanded}
        <a
          target="_blank"
          href={`https://myanimelist.net/anime/${animeMetadata.id}`}
          on:click={() =>
            submitAnalyticsEvent({
              category: 'recommendations',
              subcategory: 'mal_link_click',
              payload: { anime_id: animeMetadata.id, rank, plan_to_watch: planToWatch, surface: getSurface() },
            })}
        >
          {animeMetadata.alternative_titles.en || animeMetadata.title}
        </a>
      {:else}
        {animeMetadata.alternative_titles.en || animeMetadata.title}
      {/if}
    </div>
    {#if !expanded && metaLine}
      <div class="meta-line">{metaLine}</div>
    {/if}
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
            submitAnalyticsEvent({
              category: 'interactive_recommender',
              subcategory: 'add_ranking_from_list',
              payload: { anime_id: animeMetadata.id, rank },
            });
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
      <GenreTagList genres={animeMetadata.genres ?? []} {excludeGenre} />
    </div>
  {/if}
  {#if expanded}
    <div class="expanded-meta">
      {#if predictedRating !== null}
        <div class="predicted-line" data-tier={ratingTier ?? 'neutral'}>
          <span class="star">★</span>
          <span class="predicted-value">{predictedRating.toFixed(1)}</span>
          <span class="predicted-text">predicted for you</span>
        </div>
      {/if}
      <div class="expanded-meta-row">
        {#if metaLine}
          <span class="meta-line">{metaLine}</span>
        {/if}
        {#each animeMetadata.genres ?? [] as genre (genre.id)}
          <Tag
            size="sm"
            type="cool-gray"
            filter={!!excludeGenre}
            on:close={() => excludeGenre?.(genre.id, genre.name)}
          >
            {genre.name}
          </Tag>
        {/each}
      </div>
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
          {#each topRatingContributors as { datum } (datum.id)}
            <Tag
              style="color: white;"
              filter={!contributorsLoading && !!excludeRanking}
              skeleton={contributorsLoading}
              on:close={() => excludeRanking?.(datum.id)}
              type="outline"
            >
              You watched:
              {datum.alternative_titles.en || datum.title}
            </Tag>
          {/each}
        {:else}
          <Tag skeleton /><Tag skeleton /><Tag skeleton />
        {/if}
      </div>
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
    grid-template-areas: 'thumbnail title genres synopsis expander';
  }

  .recommendation[data-plan-to-watch='true'] {
    background-color: #55d95f19;
  }

  .recommendation[data-expanded='false'] {
    height: 120px;
    grid-template-columns: 87px 140px 190px 1fr 60px;
  }

  .recommendation[data-expanded='false'][data-show-rating='true'] {
    grid-template-areas: 'thumbnail rating title genres synopsis expander';
    grid-template-columns: 87px 64px 140px 190px 1fr 60px;
  }

  .recommendation[data-expanded='true'] {
    max-height: 800px;
    transition: max-height 0.3s ease-in-out;
    grid-template-areas:
      'thumbnail title expander'
      'thumbnail meta meta'
      'thumbnail synopsis synopsis'
      'details details details';
    grid-template-columns: 225px 1fr 60px;
    grid-template-rows: 28px auto auto auto;
  }

  @media (max-width: 768px) {
    .recommendation[data-expanded='false'] {
      grid-template-areas: 'thumbnail title genres expander';
      grid-template-columns: 90px 100px 1fr 45px;
    }

    .recommendation[data-expanded='false'][data-show-rating='true'] {
      grid-template-areas: 'thumbnail rating title genres expander';
      grid-template-columns: 90px 48px 100px 1fr 45px;
    }

    .recommendation[data-expanded='false'] .synopsis {
      display: none;
    }

    .recommendation[data-expanded='true'] {
      grid-template-areas:
        'title title expander'
        'thumbnail meta expander'
        'thumbnail synopsis expander'
        'details details details';
      grid-template-columns: 150px 1fr 45px;
      grid-template-rows: 30px auto auto auto;
    }
  }

  .recommendation .predicted-rating {
    grid-area: rating;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 1px;
    height: 100%;
    border-right: 1px solid #cccccc55;
    cursor: pointer;
  }

  .predicted-rating[data-tier='high'],
  .predicted-line[data-tier='high'] {
    --tier-color: #42be65;
  }

  .predicted-rating[data-tier='low'],
  .predicted-line[data-tier='low'] {
    --tier-color: #ff832b;
  }

  .predicted-rating[data-tier='mid'],
  .predicted-rating[data-tier='neutral'],
  .predicted-line[data-tier='mid'],
  .predicted-line[data-tier='neutral'] {
    --tier-color: #a8b0b8;
  }

  .rating-num {
    font-size: 21px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    color: var(--tier-color);
  }

  .rating-label {
    font-size: 8.5px;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #8d8d8d;
  }

  @media (max-width: 768px) {
    .rating-num {
      font-size: 17px;
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
    min-width: 0;
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
    overflow: hidden;
  }

  .title-text {
    overflow-wrap: anywhere;
  }

  .meta-line {
    font-size: 11px;
    color: #9ba3ab;
    letter-spacing: 0.02em;
    white-space: nowrap;
  }

  .recommendation[data-expanded='false'] .title .meta-line {
    flex: 0;
    padding-bottom: 3px;
  }

  .recommendation[data-expanded='false'] .title .tag {
    display: flex;
    justify-content: center;
    flex: 0;
    padding: 0 4px 4px 4px;
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

  .recommendation .expanded-meta {
    grid-area: meta;
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 8px 8px 4px;
  }

  .predicted-line {
    display: flex;
    align-items: baseline;
    gap: 5px;
  }

  .predicted-line .star {
    font-size: 14px;
    color: var(--tier-color);
  }

  .predicted-line .predicted-value {
    font-size: 17px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    color: var(--tier-color);
  }

  .predicted-line .predicted-text {
    font-size: 13px;
    color: #c9ced3;
  }

  .expanded-meta-row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 2px 4px;
  }

  .expanded-meta-row .meta-line {
    margin-right: 6px;
  }

  .expanded-meta-row :global(.bx--tag) {
    margin: 0;
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
    grid-area: genres;
    height: 120px;
    max-height: 120px;
    border-right: 1px solid #cccccc55;
    min-width: 0;
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
    -webkit-line-clamp: 5;
    line-clamp: 5;
    font-size: 13.5px;
    line-height: 1.32;
    color: #b9c0c7;
    /* no bottom padding: clamped-away lines paint into the padding box and show as a clipped stripe */
    padding: 6px 10px 0;
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
