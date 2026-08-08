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
  export let topContributors:
    | {
        animeId: number;
        datum: AnimeDetails | undefined;
        strength: number;
        presence?: number;
        rating?: number;
        probabilityWithout?: number;
        ratingDelta?: number;
        userRating?: number;
      }[]
    | undefined;
  export let contributionBaseline: number | undefined = undefined;

  // Color ramp by contribution magnitude relative to the profile-wide baseline
  // (log scale, saturating ~50x baseline); falls back to within-rec share.
  const contribColorT = (strength: number, fill: number): number => {
    if (contributionBaseline === undefined || contributionBaseline <= 0) {
      return fill;
    }
    return Math.max(0, Math.min(1, Math.log10(strength / (3 * contributionBaseline)) / 1.2));
  };
  const barColor = (t: number): string => `hsl(${145 - 35 * t}, ${35 + 55 * t}%, ${38 + 14 * t}%)`;

  const formatRel = (x: number): string => {
    const pct = Math.abs(x) * 100;
    return `${pct >= 10 ? pct.toFixed(0) : pct.toFixed(1)}%`;
  };

  const userRatingColor = (rating: number): string => `hsl(${((rating - 1) / 9) * 120}, 72%, 58%)`;
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
        <h3>Recommended because you watched:</h3>
        {#if topContributors && topContributors.length > 0}
          {@const maxStrength = Math.max(...topContributors.map((c) => c.strength))}
          {@const maxFill = 0.25 + 0.75 * contribColorT(maxStrength, 1)}
          <div class="influence-pills">
          {#each topContributors as { animeId, datum, strength, presence, rating, userRating } (animeId)}
            {#if datum}
              {@const fill = (strength / maxStrength) * maxFill}
              {@const color = barColor(contribColorT(strength, fill))}
              <span class="contributor">
                <Tag
                  style="color: white;"
                  filter={!contributorsLoading && !!excludeRanking}
                  skeleton={contributorsLoading}
                  on:close={() => excludeRanking?.(animeId)}
                  type="outline"
                >
                  <span class="tag-label">{datum.alternative_titles.en || datum.title}</span>
                </Tag>
                <span class="strength-track">
                  <span class="strength-fill" style="width: {Math.round(fill * 100)}%; background: {color};" />
                </span>
                {#if presence !== undefined && rating !== undefined}
                  {@const relScore = Math.exp(presence + rating) - 1}
                  {@const relPresence = Math.exp(presence) - 1}
                  {@const relRating = Math.exp(rating) - 1}
                  <div class="contrib-popover">
                    <div class="popover-context">
                      Because you {userRating ? 'rated' : 'watched'}
                      <b>{datum.alternative_titles.en || datum.title}</b>{#if userRating}
                        <b class="user-rating" style="color: {userRatingColor(userRating)};">{userRating}★</b>{/if}:
                    </div>
                    <div>
                      Recommendation score:
                      <span class="rel-delta" class:negative={relScore < 0} class:big={relScore >= 1}>
                        {relScore >= 0 ? '+' : '−'}{formatRel(relScore)}
                      </span>
                    </div>
                    <div class="popover-breakdown">
                      Presence
                      <span class="rel-delta" class:negative={relPresence < 0} class:big={relPresence >= 1}>
                        {relPresence >= 0 ? '+' : '−'}{formatRel(relPresence)}
                      </span>
                    </div>
                    <div class="popover-breakdown">
                      Rating
                      <span class="rel-delta" class:negative={relRating < 0} class:big={relRating >= 1}>
                        {relRating >= 0 ? '+' : '−'}{formatRel(relRating)}
                      </span>
                    </div>
                  </div>
                {/if}
              </span>
            {/if}
          {/each}
          </div>
        {:else}
          <div class="influence-pills"><Tag skeleton /><Tag skeleton /><Tag skeleton /></div>
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
    flex: 1;
    min-width: 0;
    min-height: 30px;
    align-items: center;
  }

  .top-influences h3 {
    flex: 0 0 auto;
    font-size: 13.5px;
    font-weight: 500;
    color: #ddd;
    margin-right: 8px;
  }

  .influence-pills {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 2px;
    flex: 1;
    min-width: 0;
  }

  .contributor {
    position: relative;
    display: flex;
    min-width: 0;
    flex-direction: column;
  }

  .contributor :global(.bx--tag) {
    width: calc(100% - 8px);
    min-width: 0;
    overflow: hidden;
  }

  .tag-label {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    text-align: center;
  }

  .strength-track {
    height: 5px;
    width: calc(100% - 8px);
    margin: 1px 4px 4px;
    background: #ffffff17;
    border-radius: 2.5px;
  }

  .strength-fill {
    display: block;
    height: 100%;
    border-radius: inherit;
  }

  .contrib-popover {
    display: none;
    position: absolute;
    bottom: 100%;
    left: 0;
    margin-bottom: 5px;
    padding: 7px 10px;
    background: #161616;
    border: 1px solid #444;
    border-radius: 4px;
    font-size: 12.5px;
    line-height: 1.55;
    width: max-content;
    max-width: min(340px, 84vw);
    z-index: 20;
    pointer-events: none;
    box-shadow: 0 3px 10px #000000aa;
  }

  .popover-context {
    color: #ddd;
    margin-bottom: 4px;
  }

  .user-rating {
    margin-left: 4px;
  }

  .contributor:hover .contrib-popover {
    display: block;
  }

  /* Anchor per grid column so the card's overflow:hidden can't clip the popover */
  .contributor:nth-child(3n + 2) .contrib-popover {
    left: 50%;
    transform: translateX(-50%);
  }

  .contributor:nth-child(3n) .contrib-popover {
    left: auto;
    right: 0;
  }

  .popover-breakdown {
    color: #ccc;
    font-size: 12px;
  }

  .rel-delta {
    color: #55d95f;
    font-weight: 600;
  }

  .rel-delta.big {
    color: #3aff5b;
    font-weight: 700;
  }

  .rel-delta.negative {
    color: #ff6b6b;
  }

  @media (max-width: 768px) {
    .top-influences {
      flex-direction: column;
      align-items: stretch;
    }

    .top-influences h3 {
      margin: 0 0 4px;
    }

    .influence-pills {
      grid-template-columns: minmax(0, 1fr);
    }

    .contributor .contrib-popover {
      left: 0;
      right: auto;
      transform: none;
    }
  }
</style>
