<script lang="ts">
  import { slide } from 'svelte/transition';
  import ChevronDown from 'carbon-icons-svelte/lib/ChevronDown.svelte';
  import ChevronUp from 'carbon-icons-svelte/lib/ChevronUp.svelte';
  import { Tag } from 'carbon-components-svelte';

  import { getSurface, submitAnalyticsEvent } from 'src/analytics';
  import type { AnimeDetails } from 'src/malAPI';
  import type { RatingTier } from 'src/util/ratingTiers';
  import GenreTagList from './GenreTagList.svelte';
  import GenreIcon from './GenreIcon.svelte';
  import HelpTip from './HelpTip.svelte';
  import Image from 'carbon-icons-svelte/lib/Image.svelte';
  import Launch from 'carbon-icons-svelte/lib/Launch.svelte';

  export let animeMetadata: AnimeDetails;
  export let rank: number;
  export let expanded: boolean;
  export let toggleExpanded: () => void;
  export let excludeRanking: ((animeID: number) => void) | undefined;
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
    animeMetadata.num_episodes
      ? `${animeMetadata.num_episodes} ${animeMetadata.num_episodes === 1 ? 'ep' : 'eps'}`
      : null,
  ]
    .filter(Boolean)
    .join(' · ');
</script>

<div
  class="recommendation"
  data-plan-to-watch={planToWatch.toString()}
  data-expanded={expanded.toString()}
  data-show-rating={(predictedRating !== null).toString()}
  in:slide
>
  <button
    type="button"
    class="poster"
    on:click={toggleExpanded}
    aria-label={`${expanded ? 'Collapse' : 'Expand'} details for ${animeMetadata.alternative_titles.en || animeMetadata.title}`}
  >
    {#if animeMetadata.main_picture?.medium}
      <img
        src={animeMetadata.main_picture.medium}
        alt={animeMetadata.alternative_titles.en || animeMetadata.title}
        loading="lazy"
      />
    {:else}
      <span class="poster-fallback"><Image size={24} aria-hidden="true" /><span>No image</span></span>
    {/if}
  </button>
  <div class="card-content">
    <div class="title-row">
      <div class="title-text">
        <button type="button" class="title-button" aria-expanded={expanded} on:click={toggleExpanded}
          >{animeMetadata.alternative_titles.en || animeMetadata.title}</button
        >
      </div>
      <a
        class="external-link"
        target="_blank"
        rel="noopener noreferrer"
        href={`https://myanimelist.net/anime/${animeMetadata.id}`}
        aria-label={`Open ${animeMetadata.alternative_titles.en || animeMetadata.title} on MyAnimeList (opens in a new tab)`}
        title="Open on MyAnimeList (new tab)"
        on:click={() =>
          submitAnalyticsEvent({
            category: 'recommendations',
            subcategory: 'mal_link_click',
            payload: { anime_id: animeMetadata.id, rank, plan_to_watch: planToWatch, surface: getSurface() },
          })}><Launch size={16} aria-hidden="true" /></a
      >
    </div>
    <div class="metadata-row">
      {#if predictedRating !== null}
        <span class="predicted-line" data-tier={ratingTier ?? 'neutral'}
          ><span class="predicted-value">{predictedRating.toFixed(1)}</span><span class="predicted-text"
            >predicted for you</span
          ></span
        >
      {/if}
      {#if metaLine}<span class="meta-line">{metaLine}</span>{/if}
    </div>
    <div class="card-tags">
      <div class="genres">
        {#if expanded}
          {#each animeMetadata.genres ?? [] as genre (genre.id)}
            <Tag size="sm" type="cool-gray"
              ><span class="genre-label"><GenreIcon name={genre.name} />{genre.name}</span></Tag
            >
          {/each}
        {:else}
          <GenreTagList genres={animeMetadata.genres ?? []} />
        {/if}
      </div>
      {#if planToWatch}
        <Tag style="color: white" type="green" size="sm">Plan To Watch</Tag>
      {:else if addRanking}
        <Tag
          style="color: white"
          type="outline"
          interactive
          size="sm"
          on:click={() => {
            submitAnalyticsEvent({
              category: 'interactive_recommender',
              subcategory: 'add_ranking_from_list',
              payload: { anime_id: animeMetadata.id, rank },
            });
            addRanking?.(animeMetadata.id);
          }}>Already Watched</Tag
        >
      {/if}
    </div>
  </div>
  <div class="synopsis">{animeMetadata.synopsis || 'No synopsis available.'}</div>
  {#if expanded}
    <div class="details">
      <div class="top-influences">
        <div class="influence-heading">
          <h3>Recommended because you watched</h3>
          <HelpTip
            label="Influence bars"
            text={`Longer bars mean a stronger contribution to this recommendation’s model score. Each recommendation uses its own bar scale. Brighter green indicates a stronger signal ${contributionBaseline !== undefined && contributionBaseline > 0 ? 'relative to your profile' : 'among the titles shown'}. These bars are not ratings or probabilities.${excludeRanking ? ' Use × to recalculate without that watched title. Restore it in Excluded Rankings.' : ''}`}
          />
        </div>
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
        {:else if contributorsLoading}
          <div class="influence-pills"><Tag skeleton /><Tag skeleton /><Tag skeleton /></div>
        {:else}
          <span class="influence-help">No contributor details are available for this recommendation.</span>
        {/if}
      </div>
    </div>
  {/if}
  <button
    type="button"
    class="expander"
    aria-label={expanded ? 'Show fewer details' : 'Show full synopsis and recommendation reasons'}
    aria-expanded={expanded}
    on:click={toggleExpanded}
  >
    <span>{expanded ? 'Less details' : 'More details'}</span>
    {#if expanded}<ChevronUp size={16} aria-hidden="true" />{:else}<ChevronDown size={16} aria-hidden="true" />{/if}
  </button>
</div>

<style lang="css">
  .recommendation {
    display: grid;
    border: 1px solid #ffffff18;
    border-radius: 10px;
    background: #191f1b;
    align-items: start;
    min-width: 0;
    overflow: hidden;
    grid-template-areas: 'thumbnail content' 'synopsis synopsis' 'details details' 'expander expander';
    grid-template-columns: 80px minmax(0, 1fr);
  }
  .recommendation:hover {
    border-color: #88b69755;
  }
  .recommendation[data-plan-to-watch='true'] {
    background: #1d2a20;
  }
  .poster {
    grid-area: thumbnail;
    position: relative;
    align-self: start;
    height: 120px;
    padding: 0;
    border: 0;
    border-bottom-right-radius: 10px;
    overflow: hidden;
    background: #202823;
    cursor: pointer;
  }
  .poster img {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
  }
  .poster-fallback {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 8px;
    position: absolute;
    inset: 0;
    color: #7e9487;
    font-size: 10px;
  }
  .card-content {
    grid-area: content;
    padding: 16px;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 10px;
    min-width: 0;
    width: 100%;
    box-sizing: border-box;
    min-height: 120px;
  }
  .title-row {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    width: 100%;
    min-width: 0;
  }
  .title-text {
    font-size: 15px;
    line-height: 1.4;
    font-weight: 600;
    overflow-wrap: anywhere;
    min-width: 0;
    flex: 0 1 auto;
  }
  .title-button {
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-line-clamp: 2;
    line-clamp: 2;
    overflow: hidden;
    text-align: left;
    padding: 0;
    border: 0;
    background: none;
    color: #e6eee9;
    font: inherit;
    cursor: pointer;
  }
  .title-button:hover {
    color: #98e4a3;
  }
  .recommendation[data-expanded='true'] .title-button {
    display: block;
    overflow: visible;
  }
  .external-link {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex: none;
    width: 28px;
    height: 28px;
    margin-top: -3px;
    margin-bottom: -3px;
    border: 1px solid #ffffff20;
    border-radius: 6px;
    color: #abc6b5;
    background: #ffffff04;
  }
  .external-link:hover {
    color: #98e4a3;
    border-color: #78d88670;
    background: #78d88615;
  }
  .metadata-row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px 12px;
    min-width: 0;
  }
  .meta-line {
    font-size: 11px;
    line-height: 1.5;
    color: #98aba0;
  }
  .predicted-line {
    display: inline-flex;
    align-items: baseline;
    gap: 5px;
    white-space: nowrap;
  }
  .predicted-line[data-tier='high'] {
    --tier-color: #72d58b;
  }
  .predicted-line[data-tier='low'] {
    --tier-color: #ffad70;
  }
  .predicted-line[data-tier='mid'],
  .predicted-line[data-tier='neutral'] {
    --tier-color: #b3c6ba;
  }
  .predicted-value {
    color: var(--tier-color);
    font-size: 16px;
    line-height: 1.2;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
  }
  .predicted-text {
    font-size: 10px;
    color: #9baca1;
  }
  .card-tags {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 6px;
    min-width: 0;
    width: 100%;
  }
  .card-tags :global(.bx--tag) {
    margin: 0;
  }
  .genres {
    min-width: 0;
    width: 100%;
    height: 24px;
  }
  .recommendation[data-expanded='true'] .genres {
    height: auto;
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
  }
  .genre-label {
    display: flex;
    align-items: center;
    gap: 5px;
  }
  .genres :global(.bx--tag) {
    background: #29302d;
    border: 1px solid #ffffff12;
    color: #d5ded8;
  }
  .synopsis {
    grid-area: synopsis;
    min-width: 0;
    overflow-wrap: anywhere;
    font-size: 13px;
    line-height: 1.55;
    color: #adbbb2;
    margin: 12px 16px;
    text-align: left;
    white-space: pre-line;
  }
  .recommendation[data-expanded='false'] .synopsis {
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-line-clamp: 3;
    line-clamp: 3;
    overflow: hidden;
  }
  .recommendation[data-expanded='true'] .synopsis {
    color: #c9d5cd;
  }
  .expander {
    grid-area: expander;
    display: flex;
    align-items: center;
    justify-content: flex-start;
    gap: 6px;
    min-height: 40px;
    width: 100%;
    padding: 8px 16px;
    border: 0;
    border-top: 1px solid #ffffff14;
    background: #ffffff04;
    color: #a4b8ac;
    font-size: 12px;
    cursor: pointer;
  }
  .expander:hover {
    background: #78d88615;
    color: #98e4a3;
  }
  button:focus-visible,
  a:focus-visible {
    outline: 2px solid #75dc82;
    outline-offset: -2px;
  }
  .details {
    grid-area: details;
    border-top: 1px solid #ffffff14;
    padding: 12px 16px 16px;
    background: #00000012;
    min-width: 0;
  }
  .top-influences {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    gap: 8px;
    min-width: 0;
  }
  .influence-heading {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .influence-heading h3 {
    margin: 0;
    color: #a5bcad;
    font-size: 12px;
    line-height: 1.4;
    font-weight: 500;
  }
  .influence-help {
    font-size: 12px;
    line-height: 1.5;
    color: #99aca1;
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
    text-align: left;
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

  @media (max-width: 800px) {
    .influence-pills {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .contributor .contrib-popover {
      left: 0;
      right: auto;
      transform: none;
    }
  }
  @media (max-width: 480px) {
    .recommendation {
      grid-template-columns: 72px minmax(0, 1fr);
    }
    .poster {
      height: 108px;
    }
    .card-content {
      padding: 12px 10px;
      gap: 8px;
      min-height: 108px;
    }
    .expander {
      padding-left: 12px;
    }
    .title-text {
      font-size: 14px;
    }
    .synopsis {
      margin: 12px;
    }
    .metadata-row {
      gap: 5px;
    }
    .influence-pills {
      grid-template-columns: minmax(0, 1fr);
    }
    .details {
      padding: 10px 12px 12px;
    }
  }
</style>
