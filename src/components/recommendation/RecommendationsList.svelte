<script lang="ts">
  import { fade } from 'svelte/transition';
  import { flip } from 'svelte/animate';

  import { getSurface, submitAnalyticsEvent } from 'src/analytics';
  import type { AnimeDetails } from 'src/malAPI';
  import type { Recommendation, UserRatingStats } from '../../routes/recommendation/recommendation/recommendation';
  import { getRatingTier, type RatingTier } from 'src/util/ratingTiers';
  import RecommendationListItem from './RecommendationListItem.svelte';
  import Filter from 'carbon-icons-svelte/lib/Filter.svelte';
  import Calendar from 'carbon-icons-svelte/lib/Calendar.svelte';
  import ArrowsVertical from 'carbon-icons-svelte/lib/ArrowsVertical.svelte';
  import Video from 'carbon-icons-svelte/lib/Video.svelte';
  import Catalog from 'carbon-icons-svelte/lib/Catalog.svelte';
  import GenreIcon from './GenreIcon.svelte';
  import HelpTip from './HelpTip.svelte';
  import Reset from 'carbon-icons-svelte/lib/Reset.svelte';
  import ChevronDown from 'carbon-icons-svelte/lib/ChevronDown.svelte';

  export let recommendations: Recommendation[];
  export let animeMetadataDatabase: { [animeID: number]: AnimeDetails };
  export let excludeRanking: ((animeID: number) => void) | undefined = undefined;
  export let excludeGenre: ((genreID: number, genreName: string) => void) | undefined = undefined;
  export let includeGenre: ((genreID: number) => void) | undefined = undefined;
  export let excludedGenreIDs: number[] = [];
  export let genreNames: Map<number, string> = new Map();
  export let addRanking: ((animeID: number) => void) | undefined = undefined;
  export let contributorsLoading: boolean;
  export let userRatingStats: UserRatingStats | null = null;
  export let contributionBaseline: number | undefined = undefined;

  type SortMode = 'model' | 'predicted-desc' | 'predicted-asc' | 'year-desc' | 'year-asc' | 'title-asc';

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
    unknown: 'Unknown',
  };

  let sortMode: SortMode = 'model';
  let genreFilter = '';
  let mediaTypeFilter = '';
  let yearFrom: number | undefined = undefined;
  let yearTo: number | undefined = undefined;

  // Drop contributors dwarfed by the rec's top one or below the profile-wide
  // significance scale; always keep at least the top contributor.
  const REL_CUTOFF = 0.15;
  const SIG_MULT = 3;
  const visibleContributors = (contributors: Recommendation['topContributors']) => {
    if (!contributors?.length) {
      return contributors;
    }
    const sigFloor = contributionBaseline !== undefined ? SIG_MULT * contributionBaseline : 0;
    const cutoff = Math.max(REL_CUTOFF * contributors[0].strength, sigFloor);
    const kept = contributors.filter((c) => c.strength >= cutoff);
    return kept.length > 0 ? kept : contributors.slice(0, 1);
  };

  const getTitle = (datum: AnimeDetails): string => datum.alternative_titles?.en || datum.title || '';
  const getYear = (datum: AnimeDetails): number | null => {
    const year = Number(datum.start_date?.slice(0, 4));
    return Number.isFinite(year) && year > 0 ? year : null;
  };
  const parseYearFilter = (raw: number | undefined): number | null => {
    if (raw === undefined || raw === null) {
      return null;
    }
    return Number.isFinite(raw) ? raw : null;
  };

  $: hasPredictedRatings =
    !!userRatingStats &&
    !userRatingStats.isNonRater &&
    recommendations.some((reco) => typeof reco.predictedRating === 'number');
  $: if (!hasPredictedRatings && (sortMode === 'predicted-desc' || sortMode === 'predicted-asc')) {
    sortMode = 'model';
  }

  $: availableGenres = Array.from(
    new Set(recommendations.flatMap((reco) => animeMetadataDatabase[reco.id]?.genres?.map((genre) => genre.name) ?? []))
  ).sort((a, b) => a.localeCompare(b));

  $: exclusionGenres = (() => {
    const genres = new Map(genreNames);
    for (const anime of Object.values(animeMetadataDatabase)) {
      for (const genre of anime.genres ?? []) genres.set(genre.id, genre.name);
    }
    for (const id of excludedGenreIDs) {
      if (!genres.has(id)) genres.set(id, `Genre ${id}`);
    }
    return Array.from(genres).sort((a, b) => a[1].localeCompare(b[1]));
  })();

  $: availableMediaTypes = Array.from(
    new Set(
      recommendations
        .map((reco) => animeMetadataDatabase[reco.id]?.media_type)
        .filter((mediaType): mediaType is NonNullable<typeof mediaType> => !!mediaType)
    )
  ).sort((a, b) => (MEDIA_TYPE_NAMES[a] ?? a).localeCompare(MEDIA_TYPE_NAMES[b] ?? b));

  let displayRecommendations: (Recommendation & {
    shownPredictedRating: number | null;
    ratingTier: RatingTier | null;
    modelRank: number;
  })[] = [];

  $: {
    const minYear = parseYearFilter(yearFrom);
    const maxYear = parseYearFilter(yearTo);

    displayRecommendations = recommendations
      .map((reco, modelRank) => {
        const showRating = !!userRatingStats && !userRatingStats.isNonRater && typeof reco.predictedRating === 'number';
        return {
          ...reco,
          modelRank,
          shownPredictedRating: showRating ? reco.predictedRating! : null,
          ratingTier: showRating ? getRatingTier(reco.predictedRating!, userRatingStats!) : null,
        };
      })
      .filter((reco) => {
        const metadata = animeMetadataDatabase[reco.id];
        if (!metadata) {
          return false;
        }

        if (genreFilter && !metadata.genres?.some((genre) => genre.name === genreFilter)) {
          return false;
        }
        if (mediaTypeFilter && metadata.media_type !== mediaTypeFilter) {
          return false;
        }

        const year = getYear(metadata);
        if ((minYear !== null || maxYear !== null) && year === null) {
          return false;
        }
        if (minYear !== null && year !== null && year < minYear) {
          return false;
        }
        if (maxYear !== null && year !== null && year > maxYear) {
          return false;
        }

        return true;
      })
      .sort((a, b) => {
        const metadataA = animeMetadataDatabase[a.id];
        const metadataB = animeMetadataDatabase[b.id];

        switch (sortMode) {
          case 'predicted-desc':
            if (a.shownPredictedRating === null && b.shownPredictedRating === null) return a.modelRank - b.modelRank;
            if (a.shownPredictedRating === null) return 1;
            if (b.shownPredictedRating === null) return -1;
            return b.shownPredictedRating - a.shownPredictedRating || a.modelRank - b.modelRank;
          case 'predicted-asc':
            if (a.shownPredictedRating === null && b.shownPredictedRating === null) return a.modelRank - b.modelRank;
            if (a.shownPredictedRating === null) return 1;
            if (b.shownPredictedRating === null) return -1;
            return a.shownPredictedRating - b.shownPredictedRating || a.modelRank - b.modelRank;
          case 'year-desc': {
            const yearA = getYear(metadataA) ?? Number.NEGATIVE_INFINITY;
            const yearB = getYear(metadataB) ?? Number.NEGATIVE_INFINITY;
            return yearB - yearA || a.modelRank - b.modelRank;
          }
          case 'year-asc': {
            const yearA = getYear(metadataA) ?? Number.POSITIVE_INFINITY;
            const yearB = getYear(metadataB) ?? Number.POSITIVE_INFINITY;
            return yearA - yearB || a.modelRank - b.modelRank;
          }
          case 'title-asc':
            return getTitle(metadataA).localeCompare(getTitle(metadataB)) || a.modelRank - b.modelRank;
          case 'model':
          default:
            return a.modelRank - b.modelRank;
        }
      });
  }

  const clearDisplayFilters = () => {
    genreFilter = '';
    mediaTypeFilter = '';
    yearFrom = undefined;
    yearTo = undefined;
    [...excludedGenreIDs].forEach((id) => includeGenre?.(id));
  };

  $: displayFiltersActive =
    !!genreFilter || !!mediaTypeFilter || yearFrom !== undefined || yearTo !== undefined || excludedGenreIDs.length > 0;

  let expandedAnimeID: number | null = null;
  $: if (expandedAnimeID !== null && !displayRecommendations.some((reco) => reco.id === expandedAnimeID)) {
    expandedAnimeID = null;
  }
</script>

<details class="browse-panel">
  <summary class="browse-summary">
    <h2 class="browse-heading"><Filter size={20} aria-hidden="true" />Browse recommendations</h2>
    <span class="result-count" role="status"
      >{displayRecommendations.length} / {recommendations.length} results{displayFiltersActive
        ? ' · filtered'
        : ''}</span
    >
    <ChevronDown size={20} aria-hidden="true" />
  </summary>
  <div class="browse-body">
    <div class="browse-header">
      <HelpTip
        label="List filters"
        text="Genre, Format, and Year narrow the recommendations already generated. Exclude genres requests new recommendations without the checked genres. Clear filters restores all genres and clears the list filters while keeping your chosen sort order."
      />
      <div class="control-actions">
        <button type="button" on:click={clearDisplayFilters} disabled={!displayFiltersActive}
          ><Reset size={16} aria-hidden="true" />Clear filters</button
        >
      </div>
    </div>
    <div class="browse-controls">
      <div class="control sort-control">
        <div class="control-heading">
          <label for="recommendation-sort"><ArrowsVertical size={16} aria-hidden="true" />Sort</label><HelpTip
            label="Sort order"
            text="Recommended order (default) keeps Sprout’s original recommendation order. Predicted rating sorts by the personalized 1–10 estimate shown beside each title; it is available only when the model provides rating predictions. You can also sort by year or title."
          />
        </div>
        <select id="recommendation-sort" bind:value={sortMode}>
          <option value="model">Recommended order (default)</option>
          <option value="predicted-desc" disabled={!hasPredictedRatings}>Predicted rating: high to low</option>
          <option value="predicted-asc" disabled={!hasPredictedRatings}>Predicted rating: low to high</option>
          <option value="year-desc">Year: newest first</option>
          <option value="year-asc">Year: oldest first</option>
          <option value="title-asc">Title: A–Z</option>
        </select>
      </div>

      <div class="control">
        <label for="genre-filter"
          >{#if genreFilter}<GenreIcon name={genreFilter} />{:else}<Catalog
              size={16}
              aria-hidden="true"
            />{/if}Genre</label
        >
        <select id="genre-filter" bind:value={genreFilter}>
          <option value="">All genres</option>
          {#if genreFilter && !availableGenres.includes(genreFilter)}
            <option value={genreFilter}>{genreFilter} (not in current results)</option>
          {/if}
          {#each availableGenres as genre}
            <option value={genre}>{genre}</option>
          {/each}
        </select>
      </div>

      <div class="control">
        <label for="format-filter"><Video size={16} aria-hidden="true" />Format</label>
        <select id="format-filter" bind:value={mediaTypeFilter}>
          <option value="">All enabled formats</option>
          {#if mediaTypeFilter && !availableMediaTypes.some((mediaType) => mediaType === mediaTypeFilter)}
            <option value={mediaTypeFilter}
              >{MEDIA_TYPE_NAMES[mediaTypeFilter] ?? mediaTypeFilter} (not in current results)</option
            >
          {/if}
          {#each availableMediaTypes as mediaType}
            <option value={mediaType}>{MEDIA_TYPE_NAMES[mediaType] ?? mediaType}</option>
          {/each}
        </select>
      </div>

      <div class="control year-control">
        <label for="year-from"><Calendar size={16} aria-hidden="true" />Year</label>
        <div>
          <input id="year-from" type="number" min="1900" max="2200" placeholder="From" bind:value={yearFrom} />
          <span>–</span>
          <input type="number" min="1900" max="2200" placeholder="To" aria-label="Year to" bind:value={yearTo} />
        </div>
      </div>

      {#if excludeGenre && includeGenre}
        <details class="genre-exclusions">
          <summary
            >Exclude genres{#if excludedGenreIDs.length}
              · {excludedGenreIDs.length} active{/if}</summary
          >
          <p>Checked genres are excluded from recommendations. Uncheck to restore them.</p>
          <div class="genre-options">
            {#each exclusionGenres as [id, name] (id)}
              <label class:excluded={excludedGenreIDs.includes(id)}>
                <input
                  type="checkbox"
                  checked={excludedGenreIDs.includes(id)}
                  on:change={(evt) => (evt.currentTarget.checked ? excludeGenre?.(id, name) : includeGenre?.(id))}
                />
                <GenreIcon {name} /><span>{name}</span>
              </label>
            {/each}
          </div>
        </details>
      {/if}
    </div>
  </div>
</details>

<div class="recommendations">
  {#each displayRecommendations as { id, topContributors, planToWatch, shownPredictedRating, ratingTier, modelRank } (id)}
    {@const animeMetadata = animeMetadataDatabase[id]}
    <div in:fade animate:flip={{ duration: (d) => 39 * Math.sqrt(d) }}>
      <RecommendationListItem
        {animeMetadata}
        rank={modelRank}
        expanded={expandedAnimeID === animeMetadata.id}
        toggleExpanded={() => {
          const expanded = expandedAnimeID !== animeMetadata.id;
          submitAnalyticsEvent({
            category: 'recommendations',
            subcategory: 'item_expand',
            payload: { anime_id: animeMetadata.id, rank: modelRank, expanded, surface: getSurface() },
          });
          expandedAnimeID = expanded ? animeMetadata.id : null;
        }}
        topContributors={visibleContributors(topContributors)?.map((c) => ({
          ...c,
          datum: animeMetadataDatabase[c.animeId],
        }))}
        {contributionBaseline}
        planToWatch={planToWatch ?? false}
        predictedRating={shownPredictedRating}
        {ratingTier}
        {excludeRanking}
        {addRanking}
        {contributorsLoading}
      />
    </div>
  {/each}
  {#if displayRecommendations.length === 0}
    <p class="empty-results">
      No recommendations match the current controls. Clear the list filters or enable more formats.
    </p>
  {/if}
</div>

<style lang="css">
  .browse-panel {
    padding: 20px;
    margin: 14px 0;
    border: 1px solid #ffffff18;
    border-radius: 12px;
    background: #171c19;
  }
  .browse-header {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 14px;
  }
  .browse-summary {
    display: flex;
    align-items: center;
    gap: 12px;
    cursor: pointer;
    list-style: none;
  }
  .browse-summary::-webkit-details-marker {
    display: none;
  }
  .browse-summary > :global(svg) {
    flex: none;
    color: #9caca2;
  }
  .browse-panel[open] > .browse-summary > :global(svg) {
    transform: rotate(180deg);
  }
  .browse-body {
    margin-top: 14px;
  }
  .browse-heading {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
    color: #edf3ef;
  }
  .browse-heading :global(svg) {
    flex: none;
    color: #88da95;
  }
  h2 {
    color: #edf3ef;
    font-size: 15px;
    font-weight: 600;
    line-height: 1.4;
    margin: 0;
  }
  .browse-controls {
    display: grid;
    grid-template-columns: minmax(0, 1.35fr) minmax(0, 1fr) minmax(0, 1fr) minmax(0, 1.15fr);
    gap: 14px;
    align-items: end;
  }
  .control {
    display: flex;
    flex-direction: column;
    gap: 8px;
    min-width: 0;
  }
  .control-heading {
    display: flex;
    justify-content: space-between;
    align-items: center;
    min-height: 30px;
  }
  .control label {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: #aabbb0;
    min-height: 30px;
  }
  .control select,
  .control input {
    min-height: 42px;
    width: 100%;
    min-width: 0;
    border: 1px solid #ffffff26;
    border-radius: 6px;
    background: #252d28;
    color: #e6eee9;
    padding: 8px 10px;
    box-sizing: border-box;
    font-size: 13px;
  }
  .control select:hover,
  .control input:hover {
    border-color: #9db6a566;
  }
  .control select:focus-visible,
  .control input:focus-visible,
  button:focus-visible,
  summary:focus-visible {
    outline: 2px solid #75dc82;
    outline-offset: 3px;
  }
  .year-control > div {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #82948a;
  }
  .control-actions {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .result-count {
    font-size: 11px;
    font-variant-numeric: tabular-nums;
    color: #a5b6ab;
    white-space: nowrap;
    margin-left: auto;
  }
  .control-actions button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    min-height: 34px;
    border: 1px solid #ffffff22;
    border-radius: 6px;
    background: transparent;
    color: #c5d7cc;
    padding: 6px 10px;
    font-size: 12px;
    cursor: pointer;
  }
  .control-actions button:hover:not(:disabled) {
    background: #77d58515;
    border-color: #77d58560;
  }
  .control-actions button:disabled {
    opacity: 0.4;
    cursor: default;
  }
  .genre-exclusions {
    grid-column: 1 / -1;
    border-top: 1px solid #ffffff14;
    margin-top: 2px;
    padding-top: 14px;
  }
  .genre-exclusions summary {
    cursor: pointer;
    color: #a8bcaf;
    font-size: 12px;
    line-height: 1.5;
  }
  .genre-exclusions[open] summary {
    color: #8ce295;
  }
  .genre-exclusions p {
    font-size: 12px;
    line-height: 1.5;
    color: #99a99f;
    margin: 12px 0;
  }
  .genre-options {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }
  .genre-options label {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 8px 10px;
    border: 1px solid #ffffff22;
    border-radius: 6px;
    font-size: 12px;
    color: #cad7cf;
    cursor: pointer;
  }
  .genre-options label:hover {
    background: #ffffff08;
  }
  .genre-options label.excluded {
    border-color: #bc8b61;
    background: #372b23;
  }
  .genre-options input {
    accent-color: #78d886;
  }
  .recommendations {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .empty-results {
    padding: 24px;
    border: 1px dashed #ffffff25;
    border-radius: 10px;
    color: #a5b6ab;
    font-size: 14px;
    line-height: 1.5;
  }
  @media (max-width: 800px) {
    .browse-controls {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
  @media (max-width: 560px) {
    .browse-panel {
      padding: 16px;
    }
    .browse-header {
      gap: 8px;
    }
    .control-actions {
      justify-content: flex-end;
    }
    .browse-summary {
      flex-wrap: wrap;
      gap: 8px;
    }
    .browse-heading {
      flex: 1;
    }
    .result-count {
      order: 1;
      width: 100%;
      margin-left: 28px;
    }
    .sort-control,
    .year-control {
      grid-column: 1 / -1;
    }
    .browse-controls {
      gap: 10px;
    }
  }
  @media (max-width: 360px) {
    .browse-controls {
      grid-template-columns: minmax(0, 1fr);
    }
  }
</style>
