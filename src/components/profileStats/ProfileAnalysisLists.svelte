<script lang="ts">
  import type { HoldoutItem, NormalizationStats } from 'src/routes/recommendation/recommendation/recommendation';
  import type { PartialStatsAnimeMetadatum } from 'src/routes/user/[username]/stats/+page.server';
  import { denormalizeRating } from 'src/util/ratingNormalization';

  export let mostImpactfulRatings: HoldoutItem[];
  export let mostSurprisingItems: HoldoutItem[];
  export let normalizationStats: NormalizationStats;
  export let animeData: { [animeID: number]: PartialStatsAnimeMetadatum };

  const getMALLink = (animeId: number) => `https://myanimelist.net/anime/${animeId}`;

  const getAnimeName = (animeId: number): string => {
    return animeData[animeId]?.title ?? `Anime #${animeId}`;
  };

  const formatRating = (rating: number): string => {
    return rating.toFixed(1);
  };

  const getPredictedRating = (normalizedRating: number): number => {
    return denormalizeRating(normalizedRating, normalizationStats);
  };

  // Check if there are any rated items at all
  $: hasRatedItems = mostSurprisingItems.length > 0;
  $: hasAnyItems = mostImpactfulRatings.length > 0;
</script>

{#if hasAnyItems}
  <div class="analysis-container">
    <div class="analysis-section">
      <h3>Most Impactful Ratings</h3>
      <p class="description">
        Ratings that have the largest effect on recommendation output. Removing these from your profile would cause the
        biggest changes to your recommendations.
      </p>
      <div class="table-container">
        <div class="table-header">
          <span class="col-rank">#</span>
          <span class="col-name">Title</span>
          <span class="col-rating">Your Rating</span>
          <span class="col-rating">Predicted</span>
        </div>
        <div class="scrollable-list">
          {#each mostImpactfulRatings as item, index}
            <div class="list-item">
              <span class="col-rank">{index + 1}</span>
              <a href={getMALLink(item.anime_id)} target="_blank" rel="noopener noreferrer" class="col-name anime-name">
                {getAnimeName(item.anime_id)}
              </a>
              <span class="col-rating">{item.true_rating > 0 ? item.true_rating : '-'}</span>
              <span class="col-rating predicted">{formatRating(getPredictedRating(item.predicted_rating))}</span>
            </div>
          {/each}
        </div>
      </div>
    </div>

    <div class="analysis-section">
      <h3>Most Surprising Ratings</h3>
      <p class="description">
        Anime where your rating differs most from the model's prediction. These represent your most distinctive
        preferences compared to similar users.
      </p>
      {#if hasRatedItems}
        <div class="table-container">
          <div class="table-header">
            <span class="col-rank">#</span>
            <span class="col-name">Title</span>
            <span class="col-rating">Your Rating</span>
            <span class="col-rating">Predicted</span>
            <span class="col-diff">Diff</span>
          </div>
          <div class="scrollable-list">
            {#each mostSurprisingItems as item, index}
              {@const predictedRating = getPredictedRating(item.predicted_rating)}
              {@const ratingDiff = item.true_rating - predictedRating}
              <div class="list-item">
                <span class="col-rank">{index + 1}</span>
                <a
                  href={getMALLink(item.anime_id)}
                  target="_blank"
                  rel="noopener noreferrer"
                  class="col-name anime-name"
                >
                  {getAnimeName(item.anime_id)}
                </a>
                <span class="col-rating">{item.true_rating}</span>
                <span class="col-rating predicted">{formatRating(predictedRating)}</span>
                <span class="col-diff" class:positive={ratingDiff > 0} class:negative={ratingDiff < 0}>
                  {ratingDiff > 0 ? '+' : ''}{formatRating(ratingDiff)}
                </span>
              </div>
            {/each}
          </div>
        </div>
      {:else}
        <p class="empty-notice">
          No rated anime in your profile. Add ratings to your anime list to see which of your opinions are most
          surprising to the model.
        </p>
      {/if}
    </div>
  </div>
{/if}

<style lang="css">
  .analysis-container {
    display: flex;
    flex-direction: column;
    gap: 24px;
    margin-bottom: 32px;
    width: 100%;
  }

  .analysis-section {
    background: rgba(255, 255, 255, 0.05);
    padding: 12px;
  }

  .analysis-section h3 {
    margin: 0 0 6px 0;
    font-size: 1.1rem;
    text-align: left;
  }

  .description {
    font-size: 0.85rem;
    color: #aaa;
    margin: 0 0 10px 0;
    text-align: left;
  }

  .table-container {
    display: flex;
    flex-direction: column;
  }

  .table-header {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 8px;
    font-size: 0.75rem;
    font-weight: 600;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  }

  .scrollable-list {
    max-height: 350px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
  }

  .list-item {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 8px;
    font-size: 0.9rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.03);
  }

  .list-item:last-child {
    border-bottom: none;
  }

  .col-rank {
    min-width: 24px;
    text-align: center;
    color: #666;
  }

  .col-name {
    flex: 1;
    text-align: left;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-width: 0;
  }

  .col-rating {
    min-width: 50px;
    text-align: center;
  }

  .col-diff {
    min-width: 50px;
    text-align: center;
    font-weight: 500;
  }

  .anime-name {
    color: #58a6ff;
    text-decoration: none;
  }

  .anime-name:hover {
    text-decoration: underline;
  }

  .predicted {
    color: #888;
  }

  .positive {
    color: #4caf50;
  }

  .negative {
    color: #f44336;
  }

  .empty-notice {
    font-size: 0.9rem;
    color: #888;
    font-style: italic;
    text-align: left;
    padding: 8px 0;
  }

  /* Custom scrollbar styling */
  .scrollable-list::-webkit-scrollbar {
    width: 6px;
  }

  .scrollable-list::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
  }

  .scrollable-list::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.2);
  }

  .scrollable-list::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.3);
  }

  /* Mobile adjustments */
  @media (max-width: 600px) {
    .analysis-section {
      padding: 8px;
    }

    .table-header,
    .list-item {
      gap: 4px;
      padding: 5px 4px;
      font-size: 0.8rem;
    }

    .table-header {
      font-size: 0.65rem;
    }

    .col-rank {
      min-width: 20px;
    }

    .col-rating {
      min-width: 40px;
    }

    .col-diff {
      min-width: 40px;
    }
  }
</style>
