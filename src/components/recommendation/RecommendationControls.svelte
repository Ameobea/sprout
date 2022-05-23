<script lang="ts" context="module">
  import { ModelName, PopularityAttenuationFactor } from './conf';

  const ALL_MODEL_OPTIONS: { id: ModelName; text: string }[] = [
    { id: ModelName.Model_6K, text: 'Top 6k Anime v1' },
    // { id: ModelName.Model_6K_TFLite, text: 'Top 6k TFLite' },
    { id: ModelName.Model_6K_Smaller, text: 'Top 6k Smaller' },
  ];

  const ALL_POPULARITY_ATTENUATION_FACTOR_OPTIONS: { id: PopularityAttenuationFactor; text: string }[] = [
    { id: PopularityAttenuationFactor.None, text: 'None' },
    { id: PopularityAttenuationFactor.VeryLow, text: 'Very Low' },
    { id: PopularityAttenuationFactor.Low, text: 'Low' },
    { id: PopularityAttenuationFactor.Medium, text: 'Medium' },
    { id: PopularityAttenuationFactor.High, text: 'High' },
    { id: PopularityAttenuationFactor.VeryHigh, text: 'VeryHigh' },
  ];
</script>

<script lang="ts">
  import { Dropdown, InlineLoading, Tag, Toggle } from 'carbon-components-svelte';
  import type { Writable } from 'svelte/store';

  import type { RecommendationControlParams } from './InteractiveRecommendations.svelte';
  import type { AnimeDetails } from 'src/malAPI';
  import { browser } from '$app/env';

  let innerWidth = browser ? window.innerWidth : 0;
  $: isMobile = innerWidth < 768;

  export let params: Writable<RecommendationControlParams>;
  export let animeMetadataDatabase: { [animeID: number]: AnimeDetails };
  export let isLoading: boolean;
  export let genresDB: Writable<Map<number, string>>;
</script>

<svelte:window bind:innerWidth />

<div class="root">
  <!-- {#if !isMobile}
    <div class="top">
      <div>
        <Dropdown
          style="width: 100%;"
          titleText="Model"
          selectedId={$params.modelName}
          on:select={(selected) => {
            $params.modelName = selected.detail.selectedItem.id;
          }}
          items={ALL_MODEL_OPTIONS}
        />
      </div>
      <div>
        <Dropdown
          style="width: 100%;"
          titleText="Popularity Attenuation Factor"
          selectedId={$params.popularityAttenuationFactor}
          on:select={(selected) => {
            $params.popularityAttenuationFactor = selected.detail.selectedItem.id;
          }}
          items={ALL_POPULARITY_ATTENUATION_FACTOR_OPTIONS}
        />
      </div>
    </div>
  {/if} -->
  <div class="toggles">
    <div>
      <Toggle labelText="Extra Seasons" bind:toggled={$params.includeExtraSeasons} />
    </div>
    <div>
      <Toggle labelText="Movies" bind:toggled={$params.includeMovies} />
    </div>
    <div>
      <Toggle labelText="ONAs / OVAs / Specials" bind:toggled={$params.includeONAsOVAsSpecials} />
    </div>
    <div style="position: relative">
      <Toggle labelText="Music" bind:toggled={$params.includeMusic} />
      {#if isLoading && isMobile}
        <div style="position: absolute; right: -4px; bottom: -4px; flex: 0;">
          <InlineLoading />
        </div>
      {/if}
    </div>
    {#if !isMobile && isLoading}
      <InlineLoading style="flex: 0; margin-left: 10px;" />
    {/if}
  </div>
  {#if $params.excludedRankingAnimeIDs.length > 0}
    <div>
      <label for="tags-container" class="bx--label">Excluded Rankings</label>
      <div class="tags-container" id="tags-container">
        {#each [...new Set($params.excludedRankingAnimeIDs)] as animeID (animeID)}
          {@const datum = animeMetadataDatabase[animeID]}
          <Tag
            filter
            on:close={() =>
              params.update((state) => {
                state.excludedRankingAnimeIDs = state.excludedRankingAnimeIDs.filter(
                  (oAnimeID) => oAnimeID !== animeID
                );
                return state;
              })}
          >
            {datum?.alternative_titles.en || datum?.title || ''}
          </Tag>
        {/each}
      </div>
    </div>
  {/if}
  {#if $params.excludedGenreIDs.length > 0}
    <div>
      <label for="tags-container" class="bx--label">Excluded Genres</label>
      <div class="tags-container" id="tags-container">
        {#each [...new Set($params.excludedGenreIDs)] as genreID (genreID)}
          {@const genreName = $genresDB.get(genreID) ?? genreID.toString()}
          <Tag
            filter
            on:close={() =>
              params.update((state) => {
                state.excludedGenreIDs = state.excludedGenreIDs.filter((oGenreID) => oGenreID !== genreID);
                return state;
              })}
          >
            {genreName}
          </Tag>
        {/each}
      </div>
    </div>
  {/if}
</div>

<style lang="css">
  .root {
    display: flex;
    flex-direction: column;
    gap: 14px;
    padding: 5px 0px 10px 0px;
    border-bottom: 1px solid #cccccc22;
  }

  .top {
    display: flex;
    flex-direction: row;
    min-width: 100%;
    gap: 20px;
  }

  .top > div {
    display: flex;
    flex: 1;
    flex-direction: column;
  }

  .toggles {
    display: flex;
    flex-direction: row;
    flex: 1;
  }

  .toggles > div {
    padding: 8px;
    box-sizing: border-box;
    border-right: 1px solid #cccccc22;
    border-top: 1px solid #cccccc22;
    border-bottom: 1px solid #cccccc22;
    width: 150px;
  }

  @media (max-width: 768px) {
    .toggles > div {
      display: flex;
      flex: 1;
      width: unset;
    }
  }

  .toggles > div:first-child {
    border-left: 1px solid #cccccc22;
  }

  .tags-container {
    display: flex;
    flex-direction: row;
    flex-wrap: wrap;
  }
</style>
