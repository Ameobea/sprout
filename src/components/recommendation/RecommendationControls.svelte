<script lang="ts" context="module">
  import { ModelName, PopularityAttenuationFactor } from './conf';

  const ALL_MODEL_OPTIONS: { id: ModelName; text: string }[] = [
    // { id: ModelName.Model_6K, text: 'Top 6k Anime v1' },
    // { id: ModelName.Model_6K_Smaller, text: 'Top 6k Smaller' },
    // { id: ModelName.Model_6K_Smaller_Weighted, text: 'Top 6k Smaller Weighted' },
    // { id: ModelName.Model_6_5K_New, text: 'Top 6.5K Weighted Updated' },
    // { id: ModelName.Model_6_5K_Unweighted, text: 'Top 6.5K Unweighted Updated' },
    { id: ModelName.Model_6_5k_New2, text: 'Top 6.5K Weighted Jan. 2023' },
    { id: ModelName.Model_6_5k_New2_Alt, text: 'Top 6.5K Weighted Alt. Jan. 2023' },
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
  import { Dropdown, InlineLoading, Tag, Toggle, ExpandableTile } from 'carbon-components-svelte';
  import type { Writable } from 'svelte/store';

  import type { AnimeDetails } from 'src/malAPI';
  import { browser } from '$app/environment';
  import { captureMessage } from 'src/sentry';
  import type { RecommendationControlParams } from './utils';

  let innerWidth = browser ? window.innerWidth : 0;
  $: isMobile = innerWidth < 768;

  export let params: Writable<RecommendationControlParams>;
  export let animeMetadataDatabase: { [animeID: number]: AnimeDetails };
  export let isLoading: boolean;
  export let genresDB: Writable<Map<number, string>>;
  export let forceHideTopBar: boolean | undefined = false;
</script>

<svelte:window bind:innerWidth />

<div class="root">
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
  {#if !isMobile && !forceHideTopBar}
    <ExpandableTile style="min-height: 10px">
      <div slot="above">Advanced Options</div>
      <div class="top" slot="below">
        <!-- svelte-ignore a11y-click-events-have-key-events -->
        <!-- svelte-ignore a11y-no-static-element-interactions -->
        <div on:click={(e) => e.stopPropagation()}>
          <Dropdown
            style="width: 100%;"
            titleText="Popularity Attenuation Factor"
            selectedId={$params.popularityAttenuationFactor}
            on:select={(selected) => {
              $params.popularityAttenuationFactor = selected.detail.selectedItem.id;
            }}
            items={ALL_POPULARITY_ATTENUATION_FACTOR_OPTIONS}
            helperText="Higher popularity attenuation factors result in less-popular anime being weighted higher in recommendations"
          />
        </div>
        <!-- svelte-ignore a11y-click-events-have-key-events -->
        <!-- svelte-ignore a11y-no-static-element-interactions -->
        <div on:click={(e) => e.stopPropagation()}>
          <Dropdown
            style="width: 100%;"
            titleText="Model"
            selectedId={$params.modelName}
            on:select={(selected) => {
              $params.modelName = selected.detail.selectedItem.id;
            }}
            items={ALL_MODEL_OPTIONS}
            helperText="Each model was trained slightly differently, which impacts the generated recommendations"
          />
        </div>
      </div>
    </ExpandableTile>
  {/if}
  {#if $params.excludedRankingAnimeIDs.length > 0}
    <div>
      <label for="tags-container" class="bx--label">Excluded Rankings</label>
      <div class="tags-container" id="tags-container">
        {#each [...new Set($params.excludedRankingAnimeIDs)] as animeID (animeID)}
          {@const datum = animeMetadataDatabase[animeID]}
          {@const title = datum?.alternative_titles.en || datum?.title || ''}
          <Tag
            filter
            on:close={() => {
              captureMessage('Remove excluded ranking', { id: animeID, name: datum.title });
              params.update((state) => {
                state.excludedRankingAnimeIDs = state.excludedRankingAnimeIDs.filter(
                  (oAnimeID) => oAnimeID !== animeID
                );
                return state;
              });
            }}
          >
            {title}
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
            on:close={() => {
              captureMessage('Remove excluded genre', { id: genreID, name: genreName });
              params.update((state) => {
                state.excludedGenreIDs = state.excludedGenreIDs.filter((oGenreID) => oGenreID !== genreID);
                return state;
              });
            }}
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
    padding: 4px;
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

  :global(.bx--tile--expandable) {
    background: #242424 !important;
  }

  :global(.bx--tile--expandable:focus) {
    outline: none !important;
  }

  :global(.bx--tile--expandable:hover[aria-expanded='false']) {
    background: #2b2b2b !important;
  }

  :global(.bx--tile--is-expanded.bx--tile--expandable) {
    background: #202020 !important;
  }
</style>
