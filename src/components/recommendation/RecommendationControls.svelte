<script lang="ts" context="module">
  import { ModelName } from './conf';

  const ALL_MODEL_OPTIONS: { id: ModelName; text: string }[] = [{ id: ModelName.Model_2025_jax, text: 'V2 - 2025' }];
</script>

<script lang="ts">
  import { Dropdown, InlineLoading, Tag, Toggle, ExpandableTile, Slider } from 'carbon-components-svelte';
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

  // Local state for sliders to prevent updates while dragging
  let localLogitWeight = $params.logitWeight;
  let localNicheBoostFactor = $params.nicheBoostFactor;

  // Sync local state with params store when params change from outside
  // $: localLogitWeight = $params.logitWeight;
  // $: localNicheBoostFactor = $params.nicheBoostFactor;
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
        <div class="top-row">
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
          <!-- svelte-ignore a11y-click-events-have-key-events -->
          <!-- svelte-ignore a11y-no-static-element-interactions -->
          <div on:click={(e) => e.stopPropagation()}>
            <Toggle labelText="Filter Plan to Watch" bind:toggled={$params.filterPlanToWatch} />
            <span class="helper-text">Hide shows that are already marked plan to watch</span>
          </div>
        </div>
        <div class="bottom-row">
          <!-- svelte-ignore a11y-click-events-have-key-events -->
          <!-- svelte-ignore a11y-no-static-element-interactions -->
          <div on:click={(e) => e.stopPropagation()}>
            <Slider
              labelText="Presence/Rating Weight"
              min={0}
              max={1}
              step={0.1}
              bind:value={localLogitWeight}
              on:change={() => {
                $params.logitWeight = localLogitWeight;
              }}
            />
            <span class="helper-text">
              Balance between predicted rating (0) and presence probability (1) when scoring recommendations
            </span>
          </div>
          <!-- svelte-ignore a11y-click-events-have-key-events -->
          <!-- svelte-ignore a11y-no-static-element-interactions -->
          <div on:click={(e) => e.stopPropagation()}>
            <Slider
              labelText="Niche Boost Factor"
              min={0}
              max={1}
              step={0.1}
              bind:value={localNicheBoostFactor}
              on:change={() => {
                $params.nicheBoostFactor = localNicheBoostFactor;
              }}
            />
            <span class="helper-text">
              Boosts shows that the model thinks you'll like more than their popularity suggests. Higher = more boost.
            </span>
          </div>
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
    gap: 16px;
    padding: 4px;
  }

  .top-row,
  .bottom-row {
    display: flex;
    flex-direction: row;
    gap: 16px;
  }

  .top-row > div,
  .bottom-row > div {
    display: flex;
    flex: 1;
    flex-direction: column;
    min-width: 0;
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

  .helper-text {
    font-size: 0.75rem;
    color: #a8a8a8;
    margin-top: 4px;
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
