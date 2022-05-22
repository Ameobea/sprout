<script lang="ts" context="module">
  import { ModelName, PopularityAttenuationFactor } from './conf';

  const ALL_MODEL_OPTIONS: { id: ModelName; text: string }[] = [
    { id: ModelName.Model_4K, text: '[OUT-OF-DATE FOR EMBEDDING] Top 4k Anime v1' },
    { id: ModelName.Model_4K_V2, text: '[OUT-OF-DATE FOR EMBEDDING] Top 4k Anime w. more loss weights' },
    { id: ModelName.Model_6K, text: 'Top 6k Anime v1' },
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
  import { Dropdown, Tag, Toggle } from 'carbon-components-svelte';
  import type { Writable } from 'svelte/store';

  import type { RecommendationControlParams } from './InteractiveRecommendations.svelte';
  import type { AnimeDetails } from 'src/malAPI';

  export let params: Writable<RecommendationControlParams>;
  export let animeMetadataDatabase: { [animeID: number]: AnimeDetails };
</script>

<div class="root">
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
    <div>
      <Toggle labelText="Music" bind:toggled={$params.includeMusic} />
    </div>
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
          <!-- TODO: Look up genre names -->
          <Tag
            filter
            on:close={() =>
              params.update((state) => {
                state.excludedGenreIDs = state.excludedGenreIDs.filter((oGenreID) => oGenreID !== genreID);
                return state;
              })}
          >
            {genreID}
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
