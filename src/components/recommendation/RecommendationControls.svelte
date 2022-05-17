<script lang="ts" context="module">
  import { ModelName } from './conf';

  const ALL_MODEL_OPTIONS: { id: ModelName; text: string }[] = [
    { id: ModelName.Model_4K, text: 'Top 4k Anime v1' },
    { id: ModelName.Model_4K_V2, text: 'Top 4k Anime w. more loss weights' },
  ];
</script>

<script lang="ts">
  import { Dropdown, Loading, Tag } from 'carbon-components-svelte';
  import type { Writable } from 'svelte/store';

  import type { RecommendationControlParams } from './InteractiveRecommendations.svelte';
  import type { AnimeDetails } from 'src/malAPI';

  export let params: Writable<RecommendationControlParams>;
  export let animeMetadataDatabase: { [animeID: number]: AnimeDetails };
  export let isLoading: boolean;
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
    <div />
  </div>
  {#if $params.excludedRankingAnimeIDs.length > 0}
    <div>
      <label class="bx--label">Excluded Rankings</label>
      <div class="tags-container">
        {#each $params.excludedRankingAnimeIDs as animeID (animeID)}
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
  {#if isLoading}
    <Loading withOverlay={false} small />
  {/if}
</div>

<style lang="css">
  .root {
    display: flex;
    flex-direction: column;
    min-height: 250px;
    gap: 14px;
  }

  .top > div {
    display: flex;
    flex: 1;
    flex-direction: column;
  }

  .tags-container {
    display: flex;
    flex-direction: row;
    flex-wrap: wrap;
  }
</style>
