<script lang="ts" context="module">
  import { ModelName } from './conf';

  const ALL_MODEL_OPTIONS: { id: ModelName; text: string }[] = [
    { id: ModelName.Model_4K, text: 'Top 4k Anime v1' },
    { id: ModelName.Model_4K_V2, text: 'Top 4k Anime w. more loss weights' },
  ];
</script>

<script lang="ts">
  import { Dropdown, Tag } from 'carbon-components-svelte';
  import type { Writable } from 'svelte/store';

  import type { RecommendationControlParams } from './InteractiveRecommendations.svelte';
  import type { AnimeDetails } from 'src/malAPI';

  export let state: Writable<RecommendationControlParams>;
  export let getAnimeMetadata: (animeId: number) => AnimeDetails;
</script>

<div class="root">
  <div class="top">
    <div>
      <Dropdown style="width: 100%;" titleText="Model" bind:selectedId={$state.modelName} items={ALL_MODEL_OPTIONS} />
    </div>
    <div />
  </div>
  {#if $state.excludedRankingAnimeIDs.length > 0}
    <div>
      <label class="bx--label">Excluded Rankings</label>
      <div class="tags-container">
        {#each $state.excludedRankingAnimeIDs as animeID (animeID)}
          {@const datum = getAnimeMetadata(animeID)}
          <Tag
            filter
            on:close={() =>
              state.update((state) => {
                state.excludedRankingAnimeIDs = state.excludedRankingAnimeIDs.filter(
                  (oAnimeID) => oAnimeID !== animeID
                );
                return state;
              })}
            type="red"
          >
            {datum.title}
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