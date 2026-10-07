<script lang="ts" context="module">
  import { getDefaultNicheBoostFactor, ModelName, PopularityAttenuationFactor, ProfileSource } from './conf';

  const ALL_MODEL_OPTIONS: { id: ModelName; text: string }[] = [
    { id: ModelName.Model_2026_logq, text: 'Aug. 2026' },
    { id: ModelName.Model_2026_rc, text: 'Aug. 2026 v2 (Beta)' },
    { id: ModelName.Model_2025_jax, text: 'Dec. 2025' },
    { id: ModelName.Legacy_2023, text: 'Legacy (2023)' },
  ];

  const ALL_POPULARITY_ATTENUATION_FACTOR_OPTIONS: { id: PopularityAttenuationFactor; text: string }[] = [
    { id: PopularityAttenuationFactor.None, text: 'None' },
    { id: PopularityAttenuationFactor.VeryLow, text: 'Very Low' },
    { id: PopularityAttenuationFactor.Low, text: 'Low' },
    { id: PopularityAttenuationFactor.Medium, text: 'Medium' },
    { id: PopularityAttenuationFactor.High, text: 'High' },
    { id: PopularityAttenuationFactor.VeryHigh, text: 'Very High' },
  ];
</script>

<script lang="ts">
  import { Dropdown, InlineLoading, Tag, Toggle, Slider } from 'carbon-components-svelte';
  import SettingsAdjust from 'carbon-icons-svelte/lib/SettingsAdjust.svelte';
  import Renew from 'carbon-icons-svelte/lib/Renew.svelte';
  import Layers from 'carbon-icons-svelte/lib/Layers.svelte';
  import Video from 'carbon-icons-svelte/lib/Video.svelte';
  import Playlist from 'carbon-icons-svelte/lib/Playlist.svelte';
  import Music from 'carbon-icons-svelte/lib/Music.svelte';
  import ChevronDown from 'carbon-icons-svelte/lib/ChevronDown.svelte';
  import HelpTip from './HelpTip.svelte';
  import type { Writable } from 'svelte/store';

  import type { AnimeDetails } from 'src/malAPI';
  import { getSurface, submitAnalyticsEvent } from 'src/analytics';
  import type { RecommendationControlParams } from './utils';

  export let params: Writable<RecommendationControlParams>;
  export let animeMetadataDatabase: { [animeID: number]: AnimeDetails };
  export let isLoading: boolean;
  export let forceHideTopBar: boolean | undefined = false;
  /**
   * Hides the presence/rating weight slider entirely for non-rater profiles, where rating
   * predictions are too unreliable for the control to be meaningful; the server scores those
   * profiles presence-only regardless of the param.
   */
  export let hideLogitWeight = false;
  export let onForceProfileRefresh: (() => void) | undefined = undefined;
  export let isProfileRefreshing = false;
  export let profileRefreshError = '';

  // Local state for sliders to prevent updates while dragging
  let localLogitWeight = $params.logitWeight;
  let localNicheBoostFactor = $params.nicheBoostFactor;

  const submitFilterToggle = (filter: string, enabled: boolean) =>
    submitAnalyticsEvent({
      category: 'recommendations',
      subcategory: 'filter_toggle',
      payload: { filter, enabled, surface: getSurface() },
    });
</script>

<details class="root">
  <summary class="panel-summary">
    <h2 class="section-title">
      <span class="heading-icon"><SettingsAdjust size={20} aria-hidden="true" /></span>
      <span>
        <span>Recommendation settings</span>
        <span class="section-description">Choose what to include</span>
      </span>
    </h2>
    <ChevronDown size={20} aria-hidden="true" />
  </summary>
  <div class="panel-body">
    {#if $params.profileSource === ProfileSource.MyAnimeList && onForceProfileRefresh}
      <div class="profile-refresh">
        <button
          type="button"
          disabled={isProfileRefreshing || isLoading || $params.modelName === ModelName.Legacy_2023}
          on:click={() => onForceProfileRefresh?.()}
        >
          <Renew size={16} aria-hidden="true" />
          {isProfileRefreshing ? 'Refreshing…' : 'Refresh MyAnimeList'}
        </button>
        <HelpTip
          label="MyAnimeList refresh"
          text={$params.modelName === ModelName.Legacy_2023
            ? 'Force refresh is unavailable for Legacy (2023), which is served by a separate legacy deployment.'
            : 'Sprout caches MyAnimeList lists for 60 seconds after fetching them; an open page does not update automatically. Refresh bypasses Sprout’s cache and updates recommendations using the list currently returned by MAL. Changes still depend on MAL making them available through its API.'}
        />
      </div>
    {/if}
    {#if profileRefreshError}<p class="refresh-error" role="alert">{profileRefreshError}</p>{/if}
    <div class="toggles">
      <div class:enabled={$params.includeExtraSeasons}>
        <div class="toggle-heading">
          <Layers size={20} aria-hidden="true" /><span>Extra Seasons</span><HelpTip
            label="Extra Seasons"
            text="Off hides TV/unknown recommendations connected through sequel, prequel, parent-story, or side-story relationships to anime you already watched. Other formats have their own switches. On allows these extra seasons/stories."
          />
        </div>
        <Toggle
          labelText="Extra Seasons"
          size="sm"
          hideLabel
          bind:toggled={$params.includeExtraSeasons}
          on:toggle={(evt) => submitFilterToggle('extra_seasons', evt.detail.toggled)}
        />
      </div>
      <div class:enabled={$params.includeMovies}>
        <div class="toggle-heading">
          <Video size={20} aria-hidden="true" /><span>Movies</span><HelpTip
            label="Movies"
            text="Off excludes MAL entries whose format is Movie. On allows movies."
          />
        </div>
        <Toggle
          labelText="Movies"
          size="sm"
          hideLabel
          bind:toggled={$params.includeMovies}
          on:toggle={(evt) => submitFilterToggle('movies', evt.detail.toggled)}
        />
      </div>
      <div class:enabled={$params.includeONAsOVAsSpecials}>
        <div class="toggle-heading">
          <Playlist size={20} aria-hidden="true" /><span>ONAs / OVAs / Specials</span><HelpTip
            label="ONAs / OVAs / Specials"
            text="Off excludes ONA, OVA, Special, and TV Special entries. On allows these formats."
          />
        </div>
        <Toggle
          labelText="ONAs / OVAs / Specials"
          size="sm"
          hideLabel
          bind:toggled={$params.includeONAsOVAsSpecials}
          on:toggle={(evt) => submitFilterToggle('onas_ovas_specials', evt.detail.toggled)}
        />
      </div>
      <div class:enabled={$params.includeMusic}>
        <div class="toggle-heading">
          <Music size={20} aria-hidden="true" /><span>Music</span><HelpTip
            label="Music"
            text="Off excludes Music, commercials (CM), and promotional videos (PV). On allows them."
          />
        </div>
        <Toggle
          labelText="Music"
          size="sm"
          hideLabel
          bind:toggled={$params.includeMusic}
          on:toggle={(evt) => submitFilterToggle('music', evt.detail.toggled)}
        />
      </div>
    </div>
    {#if isLoading}<div class="loading-status"><InlineLoading description="Updating recommendations…" /></div>{/if}
    {#if !forceHideTopBar}
      <details
        class="advanced-options"
        on:toggle={() =>
          submitAnalyticsEvent({
            category: 'recommendations',
            subcategory: 'advanced_options_toggle',
            payload: { surface: getSurface() },
          })}
      >
        <summary
          ><span>Advanced Options</span><span class="model-caption"
            >{ALL_MODEL_OPTIONS.find((model) => model.id === $params.modelName)?.text}</span
          ><ChevronDown size={16} aria-hidden="true" /></summary
        >
        <div class="top">
          <div class="top-row">
            <div>
              <div class="field-heading">
                <span>Model</span><HelpTip
                  label="Model"
                  text="Chooses the trained recommendation version. Aug. 2026 is the default, v2 is experimental, Dec. 2025 is the previous model, and Legacy uses the older 2023 serving path. Changing model can change scores and ranking order."
                />
              </div>
              <Dropdown
                style="width: 100%;"
                titleText="Model"
                hideLabel
                selectedId={$params.modelName}
                on:select={(selected) => {
                  const model = selected.detail.selectedItem.id;
                  if (model !== $params.modelName) {
                    submitAnalyticsEvent({
                      category: 'recommendations',
                      subcategory: 'model_select',
                      payload: { model, surface: getSurface() },
                    });
                    if ($params.nicheBoostFactor === getDefaultNicheBoostFactor($params.modelName)) {
                      $params.nicheBoostFactor = getDefaultNicheBoostFactor(model);
                      localNicheBoostFactor = getDefaultNicheBoostFactor(model);
                    }
                  }
                  $params.modelName = model;
                }}
                items={ALL_MODEL_OPTIONS}
              />
            </div>
            <div>
              <div class="field-heading">
                <span>Filter Plan to Watch</span><HelpTip
                  label="Filter Plan to Watch"
                  text="On removes anime already marked Plan to Watch in your profile. Off keeps them eligible and labels them in the results."
                />
              </div>
              <Toggle
                labelText="Filter Plan to Watch"
                hideLabel
                size="sm"
                bind:toggled={$params.filterPlanToWatch}
                on:toggle={(evt) => submitFilterToggle('plan_to_watch', evt.detail.toggled)}
              />
            </div>
          </div>
          <div class="bottom-row">
            {#if $params.modelName === ModelName.Legacy_2023}
              <div>
                <div class="field-heading">
                  <span>Popularity attenuation</span><HelpTip
                    label="Popularity attenuation"
                    text="Legacy model only. Higher values increasingly favor less-popular anime; None preserves the legacy ranking without this adjustment."
                  />
                </div>
                <Dropdown
                  style="width: 100%;"
                  titleText="Popularity Attenuation Factor"
                  hideLabel
                  selectedId={$params.popularityAttenuationFactor}
                  on:select={(selected) => {
                    const value = selected.detail.selectedItem.id;
                    if (value !== $params.popularityAttenuationFactor) {
                      submitAnalyticsEvent({
                        category: 'recommendations',
                        subcategory: 'attenuation_change',
                        payload: { value, surface: getSurface() },
                      });
                    }
                    $params.popularityAttenuationFactor = value;
                  }}
                  items={ALL_POPULARITY_ATTENUATION_FACTOR_OPTIONS}
                />
              </div>
            {:else}
              {#if !hideLogitWeight}
                <div>
                  <div class="field-heading">
                    <span>Presence / Rating Weight</span><HelpTip
                      label="Presence / Rating Weight"
                      text="0 prioritizes your predicted 1–10 rating; 1 prioritizes how strongly the model expects the anime to belong in your profile. Intermediate values blend both signals."
                    />
                  </div>
                  <Slider
                    labelText="Presence/Rating Weight"
                    hideLabel
                    min={0}
                    max={1}
                    step={0.1}
                    bind:value={localLogitWeight}
                    on:change={(evt) => {
                      if ($params.logitWeight !== evt.detail) {
                        submitAnalyticsEvent({
                          category: 'recommendations',
                          subcategory: 'logit_weight_change',
                          payload: { value: evt.detail, surface: getSurface() },
                        });
                      }
                      $params.logitWeight = evt.detail;
                    }}
                  />
                  <div class="slider-captions"><span>Predicted rating</span><span>Profile match</span></div>
                </div>
              {/if}
              <div>
                <div class="field-heading">
                  <span>Niche Boost Factor</span><HelpTip
                    label="Niche Boost Factor"
                    text="0 adds no niche boost. Higher values increasingly favor anime the model predicts for you more strongly than their overall popularity would suggest."
                  />
                </div>
                <Slider
                  labelText="Niche Boost Factor"
                  hideLabel
                  min={0}
                  max={1}
                  step={0.1}
                  bind:value={localNicheBoostFactor}
                  on:change={(evt) => {
                    if ($params.nicheBoostFactor !== evt.detail) {
                      submitAnalyticsEvent({
                        category: 'recommendations',
                        subcategory: 'niche_boost_change',
                        payload: { value: evt.detail, surface: getSurface() },
                      });
                    }
                    $params.nicheBoostFactor = evt.detail;
                  }}
                />
                <div class="slider-captions"><span>No boost</span><span>More discovery</span></div>
              </div>
            {/if}
          </div>
        </div>
      </details>
    {/if}
    {#if $params.excludedRankingAnimeIDs.length > 0}
      <div>
        <label for="excluded-rankings" class="bx--label">Excluded Rankings</label>
        <div class="tags-container" id="excluded-rankings">
          {#each [...new Set($params.excludedRankingAnimeIDs)] as animeID (animeID)}
            {@const datum = animeMetadataDatabase[animeID]}
            {@const title = datum?.alternative_titles.en || datum?.title || ''}
            <Tag
              filter
              on:close={() => {
                submitAnalyticsEvent({
                  category: 'recommendations',
                  subcategory: 'exclude_ranking_remove',
                  payload: { anime_id: animeID, surface: getSurface() },
                });
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
  </div>
</details>

<style lang="css">
  .root {
    padding: 20px;
    border: 1px solid #ffffff18;
    border-radius: 12px;
    background: #171c19;
  }
  .panel-summary,
  .section-title,
  .profile-refresh {
    display: flex;
    align-items: center;
  }
  .panel-summary {
    justify-content: space-between;
    gap: 16px;
    cursor: pointer;
    list-style: none;
  }
  .panel-summary::-webkit-details-marker {
    display: none;
  }
  .panel-summary > :global(svg) {
    flex: none;
    color: #9caca2;
  }
  .root[open] > .panel-summary > :global(svg) {
    transform: rotate(180deg);
  }
  .panel-body {
    display: flex;
    flex-direction: column;
    gap: 16px;
    margin-top: 16px;
  }
  .section-title {
    gap: 12px;
  }
  .heading-icon {
    display: flex;
    padding: 10px;
    border-radius: 10px;
    background: #77d58512;
    color: #88da95;
  }
  h2 {
    color: #edf3ef;
    font-size: 16px;
    line-height: 1.4;
    font-weight: 600;
    margin: 0;
  }
  .section-description {
    display: block;
    margin: 3px 0 0;
    font-size: 12px;
    font-weight: 400;
    line-height: 1.4;
    color: #99a69f;
  }
  .toggles {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
  }
  .toggles > div {
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 12px;
    border: 1px solid #ffffff15;
    border-radius: 8px;
    background: #202622;
    min-width: 0;
  }
  .toggles > div.enabled {
    border-color: #77d58550;
    background: #233429;
  }
  .toggle-heading {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    min-height: 30px;
    color: #c8d5cd;
  }
  .toggle-heading > span {
    flex: 1;
    padding-top: 6px;
    font-size: 12px;
    font-weight: 500;
    line-height: 1.4;
  }
  .toggle-heading > :global(svg) {
    flex: none;
    margin-top: 5px;
    color: #8fa69a;
  }
  .toggles :global(.bx--toggle-input__label) {
    font-size: 12px;
    color: #c8d5cd;
  }
  .toggles :global(.bx--form-item) {
    flex: none;
    margin-top: auto;
  }
  .profile-refresh {
    gap: 2px;
    align-self: flex-end;
  }
  .profile-refresh button {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    border: 1px solid #ffffff26;
    border-radius: 7px;
    background: #252e28;
    color: #e0eae3;
    padding: 10px 12px;
    cursor: pointer;
    font-size: 12px;
    line-height: 1.4;
  }
  .profile-refresh button:hover:not(:disabled) {
    border-color: #77d58570;
    background: #2c3930;
  }
  .profile-refresh button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  .profile-refresh button:focus-visible,
  summary:focus-visible {
    outline: 2px solid #75dc82;
    outline-offset: 3px;
  }
  .refresh-error {
    padding: 10px 12px;
    margin: 0;
    color: #ffb2b6;
    background: #ff83890c;
    border-radius: 6px;
    font-size: 13px;
    line-height: 1.5;
    overflow-wrap: anywhere;
  }
  .loading-status {
    min-width: 0;
  }
  .advanced-options {
    border-top: 1px solid #ffffff14;
  }
  .advanced-options > summary {
    display: flex;
    align-items: center;
    gap: 10px;
    cursor: pointer;
    list-style: none;
    padding: 16px 0 0;
    color: #c8d5cd;
    font-size: 13px;
  }
  .advanced-options > summary::-webkit-details-marker {
    display: none;
  }
  .advanced-options > summary > span:first-child {
    flex: 1;
  }
  .model-caption {
    font-size: 11px;
    color: #9caca2;
    background: #ffffff08;
    padding: 4px 8px;
    border-radius: 4px;
  }
  .advanced-options[open] > summary :global(svg) {
    transform: rotate(180deg);
  }
  .top {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
    padding-top: 16px;
  }
  .top-row,
  .bottom-row {
    display: contents;
  }
  .top-row > div,
  .bottom-row > div {
    display: flex;
    flex-direction: column;
    min-width: 0;
    padding: 12px 14px;
    border: 1px solid #ffffff10;
    border-radius: 8px;
    background: #1d2420;
    gap: 10px;
  }
  .field-heading {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    min-height: 30px;
    color: #c8d5cd;
    font-size: 13px;
    line-height: 1.4;
  }
  .top :global(.bx--slider-container) {
    min-width: 0;
    width: 100%;
  }
  .top :global(.bx--slider) {
    min-width: 0;
    flex: 1;
  }
  .top :global(.bx--slider-text-input) {
    min-width: 52px;
    width: 52px;
    padding: 0 5px;
    border-radius: 4px;
  }
  .top :global(.bx--dropdown) {
    border: 1px solid #ffffff25;
    border-radius: 5px;
    background: #262e29;
  }
  .top :global(.bx--label.bx--visually-hidden) {
    margin: 0;
  }
  .top :global(.bx--toggle-input__label) {
    min-height: 42px;
    display: flex;
    align-items: center;
  }
  .slider-captions {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    font-size: 11px;
    line-height: 1.4;
    color: #93a399;
  }
  .tags-container {
    display: flex;
    flex-wrap: wrap;
  }
  @media (max-width: 1000px) {
    .toggles {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
  @media (max-width: 560px) {
    .root {
      padding: 16px;
    }
    .panel-body {
      gap: 12px;
    }
    .top {
      grid-template-columns: minmax(0, 1fr);
    }
    .toggle-heading {
      gap: 5px;
    }
    .toggle-heading > :global(svg) {
      display: none;
    }
    .toggles > div {
      padding: 9px 10px;
    }
    .profile-refresh {
      width: 100%;
    }
    .profile-refresh button {
      flex: 1;
      justify-content: center;
    }
  }
  @media (max-width: 360px) {
    .toggles {
      grid-template-columns: minmax(0, 1fr);
    }
    .toggle-heading > :global(svg) {
      display: block;
    }
  }
</style>
