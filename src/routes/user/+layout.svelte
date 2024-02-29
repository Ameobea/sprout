<script context="module" lang="ts">
  enum UserTab {
    Recommendations = 0,
    Stats = 1,
    Atlas = 2,
  }

  const formatTabName = (tab: UserTab) => {
    switch (tab) {
      case UserTab.Recommendations:
        return 'Recommendations';
      case UserTab.Stats:
        return 'Stats';
      case UserTab.Atlas:
        return 'Atlas';
      default:
        return 'Unknown';
    }
  };

  const getActiveTab = (url: URL) => {
    const pathname = url.pathname;
    if (pathname.includes('/recommendations')) {
      return UserTab.Recommendations;
    } else if (pathname.includes('/stats')) {
      return UserTab.Stats;
    } else if (pathname.includes('/atlas')) {
      return UserTab.Atlas;
    } else {
      return UserTab.Recommendations;
    }
  };

  const getProfileSource = (url: URL): ProfileSource => {
    const queryParams = new URLSearchParams(url.search);
    const source = queryParams.get('source');
    switch (source) {
      case ProfileSource.MyAnimeList:
      case null:
        return ProfileSource.MyAnimeList;
      case ProfileSource.AniList:
        return ProfileSource.AniList;
      default:
        return ProfileSource.MyAnimeList;
    }
  };

  const getTabPath = (username: string, tab: UserTab, profileSource: ProfileSource) => {
    const sourceQuery = profileSource === DEFAULT_PROFILE_SOURCE ? '' : `?source=${profileSource}`;

    switch (tab) {
      case UserTab.Recommendations:
        return `/user/${username}/recommendations${sourceQuery}`;
      case UserTab.Stats:
        return `/user/${username}/stats${sourceQuery}`;
      case UserTab.Atlas:
        return `/user/${username}/atlas${sourceQuery}`;
      default:
        return `/user/${username}/recommendations${sourceQuery}`;
    }
  };
</script>

<script lang="ts">
  import { goto, preloadCode } from '$app/navigation';
  import { page } from '$app/stores';
  import { captureMessage } from 'src/sentry';
  import { Tabs, Tab } from 'carbon-components-svelte';

  import Header from 'src/components/recommendation/Header.svelte';
  import { DEFAULT_PROFILE_SOURCE, ProfileSource } from 'src/components/recommendation/conf';

  $: activeTab = getActiveTab($page.url);
  $: profileSource = getProfileSource($page.url);
  $: username = $page.params.username;
  $: title = `Anime ${(() => {
    switch (activeTab) {
      case UserTab.Recommendations:
        return 'Recommendations';
      case UserTab.Stats:
        return 'Stats';
      case UserTab.Atlas:
        return 'Atlas';
      default:
        return 'Recommendations';
    }
  })()} for ${username}`;

  const handleTabSelected = (evt: any) => {
    const newSelectedTab = evt.detail as UserTab;
    captureMessage(`User page tab click: ${formatTabName(newSelectedTab)}`);
    goto(getTabPath(username, newSelectedTab, profileSource));
  };

  const mkTabPrefetcher = (tab: UserTab) => () => {
    const path = getTabPath(username, tab, profileSource);
    preloadCode(path);
  };
</script>

<div class="content">
  <Header content={title} />

  <Tabs
    autoWidth
    selected={activeTab}
    on:change={handleTabSelected}
    style="margin-left: auto; margin-right: auto; background-color: #060606;"
  >
    <Tab on:mouseenter={mkTabPrefetcher(UserTab.Recommendations)} label="Anime Recommendations" />
    <Tab on:mouseenter={mkTabPrefetcher(UserTab.Stats)} label="Profile Stats + Analytics" />
    <Tab on:mouseenter={mkTabPrefetcher(UserTab.Atlas)} label="Atlas Visualization" />
  </Tabs>

  <slot foo="bar" />
</div>

<style lang="css">
  :global(body) {
    overflow-y: scroll;
    overflow-x: hidden;
  }

  .content {
    display: flex;
    flex-direction: column;
    max-width: calc(min(100vw, 950px));
    margin: 0 auto;
    background: #131313;
    min-height: 100vh;
  }
</style>
