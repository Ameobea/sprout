<script context="module" lang="ts">
  enum UserTab {
    Recommendations = 0,
    Stats = 1,
    Atlas = 2,
  }

  const getActiveTab = (url: URL) => {
    const pathname = url.pathname;
    if (pathname.endsWith('/recommendations')) {
      return UserTab.Recommendations;
    } else if (pathname.endsWith('/stats')) {
      return UserTab.Stats;
    } else if (pathname.endsWith('/atlas')) {
      return UserTab.Atlas;
    } else {
      return UserTab.Recommendations;
    }
  };

  const getTabPath = (username: string, tab: UserTab) => {
    switch (tab) {
      case UserTab.Recommendations:
        return `/user/${username}/recommendations`;
      case UserTab.Stats:
        return `/user/${username}/stats`;
      case UserTab.Atlas:
        return `/user/${username}/atlas`;
      default:
        return `/user/${username}/recommendations`;
    }
  };
</script>

<script lang="ts">
  import { goto, prefetch } from '$app/navigation';
  import { page } from '$app/stores';
  import { Tabs, Tab } from 'carbon-components-svelte';

  import Header from 'src/components/recommendation/Header.svelte';

  $: activeTab = getActiveTab($page.url);
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
    goto(getTabPath(username, newSelectedTab));
  };

  const mkTabPrefetcher = (tab: UserTab) => () => {
    const path = getTabPath(username, tab);
    prefetch(path);
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
