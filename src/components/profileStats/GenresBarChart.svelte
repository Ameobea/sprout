<script context="module" lang="ts">
  const buildGenresBarChartData = (
    profile: PartialStatsMALUserAnimeListItem[],
    animeData: { [animeID: number]: PartialStatsAnimeMetadatum }
  ) => {
    const genreAppearanceCounts = profile.reduce((acc, item) => {
      const datum = animeData[item.node.id];
      if (!datum) {
        console.error(`Missing anime data for ${item.node.id}`);
        return acc;
      }

      const tags = datum.tags ?? [];
      for (const tag of tags) {
        if (!acc.has(tag)) {
          acc.set(tag, 0);
        }
        acc.set(tag, acc.get(tag)! + 1);
      }

      return acc;
    }, new Map<string, number>());

    const data: { group: string; value: number }[] = [];
    for (const [genre, count] of genreAppearanceCounts) {
      data.push({ group: genre, value: count });
    }
    data.sort((a, b) => b.value - a.value);
    return data.slice(0, 25);
  };
</script>

<script lang="ts">
  import '@carbon/charts-svelte/styles.css';
  import { BarChartSimple } from '@carbon/charts-svelte';
  import type {
    PartialStatsAnimeMetadatum,
    PartialStatsMALUserAnimeListItem,
  } from 'src/routes/user/[username]/stats/+page.server';

  export let profile: PartialStatsMALUserAnimeListItem[];
  export let animeData: { [animeID: number]: PartialStatsAnimeMetadatum };

  $: genresBarChartData = buildGenresBarChartData(profile, animeData);
  $: options = {
    title: 'Top Tags',
    height: '400px',
    axes: {
      left: { mapsTo: 'value' },
      bottom: { mapsTo: 'group', scaleType: 'labels' as any },
    },
    legend: {
      enabled: false,
    },
    style: {},
    theme: 'g90',
  };
</script>

<BarChartSimple data={genresBarChartData} {options} />
