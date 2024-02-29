<script context="module" lang="ts">
  const buildRatingDistributionChartData = (profile: PartialStatsMALUserAnimeListItem[]) => {
    const ratingAppearanceCounts = profile.reduce((acc, item) => {
      acc[item.list_status.score] += 1;
      return acc;
    }, new Array<number>(11).fill(0));

    return ratingAppearanceCounts.slice(1).map((count, index) => ({
      group: `${index + 1}`,
      value: count,
    }));
  };
</script>

<script lang="ts">
  import '@carbon/charts-svelte/styles.css';
  import { BarChartSimple } from '@carbon/charts-svelte';
  import type { PartialStatsMALUserAnimeListItem } from 'src/routes/user/[username]/stats/+page.server';

  export let profile: PartialStatsMALUserAnimeListItem[];

  $: genresBarChartData = buildRatingDistributionChartData(profile);
  $: options = {
    title: 'Rating Distribution',
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
