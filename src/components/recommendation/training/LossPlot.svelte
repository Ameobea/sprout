<script lang="ts">
  import UPlot from 'uplot';
  import 'uplot/dist/uPlot.min.css';

  export let losses: number[];
  export let iters: number;

  let uPlotInst: UPlot | null = null;

  $: {
    if (uPlotInst) {
      uPlotInst.setData([new Array<number>(iters).fill(0).map((_, i) => i), losses]);
    }
  }

  const renderChart = (containerNode: HTMLElement) => {
    uPlotInst = new UPlot({
      width: window.innerWidth / 2 - 20,
      height: 600,
      series: [
        {},
        {
          label: 'Loss',
          stroke: 'red',
          // value: (u, v) => v.toFixed(2),
        },
      ],
      axes: [
        {},
        {
          show: true,
          label: 'MSE',
          stroke: '#ccc',
          ticks: { show: true, stroke: '#ccc', width: 1 },
          grid: { show: true, stroke: '#cccccc88', width: 1 },
          // values: (u, vals, space) => vals.map((v) => (+v).toFixed(3)),
        },
      ],
      scales: { x: { time: false }, y: { log: 10, distr: 3 } },
      id: 'loss-plot',
    });
    uPlotInst.setData([losses.map((_, i) => i), losses]);
    containerNode.appendChild(uPlotInst.root);
  };
</script>

<div class="root" use:renderChart />

<style lang="css">
</style>
