<script lang="ts" context="module">
  enum LogLevel {
    Debug = 'debug',
    Info = 'info',
    Warn = 'warn',
    Error = 'error',
  }

  interface LogLine {
    id: number;
    text: string;
    timestamp: Date;
    level: LogLevel;
  }

  let logLineIDCounter = 0;
</script>

<script lang="ts">
  import { onMount } from 'svelte';
  import { writable } from 'svelte/store';

  import type { Logger } from 'src/training/trainRecommender';
  import LossPlot from 'src/components/recommendation/training/LossPlot.svelte';

  let logLines = writable([] as LogLine[]);
  let losses = writable([] as number[]);
  const iters = 250;

  const logger: Logger = (() => {
    function logger(text: string) {
      console.log(text);
      logLines.update((lines) => {
        lines.push({
          id: logLineIDCounter++,
          text,
          timestamp: new Date(),
          level: LogLevel.Info,
        });
        return lines;
      });
    }
    logger.warn = (text: string) => {
      logLines.update((lines) => {
        lines.push({
          id: logLineIDCounter++,
          text,
          timestamp: new Date(),
          level: LogLevel.Warn,
        });
        return lines;
      });
    };
    logger.error = (text: string) => {
      logLines.update((lines) => {
        lines.push({
          id: logLineIDCounter++,
          text,
          timestamp: new Date(),
          level: LogLevel.Error,
        });
        return lines;
      });
    };
    return logger;
  })();

  onMount(async () => {
    logger('Initializing training...');
    const { trainRecommender } = await import('src/training/trainRecommender');
    await trainRecommender(iters, logger, (loss) => {
      losses.update((losses) => {
        losses.push(loss);
        return losses;
      });
    });
  });
</script>

<div class="root">
  <div class="console">
    {#each $logLines as line (line.id)}
      <div class={`log-line ${line.level}`}>{line.text}</div>
    {/each}
  </div>
  <LossPlot {iters} losses={$losses} />
</div>

<style lang="css">
  .root {
    display: flex;
    flex-direction: row;
  }

  .console {
    max-height: 100vh;
    min-height: 800px;
    width: calc(max(600px, 50vw));
    display: flex;
    flex-direction: column;
    padding: 2px 4px;
    box-sizing: border-box;
    font: 14px 'Input', 'Oxygen Mono', 'IBM Plex Mono', 'Consolas', monospace;
    overflow-y: auto;
    background-color: #121212;
    border: 1px solid #888;
  }

  .log-line {
    /* preserve newlines */
    white-space: pre-wrap;
  }

  .log-line.warn {
    color: #ffa500;
  }

  .log-line.error {
    color: #ff0000;
  }
</style>
