<script lang="ts">
  import { browser } from '$app/env';
  import * as Sentry from '@sentry/browser';
  import { Integrations } from '@sentry/tracing';
  import 'carbon-components-svelte/css/g100.css';

  import '../index.css';

  if (browser) {
    if (!window.location.href.includes('localhost')) {
      Sentry.init({
        dsn: 'https://0cbd7f7ac2ca48fe8fcc17cd8ba73c1d@sentry.ameo.design/12',
        integrations: [new Integrations.BrowserTracing()],

        tracesSampleRate: 1.0,
      });
    }
  }
</script>

<slot />

<style lang="css">
</style>
