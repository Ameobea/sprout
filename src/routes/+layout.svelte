<script lang="ts">
  import 'carbon-components-svelte/css/g100.css';
  import { maybeInitSentry } from '../sentry.js';
  import { maybeInitAnalytics } from '../analytics';

  import '../index.css';

  maybeInitSentry();
  maybeInitAnalytics();
</script>

<slot />

<style lang="css">
</style>
