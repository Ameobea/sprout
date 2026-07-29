<script lang="ts">
  import 'carbon-components-svelte/css/g100.css';
  import { maybeInitSentry } from 'src/sentry';
  import { maybeInitAnalytics } from 'src/analytics';

  import '../index.css';

  maybeInitSentry();
  maybeInitAnalytics();
</script>

<slot />

<style lang="css">
</style>
