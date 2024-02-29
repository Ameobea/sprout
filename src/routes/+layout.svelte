<script lang="ts">
  import 'carbon-components-svelte/css/g100.css';
  import { maybeInitSentry } from '../sentry.js';

  import '../index.css';

  maybeInitSentry();
</script>

<slot />

<style lang="css">
</style>
