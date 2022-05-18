<script lang="ts">
  import 'carbon-components-svelte/css/g100.css';

  import Header from 'src/components/recommendation/Header.svelte';
  import { HeaderTitle } from 'src/components/recommendation/HeaderTitle';
</script>

<Header content={$HeaderTitle} />
<div class="content">
  <slot foo="bar" />
</div>

<style lang="css">
  .content {
    display: flex;
    flex-direction: column;
    max-width: calc(min(100vw, 850px));
    margin: 0 auto;
    background: #141414;
    min-height: 100vh;
  }
</style>
