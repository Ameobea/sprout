<script lang="ts">
  import Atlas from 'src/components/Atlas.svelte';
  import type { PageData } from './$types';

  export let data: PageData;
</script>

<Atlas embedding={data.embedding} embeddingName={data.embeddingName} />
