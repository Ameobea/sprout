<script lang="ts" context="module">
  export const prerender = true;
</script>

<script lang="ts">
  import Atlas from 'src/components/Atlas.svelte';
  import type { Embedding } from './index';
  import '../index.css';

  export let embedding: Embedding;
</script>

<Atlas {embedding} />
