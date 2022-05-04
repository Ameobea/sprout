<script lang="ts" context="module">
  export const prerender = true;
</script>

<script lang="ts">
  import Atlas from 'src/components/Atlas.svelte';
  import type { Embedding } from './embedding';
  import '../index.css';

  export let embedding: Embedding;
</script>

<Atlas {embedding} />
