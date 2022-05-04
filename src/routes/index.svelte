<script lang="ts" context="module">
  export const prerender = true;
</script>

<script lang="ts">
  import Atlas from 'src/components/Atlas.svelte';
  import type { Embedding } from './embedding';
  import '../index.css';
  import type { EmbeddingName } from 'src/types';

  export let embedding: Embedding;
  export let embeddingName: EmbeddingName;
</script>

<Atlas {embedding} {embeddingName} />
