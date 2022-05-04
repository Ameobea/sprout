<script lang="ts" context="module">
  export const prerender = true;
</script>

<script lang="ts">
  import Atlas from 'src/components/Atlas.svelte';
  import type { Embedding } from './embedding';
  import type { EmbeddingName } from 'src/types';
  import '../index.css';

  export let embedding: Embedding;
  export let embeddingName: EmbeddingName;
</script>

<Atlas {embedding} {embeddingName} />
