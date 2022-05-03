<script lang="ts" context="module">
  import type { LoadInput } from '@sveltejs/kit/types/private';

  export const prerender = true;

  export async function load({ fetch }: LoadInput) {
    const { embedding } = await fetch('/embedding?embedding=ggvec').then(
      (res) => res.json() as Promise<{ embedding: Embedding }>
    );
    return { props: { embedding } };
  }
</script>

<script lang="ts">
  import Atlas from 'src/components/Atlas.svelte';
  import type { Embedding } from './embedding';
  import '../index.css';

  export let embedding: Embedding;
</script>

<Atlas {embedding} />
