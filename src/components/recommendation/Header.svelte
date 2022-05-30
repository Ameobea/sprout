<script lang="ts">
  import SproutLogo from 'src/components/SproutLogo.svelte';

  export let content: string;

  let innerWidth: number = 2000;
</script>

<svelte:window bind:innerWidth />

<div class="root">
  <h1>
    <a sveltekit:prefetch href="/">
      <SproutLogo
        style={`
        width: 24px;
        height: 24px;
        ${innerWidth >= 768 ? ' margin-right: 1px; margin-bottom: -3px;' : ' margin-right: -6px'}`}
      />
    </a>{content}
  </h1>
</div>

<style lang="css">
  .root {
    display: flex;
    flex-direction: row;
    justify-content: center;
    border-bottom: 1px solid #cccccc44;
    background: #060606;
  }

  h1 {
    font-size: 22px;
    margin: 0;
    padding: 5px 0;
    text-align: center;
  }

  a {
    text-decoration: none;
  }
</style>
