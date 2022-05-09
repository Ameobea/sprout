<script lang="ts">
  export let name: string;
  export let onDelete: () => void;
</script>

<div class="root">
  <div class="title"><h3>{name}</h3></div>
  <div>
    <button on:click={onDelete}>Exclude from Consideration</button>
  </div>
</div>

<style lang="css">
  .root {
    display: flex;
    flex: 1;
    flex-direction: column;
    box-sizing: border-box;
  }

  h3 {
    margin: 0;
  }

  button {
    cursor: pointer;
  }

  .title {
    text-align: center;
  }
</style>
