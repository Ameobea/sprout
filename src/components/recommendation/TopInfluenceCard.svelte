<script lang="ts">
  import 'hint.css/hint.css';

  export let name: string;
  export let onDelete: () => void;
</script>

<div class="root">
  <div class="title"><h3>{name}</h3></div>
  <div>
    <button
      class="hint--top hint--no-animate hint--medium"
      aria-label="Your rating of this anime will be ignored and it won't be treated as input for recommendations"
      on:click={onDelete}
    >
      Exclude from Consideration
    </button>
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
    height: 30px;
  }

  .title {
    text-align: center;
  }
</style>
