<script lang="ts">
  export let content: string;
</script>

<div class="root">
  <h1>{content}</h1>
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
  }
</style>
