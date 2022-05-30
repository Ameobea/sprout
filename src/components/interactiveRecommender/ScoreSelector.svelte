<script context="module" lang="ts">
  export enum Score {
    Amazing = 10,
    Good = 7,
    Meh = 5,
    Bad = 2,
  }
</script>

<script lang="ts">
  export let score: Score = Score.Good;
</script>

<div class="root">
  <div
    class="score-item"
    aria-selected={score === Score.Amazing}
    style="background-color: rgb(60, 163, 16);"
    on:click={() => {
      score = Score.Amazing;
    }}
  >
    Amazing
  </div>
  <div
    class="score-item"
    aria-selected={score === Score.Good}
    style="background-color: rgb(81, 106, 18);"
    on:click={() => {
      score = Score.Good;
    }}
  >
    Good
  </div>
  <div
    class="score-item"
    aria-selected={score === Score.Meh}
    style="background-color: rgb(143, 109, 32);"
    on:click={() => {
      score = Score.Meh;
    }}
  >
    Meh
  </div>
  <div
    class="score-item"
    aria-selected={score === Score.Bad}
    style="background-color: rgb(124, 22, 22);"
    on:click={() => {
      score = Score.Bad;
    }}
  >
    Bad
  </div>
</div>

<style lang="css">
  .root {
    display: flex;
    flex-direction: column;
    width: 120px;
  }

  .score-item {
    color: white;
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: center;
    justify-self: center;
    align-self: center;
    width: 100%;
    height: 50px;
    border: 1px solid #ccc;
    cursor: pointer;
  }

  .score-item[aria-selected='false'] {
    background-color: transparent !important;
  }
</style>
