<script lang="ts">
  import type { AnimeDetails } from '../malAPI';

  export let id: number;

  let details: { id: number; details: Promise<AnimeDetails> } = {
    id: -1,
    details: new Promise((_resolve) => {}),
  };

  $: {
    if (details.id !== id) {
      details = { id, details: fetch(`/anime?id=${id}`).then((res) => res.json()) };
    }
  }
</script>

<div class="root">
  {#await details.details}
    <div class="details">
      <div class="placeholder-image" width={225} height={332} />
      <div><h2>Loading...</h2></div>
    </div>
    <p class="synopsis">Loading anime info...</p>
  {:then details}
    <div class="details">
      <img src={details.main_picture.medium} width={225} height={332} alt={details.title} />
      <div class="info">
        <h2>{details.title}</h2>
        <p>{details.start_date ?? '-'} - {details.end_date ?? '-'}</p>
      </div>
    </div>
    <p class="synopsis">{details.synopsis}</p>
  {:catch error}
    <p style="color: red">Error loading anime info</p>
  {/await}
</div>

<style lang="css">
  .root {
    padding: 0px;
    box-sizing: border-box;
    background: #222;
    border: 1px solid #444;
    border-radius: 3px;
    position: absolute;
    bottom: 0;
    left: 0;
    width: calc(min(30vw, 320px));
    height: calc(max(40vh, 400px));
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  .details {
    width: 100%;
    display: flex;
    flex-direction: row;
    margin-bottom: 4px;
    gap: 4px;
  }

  .details * {
    flex: 1;
  }

  .info p {
    font-size: 12px;
    font-weight: bold;
  }

  h2 {
    margin-top: 2px;
    margin-bottom: 2px;
    font-size: 16px;
    text-align: center;
    line-height: 17px;
  }

  img,
  .placeholder-image {
    background-color: #161616;
    height: 213px;
    width: 153px;
    display: inline;
    object-fit: scale-down;
  }

  p {
    box-sizing: border-box;
    margin: 0;
    padding: 3px;
  }

  p.synopsis {
    font-size: 13px;
    width: 100%;
  }
</style>
