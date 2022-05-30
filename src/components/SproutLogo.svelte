<script lang="ts">
  export let style: string | undefined = undefined;
</script>

<img
  src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAJNJREFUOI1jYKAQMOKTzNpn8B9dbJrTBRQ9eA1gYGBgiFujATdkUcgNguoxwMynARiuQAYsMAa6c6c5XWBE1oxNnoEBzQvIzuUR4mDQV1dguHjzAQMDAwPDl3c/4OqQvcKCpB9FAmYYTCNN/M/AwMDARLKp1DYAw1/3C9VwOlux/xaGeqwBg80QbJoZGAZDGAy8AQCyjjMqMORWWgAAAABJRU5ErkJggg=="
  alt="Sprout logo showing a small pixel art plant sprouting out of an orange plant pot"
  class="root"
  {style}
/>

<style>
  .root {
    display: inline;
    width: 38px;
    height: 38px;
    margin-right: -8px;
    margin-bottom: -1px;
    image-rendering: pixelated;
  }

  @media (max-width: 768px) {
    .root {
      transform: scale(0.8);
      transform-origin: bottom center;
    }
  }
</style>
