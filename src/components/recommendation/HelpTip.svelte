<script lang="ts">
  import { onDestroy, tick } from 'svelte';
  import Information from 'carbon-icons-svelte/lib/Information.svelte';

  export let label: string;
  export let text: string;

  const id = `recommendation-help-${Math.random().toString(36).slice(2)}`;
  let root: HTMLSpanElement;
  let trigger: HTMLButtonElement;
  let bubble: HTMLSpanElement;
  let open = false;
  let pinned = false;
  let left = 16;
  let top = 16;
  let leaveTimer: ReturnType<typeof setTimeout> | undefined;

  const show = async () => {
    clearTimeout(leaveTimer);
    open = true;
    await tick();
    if (!open || !bubble || !trigger) return;
    const anchor = trigger.getBoundingClientRect();
    const { width, height } = bubble.getBoundingClientRect();
    left = Math.max(16, Math.min(anchor.right - width, window.innerWidth - width - 16));
    top = anchor.bottom + 8;
    if (top + height > window.innerHeight - 16) top = Math.max(16, anchor.top - height - 8);
  };

  const close = () => {
    clearTimeout(leaveTimer);
    open = false;
    pinned = false;
  };

  const leave = () => {
    if (!pinned) leaveTimer = setTimeout(close, 150);
  };
  const onOutsideClick = (event: MouseEvent) => {
    if (open && event.target instanceof Node && !root?.contains(event.target)) close();
  };
  const onOutsideFocus = (event: FocusEvent) => {
    if (open && event.target instanceof Node && !root?.contains(event.target)) close();
  };
  onDestroy(() => clearTimeout(leaveTimer));
</script>

<svelte:window
  on:keydown={(event) => {
    if (event.key === 'Escape') close();
  }}
  on:click={onOutsideClick}
  on:focusin={onOutsideFocus}
  on:resize={close}
  on:scroll={close}
/>

<span class="help" bind:this={root} on:pointerenter={show} on:pointerleave={leave}>
  <button
    bind:this={trigger}
    type="button"
    aria-label={`About ${label}`}
    aria-describedby={open ? id : undefined}
    aria-expanded={open}
    aria-controls={open ? id : undefined}
    on:focus={show}
    on:blur={leave}
    on:click={() => {
      if (pinned) close();
      else {
        pinned = true;
        show();
      }
    }}><Information size={16} aria-hidden="true" /></button
  >
  {#if open}
    <span class="bubble" role="tooltip" {id} bind:this={bubble} style:left={`${left}px`} style:top={`${top}px`}>
      <strong>{label}</strong>
      <span>{text}</span>
    </span>
  {/if}
</span>

<style>
  .help {
    display: inline-flex;
    flex: none;
  }
  button {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 30px;
    height: 30px;
    padding: 0;
    border: 0;
    border-radius: 50%;
    background: transparent;
    color: #99a69f;
    cursor: pointer;
  }
  button:hover,
  button[aria-expanded='true'] {
    color: #8ce295;
    background: #6bce7b15;
  }
  button:focus-visible {
    outline: 2px solid #75dc82;
    outline-offset: 2px;
  }
  .bubble {
    position: fixed;
    z-index: 10000;
    display: flex;
    flex-direction: column;
    gap: 7px;
    width: 300px;
    max-width: calc(100vw - 32px);
    max-height: calc(100dvh - 32px);
    overflow-y: auto;
    box-sizing: border-box;
    padding: 14px 16px;
    border-radius: 10px;
    border: 1px solid #506258;
    background: #242d28;
    color: #d0dbd4;
    box-shadow: 0 8px 32px #0009;
    font-size: 13px;
    line-height: 1.5;
    font-weight: 400;
    text-align: left;
    white-space: normal;
  }
  strong {
    color: #f3f7f4;
    font-size: 13px;
    font-weight: 600;
  }
  @media (prefers-reduced-motion: no-preference) {
    button {
      transition:
        background 120ms,
        color 120ms;
    }
  }
</style>
