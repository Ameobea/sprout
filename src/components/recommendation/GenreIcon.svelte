<script lang="ts">
  import type { ComponentType } from 'svelte';
  import Flash from 'carbon-icons-svelte/lib/Flash.svelte';
  import Explore from 'carbon-icons-svelte/lib/Explore.svelte';
  import SprayPaint from 'carbon-icons-svelte/lib/SprayPaint.svelte';
  import Trophy from 'carbon-icons-svelte/lib/Trophy.svelte';
  import FaceWink from 'carbon-icons-svelte/lib/FaceWink.svelte';
  import Theater from 'carbon-icons-svelte/lib/Theater.svelte';
  import MagicWand from 'carbon-icons-svelte/lib/MagicWand.svelte';
  import Restaurant from 'carbon-icons-svelte/lib/Restaurant.svelte';
  import Bat from 'carbon-icons-svelte/lib/Bat.svelte';
  import Search from 'carbon-icons-svelte/lib/Search.svelte';
  import Favorite from 'carbon-icons-svelte/lib/Favorite.svelte';
  import Rocket from 'carbon-icons-svelte/lib/Rocket.svelte';
  import Cafe from 'carbon-icons-svelte/lib/Cafe.svelte';
  import Basketball from 'carbon-icons-svelte/lib/Basketball.svelte';
  import Moon from 'carbon-icons-svelte/lib/Moon.svelte';
  import WarningAlt from 'carbon-icons-svelte/lib/WarningAlt.svelte';
  import View from 'carbon-icons-svelte/lib/View.svelte';
  import ViewFilled from 'carbon-icons-svelte/lib/ViewFilled.svelte';
  import Locked from 'carbon-icons-svelte/lib/Locked.svelte';
  import UserAvatar from 'carbon-icons-svelte/lib/UserAvatar.svelte';
  import DogWalker from 'carbon-icons-svelte/lib/DogWalker.svelte';
  import CropGrowth from 'carbon-icons-svelte/lib/CropGrowth.svelte';
  import ParentChild from 'carbon-icons-svelte/lib/ParentChild.svelte';
  import Box from 'carbon-icons-svelte/lib/Box.svelte';
  import UserProfile from 'carbon-icons-svelte/lib/UserProfile.svelte';
  import Warning from 'carbon-icons-svelte/lib/Warning.svelte';
  import Security from 'carbon-icons-svelte/lib/Security.svelte';
  import Education from 'carbon-icons-svelte/lib/Education.svelte';
  import FaceDizzy from 'carbon-icons-svelte/lib/FaceDizzy.svelte';
  import RainDrop from 'carbon-icons-svelte/lib/RainDrop.svelte';
  import Group from 'carbon-icons-svelte/lib/Group.svelte';
  import GameConsole from 'carbon-icons-svelte/lib/GameConsole.svelte';
  import Time from 'carbon-icons-svelte/lib/Time.svelte';
  import Microphone from 'carbon-icons-svelte/lib/Microphone.svelte';
  import MicrophoneFilled from 'carbon-icons-svelte/lib/MicrophoneFilled.svelte';
  import Exit from 'carbon-icons-svelte/lib/Exit.svelte';
  import Sprout from 'carbon-icons-svelte/lib/Sprout.svelte';
  import ChartNetwork from 'carbon-icons-svelte/lib/ChartNetwork.svelte';
  import Shuffle from 'carbon-icons-svelte/lib/Shuffle.svelte';
  import Star from 'carbon-icons-svelte/lib/Star.svelte';
  import Movement from 'carbon-icons-svelte/lib/Movement.svelte';
  import Bot from 'carbon-icons-svelte/lib/Bot.svelte';
  import Hospital from 'carbon-icons-svelte/lib/Hospital.svelte';
  import Notebook from 'carbon-icons-svelte/lib/Notebook.svelte';
  import MilitaryCamp from 'carbon-icons-svelte/lib/MilitaryCamp.svelte';
  import Music from 'carbon-icons-svelte/lib/Music.svelte';
  import IbmWatsonDiscovery from 'carbon-icons-svelte/lib/IbmWatsonDiscovery.svelte';
  import Password from 'carbon-icons-svelte/lib/Password.svelte';
  import Headset from 'carbon-icons-svelte/lib/Headset.svelte';
  import FaceSatisfied from 'carbon-icons-svelte/lib/FaceSatisfied.svelte';
  import Events from 'carbon-icons-svelte/lib/Events.svelte';
  import Fish from 'carbon-icons-svelte/lib/Fish.svelte';
  import Cognitive from 'carbon-icons-svelte/lib/Cognitive.svelte';
  import Car from 'carbon-icons-svelte/lib/Car.svelte';
  import Renew from 'carbon-icons-svelte/lib/Renew.svelte';
  import Partnership from 'carbon-icons-svelte/lib/Partnership.svelte';
  import Chat from 'carbon-icons-svelte/lib/Chat.svelte';
  import Cut from 'carbon-icons-svelte/lib/Cut.svelte';
  import Book from 'carbon-icons-svelte/lib/Book.svelte';
  import Video from 'carbon-icons-svelte/lib/Video.svelte';
  import Satellite from 'carbon-icons-svelte/lib/Satellite.svelte';
  import StrategyPlay from 'carbon-icons-svelte/lib/StrategyPlay.svelte';
  import Lightning from 'carbon-icons-svelte/lib/Lightning.svelte';
  import Mountain from 'carbon-icons-svelte/lib/Mountain.svelte';
  import Soccer from 'carbon-icons-svelte/lib/Soccer.svelte';
  import RecentlyViewed from 'carbon-icons-svelte/lib/RecentlyViewed.svelte';
  import Building from 'carbon-icons-svelte/lib/Building.svelte';
  import FaceMask from 'carbon-icons-svelte/lib/FaceMask.svelte';
  import GameWireless from 'carbon-icons-svelte/lib/GameWireless.svelte';
  import PaintBrush from 'carbon-icons-svelte/lib/PaintBrush.svelte';
  import Enterprise from 'carbon-icons-svelte/lib/Enterprise.svelte';
  import UserMultiple from 'carbon-icons-svelte/lib/UserMultiple.svelte';
  import EventsAlt from 'carbon-icons-svelte/lib/EventsAlt.svelte';
  import GenderFemale from 'carbon-icons-svelte/lib/GenderFemale.svelte';
  import Accessibility from 'carbon-icons-svelte/lib/Accessibility.svelte';
  import User from 'carbon-icons-svelte/lib/User.svelte';
  import Gift from 'carbon-icons-svelte/lib/Gift.svelte';
  import Badge from 'carbon-icons-svelte/lib/Badge.svelte';
  import CarFront from 'carbon-icons-svelte/lib/CarFront.svelte';

  export let name: string;
  export let size: 16 | 20 | 24 | 32 = 16;

  const icons: Record<string, ComponentType> = {
    action: Flash,
    adventure: Explore,
    'avant garde': SprayPaint,
    'award winning': Trophy,
    comedy: FaceWink,
    drama: Theater,
    fantasy: MagicWand,
    gourmet: Restaurant,
    horror: Bat,
    mystery: Search,
    romance: Favorite,
    'sci-fi': Rocket,
    'slice of life': Cafe,
    sports: Basketball,
    supernatural: Moon,
    suspense: WarningAlt,
    ecchi: View,
    erotica: ViewFilled,
    hentai: Locked,
    'adult cast': UserAvatar,
    anthropomorphic: DogWalker,
    cgdct: CropGrowth,
    childcare: ParentChild,
    'combat sports': Box,
    crossdressing: UserProfile,
    delinquents: Warning,
    detective: Security,
    educational: Education,
    'gag humor': FaceDizzy,
    gore: RainDrop,
    harem: Group,
    'high stakes game': GameConsole,
    historical: Time,
    'idols (female)': Microphone,
    'idols (male)': MicrophoneFilled,
    isekai: Exit,
    iyashikei: Sprout,
    'love polygon': ChartNetwork,
    'magical sex shift': Shuffle,
    'mahou shoujo': Star,
    'martial arts': Movement,
    mecha: Bot,
    medical: Hospital,
    memoir: Notebook,
    military: MilitaryCamp,
    music: Music,
    mythology: IbmWatsonDiscovery,
    'organized crime': Password,
    'otaku culture': Headset,
    parody: FaceSatisfied,
    'performing arts': Events,
    pets: Fish,
    psychological: Cognitive,
    racing: Car,
    reincarnation: Renew,
    'reverse harem': Partnership,
    'romantic subtext': Chat,
    samurai: Cut,
    school: Book,
    showbiz: Video,
    space: Satellite,
    'strategy game': StrategyPlay,
    'super power': Lightning,
    survival: Mountain,
    'team sports': Soccer,
    'time travel': RecentlyViewed,
    'urban fantasy': Building,
    vampire: FaceMask,
    'video game': GameWireless,
    'visual arts': PaintBrush,
    workplace: Enterprise,
    'boys love': UserMultiple,
    'girls love': EventsAlt,
    josei: GenderFemale,
    kids: Accessibility,
    seinen: User,
    shoujo: Gift,
    shounen: Badge,
    cars: CarFront,
  };
  $: icon = icons[name.toLowerCase()];
  // Future categories get a stable, individual geometric mark instead of sharing a fallback.
  $: seed = Array.from(name).reduce((hash, char) => Math.imul(hash ^ char.charCodeAt(0), 16777619), 2166136261) >>> 0;
  $: points = Array.from({ length: 8 }, (_, i) => {
    const angle = (i * Math.PI) / 4;
    const radius = 5 + ((seed >>> (i * 3)) & 7) / 2;
    return `${12 + Math.cos(angle) * radius},${12 + Math.sin(angle) * radius}`;
  }).join(' ');
</script>

<span class="genre-icon">
  {#if icon}
    <svelte:component this={icon} {size} aria-hidden="true" />
  {:else}
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="1.5"
      aria-hidden="true"><polygon {points} /><circle cx="12" cy="12" r="2" /></svg
    >
  {/if}
</span>

<style>
  .genre-icon {
    display: inline-flex;
    align-items: center;
    flex: 0 0 auto;
    opacity: 0.9;
  }
</style>
