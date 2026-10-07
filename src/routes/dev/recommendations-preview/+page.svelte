<script lang="ts">
  import { writable, type Writable } from 'svelte/store';
  import { browser } from '$app/environment';

  import RecommendationControls from 'src/components/recommendation/RecommendationControls.svelte';
  import RecommendationsList from 'src/components/recommendation/RecommendationsList.svelte';
  import { getDefaultRecommendationControlParams } from 'src/components/recommendation/utils';
  import { AnimeMediaType } from 'src/animeMediaType';
  import type { AnimeDetails } from 'src/malAPI';
  import { ModelName } from 'src/components/recommendation/conf';
  import type { Recommendation, UserRatingStats } from 'src/routes/recommendation/recommendation/recommendation';

  const params = writable(getDefaultRecommendationControlParams());

  const poster = (seed: number) =>
    `data:image/svg+xml,${encodeURIComponent(`<svg xmlns="http://www.w3.org/2000/svg" width="225" height="320"><rect width="225" height="320" fill="hsl(${(seed * 37) % 360},35%,25%)"/><text x="112" y="170" text-anchor="middle" fill="white" font-size="24">Sprout ${seed}</text></svg>`)}`;

  const animeMetadataDatabase: { [animeID: number]: AnimeDetails } = {
    101: {
      id: 101,
      title: 'Hoshi no Kage',
      main_picture: { medium: poster(101), large: poster(101) },
      alternative_titles: { synonyms: ['Shadow of the Stars'], en: 'Shadow of the Stars', ja: '星の影' },
      start_date: '2026-04-10',
      end_date: '2026-06-26',
      synopsis: `For twelve years, the observatory on the far side of the Moon has listened to a signal that arrives every evening at precisely the same time. Most researchers dismiss it as an equipment fault. Junior navigator Mina notices that each transmission describes a place that no longer exists on Earth—and that the voice behind it sounds like her missing brother.

When the signal suddenly changes, Mina joins a survey crew aboard a retired transport ship. Their mission is supposed to last six days: reach an abandoned relay station, replace its recorder, and return before the next supply window closes. Instead, they discover a collection of messages addressed to people who have not yet been born. Some predict small, ordinary events. Others warn of a disaster the crew can no longer ignore.

The captain wants to report the discovery and turn back. The engineer believes the station is a trap. Mina is certain that someone is waiting for them beyond the permitted navigation zone, but following the voice means risking the only ship that can bring everyone home. As old friendships break under the pressure, the crew must decide whose memories they can trust.

Far from the cities and families they left behind, these reluctant explorers begin to uncover why the observatory was built, why its original team disappeared, and why the same message has been travelling through the dark for twelve years. Their answers may change the way humanity remembers its first journey into space.`,
      media_type: AnimeMediaType.TV,
      num_episodes: 12,
      genres: [
        { id: 1, name: 'Sci-Fi' },
        { id: 2, name: 'Drama' },
        { id: 3, name: 'Mystery' },
        { id: 8, name: 'Fantasy' },
        { id: 4, name: 'Action' },
      ],
    },
    102: {
      id: 102,
      title: 'Kuroi Hanabi',
      main_picture: { medium: poster(102), large: poster(102) },
      alternative_titles: { synonyms: ['Black Fireworks'], en: 'Black Fireworks', ja: '黒い花火' },
      start_date: '2023-07-14',
      end_date: '2023-07-14',
      synopsis: 'A feature-length mystery in which an old summer festival hides a decades-old disappearance.',
      media_type: AnimeMediaType.Movie,
      num_episodes: 1,
      genres: [
        { id: 2, name: 'Drama' },
        { id: 3, name: 'Mystery' },
      ],
    },
    103: {
      id: 103,
      title: 'Neon Courier',
      main_picture: { medium: poster(103), large: poster(103) },
      alternative_titles: { synonyms: ['NC'], en: 'Neon Courier', ja: 'ネオンクーリエ' },
      start_date: '2025-01-08',
      end_date: '2025-03-26',
      synopsis: `A bike courier crosses a dense cyberpunk city while trying to keep one impossible package alive. Ren usually delivers spare parts and late dinners, keeping his head down whenever the city patrols pass. A stranger's offer of a month's wages for a single trip seems like the chance to finally leave his debt behind.

But the package speaks, the delivery address was demolished years ago, and someone has locked every bridge out of the district. With help from an old racing rival and a mechanic who refuses to answer questions, Ren follows a trail through crowded markets and abandoned transit tunnels. Before sunrise, he must choose between completing the delivery and protecting the person who has become its unexpected passenger.`,
      media_type: AnimeMediaType.TV,
      num_episodes: 12,
      genres: [
        { id: 1, name: 'Sci-Fi' },
        { id: 4, name: 'Action' },
      ],
    },
    104: {
      id: 104,
      title: 'Ame no Tegami',
      main_picture: { medium: poster(104), large: poster(104) },
      alternative_titles: { synonyms: ['Letters in the Rain'], en: 'Letters in the Rain', ja: '雨の手紙' },
      start_date: '2019-10-03',
      end_date: '2019-12-19',
      synopsis: 'Two students begin exchanging anonymous letters and gradually realize they already know each other.',
      media_type: AnimeMediaType.TV,
      num_episodes: 12,
      genres: [
        { id: 2, name: 'Drama' },
        { id: 5, name: 'Romance' },
      ],
    },
    105: {
      id: 105,
      title: 'Orbital Kitchen OVA',
      main_picture: { medium: poster(105), large: poster(105) },
      alternative_titles: { synonyms: [], en: 'Orbital Kitchen OVA', ja: 'オービタルキッチン' },
      start_date: '2021-05-20',
      end_date: '2021-05-20',
      synopsis: 'A short OVA about a dysfunctional restaurant crew trying to serve dinner in zero gravity.',
      media_type: AnimeMediaType.OVA,
      num_episodes: 1,
      genres: [
        { id: 1, name: 'Sci-Fi' },
        { id: 6, name: 'Comedy' },
      ],
    },
    106: {
      id: 106,
      title: 'Signal Bloom',
      main_picture: { medium: poster(106), large: poster(106) },
      alternative_titles: { synonyms: [], en: 'Signal Bloom', ja: 'シグナルブルーム' },
      start_date: '2024-11-02',
      end_date: '2024-11-02',
      synopsis: 'A stylized music video built around a city that changes shape with every note.',
      media_type: AnimeMediaType.Music,
      num_episodes: 1,
      genres: [{ id: 7, name: 'Music' }],
    },
    107: {
      id: 107,
      title: 'Glass District',
      main_picture: { medium: poster(107), large: poster(107) },
      alternative_titles: { synonyms: ['The Glass District'], en: 'Glass District', ja: '硝子区' },
      start_date: '2017-04-06',
      end_date: '2017-09-28',
      synopsis: 'Detectives investigate crimes in a city where memories can be copied, traded, and forged.',
      media_type: AnimeMediaType.TV,
      num_episodes: 24,
      genres: [
        { id: 3, name: 'Mystery' },
        { id: 1, name: 'Sci-Fi' },
      ],
    },
    108: {
      id: 108,
      title: 'Pocket Spirits',
      main_picture: { medium: poster(108), large: poster(108) },
      alternative_titles: { synonyms: [], en: 'Pocket Spirits', ja: 'ポケットスピリッツ' },
      start_date: '2022-02-12',
      end_date: '2022-04-30',
      synopsis: 'A light adventure about tiny spirits that inhabit forgotten objects around a seaside town.',
      media_type: AnimeMediaType.ONA,
      num_episodes: 10,
      genres: [
        { id: 6, name: 'Comedy' },
        { id: 8, name: 'Fantasy' },
      ],
    },
  };

  const sampleRecommendations: Recommendation[] = [
    { id: 101, score: 0.83, predictedRating: 8.4 },
    { id: 102, score: 0.79, predictedRating: 8.9 },
    { id: 103, score: 0.77, predictedRating: 7.8, planToWatch: true },
    { id: 104, score: 0.72, predictedRating: 9.1 },
    { id: 105, score: 0.68, predictedRating: 7.2 },
    { id: 106, score: 0.64, predictedRating: 6.9 },
    { id: 107, score: 0.61, predictedRating: 8.1 },
    { id: 108, score: 0.57, predictedRating: 7.5 },
  ];

  // Optional development-only fixture for responsive edge cases: ?scenario=layout.
  const stressLayout = browser && new URLSearchParams(window.location.search).get('scenario') === 'layout';
  if (stressLayout) {
    animeMetadataDatabase[101].alternative_titles.en =
      'Shadow of the Stars: An Unexpected Journey Across the Entire Universe and Back Again';
    animeMetadataDatabase[101].genres = Object.values(animeMetadataDatabase)
      .flatMap((anime) => anime.genres ?? [])
      .filter((genre, index, genres) => genres.findIndex((item) => item.id === genre.id) === index);
    animeMetadataDatabase[103].synopsis = '';
    animeMetadataDatabase[103].main_picture = { medium: '', large: '' };
    animeMetadataDatabase[104].start_date = '';
    animeMetadataDatabase[104].num_episodes = 0;
    sampleRecommendations[2].predictedRating = undefined;
  }

  // Sample 107 represents a TV story related to a watched title. Model sliders are
  // deliberately not simulated: their scores require the real model server.
  $: recommendations = sampleRecommendations
    .filter((reco) => {
      const metadata = animeMetadataDatabase[reco.id];
      if (!$params.includeExtraSeasons && reco.id === 107) return false;
      if (!$params.includeMovies && metadata.media_type === AnimeMediaType.Movie) return false;
      if (!$params.includeONAsOVAsSpecials && [AnimeMediaType.ONA, AnimeMediaType.OVA].includes(metadata.media_type))
        return false;
      if (!$params.includeMusic && metadata.media_type === AnimeMediaType.Music) return false;
      if ($params.filterPlanToWatch && reco.planToWatch) return false;
      return !metadata.genres?.some((genre) => $params.excludedGenreIDs.includes(genre.id));
    })
    .map((reco) => ({
      ...reco,
      predictedRating: $params.modelName === ModelName.Legacy_2023 ? undefined : reco.predictedRating,
      topContributors: [
        { animeId: 101, strength: 0.8 },
        { animeId: 104, strength: 0.4 },
      ].filter((contributor) => !$params.excludedRankingAnimeIDs.includes(contributor.animeId)),
    }));

  const userRatingStats: UserRatingStats = {
    mean: 7.2,
    stdDev: 1.1,
    ratedCount: 184,
    isNonRater: false,
  };

  const genresDB: Writable<Map<number, string>> = writable(
    new Map(
      Array.from(
        new Map(
          Object.values(animeMetadataDatabase)
            .flatMap((anime) => anime.genres ?? [])
            .map((genre) => [genre.id, genre.name] as const)
        ).entries()
      )
    )
  );

  let isProfileRefreshing = false;
  let profileRefreshError = '';

  const previewRefresh = async () => {
    isProfileRefreshing = true;
    profileRefreshError = '';
    await new Promise((resolve) => setTimeout(resolve, 900));
    isProfileRefreshing = false;
  };

  const excludeRanking = (animeID: number) => {
    $params.excludedRankingAnimeIDs = [...new Set([...$params.excludedRankingAnimeIDs, animeID])];
  };
  const excludeGenre = (genreID: number) => {
    $params.excludedGenreIDs = [...new Set([...$params.excludedGenreIDs, genreID])];
  };

  const includeGenre = (genreID: number) => {
    $params.excludedGenreIDs = $params.excludedGenreIDs.filter((id) => id !== genreID);
  };
</script>

<svelte:head>
  <title>Sprout Recommendation Controls Preview</title>
</svelte:head>

<div class="preview-shell">
  <div class="preview-note">
    <strong>Development preview</strong>
    <span
      >Uses local sample data only. Toggles filter the samples; refresh simulates loading. Model settings do not
      recompute sample scores. No MyAnimeList or model-server request is made.</span
    >
  </div>

  <RecommendationControls
    {params}
    {animeMetadataDatabase}
    isLoading={false}
    onForceProfileRefresh={previewRefresh}
    {isProfileRefreshing}
    {profileRefreshError}
  />

  <RecommendationsList
    {recommendations}
    {animeMetadataDatabase}
    contributorsLoading={false}
    {userRatingStats}
    {excludeRanking}
    {excludeGenre}
    {includeGenre}
    excludedGenreIDs={$params.excludedGenreIDs}
    genreNames={$genresDB}
  />
</div>

<style>
  .preview-shell {
    max-width: 950px;
    min-height: 100vh;
    margin: 0 auto;
    padding: 12px;
    background: #131313;
  }

  .preview-note {
    display: flex;
    flex-direction: column;
    gap: 3px;
    margin-bottom: 10px;
    padding: 10px 12px;
    border: 1px solid #393939;
    background: #1b1b1b;
    color: #f4f4f4;
  }

  .preview-note span {
    color: #a8a8a8;
    font-size: 0.85rem;
  }
</style>
