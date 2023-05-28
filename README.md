# Sprout

[![Sprout logo, a pixel art plant sprouting out of an orange pot](https://i.ameo.link/b3r.png)](https://anime.ameo.dev)

<https://anime.ameo.dev>

Sprout is an anime recommendation website that uses a neural network to suggest animes to watch based on watch history and past ratings.  It can pull public profiles directly from [MyAnimeList](https://myanimelist.net/) or [AniList](https://anilist.co/), and it also has an [interactive recommender](https://anime.ameo.dev/interactive-recommender) where you can provide your ratings directly on the site.

![](https://i.ameo.link/b3t.png)

In addition to recommendations, it also has some profile analytics and an interactive visualzation of the whole world of anime, [Anime Atlas](https://anime.ameo.dev/pymde_4d_40n):

![](https://i.ameo.link/b3s.png)

## Tech

The site is written in TypeScript and built with [SvelteKit](https://kit.svelte.dev/) using the Node.JS adapter.

### Recommendation Model

The recommendation model was trained completely in the browser using [Tensorflow.JS](https://www.tensorflow.org/js/).  It is a rather large model of ~100M parameters but uses a very simple architecture.  It was trained to learn associations between anime as well as other patterns in user profiles and anime preferences using data collected from a very large number of MyAnimeList profiles.

The design is similar to a [denoising autoencoder](https://paperswithcode.com/method/denoising-autoencoder).  User profiles were provided as input with some percentage of ratings held out, and the model was trained to predict the full profile.

The model is served using Tensorflow.JS as well and runs on the CPU.  To determine which ratings in a users' profile contribute most to individual recommendations, the model is re-run many times while holding out each rating to see which causes the recommendation's score to go down the most.

I've written a bit more about the model's design, architecture, and training process in this Reddit comment and thread: <https://www.reddit.com/r/anime/comments/vvbxjy/releasing_sprout_an_aipowered_anime/iftcxfn/>

### Atlas

The atlas was created by producing an embedding out of a weighted co-occurence graph of anime<->anime pairs based on their presences and ratings in users' profiles.  If a user watched two anime and rated both highly, that edge in the graph gains more weight.  The embedding is produced using [PyMDE](https://pymde.org/) and then projected down to 2D using t-SNE.

See the `notebooks` directory for experiments, research, and notebooks used for the embedding generation process.

The web portion of the atlas is built with [PixiJS](https://pixijs.com/), a WebGL-powered 2D graphics framework for interactive web apps.
