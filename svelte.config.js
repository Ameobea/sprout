import adapter from '@sveltejs/adapter-node';
import preprocess from 'svelte-preprocess';
import { resolve } from 'path';

/** @type {import('@sveltejs/kit').Config} */
const config = {
  // Consult https://github.com/sveltejs/svelte-preprocess
  // for more information about preprocessors
  preprocess: preprocess(),

  kit: {
    inlineStyleThreshold: 2048,
    adapter: adapter({
      precompress: true,
    }),
    vite: {
      resolve: {
        alias: {
          src: resolve('./src'),
        },
      },
    },
  },
};

export default config;
