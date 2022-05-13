import adapter from '@sveltejs/adapter-node';
import preprocess from 'svelte-preprocess';
import { resolve } from 'path';
import { optimizeImports, optimizeCss, icons, elements } from 'carbon-preprocess-svelte';

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
      plugins: [optimizeImports(), /*process.env.NODE_ENV === 'production' && optimizeCss(),*/ icons(), elements()],
      resolve: {
        alias: {
          src: resolve('./src'),
        },
      },
      build: {
        sourcemap: true,
      },
    },
  },
};

export default config;
