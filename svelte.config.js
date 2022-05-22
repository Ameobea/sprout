import adapter from '@sveltejs/adapter-node';
import preprocess from 'svelte-preprocess';
import { resolve } from 'path';
import { optimizeImports, optimizeCss, icons, elements } from 'carbon-preprocess-svelte';

const production = process.env.NODE_ENV === 'production';

/** @type {import('@sveltejs/kit').Config} */
const config = {
  // Consult https://github.com/sveltejs/svelte-preprocess
  // for more information about preprocessors
  preprocess: [preprocess(), optimizeImports()],

  kit: {
    inlineStyleThreshold: 2048,
    adapter: adapter({
      precompress: true,
    }),
    vite: {
      plugins: [
        // production && optimizeCss({ safelist: { deep: [/.*data-*$/] } }),
        icons(),
        elements(),
      ],
      resolve: {
        alias: {
          src: resolve('./src'),
        },
      },
      build: {
        // sourcemap: true,
      },
      optimizeDeps: {
        include: ['@carbon/charts'],
      },
      ssr: {
        noExternal: [production && '@carbon/charts'].filter(Boolean),
      },
    },
    prerender: {
      concurrency: 6,
    },
    floc: true,
  },
  experimental: {
    inspector: {
      holdMode: true,
    },
    prebundleSvelteLibraries: true,
  },
};

export default config;
