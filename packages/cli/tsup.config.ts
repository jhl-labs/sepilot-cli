import { defineConfig } from 'tsup'

export default defineConfig({
  entry: ['src/index.ts', 'src/lib.ts'],
  format: ['esm'],
  dts: true,
  sourcemap: false,
  minify: true,
  clean: true,
  splitting: false,
  noExternal: [/@sepilotd\/api-client/, /@sepilotd\/core/, /@sepilotd\/presentation/],
  banner: {
    js: '#!/usr/bin/env node',
  },
  esbuildOptions(options) {
    options.jsx = 'automatic'
  },
})
