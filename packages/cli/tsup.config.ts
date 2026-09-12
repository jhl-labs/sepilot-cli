import { defineConfig } from 'tsup'

export default defineConfig({
  entry: ['src/index.ts', 'src/lib.ts'],
  format: ['esm'],
  dts: true,
  sourcemap: false,
  minify: true,
  clean: true,
  // Keep each entry self-contained so the bin guard in src/index.ts runs from
  // dist/index.js (code splitting would hoist top-level statements into a
  // shared chunk whose import.meta.url no longer matches the bin path).
  splitting: false,
  noExternal: [/@sepilotd\/api-client/, /@sepilotd\/core/, /@sepilotd\/presentation/],
  banner: {
    js: '#!/usr/bin/env node',
  },
  esbuildOptions(options) {
    options.jsx = 'automatic'
  },
})
