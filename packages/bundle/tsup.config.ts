import { defineConfig } from 'tsup'

export default defineConfig({
  entry: ['src/main.ts'],
  format: ['esm'],
  target: 'node20',
  platform: 'node',
  clean: true,
  sourcemap: false,
  minify: true,
  banner: { js: '#!/usr/bin/env node' },
  skipNodeModulesBundle: true,
  external: [/^@sepilotd\//],
})
