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
  // Node-first artifact: transpile `src/main.ts` and leave every import
  // (`@sepilotd/*` and their transitive third-party deps) external so it runs
  // under plain `node` against the monorepo's `node_modules`. The real
  // single-file binary (Phase 6) is produced by `bun build --compile
  // src/main.ts`, which resolves and bundles everything itself — it does not
  // consume this `dist/main.js`.
  skipNodeModulesBundle: true,
  external: [/^@sepilotd\//],
})
