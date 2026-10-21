import { defineConfig } from 'tsup'

const isRelease =
  process.env.SEPILOTD_RELEASE_BUILD === '1' || process.env.NODE_ENV === 'production'

const nativeBindings = [
  'node-pty',
  'better-sqlite3',
  /^sqlite-vec/,
  'bufferutil',
  'utf-8-validate',
  'cpu-features',
  'ssh2',
  'dockerode',
  'docker-modem',
]

const releaseExternal = [
  ...nativeBindings,
  'pdf-parse',
  'pdfjs-dist',
  'playwright',
  'pino-pretty',
  'node-telegram-bot-api',
]

export default defineConfig((options) => {
  const isWatch = Boolean(options.watch)

  return {
    entry: ['src/index.ts', 'src/benchmark.ts', 'src/embedded.ts', 'src/main.ts'],
    format: ['esm'],
    dts: isRelease || !isWatch,
    sourcemap: false,
    minify: isRelease,
    treeshake: isRelease,
    splitting: !isRelease,
    clean: true,
    banner: {
      js: [
        '#!/usr/bin/env node',
        "import { createRequire as __sepilotCreateRequire } from 'module';",
        'const require = __sepilotCreateRequire(import.meta.url);',
      ].join('\n'),
    },
    external: isRelease ? releaseExternal : ['playwright', 'yaml', ...nativeBindings],
    noExternal: isRelease
      ? [
          /^(?!node-pty$|better-sqlite3$|sqlite-vec|bufferutil$|utf-8-validate$|cpu-features$|ssh2$|dockerode$|docker-modem$|pdf-parse$|pdfjs-dist$|playwright$|pino-pretty$|node-telegram-bot-api$|^node:).+/,
        ]
      : [/^ajv(?:\/|$)/],
  }
})
