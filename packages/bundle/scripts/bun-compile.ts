#!/usr/bin/env bun
import { readdirSync, readFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const EXTERNAL = [
  'node-pty',
  'better-sqlite3',
  'cpu-features',
  'chromium-bidi',
]

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url))
const DAEMON_SOURCE_DIR = resolve(SCRIPT_DIR, '..', '..', 'daemon', 'src')

const NPM_UNDICI_ENTRY = resolve(
  SCRIPT_DIR,
  '..',
  '..',
  'daemon',
  'node_modules',
  'undici',
  'index.js',
)

const pinNpmUndiciSubpath = {
  name: 'pin-npm-undici-subpath',
  setup(build: import('bun').PluginBuilder) {
    build.onResolve({ filter: /^undici\/index\.js$/ }, () => ({
      path: NPM_UNDICI_ENTRY,
    }))
  },
}

function runtimeSourceFiles(root: string): string[] {
  const files: string[] = []
  for (const entry of readdirSync(root, { withFileTypes: true })) {
    if (entry.name === '__tests__') continue
    const path = join(root, entry.name)
    if (entry.isDirectory()) {
      files.push(...runtimeSourceFiles(path))
    } else if (/\.[cm]?[jt]sx?$/u.test(entry.name)) {
      files.push(path)
    }
  }
  return files
}

function importDeclarations(source: string): string[] {
  const declarations: string[] = []
  let current = ''
  for (const line of source.split(/\r?\n/u)) {
    if (!current) {
      if (!/^\s*import\b/u.test(line)) continue
      current = line
    } else {
      current += `\n${line}`
    }
    if (
      /^\s*import\s+['"][^'"]+['"]\s*;?\s*$/u.test(current)
      || /\bfrom\s+['"][^'"]+['"]\s*;?\s*$/u.test(current)
    ) {
      declarations.push(current)
      current = ''
    }
  }
  return declarations
}

export function assertNoBareUndiciRuntimeImports(
  sourceRoot: string = DAEMON_SOURCE_DIR,
): void {
  const violations: string[] = []
  for (const path of runtimeSourceFiles(sourceRoot)) {
    for (const declaration of importDeclarations(readFileSync(path, 'utf8'))) {
      const sideEffectImport = /^\s*import\s+['"]undici['"]/u.test(declaration)
      if (!sideEffectImport && !/\bfrom\s+['"]undici['"]/u.test(declaration)) continue
      const clause = declaration.replace(/^\s*import\s+/u, '').split(/\bfrom\b/u, 1)[0]?.trim()
      if (!sideEffectImport && clause?.startsWith('type ')) continue
      violations.push(path)
    }
  }
  if (violations.length > 0) {
    throw new Error(
      'Bun standalone runtime imports must use "undici/index.js"; bare "undici" '
        + `would bypass the configured dispatcher:\n${[...new Set(violations)].join('\n')}`,
    )
  }
}

const stubReactDevtools = {
  name: 'stub-react-devtools-core',
  setup(build: import('bun').PluginBuilder) {
    build.onResolve({ filter: /^react-devtools-core$/ }, () => ({
      path: 'react-devtools-core',
      namespace: 'sepilot-react-devtools-stub',
    }))
    build.onLoad({ filter: /.*/, namespace: 'sepilot-react-devtools-stub' }, () => ({
      contents: 'export default { connectToDevTools() {} }\n',
      loader: 'js',
    }))
  },
}

const stubSsh2 = {
  name: 'stub-ssh2',
  setup(build: import('bun').PluginBuilder) {
    build.onResolve({ filter: /^ssh2$/ }, () => ({
      path: 'ssh2',
      namespace: 'sepilot-ssh2-stub',
    }))
    build.onLoad({ filter: /.*/, namespace: 'sepilot-ssh2-stub' }, () => ({
      contents: [
        'class UnsupportedSsh2Client {',
        '  once() { return this }',
        '  on() { return this }',
        '  connect() { throw new Error("Docker over SSH is not available in the standalone binary.") }',
        '  exec(_command, callback) { callback?.(new Error("Docker over SSH is not available in the standalone binary.")) }',
        '  end() {}',
        '}',
        'export const Client = UnsupportedSsh2Client',
        'export default { Client: UnsupportedSsh2Client }',
        '',
      ].join('\n'),
      loader: 'js',
    }))
  },
}

const stubElectron = {
  name: 'stub-electron',
  setup(build: import('bun').PluginBuilder) {
    build.onResolve({ filter: /^electron$/ }, () => ({
      path: 'electron',
      namespace: 'sepilot-electron-stub',
    }))
    build.onLoad({ filter: /.*/, namespace: 'sepilot-electron-stub' }, () => ({
      contents: [
        'export const app = new Proxy({}, {',
        '  get() { throw new Error("Electron automation is not available in the standalone binary.") },',
        '})',
        'export default { app }',
        '',
      ].join('\n'),
      loader: 'js',
    }))
  },
}

const stubProtobufInquire = {
  name: 'stub-protobufjs-inquire',
  setup(build: import('bun').PluginBuilder) {
    build.onResolve({ filter: /^@protobufjs\/inquire$/ }, () => ({
      path: '@protobufjs/inquire',
      namespace: 'sepilot-protobufjs-inquire-stub',
    }))
    build.onLoad({ filter: /.*/, namespace: 'sepilot-protobufjs-inquire-stub' }, () => ({
      contents: [
        'module.exports = function inquire(moduleName) {',
        '  if (moduleName === "long") return null',
        '  try {',
        '    if (typeof require !== "function") return null',
        '    const mod = require(moduleName)',
        '    return mod && (mod.length || Object.keys(mod).length) ? mod : null',
        '  } catch {',
        '    return null',
        '  }',
        '}',
        '',
      ].join('\n'),
      loader: 'js',
    }))
  },
}

const makePlaywrightCoreDirStandaloneSafe = {
  name: 'playwright-core-dir-standalone-safe',
  setup(build: import('bun').PluginBuilder) {
    build.onLoad({
      filter: /playwright-core[\\/]lib[\\/]server[\\/]utils[\\/]nodePlatform\.js$/,
    }, (args) => {
      const source = readFileSync(args.path, 'utf8')
      const marker = 'require.resolve("../../../package.json")'
      if (!source.includes(marker)) {
        throw new Error(`Playwright nodePlatform package marker changed: ${args.path}`)
      }
      return {
        contents: source.replace(marker, '__filename'),
        loader: 'js',
      }
    })
  },
}

type BunBuildLike = {
  build(options: {
    entrypoints: string[]
    target: string
    external: string[]
    plugins: Array<
      | typeof pinNpmUndiciSubpath
      | typeof stubReactDevtools
      | typeof stubSsh2
      | typeof stubElectron
      | typeof stubProtobufInquire
      | typeof makePlaywrightCoreDirStandaloneSafe
    >
    compile: { target: string; outfile: string }
  }): Promise<{ success: boolean; logs: unknown[] }>
}

export async function compileWithBun(
  target: string,
  entry: string,
  outfile: string,
  bun: BunBuildLike,
): Promise<void> {
  assertNoBareUndiciRuntimeImports()
  const result = await bun.build({
    entrypoints: [entry],
    target: 'bun',
    external: EXTERNAL,
    plugins: [
      pinNpmUndiciSubpath,
      stubReactDevtools,
      stubSsh2,
      stubElectron,
      stubProtobufInquire,
      makePlaywrightCoreDirStandaloneSafe,
    ],
    // @ts-expect-error `compile` is supported by `bun build --compile` (Bun 1.2+);
    compile: { target, outfile },
  })

  if (!result.success) {
    console.error('bun build --compile failed:')
    for (const message of result.logs) console.error(message)
    process.exit(1)
  }
  console.log(`compiled -> ${outfile}`)
}

export async function main(argv: string[] = process.argv): Promise<void> {
  const [, , target, entry, outfile] = argv
  if (!target || !entry || !outfile) {
    console.error('usage: bun run scripts/bun-compile.ts <bun-target> <entry> <outfile>')
    process.exit(1)
  }
  await compileWithBun(target, entry, outfile, Bun)
}

export const __testables = {
  EXTERNAL,
  DAEMON_SOURCE_DIR,
  NPM_UNDICI_ENTRY,
  pinNpmUndiciSubpath,
  stubReactDevtools,
  stubSsh2,
  stubElectron,
  stubProtobufInquire,
  makePlaywrightCoreDirStandaloneSafe,
}

const invokedPath = process.argv[1] ? pathToFileURL(resolve(process.argv[1])).href : undefined
if (invokedPath === import.meta.url) {
  await main()
}
