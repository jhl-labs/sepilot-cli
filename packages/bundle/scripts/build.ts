#!/usr/bin/env tsx
import { execFileSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { createRequire } from 'node:module'
import { tmpdir } from 'node:os'
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { scanArtifact } from './secret-scan.js'
import { generateFeatureRegistration } from './generate-feature-registration.js'
import { FEATURE_CATALOG } from '../../daemon/src/features/feature-catalog.js'

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url))
const BUNDLE_DIR = resolve(SCRIPT_DIR, '..')
const REPO_ROOT = resolve(BUNDLE_DIR, '..', '..')

type TargetName = 'linux-x64' | 'linux-arm64' | 'windows-x64' | 'darwin-x64' | 'darwin-arm64'

interface TargetSpec {
  bunTarget: string
  outFile: string
  vecPkg: string
  vecFile: string
  host: { platform: NodeJS.Platform; arch: string }
}

const TARGETS: Record<TargetName, TargetSpec> = {
  'linux-x64': {
    bunTarget: 'bun-linux-x64',
    outFile: 'sepilot-linux-x64',
    vecPkg: 'sqlite-vec-linux-x64',
    vecFile: 'vec0.so',
    host: { platform: 'linux', arch: 'x64' },
  },
  'linux-arm64': {
    bunTarget: 'bun-linux-arm64',
    outFile: 'sepilot-linux-arm64',
    vecPkg: 'sqlite-vec-linux-arm64',
    vecFile: 'vec0.so',
    host: { platform: 'linux', arch: 'arm64' },
  },
  'windows-x64': {
    bunTarget: 'bun-windows-x64',
    outFile: 'sepilot-windows-x64.exe',
    vecPkg: 'sqlite-vec-windows-x64',
    vecFile: 'vec0.dll',
    host: { platform: 'win32', arch: 'x64' },
  },
  'darwin-x64': {
    bunTarget: 'bun-darwin-x64',
    outFile: 'sepilot-darwin-x64',
    vecPkg: 'sqlite-vec-darwin-x64',
    vecFile: 'vec0.dylib',
    host: { platform: 'darwin', arch: 'x64' },
  },
  'darwin-arm64': {
    bunTarget: 'bun-darwin-arm64',
    outFile: 'sepilot-darwin-arm64',
    vecPkg: 'sqlite-vec-darwin-arm64',
    vecFile: 'vec0.dylib',
    host: { platform: 'darwin', arch: 'arm64' },
  },
}

const SQLITE_VEC_VERSION = (() => {
  const candidates = [
    join(REPO_ROOT, 'packages/daemon/node_modules/sqlite-vec/package.json'),
    join(REPO_ROOT, 'node_modules/sqlite-vec/package.json'),
  ]
  for (const p of candidates) {
    try {
      if (existsSync(p)) {
        const v = (JSON.parse(readFileSync(p, 'utf8')) as { version?: string }).version
        if (v) return v
      }
    } catch {
    }
  }
  return '0.1.9'
})()

function log(msg: string): void {
  process.stdout.write(`[build:binary] ${msg}\n`)
}

function fail(msg: string): never {
  process.stderr.write(`[build:binary] ERROR: ${msg}\n`)
  process.exit(1)
}

const SHELL_REQUIRED_ON_WINDOWS = new Set(['pnpm', 'npm'])

function run(cmd: string, args: string[], cwd: string = REPO_ROOT): void {
  log(`$ ${cmd} ${args.join(' ')}  (cwd=${cwd})`)
  const shell = process.platform === 'win32' && SHELL_REQUIRED_ON_WINDOWS.has(cmd)
  execFileSync(cmd, args, { cwd, stdio: 'inherit', shell })
}

function parseTargets(arg: string | undefined): TargetName[] {
  const value = (arg ?? 'linux-x64').trim()
  if (value === 'all') return Object.keys(TARGETS) as TargetName[]
  if (value in TARGETS) return [value as TargetName]
  fail(`unsupported target "${value}" — expected one of: ${Object.keys(TARGETS).join(', ')}, all`)
}

function buildDeps(): void {
  log('Step 2/8: building @sepilotd/cli + @sepilotd/daemon (+ deps) via turbo...')
  run('pnpm', ['turbo', 'build', '--filter', '@sepilotd/bundle...'])
  const required = [
    join(REPO_ROOT, 'packages/cli/dist/lib.js'),
    join(REPO_ROOT, 'packages/daemon/dist/main.js'),
    join(REPO_ROOT, 'packages/daemon/dist/embedded.js'),
  ]
  const missing = required.filter((p) => !existsSync(p))
  if (missing.length > 0) {
    fail(`required dist artifacts missing after build:\n  ${missing.join('\n  ')}`)
  }
  log('Step 2/8: OK — cli + daemon dist artifacts present.')
}

function verifyBunNetworkDispatcherBridge(
  bundleDir: string = BUNDLE_DIR,
  scriptDir: string = SCRIPT_DIR,
): void {
  const smokeScript = join(scriptDir, 'bun-network-dispatcher-smoke.ts')
  log('Network preflight: verifying Bun compiled fetch uses the daemon dispatcher...')
  run('bun', ['run', smokeScript], bundleDir)
  log('Network preflight: OK')
}

function resolveHostVecLoadable(): string | undefined {
  try {
    const requireFromDaemon = createRequire(join(REPO_ROOT, 'packages/daemon/package.json'))
    const sqliteVec = requireFromDaemon('sqlite-vec') as { getLoadablePath(): string }
    const p = sqliteVec.getLoadablePath()
    return existsSync(p) ? p : undefined
  } catch {
    return undefined
  }
}

function fetchVecLoadable(spec: TargetSpec): string {
  const tmp = mkdtempSync(join(tmpdir(), 'sepilot-vec-pack-'))
  const tgz = join(tmp, `${spec.vecPkg}-${SQLITE_VEC_VERSION}.tgz`)
  const url = `https://registry.npmjs.org/${spec.vecPkg}/-/${spec.vecPkg}-${SQLITE_VEC_VERSION}.tgz`
  let fetched = false
  try {
    run('npm', ['pack', `${spec.vecPkg}@${SQLITE_VEC_VERSION}`, '--pack-destination', tmp], tmp)
    if (existsSync(tgz)) {
      fetched = true
    } else {
      const candidates = readdirSync(tmp).filter((f) => f.endsWith('.tgz'))
      if (candidates.length === 1) {
        copyFileSync(join(tmp, candidates[0]), tgz)
        fetched = true
      }
    }
  } catch (err) {
    log(
      `npm pack ${spec.vecPkg}@${SQLITE_VEC_VERSION} failed (${
        err instanceof Error ? err.message : String(err)
      }) — falling back to direct registry fetch.`,
    )
  }
  if (!fetched) {
    try {
      run('curl', ['-fsSL', '-o', tgz, url], tmp)
      fetched = existsSync(tgz)
    } catch (err) {
      rmSync(tmp, { recursive: true, force: true })
      fail(
        `could not obtain ${spec.vecPkg}@${SQLITE_VEC_VERSION} — neither \`npm pack\` nor \`curl ${url}\` worked: ${
          err instanceof Error ? err.message : String(err)
        }`,
      )
    }
  }
  if (!fetched) {
    rmSync(tmp, { recursive: true, force: true })
    fail(`could not obtain ${spec.vecPkg}@${SQLITE_VEC_VERSION}`)
  }
  run('tar', ['-xzf', tgz, '-C', tmp, `package/${spec.vecFile}`], tmp)
  const extracted = join(tmp, 'package', spec.vecFile)
  if (!existsSync(extracted)) {
    rmSync(tmp, { recursive: true, force: true })
    fail(`tarball ${spec.vecPkg}@${SQLITE_VEC_VERSION} did not contain package/${spec.vecFile}`)
  }
  return extracted
}

function obtainVecLoadable(
  target: TargetName,
  spec: TargetSpec,
  bundleDir: string = BUNDLE_DIR,
): { destPath: string; ext: string } {
  log(`Step 3/8: obtaining the sqlite-vec vec0 loadable for ${target} (${spec.vecPkg})...`)
  const ext = spec.vecFile.replace(/^vec0\./, '')
  let sourcePath: string | undefined
  if (process.platform === spec.host.platform && process.arch === spec.host.arch) {
    sourcePath = resolveHostVecLoadable()
    if (sourcePath) log(`Step 3/8: using host-installed loadable ${sourcePath}`)
  }
  let tmpToCleanup: string | undefined
  if (!sourcePath) {
    sourcePath = fetchVecLoadable(spec)
    tmpToCleanup = resolve(sourcePath, '..', '..')
  }
  const assetsDir = join(bundleDir, 'assets')
  mkdirSync(assetsDir, { recursive: true })
  const destPath = join(assetsDir, `vec0-${target}.${ext}`)
  copyFileSync(sourcePath, destPath)
  if (tmpToCleanup) rmSync(tmpToCleanup, { recursive: true, force: true })
  const sizeKb = (readFileSync(destPath).byteLength / 1024).toFixed(0)
  log(`Step 3/8: OK — vec0-${target}.${ext} ready (${sizeKb} KB)`)
  return { destPath, ext }
}

function writeGeneratedVecAsset(
  target: TargetName,
  ext: string,
  bundleDir: string = BUNDLE_DIR,
): string {
  log('Step 4/8: writing src/generated/vec-asset.ts for the embedded asset...')
  const generatedDir = join(bundleDir, 'src', 'generated')
  mkdirSync(generatedDir, { recursive: true })
  const generatedPath = join(generatedDir, 'vec-asset.ts')
  const contents = `// GENERATED by scripts/build.ts for target ${target} — do not edit; do not commit.
// @ts-expect-error bun handles \`with { type: 'file' }\`; tsc/node don't resolve binary modules.
import vec0 from '../../assets/vec0-${target}.${ext}' with { type: 'file' }
export const VEC0_PATH: string | undefined = vec0 as unknown as string
`
  writeFileSync(generatedPath, contents)
  log(`Step 4/8: OK — wrote ${generatedPath}`)
  return generatedPath
}

function bunCompile(
  target: TargetName,
  spec: TargetSpec,
  bundleDir: string = BUNDLE_DIR,
  scriptDir: string = SCRIPT_DIR,
): string {
  log(`Step 5/8: bun build --compile --target=${spec.bunTarget} ...`)
  const outDir = join(bundleDir, 'out')
  mkdirSync(outDir, { recursive: true })
  const outFile = join(outDir, spec.outFile)
  const entry = join(bundleDir, 'src', 'main.ts')
  const compileScript = join(scriptDir, 'bun-compile.ts')
  run('bun', ['run', compileScript, spec.bunTarget, entry, outFile], bundleDir)
  if (!existsSync(outFile)) {
    fail(`bun --compile reported success but ${outFile} does not exist.`)
  }
  const bytes = readFileSync(outFile).byteLength
  const MIN_BYTES = 20 * 1024 * 1024
  if (bytes < MIN_BYTES) {
    fail(
      `compiled ${outFile} is suspiciously small (${(bytes / (1024 * 1024)).toFixed(1)} MB < 20 MB) — something went wrong.`,
    )
  }
  log(`Step 5/8: OK — compiled ${outFile} (${(bytes / (1024 * 1024)).toFixed(1)} MB)`)
  return outFile
}

function secretScanGate(outFile: string): void {
  log('Step 6/8: scanning the binary for secret-like content...')
  const result = scanArtifact(outFile)
  if (!result.ok) {
    process.stderr.write(
      `[build:binary] ERROR: secret-scan FAILED for ${outFile} — release blocked:\n`,
    )
    for (const finding of result.findings) {
      process.stderr.write(`  - ${finding}\n`)
    }
    process.stderr.write(
      '[build:binary] (see CLAUDE.md "Release Artifact Secret Policy"). Aborting.\n',
    )
    process.exit(1)
  }
  log('Step 6/8: OK — no secret-like content found.')
}

function exclusionScanGate(outFile: string, disabled: string[]): void {
  if (disabled.length === 0) return
  log(
    `Step 6b/8: scanning the binary to prove ${disabled.length} disabled feature(s) are absent...`,
  )
  const binary = readFileSync(outFile)
  const leaks: string[] = []
  for (const id of disabled) {
    const feature = FEATURE_CATALOG.find((f) => f.id === id)!
    for (const marker of feature.binaryMarkers) {
      if (binary.includes(Buffer.from(marker))) leaks.push(`${id}: '${marker}'`)
    }
  }
  if (leaks.length > 0) {
    throw new Error(`disabled feature code leaked into the binary:\n${leaks.join('\n')}`)
  }
  log(`Step 6b/8: OK — exclusion scan OK — ${disabled.length} disabled feature(s) verified absent`)
}

const DEV_ONLY_MARKERS = ['sepilot-dev-tools', 'dev-tools/src/']

function findDevOnlyMarkerLeaks(binary: Buffer): string[] {
  const leaks: string[] = []
  for (const marker of DEV_ONLY_MARKERS) {
    if (binary.includes(Buffer.from(marker))) leaks.push(`dev-only: '${marker}'`)
  }
  return leaks
}

function devToolsExclusionScanGate(outFile: string): void {
  const binary = readFileSync(outFile)
  const leaks = findDevOnlyMarkerLeaks(binary)
  if (leaks.length > 0) {
    throw new Error(`dev-only code leaked into the binary:\n${leaks.join('\n')}`)
  }
}

function writeSha256(outFile: string, spec: TargetSpec): void {
  log('Step 7/8: computing SHA-256 of the binary...')
  const hash = createHash('sha256').update(readFileSync(outFile)).digest('hex')
  const shaFile = `${outFile}.sha256`
  writeFileSync(shaFile, `${hash}  ${spec.outFile}\n`)
  log(`Step 7/8: OK — ${hash}  ${spec.outFile}  (-> ${shaFile})`)
}

const STUB_CONTENTS =
  '// Overwritten by scripts/build.ts during `bun build --compile`. Default: no embedded vec0 asset.\n' +
  'export const VEC0_PATH: string | undefined = undefined\n'

const VERSION_STUB_CONTENTS =
  '// Overwritten by scripts/build.ts during `bun build --compile`. Default: no\n' +
  '// build-time version override (the CLI reads its own package.json instead).\n' +
  '// The generated form sets `process.env.SEPILOT_VERSION` on import; the\n' +
  '// committed stub deliberately does nothing on import.\n' +
  'export {}\n'

function restoreStub(generatedPath: string): void {
  log('Step 8/8: restoring the committed stub src/generated/vec-asset.ts...')
  try {
    execFileSync('git', ['checkout', '--', generatedPath], { cwd: REPO_ROOT, stdio: 'inherit' })
    log('Step 8/8: OK — stub restored via `git checkout --`.')
  } catch {
    mkdirSync(dirname(generatedPath), { recursive: true })
    writeFileSync(generatedPath, STUB_CONTENTS)
    log('Step 8/8: no committed stub yet — wrote the canonical stub contents.')
  }
}

function writeGeneratedFeatureRegistration(repoRoot: string = REPO_ROOT): {
  generatedPath: string
  disabled: string[]
  originalContents: string
} {
  log('Step 1/8: generating daemon feature-registration manifest...')
  const featuresFile =
    process.env.SEPILOT_BUILD_FEATURES_FILE ?? join(repoRoot, 'build.features.yaml')
  const generatedPath = join(repoRoot, 'packages/daemon/src/generated/feature-registration.ts')
  const originalContents = readFileSync(generatedPath, 'utf8')
  const { disabled } = generateFeatureRegistration({ featuresFile, outFile: generatedPath })
  log(
    `Step 1/8: build features: ${
      disabled.length === 0 ? 'all enabled' : `disabled = ${disabled.join(', ')}`
    }`,
  )
  return { generatedPath, disabled, originalContents }
}

function restoreFeatureRegistrationStub(generatedPath: string, originalContents: string): void {
  mkdirSync(dirname(generatedPath), { recursive: true })
  writeFileSync(generatedPath, originalContents)
  log('Step 1/8: restored the pre-build feature-registration.ts contents.')
}

function writeGeneratedVersion(bundleDir: string = BUNDLE_DIR): string {
  const candidates = [join(REPO_ROOT, 'package.json'), join(REPO_ROOT, 'packages/cli/package.json')]
  let version: string | undefined
  for (const p of candidates) {
    try {
      if (existsSync(p)) {
        const v = (JSON.parse(readFileSync(p, 'utf8')) as { version?: string }).version?.trim()
        if (v) {
          version = v
          break
        }
      }
    } catch {
    }
  }
  const generatedDir = join(bundleDir, 'src', 'generated')
  mkdirSync(generatedDir, { recursive: true })
  const generatedPath = join(generatedDir, 'version.ts')
  if (version) {
    const v = JSON.stringify(version)
    writeFileSync(
      generatedPath,
      `// GENERATED by scripts/build.ts — do not edit; do not commit.\n` +
        `delete process.env.NODE_PATH\n` +
        `const SEPILOT_VERSION: string | undefined = ${v}\n` +
        `if (!process.env.SEPILOT_VERSION) process.env.SEPILOT_VERSION = SEPILOT_VERSION\n` +
        `export {}\n`,
    )
    log(`version override: SEPILOT_VERSION=${v} -> ${generatedPath}`)
  } else {
    writeFileSync(generatedPath, VERSION_STUB_CONTENTS)
    log('version override: could not read a monorepo version — left the stub in place.')
  }
  return generatedPath
}

function restoreVersionStub(generatedPath: string): void {
  try {
    execFileSync('git', ['checkout', '--', generatedPath], { cwd: REPO_ROOT, stdio: 'inherit' })
    log('version override: restored the committed stub via `git checkout --`.')
  } catch {
    mkdirSync(dirname(generatedPath), { recursive: true })
    writeFileSync(generatedPath, VERSION_STUB_CONTENTS)
    log('version override: no committed stub yet — wrote the canonical stub contents.')
  }
}

function maybeSmoke(target: TargetName, outFile: string): void {
  if (process.platform !== 'linux' && process.platform !== 'darwin') return
  if (target !== `${process.platform}-${process.arch}`) return
  const smokeScript = join(SCRIPT_DIR, 'smoke.sh')
  if (!existsSync(smokeScript)) throw new Error(`Native smoke script is missing: ${smokeScript}`)
  log(`running smoke test: bash ${smokeScript} ${outFile}`)
  run('env', ['-u', 'SEPILOTD_HOME', 'bash', smokeScript, outFile], BUNDLE_DIR)
  log('smoke test: OK')
}

function buildOne(
  target: TargetName,
  bundleDir: string = BUNDLE_DIR,
  scriptDir: string = SCRIPT_DIR,
  disabledFeatures: string[] = [],
): void {
  const spec = TARGETS[target]
  log(`=== target: ${target} (${spec.bunTarget}) ===`)
  const { ext } = obtainVecLoadable(target, spec, bundleDir)
  const generatedPath = writeGeneratedVecAsset(target, ext, bundleDir)
  try {
    const outFile = bunCompile(target, spec, bundleDir, scriptDir)
    secretScanGate(outFile)
    exclusionScanGate(outFile, disabledFeatures)
    devToolsExclusionScanGate(outFile)
    writeSha256(outFile, spec)
    const sizeMb = (readFileSync(outFile).byteLength / (1024 * 1024)).toFixed(1)
    log(`built ${outFile} (${sizeMb} MB) + ${outFile}.sha256`)
    maybeSmoke(target, outFile)
  } finally {
    restoreStub(generatedPath)
  }
}

function main(): void {
  const targets = parseTargets(process.argv[2])
  log(`targets: ${targets.join(', ')}`)
  const {
    generatedPath: featureRegistrationPath,
    disabled: disabledFeatures,
    originalContents: featureRegistrationStub,
  } = writeGeneratedFeatureRegistration()
  try {
    buildDeps()
    verifyBunNetworkDispatcherBridge()
    const versionPath = writeGeneratedVersion()
    try {
      for (const target of targets) buildOne(target, BUNDLE_DIR, SCRIPT_DIR, disabledFeatures)
    } finally {
      restoreVersionStub(versionPath)
    }
  } finally {
    restoreFeatureRegistrationStub(featureRegistrationPath, featureRegistrationStub)
  }
  log(
    `build features: ${
      disabledFeatures.length === 0 ? 'all enabled' : `disabled = ${disabledFeatures.join(', ')}`
    }`,
  )
  log(`DONE — built ${targets.length} target(s): ${targets.join(', ')}`)
}

export const __testables = {
  TARGETS,
  STUB_CONTENTS,
  VERSION_STUB_CONTENTS,
  parseTargets,
  run,
  buildDeps,
  verifyBunNetworkDispatcherBridge,
  obtainVecLoadable,
  resolveHostVecLoadable,
  fetchVecLoadable,
  bunCompile,
  secretScanGate,
  exclusionScanGate,
  findDevOnlyMarkerLeaks,
  devToolsExclusionScanGate,
  restoreStub,
  restoreVersionStub,
  writeGeneratedVecAsset,
  writeGeneratedVersion,
  writeGeneratedFeatureRegistration,
  restoreFeatureRegistrationStub,
  writeSha256,
  maybeSmoke,
  buildOne,
}

const invokedPath = process.argv[1] ? pathToFileURL(resolve(process.argv[1])).href : undefined
if (invokedPath === import.meta.url) {
  main()
}
