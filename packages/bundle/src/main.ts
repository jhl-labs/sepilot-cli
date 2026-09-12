// packages/bundle/src/main.ts
//
// Single entry point for the standalone `sepilot` build — the one place in the
// monorepo allowed to import BOTH `@sepilotd/cli` and `@sepilotd/daemon`.
// It registers the standalone daemon hooks (so the CLI's ensure-daemon can
// re-launch this executable as the daemon and dispatch the hidden `__daemon`
// subcommand to the daemon's foreground entry), then runs the CLI.

// IMPORTANT: this side-effecting import must come *first*, before `@sepilotd/cli`.
// `bun build --compile` can't read `package.json` from inside Bun's virtual
// filesystem, so package.json-based version loaders would report `0.0.0`.
// `scripts/build.ts` generates a `version.ts` that sets
// `process.env.SEPILOT_VERSION` on import; the CLI builds its Commander program
// and the daemon computes its health version at module load time, so the env var
// has to be set before `@sepilotd/cli` or `@sepilotd/daemon/*` is imported. The
// committed stub does nothing on import.
import './generated/version.js'
import { createHash } from 'node:crypto'
import {
  mkdtempSync,
  readFileSync,
  realpathSync,
  writeFileSync,
} from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { VEC0_PATH } from './generated/vec-asset.js'

const materializedVec0ByHash = new Map<string, string>()

// Parent Node tooling such as `tsx` can inject NODE_PATH. The standalone binary
// must resolve its embedded dependency graph deterministically, so clear it
// before dynamically importing the CLI/daemon graph below.
delete process.env.NODE_PATH

/**
 * `true` when this is a `bun build --compile` single-file binary.
 *
 * The reliable signal is the entry module living in Bun's virtual `$bunfs`
 * filesystem. Linux builds commonly expose `file:///$bunfs/root/...`; Windows
 * URL formatting can include a drive-root prefix, so match `$bunfs` anywhere in
 * the module URL instead of only the Linux-shaped prefix.
 * (Don't probe `process.argv[1]` with `existsSync` — Bun's fs *can* stat bunfs
 * paths, so that always looks "real" and would mis-detect the binary as a
 * script running under an interpreter.)
 */
export function isCompiledBinaryRuntime(
  moduleUrl: string = import.meta.url,
  entry: string | undefined = process.argv[1],
  execPath: string = process.execPath,
): boolean {
  if (moduleUrl.includes('$bunfs')) return true
  if (!entry) return false
  try {
    return realpathSync(entry) === realpathSync(execPath)
  } catch {
    return entry === execPath
  }
}

const IS_COMPILED_BINARY = isCompiledBinaryRuntime()

/**
 * Decide how to re-launch *this* program as the daemon.
 *
 *  - `bun build --compile` single-file binary: `process.execPath` is the binary
 *    itself, so the daemon re-entry is `<binary> __daemon` (no prefix args).
 *  - plain `dist/main.js` running under a Node/Bun interpreter: `process.execPath`
 *    is the interpreter and `process.argv[1]` is the script, so the re-entry is
 *    `<interpreter> <main.js> __daemon`.
 */
export function resolveSelfExecForRuntime(
  moduleUrl: string = import.meta.url,
  entry: string | undefined = process.argv[1],
  execPath: string = process.execPath,
): { command: string; prefixArgs: string[] } {
  if (isCompiledBinaryRuntime(moduleUrl, entry, execPath)) {
    return { command: execPath, prefixArgs: [] }
  }
  if (entry) return { command: execPath, prefixArgs: [entry] }
  // No entry script and not a compiled binary — best effort: re-exec the
  // interpreter with no prefix (callers can still override via the hook).
  return { command: execPath, prefixArgs: [] }
}

function resolveSelfExec(): { command: string; prefixArgs: string[] } {
  return resolveSelfExecForRuntime()
}

/**
 * Materialize the embedded `sqlite-vec` vec0 loadable to a *real* file on disk.
 *
 * `bun build --compile` exposes the embedded asset at a virtual `/$bunfs/...`
 * path. `dlopen(3)` (which SQLite's `sqlite3_load_extension` uses) cannot load
 * from bunfs, so we copy the bytes out to a real temp path the daemon's
 * sqlite-vec backend can `loadExtension()`.
 *
 * The file is named exactly `vec0.so` (matching what `sqlite-vec` ships) inside
 * a securely-created per-process directory: SQLite derives the extension's init
 * function from the *base filename*, so it must stay `vec0.so` for
 * `sqlite3_vec_init` to be found. The content hash cache avoids duplicate writes
 * inside one process without trusting a predictable path in the shared OS temp
 * directory. Returns the real path, or `undefined` if anything goes wrong (the
 * daemon then degrades to the non-vector `sqlite-scan` backend).
 */
function materializeVec0(bunfsPath: string): string | undefined {
  try {
    const bytes = readFileSync(bunfsPath)
    const hash = createHash('sha256').update(bytes).digest('hex').slice(0, 16)
    const cached = materializedVec0ByHash.get(hash)
    if (cached) return cached
    const dir = mkdtempSync(join(tmpdir(), `sepilot-vec-${hash}-`))
    const realPath = join(dir, 'vec0.so')
    writeFileSync(realPath, bytes, { flag: 'wx', mode: 0o700 })
    materializedVec0ByHash.set(hash, realPath)
    return realPath
  } catch {
    return undefined
  }
}

export async function bundleMain(argv: string[] = process.argv.slice(2)): Promise<void> {
  const [
    { registerStandaloneDaemon, runCli },
    { runDaemonMain },
    { startEmbeddedDaemon },
  ] = await Promise.all([
    import('@sepilotd/cli'),
    import('@sepilotd/daemon/main'),
    import('@sepilotd/daemon/embedded'),
  ])

  // When this is a `bun build --compile` binary the vec0 loadable extension was
  // embedded as a Bun file asset (a virtual `/$bunfs/...` path). `dlopen` can't
  // load from bunfs, so copy it to a real temp file and point the daemon's
  // sqlite-vec backend there. Under plain Node `VEC0_PATH` is `undefined` (the
  // committed stub) and the daemon keeps using `node_modules/sqlite-vec-*`.
  if (VEC0_PATH) {
    const realVecPath = materializeVec0(VEC0_PATH)
    if (realVecPath) process.env.SEPILOTD_SQLITE_VEC_PATH = realVecPath
  }

  const selfExec = resolveSelfExec()
  registerStandaloneDaemon({
    selfExec: true,
    selfExecCommand: selfExec.command,
    selfExecPrefixArgs: selfExec.prefixArgs,
    // Only a `bun build --compile` binary can self-update: `process.execPath`
    // is the binary itself. Under a Node/Bun interpreter it's the interpreter,
    // which `sepilot upgrade` must not touch — leave `selfBinaryPath` undefined.
    selfBinaryPath: IS_COMPILED_BINARY ? process.execPath : undefined,
    foregroundMain: (daemonArgv) => runDaemonMain(daemonArgv),
    embeddedFactory: (opts) => startEmbeddedDaemon(opts),
  })
  await runCli(argv)
}

export const __testables = {
  materializeVec0,
}

function isMainModule(): boolean {
  // A `bun build --compile` binary always executes the embedded entry.
  if (IS_COMPILED_BINARY) return true
  const entry = process.argv[1]
  if (!entry) return false
  try {
    return realpathSync(entry) === realpathSync(fileURLToPath(import.meta.url))
  } catch {
    return import.meta.url.endsWith('/main.js')
  }
}

if (isMainModule()) {
  bundleMain().catch((error) => {
    console.error(error)
    process.exit(1)
  })
}
