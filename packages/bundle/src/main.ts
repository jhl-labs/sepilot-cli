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

delete process.env.NODE_PATH

const BUN_VIRTUAL_ENTRY_MARKER_PATTERN = /\$bunfs|~BUN/i

export function isCompiledBinaryRuntime(
  moduleUrl: string = import.meta.url,
  entry: string | undefined = process.argv[1],
  execPath: string = process.execPath,
): boolean {
  if (BUN_VIRTUAL_ENTRY_MARKER_PATTERN.test(moduleUrl)) return true
  if (!entry) return false
  if (BUN_VIRTUAL_ENTRY_MARKER_PATTERN.test(entry)) return true
  try {
    return realpathSync(entry) === realpathSync(execPath)
  } catch {
    return entry === execPath
  }
}

const IS_COMPILED_BINARY = isCompiledBinaryRuntime()

export function resolveSelfExecForRuntime(
  moduleUrl: string = import.meta.url,
  entry: string | undefined = process.argv[1],
  execPath: string = process.execPath,
): { command: string; prefixArgs: string[] } {
  if (isCompiledBinaryRuntime(moduleUrl, entry, execPath)) {
    return { command: execPath, prefixArgs: [] }
  }
  if (entry) return { command: execPath, prefixArgs: [entry] }
  return { command: execPath, prefixArgs: [] }
}

function resolveSelfExec(): { command: string; prefixArgs: string[] } {
  return resolveSelfExecForRuntime()
}

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

  if (VEC0_PATH) {
    const realVecPath = materializeVec0(VEC0_PATH)
    if (realVecPath) process.env.SEPILOTD_SQLITE_VEC_PATH = realVecPath
  }

  const selfExec = resolveSelfExec()
  registerStandaloneDaemon({
    selfExec: true,
    selfExecCommand: selfExec.command,
    selfExecPrefixArgs: selfExec.prefixArgs,
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
