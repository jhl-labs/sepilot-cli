import { randomUUID } from 'node:crypto'
import { chmod, mkdir, readFile, rename, unlink, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { resolveDaemonDataDir } from './token.js'

const FORWARD_ENV_MARKER = 'SEPILOTD_FORWARD_ENV'
const FORWARD_ENV_METADATA_VERSION = 1
const FORWARD_ENV_METADATA_FILE = 'daemon-forward-env.json'
const MAX_FORWARD_ENV_NAMES = 128
const MAX_ENV_NAME_LENGTH = 128
const ENV_NAME_PATTERN = /^[A-Z_][A-Z0-9_]*$/i
const DAEMON_CHILD_ENV_ALLOWLIST = new Set([
  'COMSPEC',
  'DISPLAY',
  'APPDATA',
  'HOME',
  'HOMEDRIVE',
  'HOMEPATH',
  'LANG',
  'LC_ALL',
  'LOCALAPPDATA',
  'LOGNAME',
  'PATH',
  'PATHEXT',
  'Path',
  'ProgramData',
  'SHELL',
  'SSH_AUTH_SOCK',
  'SystemRoot',
  'TEMP',
  'TERM',
  'TMP',
  'TMPDIR',
  'USER',
  'USERPROFILE',
  'WAYLAND_DISPLAY',
  'WINDIR',
  'XDG_CACHE_HOME',
  'XDG_CONFIG_HOME',
  'XDG_DATA_HOME',
  'XDG_RUNTIME_DIR',
])

interface ForwardEnvMetadata {
  version: typeof FORWARD_ENV_METADATA_VERSION
  names: string[]
}

export interface PreparedDaemonEnvironment {
  env: Record<string, string>
  forwardNames: string[]
}

function isNodeFsError(error: unknown, code: string): boolean {
  return typeof error === 'object'
    && error !== null
    && 'code' in error
    && (error as { code?: unknown }).code === code
}

function validateForwardEnvName(rawName: string, source: string): string {
  const name = rawName.trim()
  if (
    !ENV_NAME_PATTERN.test(name)
    || name.length > MAX_ENV_NAME_LENGTH
    || name === FORWARD_ENV_MARKER
  ) {
    const label = source === '--forward-env'
      ? 'Invalid --forward-env name'
      : `Invalid ${source} environment variable name`
    throw new Error(`${label}: ${rawName}`)
  }
  return name
}

export function normalizeForwardEnvNames(
  rawNames: readonly string[],
  source = '--forward-env',
): string[] {
  const names = [...new Set(rawNames.map((name) => validateForwardEnvName(name, source)))]
    .sort()
  if (names.length > MAX_FORWARD_ENV_NAMES) {
    throw new Error(`${source} accepts at most ${MAX_FORWARD_ENV_NAMES} variable names.`)
  }
  return names
}

function parseForwardEnvMarker(raw: string | undefined): string[] {
  if (!raw?.trim()) return []
  return normalizeForwardEnvNames(
    raw.split(/[,\s]+/).map((name) => name.trim()).filter(Boolean),
    FORWARD_ENV_MARKER,
  )
}

// Non-secret, fixed-shape flags that are safe to always forward: they carry
// no ambient-shell secret material and CLAUDE.md's Operational Knobs
// document setting them on the same `sepilot start|restart` invocation that
// spawns the daemon (e.g. `SEPILOT_DEBUG=1 sepilot restart`). Without this,
// the child-env allowlist silently drops them and the documented workflow
// produces a daemon that never writes the trace file the docs promise.
const ALWAYS_FORWARDED_DAEMON_FLAGS = new Set(['SEPILOT_VERSION', 'SEPILOT_DEBUG'])

function shouldForwardDaemonChildEnv(key: string): boolean {
  return DAEMON_CHILD_ENV_ALLOWLIST.has(key)
    || key.startsWith('SEPILOTD_')
    || ALWAYS_FORWARDED_DAEMON_FLAGS.has(key)
}

function buildDaemonChildEnvForNames(
  env: NodeJS.ProcessEnv,
  overrides: Record<string, string>,
  forwardNames: readonly string[],
): PreparedDaemonEnvironment {
  const names = normalizeForwardEnvNames(forwardNames)
  const explicit = new Set(names)
  for (const name of names) {
    if (typeof env[name] !== 'string') {
      throw new Error(`--forward-env ${name} is not set in an approved environment source.`)
    }
  }

  const childEnv: Record<string, string> = {}
  for (const [key, value] of Object.entries(env)) {
    if (typeof value !== 'string' || key === FORWARD_ENV_MARKER) continue
    if (shouldForwardDaemonChildEnv(key) || explicit.has(key)) childEnv[key] = value
  }
  const finalEnv = { ...childEnv, ...overrides }
  delete finalEnv[FORWARD_ENV_MARKER]
  if (names.length > 0) finalEnv[FORWARD_ENV_MARKER] = names.join(',')

  return {
    env: finalEnv,
    forwardNames: names,
  }
}

export function buildDaemonChildEnv(
  env: NodeJS.ProcessEnv = process.env,
  overrides: Record<string, string> = {},
  requestedForwardNames: readonly string[] = [],
): Record<string, string> {
  const names = normalizeForwardEnvNames([
    ...parseForwardEnvMarker(env[FORWARD_ENV_MARKER]),
    ...requestedForwardNames,
  ])
  return buildDaemonChildEnvForNames(env, overrides, names).env
}

export function daemonForwardEnvMetadataPath(dataDir = resolveDaemonDataDir()): string {
  return join(dataDir, 'security', FORWARD_ENV_METADATA_FILE)
}

async function readForwardEnvMetadata(dataDir: string): Promise<string[] | null> {
  const path = daemonForwardEnvMetadataPath(dataDir)
  let raw: string
  try {
    raw = await readFile(path, 'utf-8')
  } catch (error) {
    if (isNodeFsError(error, 'ENOENT')) return null
    throw new Error(
      `Failed to read daemon forwarding metadata at ${path}: ${
        error instanceof Error ? error.message : String(error)
      }`,
      { cause: error },
    )
  }

  try {
    const parsed = JSON.parse(raw) as Partial<ForwardEnvMetadata>
    if (
      parsed.version !== FORWARD_ENV_METADATA_VERSION
      || !Array.isArray(parsed.names)
      || !parsed.names.every((name) => typeof name === 'string')
    ) {
      throw new Error('unsupported schema')
    }
    return normalizeForwardEnvNames(parsed.names, 'daemon forwarding metadata')
  } catch (error) {
    throw new Error(
      `Refusing to use invalid daemon forwarding metadata at ${path}: ${
        error instanceof Error ? error.message : String(error)
      }`,
      { cause: error },
    )
  }
}

export async function persistDaemonForwardEnvNames(
  names: readonly string[],
  dataDir = resolveDaemonDataDir(),
): Promise<string> {
  const normalized = normalizeForwardEnvNames(names)
  const path = daemonForwardEnvMetadataPath(dataDir)
  const directory = join(dataDir, 'security')
  const temporaryPath = `${path}.tmp-${process.pid}-${randomUUID()}`
  const metadata: ForwardEnvMetadata = {
    version: FORWARD_ENV_METADATA_VERSION,
    names: normalized,
  }

  await mkdir(directory, { recursive: true, mode: 0o700 })
  await chmod(directory, 0o700)
  try {
    await writeFile(temporaryPath, `${JSON.stringify(metadata, null, 2)}\n`, {
      encoding: 'utf-8',
      flag: 'wx',
      mode: 0o600,
    })
    await rename(temporaryPath, path)
    await chmod(path, 0o600)
  } catch (error) {
    await unlink(temporaryPath).catch(() => undefined)
    throw error
  }
  return path
}

function parseProcessEnvironment(raw: Buffer): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {}
  for (const entry of raw.toString('utf-8').split('\0')) {
    const separator = entry.indexOf('=')
    if (separator <= 0) continue
    env[entry.slice(0, separator)] = entry.slice(separator + 1)
  }
  return env
}

async function readRunningDaemonEnvironment(dataDir: string): Promise<NodeJS.ProcessEnv> {
  if (process.platform !== 'linux') return {}
  let pid: number
  try {
    const rawPid = await readFile(join(dataDir, 'sepilotd.pid'), 'utf-8')
    pid = Number(rawPid.trim())
    if (!Number.isSafeInteger(pid) || pid <= 0) return {}
  } catch {
    return {}
  }

  try {
    return parseProcessEnvironment(await readFile(`/proc/${pid}/environ`))
  } catch {
    return {}
  }
}

export async function prepareDaemonEnvironment(options: {
  requestedForwardNames?: readonly string[]
  clearForwardEnv?: boolean
  dataDir?: string
  env?: NodeJS.ProcessEnv
  overrides?: Record<string, string>
} = {}): Promise<PreparedDaemonEnvironment> {
  const env = options.env ?? process.env
  const dataDir = options.dataDir ?? resolveDaemonDataDir()
  const requested = options.requestedForwardNames ?? []
  if (options.clearForwardEnv && requested.length > 0) {
    throw new Error('--clear-forward-env cannot be combined with --forward-env.')
  }

  if (options.clearForwardEnv) {
    return buildDaemonChildEnvForNames(env, options.overrides ?? {}, [])
  }
  if (requested.length > 0) {
    return buildDaemonChildEnvForNames(env, options.overrides ?? {}, requested)
  }

  const persistedNames = await readForwardEnvMetadata(dataDir)
  const names = persistedNames ?? parseForwardEnvMarker(env[FORWARD_ENV_MARKER])
  if (names.length === 0) {
    return buildDaemonChildEnvForNames(env, options.overrides ?? {}, [])
  }

  const sourceEnv: NodeJS.ProcessEnv = { ...env }
  const missing = names.filter((name) => typeof sourceEnv[name] !== 'string')
  if (missing.length > 0) {
    const runningEnv = await readRunningDaemonEnvironment(dataDir)
    const runningNames = new Set(parseForwardEnvMarker(runningEnv[FORWARD_ENV_MARKER]))
    for (const name of missing) {
      if (runningNames.has(name) && typeof runningEnv[name] === 'string') {
        sourceEnv[name] = runningEnv[name]
      }
    }
  }

  const unavailable = names.filter((name) => typeof sourceEnv[name] !== 'string')
  if (unavailable.length > 0) {
    throw new Error(
      `Cannot preserve daemon forwarding for ${unavailable.join(', ')}. `
        + 'Export the missing variable(s) and retry with --forward-env, or intentionally remove '
        + 'the saved forwarding contract with `sepilot restart --clear-forward-env`.',
    )
  }
  return buildDaemonChildEnvForNames(sourceEnv, options.overrides ?? {}, names)
}
