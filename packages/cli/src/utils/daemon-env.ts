import { readFile, rename, writeFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { dirname, join } from 'node:path'

const MANAGED_DAEMON_ENV_FILENAME = '.env'

export function daemonDataDir(homeDir = homedir()): string {
  return join(homeDir, '.sepilotd')
}

export function daemonEnvPath(options: {
  homeDir?: string
  configFilePath?: string
} = {}): string {
  if (options.configFilePath) {
    return join(dirname(options.configFilePath), MANAGED_DAEMON_ENV_FILENAME)
  }

  return join(daemonDataDir(options.homeDir), MANAGED_DAEMON_ENV_FILENAME)
}

function parseQuotedEnvValue(value: string): string {
  if (value.length < 2) {
    return value
  }

  const quote = value[0]
  if ((quote !== '"' && quote !== '\'') || value[value.length - 1] !== quote) {
    return value
  }

  if (quote === '\'') {
    return value.slice(1, -1)
  }

  try {
    return JSON.parse(value)
  } catch {
    return value
      .slice(1, -1)
      .replace(/\\n/g, '\n')
      .replace(/\\r/g, '\r')
      .replace(/\\t/g, '\t')
      .replace(/\\"/g, '"')
      .replace(/\\\\/g, '\\')
  }
}

export function parseDaemonEnv(text: string): Record<string, string> {
  const entries: Record<string, string> = {}

  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith('#')) {
      continue
    }

    const separator = line.indexOf('=')
    if (separator <= 0) {
      continue
    }

    const key = line.slice(0, separator).trim()
    if (!key) {
      continue
    }

    entries[key] = parseQuotedEnvValue(line.slice(separator + 1).trim())
  }

  return entries
}

function serializeDaemonEnvValue(value: string): string {
  if (/^[A-Za-z0-9_./:@-]+$/.test(value)) {
    return value
  }

  return JSON.stringify(value)
}

export function serializeDaemonEnv(entries: Record<string, string>): string {
  return Object.entries(entries)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, value]) => `${key}=${serializeDaemonEnvValue(value)}`)
    .join('\n')
}

function isNodeFsError(err: unknown, code: string): boolean {
  return (
    typeof err === 'object'
    && err !== null
    && 'code' in err
    && (err as { code?: unknown }).code === code
  )
}

/**
 * Read the daemon-managed `.env` file. Returns an empty record only
 * when the file genuinely does not exist (ENOENT). Every other read
 * error — EACCES, EBUSY, EIO — is rethrown so callers like
 * `updateDaemonEnvFile` cannot silently merge updates into `{}` and
 * truncate the user's existing `.env`.
 */
export async function readDaemonEnvFile(options: {
  homeDir?: string
  configFilePath?: string
} = {}): Promise<Record<string, string>> {
  const envPath = daemonEnvPath(options)
  try {
    return parseDaemonEnv(await readFile(envPath, 'utf-8'))
  } catch (err) {
    if (isNodeFsError(err, 'ENOENT')) {
      return {}
    }
    throw new Error(
      `Failed to read daemon env file at ${envPath}: ${
        err instanceof Error ? err.message : String(err)
      }`,
      { cause: err },
    )
  }
}

export async function updateDaemonEnvFile(
  updates: Record<string, string>,
  options: {
    homeDir?: string
    configFilePath?: string
  } = {},
): Promise<string> {
  const envPath = daemonEnvPath(options)
  const tempPath = `${envPath}.tmp`
  // Aborts the update on a non-ENOENT read failure rather than
  // merging into `{}` and writing a near-empty file over the
  // existing `.env`. The caller surfaces the error to the operator
  // who can fix the underlying permission/IO problem before retrying.
  const current = await readDaemonEnvFile(options)
  const next = {
    ...current,
    ...updates,
  }
  const serialized = serializeDaemonEnv(next)

  await writeFile(tempPath, serialized ? `${serialized}\n` : '', { mode: 0o600 })
  await rename(tempPath, envPath)

  return envPath
}

export function mergeDaemonEnv(
  shellEnv: Record<string, string | undefined>,
  managedEnv: Record<string, string | undefined>,
): Record<string, string | undefined> {
  const merged: Record<string, string | undefined> = { ...managedEnv }
  for (const [key, value] of Object.entries(shellEnv)) {
    if (value !== undefined) {
      merged[key] = value
    }
  }
  return merged
}
