import { readFile, rename, writeFile } from 'node:fs/promises'
import { join } from 'node:path'

export const MANAGED_ENV_FILENAME = '.env'

const managedEnvState = new Map<string, string>()

function normalizeEnvKey(key: string): string | null {
  const trimmed = key.trim()
  if (!trimmed || !/^[A-Za-z_][A-Za-z0-9_]*$/.test(trimmed)) {
    return null
  }
  return trimmed
}

function parseQuotedEnvValue(
  value: string,
): string {
  if (value.length < 2) {
    return value
  }

  const quote = value[0]
  if ((quote !== '"' && quote !== '\'') || value[value.length - 1] !== quote) {
    return value
  }

  const inner = value.slice(1, -1)
  if (quote === '\'') {
    return inner
  }

  try {
    return JSON.parse(value)
  } catch {
    return inner
      .replace(/\\n/g, '\n')
      .replace(/\\r/g, '\r')
      .replace(/\\t/g, '\t')
      .replace(/\\"/g, '"')
      .replace(/\\\\/g, '\\')
  }
}

export function parseManagedEnv(text: string): Record<string, string> {
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

    const key = normalizeEnvKey(line.slice(0, separator))
    if (!key) {
      continue
    }

    const rawValue = line.slice(separator + 1).trim()
    entries[key] = parseQuotedEnvValue(rawValue)
  }

  return entries
}

function serializeManagedEnvValue(value: string): string {
  if (/^[A-Za-z0-9_./:@-]+$/.test(value)) {
    return value
  }

  return JSON.stringify(value)
}

export function serializeManagedEnv(entries: Record<string, string>): string {
  return Object.entries(entries)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, value]) => `${key}=${serializeManagedEnvValue(value)}`)
    .join('\n')
}

export function managedEnvPath(dataDir: string): string {
  return join(dataDir, MANAGED_ENV_FILENAME)
}

export async function readManagedEnvFile(dataDir: string): Promise<Record<string, string>> {
  try {
    return parseManagedEnv(await readFile(managedEnvPath(dataDir), 'utf-8'))
  } catch {
    return {}
  }
}

export function applyManagedEnvEntries(
  entries: Record<string, string>,
  options: { overrideExisting?: boolean } = {},
): void {
  const overrideExisting = options.overrideExisting ?? false
  for (const [key, value] of Object.entries(entries)) {
    if (!overrideExisting && process.env[key] != null) {
      continue
    }
    process.env[key] = value
  }
}

export async function loadManagedEnvFile(dataDir: string): Promise<Record<string, string>> {
  const entries = await readManagedEnvFile(dataDir)
  applyManagedEnvEntries(entries)
  managedEnvState.clear()
  for (const [key, value] of Object.entries(entries)) {
    managedEnvState.set(key, value)
  }
  return entries
}

export interface UpdateManagedEnvFileResult {
  path: string
  updated: string[]
  removed: string[]
}

export async function updateManagedEnvFile(
  dataDir: string,
  updates: Record<string, string | null | undefined>,
): Promise<UpdateManagedEnvFileResult> {
  const current = await readManagedEnvFile(dataDir)
  const next = { ...current }
  const updated = new Set<string>()
  const removed = new Set<string>()

  for (const [rawKey, rawValue] of Object.entries(updates)) {
    const key = normalizeEnvKey(rawKey)
    if (!key) {
      continue
    }

    if (rawValue == null || rawValue.length === 0) {
      if (Object.hasOwn(next, key)) {
        delete next[key]
        removed.add(key)
      }
      updated.delete(key)
      continue
    }

    next[key] = rawValue
    updated.add(key)
    removed.delete(key)
  }

  const envPath = managedEnvPath(dataDir)
  const tempPath = `${envPath}.tmp`
  const serialized = serializeManagedEnv(next)
  await writeFile(tempPath, serialized ? `${serialized}\n` : '', { mode: 0o600 })
  await rename(tempPath, envPath)

  for (const [key, previousValue] of managedEnvState.entries()) {
    if (!Object.hasOwn(next, key) && process.env[key] === previousValue) {
      delete process.env[key]
    }
  }
  for (const [key, value] of Object.entries(next)) {
    process.env[key] = value
  }

  managedEnvState.clear()
  for (const [key, value] of Object.entries(next)) {
    managedEnvState.set(key, value)
  }

  return {
    path: envPath,
    updated: Array.from(updated).sort(),
    removed: Array.from(removed).sort(),
  }
}
