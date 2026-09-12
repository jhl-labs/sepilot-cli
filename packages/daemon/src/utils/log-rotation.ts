import { copyFile, rename, rm, stat, truncate } from 'node:fs/promises'

const ROTATED_FILE_MODE = 0o600

export interface LogRotationOptions {
  enabled: boolean
  maxBytes: number
  maxFiles: number
}

export const DEFAULT_LOG_ROTATION: LogRotationOptions = {
  enabled: true,
  maxBytes: 10 * 1024 * 1024,
  maxFiles: 3,
}

// copy + truncate preserves the file's inode and offset for whatever
// process is currently writing into it. `appendFile` in Node opens a fresh
// handle per call, so a plain rename would still be safe — but copy+truncate
// is the same pattern service-supervisor uses for its managed services, and
// matching it keeps the rotation behaviour predictable across both layers.
export async function rotateLogFileIfNeeded(
  path: string,
  options: LogRotationOptions,
): Promise<boolean> {
  if (!options.enabled) return false

  let size = 0
  try {
    size = (await stat(path)).size
  } catch {
    return false
  }
  if (size <= options.maxBytes) return false

  await rm(`${path}.${options.maxFiles}`, { force: true }).catch(() => {})
  for (let index = options.maxFiles - 1; index >= 1; index -= 1) {
    await rename(`${path}.${index}`, `${path}.${index + 1}`).catch(() => {})
  }

  await copyFile(path, `${path}.1`)
  try {
    const { chmod } = await import('node:fs/promises')
    await chmod(`${path}.1`, ROTATED_FILE_MODE).catch(() => {})
  } catch {
    // chmod is best-effort; Windows ignores POSIX modes anyway.
  }
  await truncate(path, 0)
  return true
}

export function normalizeLogRotationOptions(
  options: Partial<LogRotationOptions> | undefined,
  fallback: LogRotationOptions = DEFAULT_LOG_ROTATION,
): LogRotationOptions {
  return {
    enabled: typeof options?.enabled === 'boolean' ? options.enabled : fallback.enabled,
    maxBytes: normalizeBytes(options?.maxBytes, fallback.maxBytes),
    maxFiles: normalizeFiles(options?.maxFiles, fallback.maxFiles),
  }
}

function normalizeBytes(value: number | undefined, fallback: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) return fallback
  return Math.max(1024, Math.trunc(value))
}

function normalizeFiles(value: number | undefined, fallback: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) return fallback
  return Math.max(1, Math.min(Math.trunc(value), 50))
}
