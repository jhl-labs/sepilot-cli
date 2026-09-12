import { chmod, copyFile, open, rename, rm, stat, truncate } from 'node:fs/promises'

export const MAX_SERVICE_LOG_READ_BYTES = 256 * 1024
export const MAX_SERVICE_LOG_FOLLOW_MS = 30_000
export const DEFAULT_SERVICE_LOG_ROTATE_BYTES = 10 * 1024 * 1024
export const DEFAULT_SERVICE_LOG_ROTATE_FILES = 3
const DEFAULT_SERVICE_LOG_READ_BYTES = 64 * 1024
const DEFAULT_SERVICE_LOG_POLL_INTERVAL_MS = 250
const SERVICE_LOG_FILE_MODE = 0o600

export interface ServiceLogReadOptions {
  offset?: number
  limitBytes?: number
  tailBytes?: number
}

export interface ServiceLogRotationOptions {
  maxBytes?: number
  maxFiles?: number
}

export interface ServiceLogReadResult {
  text: string
  nextOffset: number
}

interface SelectedLogRange {
  start: number
  count: number
}

export async function readServiceLogFile(
  path: string,
  options: ServiceLogReadOptions = {},
): Promise<ServiceLogReadResult> {
  let size = 0
  try {
    size = (await stat(path)).size
  } catch {
    return { text: '', nextOffset: 0 }
  }

  const range = selectLogRange(size, options)
  if (range.count <= 0) return { text: '', nextOffset: size }

  const file = await open(path, 'r')
  try {
    const buffer = Buffer.alloc(range.count)
    const { bytesRead } = await file.read(buffer, 0, range.count, range.start)
    return {
      text: buffer.subarray(0, bytesRead).toString('utf-8'),
      nextOffset: range.start + bytesRead,
    }
  } finally {
    await file.close()
  }
}

export function readServiceLogText(
  text: string,
  options: ServiceLogReadOptions = {},
): ServiceLogReadResult {
  const buffer = Buffer.from(text, 'utf-8')
  const range = selectLogRange(buffer.length, options)
  if (range.count <= 0) return { text: '', nextOffset: buffer.length }

  return {
    text: buffer.subarray(range.start, range.start + range.count).toString('utf-8'),
    nextOffset: range.start + range.count,
  }
}

export function hasServiceLogOutput(chunk: { stdout: string; stderr: string }): boolean {
  return chunk.stdout.length > 0 || chunk.stderr.length > 0
}

export function normalizeServiceLogFollowMs(value: number | undefined): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return 0
  }
  return Math.max(0, Math.min(Math.trunc(value), MAX_SERVICE_LOG_FOLLOW_MS))
}

export function normalizeServiceLogPollIntervalMs(value: number | undefined): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return DEFAULT_SERVICE_LOG_POLL_INTERVAL_MS
  }
  return Math.max(50, Math.min(Math.trunc(value), 5_000))
}

export async function sleep(ms: number): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, ms))
}

export function normalizeServiceLogRotation(
  options: ServiceLogRotationOptions | undefined,
): Required<ServiceLogRotationOptions> {
  return {
    maxBytes: normalizeRotationByteCount(options?.maxBytes),
    maxFiles: normalizeRotationFileCount(options?.maxFiles),
  }
}

export async function rotateServiceLogFile(
  path: string,
  options: Required<ServiceLogRotationOptions>,
): Promise<boolean> {
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

  // Copy+truncate preserves the active file inode, so already-running
  // process, systemd, or launchd writers that hold the file open keep writing
  // to the same path after rotation.
  await copyFile(path, `${path}.1`)
  await chmod(`${path}.1`, SERVICE_LOG_FILE_MODE).catch(() => {})
  await truncate(path, 0)
  await chmod(path, SERVICE_LOG_FILE_MODE).catch(() => {})
  return true
}

function selectLogRange(size: number, options: ServiceLogReadOptions): SelectedLogRange {
  if (size <= 0) return { start: 0, count: 0 }

  const limitBytes = normalizeByteCount(options.limitBytes, DEFAULT_SERVICE_LOG_READ_BYTES)
  const offset = normalizeOffset(options.offset)
  const shouldTail = offset === undefined && typeof options.tailBytes === 'number' && Number.isFinite(options.tailBytes)

  const start = shouldTail
    ? Math.max(0, size - Math.min(normalizeByteCount(options.tailBytes, limitBytes), limitBytes))
    : offset !== undefined && offset <= size
      ? offset
      : 0

  return {
    start,
    count: Math.min(limitBytes, size - start),
  }
}

function normalizeByteCount(value: number | undefined, fallback: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return fallback
  }
  return Math.max(1, Math.min(Math.trunc(value), MAX_SERVICE_LOG_READ_BYTES))
}

function normalizeOffset(value: number | undefined): number | undefined {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return undefined
  }
  return Math.max(0, Math.trunc(value))
}

function normalizeRotationByteCount(value: number | undefined): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return DEFAULT_SERVICE_LOG_ROTATE_BYTES
  }
  return Math.max(1, Math.trunc(value))
}

function normalizeRotationFileCount(value: number | undefined): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return DEFAULT_SERVICE_LOG_ROTATE_FILES
  }
  return Math.max(1, Math.min(Math.trunc(value), 20))
}
