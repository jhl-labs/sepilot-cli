import { appendFile, chmod, mkdir, rename, stat } from 'node:fs/promises'
import { homedir } from 'node:os'
import { dirname, join } from 'node:path'

const MAX_LOG_BYTES = 5 * 1024 * 1024

function logFile(): string {
  return (
    process.env.SEPILOT_CLI_LOG_FILE || join(homedir(), '.sepilotd', 'logs', 'cli-tui.jsonl')
  )
}

const dirReadyByPath = new Map<string, Promise<void>>()
let writeChain: Promise<void> = Promise.resolve()
const instanceId = Math.random().toString(36).slice(2, 10)

function enabled(): boolean {
  return process.env.SEPILOT_DEBUG === '1' && process.env.SEPILOT_CLI_LOG !== '0'
}

export function isTuiDebugLevel(): boolean {
  return (process.env.SEPILOT_CLI_LOG_LEVEL ?? '').toLowerCase() === 'debug'
}

async function ensureDir(file: string): Promise<void> {
  const dir = dirname(file)
  let ready = dirReadyByPath.get(dir)
  if (!ready) {
    ready = mkdir(dir, { recursive: true, mode: 0o700 })
      .then(() => chmod(dir, 0o700).catch(() => undefined))
      .then(() => undefined)
    dirReadyByPath.set(dir, ready)
  }
  await ready
}

async function rotateIfNeeded(file: string): Promise<void> {
  try {
    const info = await stat(file)
    if (info.size > MAX_LOG_BYTES) {
      await rename(file, `${file}.1`)
    }
  } catch {
  }
}

export function tuiLog(event: string, data?: Record<string, unknown>): void {
  if (!enabled()) return
  const line = `${JSON.stringify({
    ts: new Date().toISOString(),
    pid: process.pid,
    ins: instanceId,
    event,
    ...data,
  })}\n`
  writeChain = writeChain
    .then(async () => {
      const file = logFile()
      await ensureDir(file)
      await rotateIfNeeded(file)
      await appendFile(file, line, { encoding: 'utf8', mode: 0o600 })
      await chmod(file, 0o600).catch(() => {})
    })
    .catch(() => {
    })
}

export function flushTuiLog(): Promise<void> {
  return writeChain
}

export function tuiLogError(
  event: string,
  error: unknown,
  data?: Record<string, unknown>,
): void {
  const detail =
    error instanceof Error
      ? { message: error.message, name: error.name, stack: error.stack?.split('\n').slice(0, 4) }
      : { message: String(error) }
  tuiLog(event, { ...data, error: detail })
}

export function logPreview(text: string | undefined | null, max = 120): string | undefined {
  if (!text) return undefined
  const singleLine = text.replace(/\s+/g, ' ').trim()
  return singleLine.length <= max ? singleLine : `${singleLine.slice(0, max)}…`
}
