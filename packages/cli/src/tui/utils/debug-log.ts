// Structured JSONL diagnostics for the interactive TUI.
//
// Field reports like "Esc then /resume failed then BUSY" are impossible to
// reconstruct without a client-side trace: the daemon logs its side, but the
// TUI's view (what it sent, which stream events arrived, when the user
// cancelled, what the error payload actually was) lived only in the terminal
// scrollback. This logger appends one JSON object per line to
// `~/.sepilotd/logs/cli-tui.jsonl` so a session can be replayed after the
// fact with `jq`. No file is created unless SEPILOT_DEBUG=1.
//
// Knobs:
//   SEPILOT_DEBUG=1            enable debug logging
//   SEPILOT_CLI_LOG=0          disable only the TUI debug log
//   SEPILOT_CLI_LOG_LEVEL=debug  also log every stream payload type
//                                (default `info` logs lifecycle + decisions
//                                + errors only)
//
// Never throws and never blocks the render path: writes are fire-and-forget
// appendFile calls, errors are swallowed (a broken log must not break chat).

import { appendFile, chmod, mkdir, rename, stat } from 'node:fs/promises'
import { homedir } from 'node:os'
import { dirname, join } from 'node:path'

const MAX_LOG_BYTES = 5 * 1024 * 1024

/** Resolved per call so tests (and operators) can redirect via env. */
function logFile(): string {
  return (
    process.env.SEPILOT_CLI_LOG_FILE || join(homedir(), '.sepilotd', 'logs', 'cli-tui.jsonl')
  )
}

const dirReadyByPath = new Map<string, Promise<void>>()
let writeChain: Promise<void> = Promise.resolve()
/** Correlates all lines from one TUI process. */
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
    // Missing file (first write) or rename race — either way, keep writing.
  }
}

/**
 * Append one structured line. `data` values should already be small
 * (ids, enum strings, short previews) — this is a trace, not a transcript.
 */
export function tuiLog(event: string, data?: Record<string, unknown>): void {
  if (!enabled()) return
  const line = `${JSON.stringify({
    ts: new Date().toISOString(),
    pid: process.pid,
    ins: instanceId,
    event,
    ...data,
  })}\n`
  // Serialize writes so rotation and appends can't interleave mid-line.
  writeChain = writeChain
    .then(async () => {
      const file = logFile()
      await ensureDir(file)
      await rotateIfNeeded(file)
      await appendFile(file, line, { encoding: 'utf8', mode: 0o600 })
      await chmod(file, 0o600).catch(() => {})
    })
    .catch(() => {
      // Logging must never surface as a chat error.
    })
}

/** Await all queued writes — for tests only. */
export function flushTuiLog(): Promise<void> {
  return writeChain
}

/** Convenience wrapper that serializes an unknown error safely. */
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

/** Trim potentially long user/model text to a loggable preview. */
export function logPreview(text: string | undefined | null, max = 120): string | undefined {
  if (!text) return undefined
  const singleLine = text.replace(/\s+/g, ' ').trim()
  return singleLine.length <= max ? singleLine : `${singleLine.slice(0, max)}…`
}
