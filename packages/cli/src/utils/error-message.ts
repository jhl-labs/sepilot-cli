import chalk from 'chalk'
import { ApiHttpError } from '@sepilotd/api-client'
import { resolveCliStreamIdleMs } from './stream-idle.js'
import { resolveCliStreamConnectMs } from './stream-connect.js'

export function errorMessage(err: unknown): string {
  if (err instanceof Error) return err.message
  if (typeof err === 'string') return err
  return String(err)
}

/**
 * Render an api-client error as something a human reads. Handles three
 * shapes, in order of preference:
 *
 *  1. `ApiHttpError` (new) — daemon/gateway returned a structured
 *     `{ error: { code, message, requestId } }` envelope; we render
 *     `<status> <code>: <message>` so the user sees the diagnostic code
 *     once and the message once, no JSON.
 *  2. Legacy `<status>: <json>` Error message — older transports throw a
 *     plain `Error` whose `message` carries the body verbatim. We unwrap
 *     it the same way for backward compatibility.
 *  3. Anything else — fall through to the raw `Error.message`.
 */
export function friendlyErrorMessage(err: unknown): string {
  if (err instanceof ApiHttpError) {
    if (err.code) return `${err.status} ${err.code}: ${err.message}`
    // The fallback ApiHttpError (no envelope) already encodes the
    // status as a prefix on `.message` — surfacing it again would
    // produce "404: 404: ..." double prefixes. Detect that shape and
    // either run it through the legacy unwrap below or return as-is.
    const legacyMatch = /^(\d{3}):\s*(\{.*\})\s*$/s.exec(err.message)
    if (legacyMatch) {
      try {
        const body = JSON.parse(legacyMatch[2])
        const inner = body?.error
        const message =
          typeof inner === 'string'
            ? inner
            : typeof inner?.message === 'string'
              ? inner.message
              : typeof body?.message === 'string'
                ? body.message
                : null
        if (message) return `${legacyMatch[1]}: ${message}`
      } catch { /* ignore */ }
    }
    return err.message
  }

  const raw = errorMessage(err)
  const match = /^(\d{3}):\s*(\{.*\})\s*$/s.exec(raw)
  if (!match) return raw
  try {
    const body = JSON.parse(match[2])
    const inner = body?.error
    const message =
      typeof inner === 'string'
        ? inner
        : typeof inner?.message === 'string'
          ? inner.message
          : typeof body?.message === 'string'
            ? body.message
            : null
    if (message) return `${match[1]}: ${message}`
  } catch {
    // ignore JSON parse failures and surface the raw daemon string
  }
  return raw
}

/**
 * Extract status, user-facing text, and optional contract recovery details
 * from typed HTTP errors and foreground SSE rejection strings. Returns null
 * when the error is not recognisable as HTTP so the caller can fall through.
 */
interface ParsedJsonError {
  inner: string
  code?: string
  details?: Readonly<Record<string, unknown>>
}

function parseJsonError(text: string): ParsedJsonError {
  try {
    const parsed = JSON.parse(text) as Record<string, unknown> | null
    if (!parsed || typeof parsed !== 'object') return { inner: text }
    const e = parsed.error
    if (typeof e === 'string') {
      const hasContractMessage = typeof parsed.message === 'string'
      const details = Object.fromEntries(
        Object.entries(parsed).filter(([key]) => !['error', 'message', 'requestId'].includes(key)),
      )
      return {
        inner: typeof parsed.message === 'string' ? parsed.message : e,
        code: hasContractMessage ? e : undefined,
        details: Object.keys(details).length > 0 ? details : undefined,
      }
    }
    if (e && typeof e === 'object' && !Array.isArray(e)) {
      const errorRecord = e as Record<string, unknown>
      const details = Object.fromEntries(
        Object.entries(errorRecord).filter(([key]) => !['code', 'message', 'requestId'].includes(key)),
      )
      return {
        inner: typeof errorRecord.message === 'string'
          ? errorRecord.message
          : typeof errorRecord.code === 'string'
            ? errorRecord.code
            : text,
        code: typeof errorRecord.code === 'string' ? errorRecord.code : undefined,
        details: Object.keys(details).length > 0 ? details : undefined,
      }
    }
    if (typeof parsed.message === 'string') return { inner: parsed.message }
  } catch { /* keep raw */ }
  return { inner: text }
}

function extractStatus(err: unknown): {
  status: number
  inner: string
  code?: string
  details?: Readonly<Record<string, unknown>>
} | null {
  if (err instanceof ApiHttpError) {
    // The legacy fallback packs `<status>: <body>` into `.message`. Strip
    // the prefix so the caller doesn't print "503: 503: ..." double, and
    // unwrap any `{"error":"..."}` body the same way friendlyErrorMessage
    // does so the printed copy doesn't show raw braces.
    const stripped = err.message.replace(/^\d{3}:\s*/, '')
    const parsed = parseJsonError(stripped)
    return {
      status: err.status,
      inner: parsed.inner,
      code: err.code ?? parsed.code,
      details: err.details ?? parsed.details,
    }
  }
  const raw = errorMessage(err)
  const streamFailure = /^chat stream failed \((\d{3})\):\s*(.+)$/s.exec(raw)
  if (streamFailure) {
    return {
      status: Number.parseInt(streamFailure[1], 10),
      ...parseJsonError(streamFailure[2]),
    }
  }
  const m = /^(\d{3}):\s*(.+)$/s.exec(raw)
  if (!m) return null
  return { status: Number.parseInt(m[1], 10), ...parseJsonError(m[2]) }
}

export interface PrintApiErrorOptions {
  /**
   * Optional context-specific suggestion appended in gray. Use for retry
   * guidance unique to the calling command (e.g., 'Re-run with --session
   * <id>' for ask). Skip for idempotent commands where retry is obvious.
   */
  hint?: string
}

export interface PrintStreamErrorOptions {
  /**
   * Current --session id, if the caller is mid-stream on a known
   * session. When present we tell the user how to resume from the same
   * session id; when absent we suggest passing one next time.
   */
  sessionId?: string
}

function isTransportDropMessage(message: string): boolean {
  return (
    /^terminated$/i.test(message)
    || /socket connection was closed unexpectedly/i.test(message)
    || /\bfetch failed\b/i.test(message)
    || /\bECONNRESET\b/i.test(message)
    || /\bECONNREFUSED\b/i.test(message)
  )
}

function sanitizeTransportDropDetail(message: string): string {
  return message
    .replace(
      /\s*For more information, pass ['"]?verbose['"]?\s*:\s*true\b[\s\S]*$/i,
      '',
    )
    .trim()
}

export function isRecoverableStreamDrop(err: unknown): boolean {
  const message = errorMessage(err)
  const cause = (err as { cause?: { code?: string } }).cause
  return (
    isTransportDropMessage(message)
    || cause?.code === 'UND_ERR_SOCKET'
    || cause?.code === 'ECONNRESET'
    || cause?.code === 'ECONNREFUSED'
  )
}

/**
 * Build the multi-line friendly text for an SSE-stream failure. Returns
 * null when the error doesn't match a known stream-failure shape so the
 * caller can fall through. Plain text (no chalk) so React surfaces (TUI)
 * can dispatch it through their own rendering pipeline; the
 * `printStreamError` wrapper colourises for the cli stderr path.
 */
export function formatStreamError(
  err: unknown,
  opts: PrintStreamErrorOptions = {},
): string | null {
  const message = errorMessage(err)
  if (/stream-connect-timeout/i.test(message)) {
    const connectSec = Math.round(resolveCliStreamConnectMs() / 1000)
    return [
      `The daemon did not open the chat stream within ${connectSec}s — aborted.`,
      'The request stalled during daemon preflight or intent routing, before SSE keepalive began.',
      'Check `sepilot status`, then retry. Tune SEPILOTD_STREAM_CONNECT_MS only when slow preflight is expected.',
    ].join('\n')
  }
  if (
    /stream-idle-timeout/i.test(message)
    || /This operation was aborted/i.test(message)
  ) {
    const idleSec = Math.round(
      resolveCliStreamIdleMs() / 1000,
    )
    const hint = opts.sessionId
      ? `Resume with \`sepilot sessions resume ${opts.sessionId}\`, or re-run with --session ${opts.sessionId} to retry. Tune SEPILOTD_STREAM_IDLE_MS=900000 or higher for slower models.`
      : 'Pass --session <id> next time, and tune SEPILOTD_STREAM_IDLE_MS=900000 or higher for slower models.'
    return [
      `Stream went silent for ${idleSec}s — aborted.`,
      'The daemon may be alive but the SSE reader did not see EOF.',
      hint,
    ].join('\n')
  }
  if (/Daemon chat stream ended before completion/i.test(message)) {
    const hint = opts.sessionId
      ? `Resume with \`sepilot sessions resume ${opts.sessionId}\`, or re-run with --session ${opts.sessionId} to retry.`
      : 'Pass --session <id> next time so an interrupted run can be resumed.'
    return [
      'Stream ended before the daemon sent completion.',
      'The visible response may be partial, so the CLI is treating this run as failed.',
      hint,
    ].join('\n')
  }
  if (isRecoverableStreamDrop(err)) {
    const lines = [
      'Connection to the daemon was lost mid-run.',
      'Check the daemon with `sepilot status`, then resume with /resume.',
    ]
    if (opts.sessionId) {
      lines.push(`CLI resume: \`sepilot sessions resume ${opts.sessionId}\`.`)
    } else {
      lines.push('For non-interactive retries, pass --session <id> next time.')
    }
    const detail = sanitizeTransportDropDetail(message)
    if (detail) lines.push(`Detail: ${detail}`)
    return lines.join('\n')
  }
  return null
}

/**
 * Build the multi-line friendly text for an api-client http error.
 * Returns null when the error isn't an http error so the caller can
 * fall through. Mirrors `formatStreamError` — the cli wrapper
 * `printApiError` colourises, surfaces with their own renderer reuse
 * the plain text form via `normalizeError`.
 */
export function formatApiError(
  err: unknown,
  opts: PrintApiErrorOptions = {},
): string | null {
  const parsed = extractStatus(err)
  if (!parsed) return null
  const { status, inner } = parsed
  if (parsed.code === 'SKILL_AUTONOMY_REQUIRED') {
    const name = typeof parsed.details?.name === 'string' ? parsed.details.name : null
    const required = typeof parsed.details?.requiredAutonomy === 'string'
      ? parsed.details.requiredAutonomy
      : null
    const current = typeof parsed.details?.currentAutonomy === 'string'
      ? parsed.details.currentAutonomy
      : null
    const lines = ['Selected skill requires a higher autonomy level (403).']
    lines.push(name && required
      ? `  Skill "${name}" requires autonomy "${required}"${current ? ` (current: "${current}")` : ''}.`
      : `  ${inner}`)
    if (required) lines.push(`Re-run with --autonomy ${required}.`)
    return lines.join('\n')
  }
  let header: string
  let detail = `  ${inner}`
  let trailer: string | null = null
  if (status === 503) {
    header = 'Daemon is unable to serve the request right now (503).'
    trailer = 'All providers may be busy or rate-limited. Retry in a moment.'
  } else if (status >= 500) {
    header = `Daemon error (${status}).`
  } else if (status >= 400) {
    header = `Request rejected (${status}).`
  } else {
    header = `Error: ${status}: ${inner}`
    detail = ''
  }
  const lines = [header]
  if (detail) lines.push(detail)
  if (trailer) lines.push(trailer)
  if (opts.hint) lines.push(opts.hint)
  return lines.join('\n')
}

/**
 * Print a multi-line friendly text to stderr, colourising the first line
 * (red header) and the rest (gray detail/hint). Internal helper shared
 * by `printStreamError` and `printApiError`.
 */
function emitFriendlyMultiline(text: string): void {
  const [first, ...rest] = text.split('\n')
  console.error(chalk.red(first))
  for (const line of rest) console.error(chalk.gray(line))
}

/**
 * Render an SSE-stream failure to stderr. Returns true when the error
 * matched a known stream-failure shape (caller should `process.exit(1)`
 * / abort the loop). Returns false on unrelated errors.
 */
export function printStreamError(
  err: unknown,
  opts: PrintStreamErrorOptions = {},
): boolean {
  const text = formatStreamError(err, opts)
  if (text == null) return false
  emitFriendlyMultiline(text)
  return true
}

/**
 * Render an api-client error to stderr. Returns true when the error
 * matched a known http status (caller should exit). Returns false on
 * unrelated errors.
 */
export function printApiError(err: unknown, opts: PrintApiErrorOptions = {}): boolean {
  const text = formatApiError(err, opts)
  if (text == null) return false
  emitFriendlyMultiline(text)
  return true
}
