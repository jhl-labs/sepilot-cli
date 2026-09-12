import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { getProgressIndicator } from './streaming.js'

const execFileAsync = promisify(execFile)

const DEFAULT_TIMEOUT_MS = 5 * 60 * 1000
const DEFAULT_MAX_BUFFER_BYTES = 256 * 1024
const DEFAULT_SUMMARY_MAX_LINES = 24
const DEFAULT_SUMMARY_MAX_CHARS = 4_000

export interface LocalShellInvocation {
  shell: string
  args: string[]
}

export interface LocalShellResult {
  command: string
  cwd: string
  shell: string
  args: string[]
  stdout: string
  stderr: string
  exitCode: number
  signal: string | null
  durationMs: number
  timedOut: boolean
  maxBufferExceeded: boolean
}

export type LocalShellGate = 'run' | 'block'

export function resolveLocalShellGate(input: {
  autonomy: string | null | undefined
}): LocalShellGate {
  const autonomy = input.autonomy?.toLowerCase() ?? ''
  if (autonomy === 'readonly') return 'block'
  // `!command` is typed directly by the user, so entering it is already the
  // human approval. Autonomy controls agent-initiated tools; requiring `!!`
  // here would ask the same human to confirm the same command twice.
  return 'run'
}

export function buildLocalShellInvocation(command: string): LocalShellInvocation {
  if (process.platform === 'win32') {
    const shell = process.env.ComSpec ?? 'cmd.exe'
    return {
      shell,
      args: ['/d', '/s', '/c', command],
    }
  }

  const shell = process.env.SHELL || '/bin/bash'
  return {
    shell,
    args: ['-lc', command],
  }
}

function normalizeExecError(error: unknown): {
  stdout: string
  stderr: string
  exitCode: number
  signal: string | null
  timedOut: boolean
  maxBufferExceeded: boolean
} {
  const execError = error as Error & {
    stdout?: string
    stderr?: string
    code?: string | number | null
    signal?: string | null
    killed?: boolean
  }

  const timedOut = Boolean(execError.killed) && /timed out/i.test(execError.message)
  const aborted = execError.code === 'ABORT_ERR'
  const maxBufferExceeded = execError.code === 'ERR_CHILD_PROCESS_STDIO_MAXBUFFER'
  const exitCode = typeof execError.code === 'number' ? execError.code : 1

  return {
    stdout: execError.stdout ?? '',
    stderr: execError.stderr ?? (aborted ? 'cancelled by user' : execError.message),
    exitCode,
    signal: execError.signal ?? null,
    timedOut,
    maxBufferExceeded,
  }
}

export async function runLocalShellCommand(
  command: string,
  options?: {
    cwd?: string
    timeoutMs?: number
    maxBufferBytes?: number
    signal?: AbortSignal
  },
): Promise<LocalShellResult> {
  const cwd = options?.cwd ?? process.cwd()
  const timeoutMs = options?.timeoutMs ?? DEFAULT_TIMEOUT_MS
  const maxBufferBytes = options?.maxBufferBytes ?? DEFAULT_MAX_BUFFER_BYTES
  const invocation = buildLocalShellInvocation(command)
  const startedAt = Date.now()

  try {
    const { stdout, stderr } = await execFileAsync(invocation.shell, invocation.args, {
      cwd,
      encoding: 'utf8',
      timeout: timeoutMs,
      maxBuffer: maxBufferBytes,
      signal: options?.signal,
      windowsHide: true,
    })

    return {
      command,
      cwd,
      shell: invocation.shell,
      args: invocation.args,
      stdout,
      stderr,
      exitCode: 0,
      signal: null,
      durationMs: Date.now() - startedAt,
      timedOut: false,
      maxBufferExceeded: false,
    }
  } catch (error) {
    const normalized = normalizeExecError(error)
    return {
      command,
      cwd,
      shell: invocation.shell,
      args: invocation.args,
      stdout: normalized.stdout,
      stderr: normalized.stderr,
      exitCode: normalized.exitCode,
      signal: normalized.signal,
      durationMs: Date.now() - startedAt,
      timedOut: normalized.timedOut,
      maxBufferExceeded: normalized.maxBufferExceeded,
    }
  }
}

function formatDuration(durationMs: number): string {
  if (durationMs < 1_000) {
    return `${durationMs}ms`
  }

  return `${(durationMs / 1_000).toFixed(durationMs >= 10_000 ? 0 : 1)}s`
}

function truncateLines(
  value: string,
  maxLines: number,
  maxChars: number,
): {
  text: string
  omittedLines: number
  truncatedChars: boolean
} {
  const normalized = value.replace(/\r\n/g, '\n').trim()
  if (!normalized) {
    return {
      text: '',
      omittedLines: 0,
      truncatedChars: false,
    }
  }

  const lines = normalized.split('\n')
  const limitedLines = lines.slice(0, maxLines)
  let text = limitedLines.join('\n')
  let truncatedChars = false

  if (text.length > maxChars) {
    text = `${text.slice(0, Math.max(0, maxChars - 3))}...`
    truncatedChars = true
  }

  return {
    text,
    omittedLines: Math.max(0, lines.length - limitedLines.length),
    truncatedChars,
  }
}

function buildOutputSection(
  label: string,
  value: string,
  maxLines: number,
  maxChars: number,
): string[] {
  const summary = truncateLines(value, maxLines, maxChars)
  if (!summary.text) {
    return []
  }

  const lines = [`${label}:`, summary.text]

  if (summary.omittedLines > 0 || summary.truncatedChars) {
    const suffix = []
    if (summary.omittedLines > 0) {
      suffix.push(`${summary.omittedLines} more line${summary.omittedLines === 1 ? '' : 's'}`)
    }
    if (summary.truncatedChars) {
      suffix.push('truncated')
    }
    lines.push(`... ${suffix.join(', ')}`)
  }

  return lines
}

export function buildLocalShellTranscript(result: LocalShellResult): string {
  const lines = [
    `Local shell finished with exit ${result.exitCode}${result.signal ? ` (${result.signal})` : ''} in ${formatDuration(result.durationMs)}.`,
    `$ ${result.command}`,
  ]

  if (result.timedOut) {
    lines.push('Command timed out before completion.')
  }
  if (result.maxBufferExceeded) {
    lines.push('Command output exceeded the local capture limit and was truncated.')
  }

  const stdoutSection = buildOutputSection(
    'stdout',
    result.stdout,
    DEFAULT_SUMMARY_MAX_LINES,
    DEFAULT_SUMMARY_MAX_CHARS,
  )
  const stderrSection = buildOutputSection(
    'stderr',
    result.stderr,
    DEFAULT_SUMMARY_MAX_LINES,
    DEFAULT_SUMMARY_MAX_CHARS,
  )

  if (stdoutSection.length === 0 && stderrSection.length === 0) {
    lines.push('No output.')
    return lines.join('\n')
  }

  return [...lines, ...stdoutSection, ...stderrSection].join('\n')
}

export function buildLocalShellProgressLabel(
  command: string,
  startedAt: number,
  now: number,
): string {
  const elapsedSeconds = Math.max(0, Math.floor((now - startedAt) / 1_000))
  const indicator = getProgressIndicator(now)
  return `${indicator} in progress • shell • ${elapsedSeconds}s • ${command}`
}
