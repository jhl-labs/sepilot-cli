import chalk from 'chalk'
import {
  forwardDaemonStream,
  isTerminalDaemonChatPayload,
  type DaemonChatStreamPayload,
} from '@sepilotd/api-client'
import { formatChatStreamFailure } from './chat-stream-error.js'
import { isRecoverableStreamDrop } from './error-message.js'

const DEFAULT_RECONNECT_ATTEMPTS = 6
const DEFAULT_RECONNECT_BASE_MS = 1000
const MAX_RECONNECT_DELAY_MS = 10_000

export interface StreamResumeRecoveryOptions<T> {
  aborter: AbortController
  getSessionId: () => string | undefined
  onEvent: (event: T) => void
  openResumeStream: (sessionId: string) => Promise<Response>
  streamIdleMs: number
  quiet?: boolean
  isTerminalEvent?: (event: T) => boolean
}

export function isTerminalCliDaemonChatEvent(event: unknown): boolean {
  return Boolean(
    event
      && typeof event === 'object'
      && isTerminalDaemonChatPayload(event as DaemonChatStreamPayload),
  )
}

function positiveEnvInt(name: string, fallback: number): number {
  const value = Number.parseInt(process.env[name] ?? '', 10)
  return Number.isFinite(value) && value > 0 ? value : fallback
}

function reconnectDelayMs(attempt: number): number {
  const base = positiveEnvInt('SEPILOTD_STREAM_RECONNECT_BASE_MS', DEFAULT_RECONNECT_BASE_MS)
  return Math.min(MAX_RECONNECT_DELAY_MS, base * 2 ** Math.max(0, attempt - 1))
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

async function openResumeResponse(
  sessionId: string,
  options: StreamResumeRecoveryOptions<unknown>,
): Promise<Response> {
  let timer: ReturnType<typeof setTimeout> | null = null
  try {
    const response = await Promise.race([
      options.openResumeStream(sessionId),
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => {
          const error = new Error('stream-idle-timeout')
          options.aborter.abort(error)
          reject(error)
        }, Math.max(0, options.streamIdleMs))
        timer.unref?.()
      }),
    ])
    if (!response.ok || !response.body) {
      throw new Error(await formatChatStreamFailure(response))
    }
    return response
  } finally {
    if (timer) {
      clearTimeout(timer)
    }
  }
}

export async function forwardDaemonStreamWithResumeRecovery<T>(
  initialResponse: Response,
  options: StreamResumeRecoveryOptions<T>,
): Promise<number> {
  let response = initialResponse
  let reconnects = 0
  const maxReconnects = positiveEnvInt(
    'SEPILOTD_STREAM_RECONNECT_ATTEMPTS',
    DEFAULT_RECONNECT_ATTEMPTS,
  )

  for (;;) {
    try {
      await forwardDaemonStream<T>(
        response,
        options.onEvent,
        options.isTerminalEvent ? { isTerminalEvent: options.isTerminalEvent } : undefined,
      )
      return reconnects
    } catch (err) {
      let resumeError = err
      for (;;) {
        const sessionId = options.getSessionId()
        if (
          options.aborter.signal.aborted
          || !sessionId
          || reconnects >= maxReconnects
          || !isRecoverableStreamDrop(resumeError)
        ) {
          throw resumeError
        }
        reconnects += 1
        const delayMs = reconnectDelayMs(reconnects)
        if (!options.quiet) {
          process.stderr.write(chalk.gray(
            `Stream dropped; resuming session ${sessionId} (${reconnects}/${maxReconnects})...\n`,
          ))
        }
        await wait(delayMs)
        try {
          response = await openResumeResponse(
            sessionId,
            options as StreamResumeRecoveryOptions<unknown>,
          )
          break
        } catch (nextErr) {
          resumeError = nextErr
        }
      }
    }
  }
}
