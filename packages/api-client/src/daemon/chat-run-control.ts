export const DEFAULT_CHAT_RUN_TIMEOUT_MS = 60 * 60_000

export type DaemonChatRunAbortKind = 'cancelled' | 'timeout'

const DAEMON_CHAT_RUN_ABORT_TAG = Symbol('DaemonChatRunAbortError')

export class DaemonChatRunAbortError extends Error {
  readonly kind: DaemonChatRunAbortKind
  readonly timeoutMs?: number
  readonly [DAEMON_CHAT_RUN_ABORT_TAG] = true

  constructor(
    kind: DaemonChatRunAbortKind,
    message: string,
    options?: { timeoutMs?: number },
  ) {
    super(message)
    this.name = 'DaemonChatRunAbortError'
    this.kind = kind
    this.timeoutMs = options?.timeoutMs
  }
}

export interface DaemonChatRunAbortMeta {
  kind: DaemonChatRunAbortKind
  label: string
  detail: string
  status: 'neutral' | 'error'
  assistantFallbackText: string
}

export interface DaemonChatRunController {
  readonly signal: AbortSignal
  readonly timeoutMs: number
  markActivity: () => void
  cancel: (reason?: Error) => void
  dispose: () => void
}

function formatTimeoutLabel(timeoutMs: number): string {
  const totalSeconds = Math.max(1, Math.round(timeoutMs / 1000))
  if (totalSeconds < 60) {
    return `${totalSeconds}s`
  }

  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return seconds > 0 ? `${minutes}m ${seconds}s` : `${minutes}m`
}

export function createCancelledChatRunError(
  message = 'Run cancelled by user.',
): DaemonChatRunAbortError {
  return new DaemonChatRunAbortError('cancelled', message)
}

export function createTimedOutChatRunError(
  timeoutMs = DEFAULT_CHAT_RUN_TIMEOUT_MS,
): DaemonChatRunAbortError {
  return new DaemonChatRunAbortError(
    'timeout',
    `Run timed out after ${formatTimeoutLabel(timeoutMs)}.`,
    { timeoutMs },
  )
}

export function isDaemonChatRunAbortError(
  error: unknown,
): error is DaemonChatRunAbortError {
  return error instanceof DaemonChatRunAbortError
    || Boolean(
      error
      && typeof error === 'object'
      && DAEMON_CHAT_RUN_ABORT_TAG in error,
    )
}

export function resolveDaemonChatRunAbortReason(
  signal?: AbortSignal | null,
  fallback?: unknown,
): DaemonChatRunAbortError | null {
  if (isDaemonChatRunAbortError(fallback)) {
    return fallback
  }

  if (!(signal?.aborted ?? false)) {
    return null
  }

  const abortedSignal = signal
  if (!abortedSignal) {
    return null
  }

  if (isDaemonChatRunAbortError(abortedSignal.reason)) {
    return abortedSignal.reason
  }

  if (fallback instanceof Error && /timeout/i.test(fallback.message)) {
    return createTimedOutChatRunError()
  }

  return createCancelledChatRunError(
    fallback instanceof Error && fallback.message.trim()
      ? fallback.message
      : 'Run cancelled by user.',
  )
}

export function getDaemonChatRunAbortMeta(
  error: unknown,
): DaemonChatRunAbortMeta | null {
  if (!isDaemonChatRunAbortError(error)) {
    return null
  }

  if (error.kind === 'timeout') {
    const timeoutMs = error.timeoutMs ?? DEFAULT_CHAT_RUN_TIMEOUT_MS
    return {
      kind: 'timeout',
      label: 'Run timed out',
      detail: `No response arrived within ${formatTimeoutLabel(timeoutMs)}.`,
      status: 'error',
      assistantFallbackText: 'Run timed out before a response completed.',
    }
  }

  return {
    kind: 'cancelled',
    label: 'Run cancelled',
    detail:
      'Cancelled by user (Esc). Send a new message to continue; /resume replays only if an interrupted tool run left a checkpoint.',
    status: 'neutral',
    assistantFallbackText: 'Run cancelled. Send a new message to continue.',
  }
}

export function createDaemonChatRunController(
  timeoutMs = DEFAULT_CHAT_RUN_TIMEOUT_MS,
): DaemonChatRunController {
  const controller = new AbortController()
  let timer: ReturnType<typeof setTimeout> | null = null

  const dispose = () => {
    if (timer) {
      clearTimeout(timer)
      timer = null
    }
  }

  const scheduleTimeout = () => {
    dispose()
    timer = setTimeout(() => {
      controller.abort(createTimedOutChatRunError(timeoutMs))
    }, timeoutMs)
    timer.unref?.()
  }

  const markActivity = () => {
    if (!controller.signal.aborted) {
      scheduleTimeout()
    }
  }

  controller.signal.addEventListener('abort', dispose, { once: true })
  scheduleTimeout()

  return {
    signal: controller.signal,
    timeoutMs,
    markActivity,
    cancel: (reason = createCancelledChatRunError()) => {
      if (!controller.signal.aborted) {
        controller.abort(reason)
      }
    },
    dispose,
  }
}
