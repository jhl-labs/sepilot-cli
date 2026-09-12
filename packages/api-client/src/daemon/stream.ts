import type { DaemonChatStreamPayload } from './types.js'

export interface ParsedSseEvent<T = unknown> {
  event: string
  data: T
  id?: string
  retry?: number
}

export interface SseMalformedJsonErrorOptions {
  event: string
  id?: string
  payload: string
  cause?: unknown
}

export class SseMalformedJsonError extends Error {
  readonly event: string
  readonly id?: string
  readonly payloadPreview: string
  readonly cause?: unknown

  constructor(options: SseMalformedJsonErrorOptions) {
    super(`SSE event "${options.event}" returned malformed JSON`)
    this.name = 'SseMalformedJsonError'
    this.event = options.event
    if (options.id !== undefined) {
      this.id = options.id
    }
    this.payloadPreview = previewSsePayload(options.payload)
    this.cause = options.cause
  }
}

export interface ReconnectingJsonSseStreamOptions {
  signal?: AbortSignal
  lastEventId?: string
  reconnect?: boolean
  reconnectWithoutLastEventId?: boolean
  maxReconnects?: number
  minRetryMs?: number
  maxRetryMs?: number
}

export interface ReconnectingJsonSseOpenOptions {
  signal: AbortSignal
  lastEventId?: string
}

function previewSsePayload(payload: string): string {
  const normalized = payload.replace(/\s+/g, ' ').trim()
  return normalized.length > 160
    ? `${normalized.slice(0, 157)}...`
    : normalized
}

export async function* parseJsonSseStream<T = unknown>(
  body: ReadableStream<Uint8Array>,
): AsyncIterable<ParsedSseEvent<T>> {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const chunks = buffer.split('\n\n')
    buffer = chunks.pop() ?? ''

    for (const chunk of chunks) {
      let id: string | undefined
      let eventName = 'message'
      let retry: number | undefined
      let payload = ''

      for (const line of chunk.split('\n')) {
        const normalizedLine = line.endsWith('\r') ? line.slice(0, -1) : line
        if (normalizedLine.startsWith('id: ')) {
          id = normalizedLine.slice(4).trim()
        } else if (normalizedLine.startsWith('event: ')) {
          eventName = normalizedLine.slice(7).trim()
        } else if (normalizedLine.startsWith('retry: ')) {
          const parsedRetry = Number.parseInt(normalizedLine.slice(7).trim(), 10)
          if (Number.isFinite(parsedRetry) && parsedRetry >= 0) {
            retry = parsedRetry
          }
        } else if (normalizedLine.startsWith('data: ')) {
          payload += normalizedLine.slice(6)
        }
      }

      if (!payload) continue

      try {
        const parsedEvent: ParsedSseEvent<T> = {
          event: eventName,
          data: JSON.parse(payload) as T,
        }
        if (id) {
          parsedEvent.id = id
        }
        if (retry != null) {
          parsedEvent.retry = retry
        }
        yield parsedEvent
      } catch (error) {
        const errorOptions: SseMalformedJsonErrorOptions = {
          event: eventName,
          payload,
          cause: error,
        }
        if (id !== undefined) {
          errorOptions.id = id
        }
        throw new SseMalformedJsonError(errorOptions)
      }
    }
  }
}

export async function* reconnectingJsonSseStream<T = unknown>(
  open: (options: ReconnectingJsonSseOpenOptions) => Promise<Response>,
  options: ReconnectingJsonSseStreamOptions = {},
): AsyncIterable<ParsedSseEvent<T>> {
  const reconnect = options.reconnect ?? true
  const maxReconnects = options.maxReconnects ?? Number.POSITIVE_INFINITY
  const minRetryMs = options.minRetryMs ?? 250
  const maxRetryMs = options.maxRetryMs ?? 30_000
  let retryMs = clampRetryMs(2000, minRetryMs, maxRetryMs)
  let lastEventId = options.lastEventId
  let reconnects = 0
  const canReconnectFromCurrentCursor = () =>
    options.reconnectWithoutLastEventId !== false || Boolean(lastEventId)

  while (!options.signal?.aborted) {
    const controller = new AbortController()
    const abortAttempt = () => {
      controller.abort(options.signal?.reason)
    }
    options.signal?.addEventListener('abort', abortAttempt, { once: true })

    let shouldReconnect = false
    let retryableError: unknown = null

    try {
      const response = await open({
        signal: controller.signal,
        lastEventId,
      })
      if (!response.ok) {
        throw await buildSseHttpError(response)
      }
      if (!response.body) {
        throw new Error('Streaming response body is unavailable')
      }

      for await (const event of parseJsonSseStream<T>(response.body)) {
        reconnects = 0
        if (event.id) {
          lastEventId = event.id
        }
        if (event.retry != null) {
          retryMs = clampRetryMs(event.retry, minRetryMs, maxRetryMs)
        }
        yield event
      }
      shouldReconnect = canReconnectFromCurrentCursor()
    } catch (error) {
      if (options.signal?.aborted || isAbortError(error)) {
        return
      }
      retryableError = error
      shouldReconnect = isRetryableSseError(error) && canReconnectFromCurrentCursor()
    } finally {
      options.signal?.removeEventListener('abort', abortAttempt)
      controller.abort()
    }

    if (!reconnect || !shouldReconnect) {
      if (retryableError) {
        throw retryableError
      }
      return
    }

    reconnects += 1
    if (reconnects > maxReconnects) {
      if (retryableError) {
        throw retryableError
      }
      return
    }

    await delayWithAbort(retryMs, options.signal)
  }
}

export async function* streamDaemonChatEvents<T = DaemonChatStreamPayload>(
  response: Response,
): AsyncIterable<T> {
  if (!response.body) {
    throw new Error('Streaming response body is unavailable')
  }

  for await (const event of parseJsonSseStream<T>(response.body)) {
    yield event.data
  }
}

async function buildSseHttpError(response: Response): Promise<Error> {
  const body = await response.text().catch(() => '')
  const error = new Error(`${response.status}: ${body || response.statusText}`)
  ;(error as Error & { status?: number }).status = response.status
  return error
}

function isRetryableSseError(error: unknown): boolean {
  if (error instanceof SseMalformedJsonError) {
    return false
  }
  const status = (error as { status?: unknown } | null)?.status
  if (typeof status === 'number' && status >= 400 && status < 500) {
    return false
  }
  return true
}

function isAbortError(error: unknown): boolean {
  return (
    error != null
    && typeof error === 'object'
    && (error as { name?: unknown }).name === 'AbortError'
  )
}

function clampRetryMs(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) {
    return min
  }
  return Math.max(min, Math.min(max, Math.floor(value)))
}

function delayWithAbort(ms: number, signal?: AbortSignal): Promise<void> {
  if (signal?.aborted) {
    return Promise.resolve()
  }

  return new Promise((resolve) => {
    const timeout = setTimeout(done, ms)
    const abort = () => done()
    function done() {
      clearTimeout(timeout)
      signal?.removeEventListener('abort', abort)
      resolve()
    }
    signal?.addEventListener('abort', abort, { once: true })
  })
}
