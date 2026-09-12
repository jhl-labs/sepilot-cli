const DEFAULT_HARD_BUFFER_LIMIT_BYTES = 8 * 1024 * 1024
const DEFAULT_WS_SOFT_BUFFER_LIMIT_BYTES = 1 * 1024 * 1024
const DEFAULT_WS_HARD_BUFFER_LIMIT_BYTES = 8 * 1024 * 1024

export class StreamWriteAbortedError extends Error {
  constructor(message = 'Stream write aborted') {
    super(message)
    this.name = 'StreamWriteAbortedError'
  }
}

export class StreamBackpressureError extends Error {
  constructor(message = 'Stream buffer exceeded hard limit') {
    super(message)
    this.name = 'StreamBackpressureError'
  }
}

export function isStreamWriteStoppedError(error: unknown): boolean {
  return error instanceof StreamWriteAbortedError
    || error instanceof StreamBackpressureError
}

type WritableLike = {
  write(chunk: string): boolean
  end?: () => void
  destroy?: (error?: Error) => void
  writableEnded?: boolean
  writableLength?: number
  once(event: string, listener: (...args: unknown[]) => void): unknown
  off?(event: string, listener: (...args: unknown[]) => void): unknown
  removeListener?(event: string, listener: (...args: unknown[]) => void): unknown
}

export interface StreamWriteOptions {
  signal?: AbortSignal
  hardBufferLimitBytes?: number
}

export interface SseFrameWriteOptions extends StreamWriteOptions {
  id?: string
}

export async function writeSseFrame(
  raw: WritableLike,
  event: string,
  data: unknown,
  options: SseFrameWriteOptions = {},
): Promise<void> {
  const idLine = options.id ? `id: ${options.id}\n` : ''
  await writeWithBackpressure(
    raw,
    `${idLine}event: ${event}\ndata: ${JSON.stringify(data)}\n\n`,
    options,
  )
}

export async function writeSseComment(
  raw: WritableLike,
  comment: string,
  options: StreamWriteOptions = {},
): Promise<void> {
  await writeWithBackpressure(raw, `: ${comment}\n\n`, options)
}

async function writeWithBackpressure(
  raw: WritableLike,
  frame: string,
  options: StreamWriteOptions,
): Promise<void> {
  if (raw.writableEnded) return
  const hardLimit = options.hardBufferLimitBytes ?? DEFAULT_HARD_BUFFER_LIMIT_BYTES
  enforceWritableLimit(raw, hardLimit)

  const accepted = raw.write(frame)
  enforceWritableLimit(raw, hardLimit)
  if (accepted) return

  await waitForDrain(raw, options.signal)
  enforceWritableLimit(raw, hardLimit)
}

function enforceWritableLimit(raw: WritableLike, hardLimit: number): void {
  if ((raw.writableLength ?? 0) <= hardLimit) return
  const error = new StreamBackpressureError()
  try {
    raw.destroy?.(error)
  } catch {
    try { raw.end?.() } catch { /* noop */ }
  }
  throw error
}

async function waitForDrain(
  raw: WritableLike,
  signal: AbortSignal | undefined,
): Promise<void> {
  if (raw.writableEnded) {
    throw new StreamWriteAbortedError('Stream closed while waiting for drain')
  }
  if (signal?.aborted) {
    throw new StreamWriteAbortedError('Stream write aborted before drain')
  }

  await new Promise<void>((resolve, reject) => {
    const removeListener = raw.off?.bind(raw) ?? raw.removeListener?.bind(raw)
    const cleanup = () => {
      removeListener?.('drain', onDrain)
      removeListener?.('close', onClose)
      removeListener?.('finish', onFinish)
      removeListener?.('error', onError)
      signal?.removeEventListener('abort', onAbort)
    }
    const finishWith = (fn: () => void) => {
      cleanup()
      fn()
    }
    const onDrain = () => finishWith(resolve)
    const onClose = () => finishWith(() =>
      reject(new StreamWriteAbortedError('Stream closed while waiting for drain')),
    )
    const onFinish = () => finishWith(() =>
      reject(new StreamWriteAbortedError('Stream finished while waiting for drain')),
    )
    const onError = (cause: unknown) => finishWith(() => {
      if (cause instanceof Error) {
        reject(cause)
        return
      }
      reject(new StreamWriteAbortedError('Stream errored while waiting for drain'))
    })
    const onAbort = () => finishWith(() =>
      reject(new StreamWriteAbortedError('Stream write aborted before drain')),
    )

    raw.once('drain', onDrain)
    raw.once('close', onClose)
    raw.once('finish', onFinish)
    raw.once('error', onError)
    signal?.addEventListener('abort', onAbort, { once: true })
  })
}

type WebSocketLike = {
  readyState: number
  bufferedAmount?: number
  send(data: string): void
  close(code?: number, reason?: string): void
}

export interface WsSendOptions {
  dropIfBackpressured?: boolean
  softBufferLimitBytes?: number
  hardBufferLimitBytes?: number
}

export function sendJsonWs(
  socket: WebSocketLike,
  data: unknown,
  options: WsSendOptions = {},
): boolean {
  if (socket.readyState !== 1) return false
  const bufferedAmount = socket.bufferedAmount ?? 0
  const softLimit = options.softBufferLimitBytes ?? DEFAULT_WS_SOFT_BUFFER_LIMIT_BYTES
  const hardLimit = options.hardBufferLimitBytes ?? DEFAULT_WS_HARD_BUFFER_LIMIT_BYTES

  if (bufferedAmount >= hardLimit) {
    try {
      socket.close(1013, 'WebSocket buffer limit exceeded')
    } catch {
      /* noop */
    }
    return false
  }
  if (options.dropIfBackpressured && bufferedAmount >= softLimit) {
    return false
  }

  socket.send(typeof data === 'string' ? data : JSON.stringify(data))
  return true
}
