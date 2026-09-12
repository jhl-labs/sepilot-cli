export const DEFAULT_CLI_STREAM_CONNECT_MS = 60_000

export function resolveCliStreamConnectMs(
  raw = process.env.SEPILOTD_STREAM_CONNECT_MS,
): number {
  const parsed = Number.parseInt(raw ?? String(DEFAULT_CLI_STREAM_CONNECT_MS), 10)
  return Number.isFinite(parsed) && parsed > 0
    ? parsed
    : DEFAULT_CLI_STREAM_CONNECT_MS
}

export function createStreamConnectTimeoutError(timeoutMs: number): Error {
  return new Error(`stream-connect-timeout after ${timeoutMs}ms`)
}

/**
 * Bound the wait for HTTP response headers separately from the long-lived
 * stream idle window. Before the daemon opens SSE, keepalive frames do not
 * exist, so a stuck pre-flight/router call otherwise looks like an endless
 * pending fetch.
 */
export async function openChatStreamWithConnectTimeout(
  responsePromise: Promise<Response>,
  options: {
    timeoutMs?: number
    abort?: (error: Error) => void
    timeoutError?: Error
  } = {},
): Promise<Response> {
  const timeoutMs = options.timeoutMs ?? resolveCliStreamConnectMs()
  const effectiveTimeoutMs = Number.isFinite(timeoutMs) && timeoutMs > 0
    ? timeoutMs
    : DEFAULT_CLI_STREAM_CONNECT_MS
  let timer: ReturnType<typeof setTimeout> | null = null
  const timeoutPromise = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      const error = options.timeoutError ?? createStreamConnectTimeoutError(effectiveTimeoutMs)
      options.abort?.(error)
      reject(error)
    }, effectiveTimeoutMs)
    timer.unref?.()
  })

  try {
    return await Promise.race([responsePromise, timeoutPromise])
  } finally {
    if (timer) clearTimeout(timer)
  }
}
