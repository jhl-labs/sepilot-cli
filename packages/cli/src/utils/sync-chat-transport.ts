import { Agent, fetch as undiciFetch } from 'undici/index.js'

const TRANSPORT_TIMEOUT_MARGIN_MS = 30_000
const MAX_TIMER_MS = 2_147_483_647

const dispatchers = new Map<number, Agent>()

/**
 * The API abort signal owns the user-visible wall-clock timeout. Keep the
 * underlying Node/Bun HTTP headers/body timers slightly wider so a runtime
 * default (notably Undici's five-minute headers timeout) cannot terminate a
 * live synchronous JSON chat first with an untyped transport error.
 */
export function resolveCliSyncChatTransportTimeoutMs(timeoutMs: number): number {
  const normalized = Number.isFinite(timeoutMs) && timeoutMs > 0
    ? Math.floor(timeoutMs)
    : 1
  return Math.min(MAX_TIMER_MS, normalized + TRANSPORT_TIMEOUT_MARGIN_MS)
}

function dispatcherFor(timeoutMs: number): Agent {
  const transportTimeoutMs = resolveCliSyncChatTransportTimeoutMs(timeoutMs)
  const existing = dispatchers.get(transportTimeoutMs)
  if (existing) return existing
  const dispatcher = new Agent({
    headersTimeout: transportTimeoutMs,
    bodyTimeout: transportTimeoutMs,
  })
  dispatchers.set(transportTimeoutMs, dispatcher)
  return dispatcher
}

/** Request-scoped transport used only by synchronous JSON chat. */
export function getCliSyncChatFetch(timeoutMs: number): typeof fetch {
  const dispatcher = dispatcherFor(timeoutMs)
  return (async (input: RequestInfo | URL, init?: RequestInit) => {
    return undiciFetch(input as never, {
      ...init,
      dispatcher,
    } as never) as unknown as Response
  }) as typeof fetch
}

/** Test/process-lifecycle helper; production command exit also closes sockets. */
export async function closeCliSyncChatDispatchers(): Promise<void> {
  const active = [...dispatchers.values()]
  dispatchers.clear()
  await Promise.allSettled(active.map((dispatcher) => dispatcher.close()))
}
