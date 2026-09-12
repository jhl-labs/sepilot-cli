export const RECONNECT_BACKOFF_MS = [1000, 2000, 4000, 8000, 16000, 32000]
export const MAX_RECONNECT_ATTEMPTS = 5

export function backoffFor(attempt: number): number {
  const index = Math.min(Math.max(attempt, 0), RECONNECT_BACKOFF_MS.length - 1)
  return RECONNECT_BACKOFF_MS[index]
}

export async function sleepWithAbort(ms: number, signal: AbortSignal): Promise<void> {
  if (signal.aborted) throw new Error('aborted')
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', onAbort)
      resolve()
    }, ms)
    const onAbort = () => {
      clearTimeout(timer)
      reject(new Error('aborted'))
    }
    signal.addEventListener('abort', onAbort, { once: true })
  })
}
