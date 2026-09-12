import { isNodeFsError } from './fs-error.js'

const DEFAULT_RETRY_CODES = ['EACCES', 'EBUSY', 'EPERM'] as const

export interface TransientFsRetryOptions {
  maxRetries?: number
  initialDelayMs?: number
  maxDelayMs?: number
  retryCodes?: readonly string[]
}

const sleep = (delayMs: number): Promise<void> =>
  new Promise((resolve) => {
    setTimeout(resolve, delayMs)
  })

function nonNegativeInteger(value: number | undefined, fallback: number): number {
  if (!Number.isFinite(value)) return fallback
  return Math.max(0, Math.floor(value ?? fallback))
}

/**
 * Retry only filesystem failures that are commonly transient while another
 * Windows process briefly holds a sharing lock. The operation itself must be
 * safe to repeat; callers should retry an open/rename, never a possibly-partial
 * append or direct write.
 */
export async function retryTransientFsOperation<T>(
  operation: () => Promise<T>,
  options: TransientFsRetryOptions = {},
): Promise<T> {
  const maxRetries = nonNegativeInteger(options.maxRetries, 6)
  const initialDelayMs = nonNegativeInteger(options.initialDelayMs, 10)
  const maxDelayMs = nonNegativeInteger(options.maxDelayMs, 250)
  const retryCodes = options.retryCodes ?? DEFAULT_RETRY_CODES

  for (let attempt = 0; ; attempt += 1) {
    try {
      return await operation()
    } catch (error) {
      const retryable = retryCodes.some((code) => isNodeFsError(error, code))
      if (!retryable || attempt >= maxRetries) throw error

      const delayMs = Math.min(maxDelayMs, initialDelayMs * 2 ** attempt)
      if (delayMs > 0) await sleep(delayMs)
    }
  }
}
