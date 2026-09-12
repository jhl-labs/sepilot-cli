export function isAbortError(error: unknown): boolean {
  if (error instanceof DOMException) {
    return error.name === 'AbortError'
  }

  if (!(error instanceof Error)) {
    return false
  }

  return error.name === 'AbortError'
    || error.name === 'TimeoutError'
    || /\b(abort|aborted|cancelled|canceled)\b/i.test(error.message)
}

export function createAbortError(message = 'Operation aborted'): Error {
  const error = new Error(message)
  error.name = 'AbortError'
  return error
}

export function getAbortError(
  signal?: AbortSignal | null,
  fallbackMessage = 'Operation aborted',
): Error {
  const reason = signal?.reason
  if (reason instanceof Error) {
    if (reason.name === 'AbortError') {
      return reason
    }
    const wrapped = new Error(reason.message)
    wrapped.name = 'AbortError'
    wrapped.cause = reason
    return wrapped
  }

  if (typeof reason === 'string' && reason.trim()) {
    return createAbortError(reason)
  }

  return createAbortError(fallbackMessage)
}

export function throwIfAborted(
  signal?: AbortSignal | null,
  fallbackMessage?: string,
): void {
  if (signal?.aborted) {
    throw getAbortError(signal, fallbackMessage)
  }
}

/**
 * Settle with the caller's AbortSignal even when the wrapped operation ignores
 * that signal. Passing the signal down is still required for resource cleanup;
 * this helper independently guarantees that the awaiting control flow is not
 * held open by an uncooperative transport or runtime adapter.
 */
export function raceWithAbort<T>(
  promise: PromiseLike<T>,
  signal?: AbortSignal | null,
  fallbackMessage?: string,
): Promise<T> {
  try {
    throwIfAborted(signal, fallbackMessage)
  } catch (error) {
    return Promise.reject(error)
  }
  if (!signal) return Promise.resolve(promise)

  return new Promise<T>((resolve, reject) => {
    const onAbort = () => {
      cleanup()
      reject(getAbortError(signal, fallbackMessage))
    }
    const cleanup = () => signal.removeEventListener('abort', onAbort)

    signal.addEventListener('abort', onAbort, { once: true })
    void Promise.resolve(promise).then(
      (value) => {
        cleanup()
        resolve(value)
      },
      (error) => {
        cleanup()
        reject(error)
      },
    )
  })
}

export async function abortableDelay(
  ms: number,
  signal?: AbortSignal | null,
): Promise<void> {
  throwIfAborted(signal)

  if (!signal) {
    await new Promise<void>((resolve) => {
      setTimeout(resolve, ms)
    })
    return
  }

  await new Promise<void>((resolve, reject) => {
    const timer = setTimeout(() => {
      cleanup()
      resolve()
    }, ms)

    const onAbort = () => {
      cleanup()
      reject(getAbortError(signal))
    }

    const cleanup = () => {
      clearTimeout(timer)
      signal.removeEventListener('abort', onAbort)
    }

    signal.addEventListener('abort', onAbort, { once: true })
  })
}
