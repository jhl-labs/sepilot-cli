export interface RunLimiterOptions {
  maxActive?: number
  maxQueued?: number
  queueTimeoutMs?: number
}

export interface RunLimiterStats {
  active: number
  queued: number
  maxActive: number
  maxQueued: number
  queueTimeoutMs: number
  accepting: boolean
  stopReason?: string
}

export interface RunLease {
  release(): void
}

type RunLimiterErrorCode =
  | 'RUN_LIMITER_STOPPED'
  | 'RUN_LIMIT_EXCEEDED'
  | 'RUN_QUEUE_TIMEOUT'

interface QueueEntry {
  id: number
  resolve: (lease: RunLease) => void
  reject: (error: Error) => void
  timeout: ReturnType<typeof setTimeout>
  signal?: AbortSignal
  onAbort?: () => void
}

export class RunLimiterError extends Error {
  readonly code: RunLimiterErrorCode

  constructor(code: RunLimiterErrorCode, message: string) {
    super(message)
    this.name = 'RunLimiterError'
    this.code = code
  }
}

export function isRunLimiterError(error: unknown): error is RunLimiterError {
  return error instanceof RunLimiterError
}

export function toRunLimiterApiError(
  error: unknown,
): { code: string; message: string } {
  if (isRunLimiterError(error)) {
    return {
      code: 'SERVICE_UNAVAILABLE',
      message: error.message,
    }
  }

  if (error instanceof Error) {
    return {
      code: 'INTERNAL_ERROR',
      message: error.message,
    }
  }

  return {
    code: 'INTERNAL_ERROR',
    message: String(error),
  }
}

export class RunLimiter {
  private readonly maxActive: number
  private readonly maxQueued: number
  private readonly queueTimeoutMs: number
  private active = 0
  private accepting = true
  private stopReason?: string
  private nextQueueId = 0
  private readonly queue: QueueEntry[] = []

  constructor(options: RunLimiterOptions = {}) {
    this.maxActive = options.maxActive ?? 4
    this.maxQueued = options.maxQueued ?? 4
    this.queueTimeoutMs = options.queueTimeoutMs ?? 1_500
  }

  async acquire(signal?: AbortSignal): Promise<RunLease> {
    if (!this.accepting) {
      throw new RunLimiterError(
        'RUN_LIMITER_STOPPED',
        this.stopReason ?? 'Agent runs are temporarily unavailable',
      )
    }

    if (this.active < this.maxActive) {
      this.active++
      return this.createLease()
    }

    if (this.queue.length >= this.maxQueued) {
      throw new RunLimiterError(
        'RUN_LIMIT_EXCEEDED',
        `Agent run capacity exhausted (${this.active}/${this.maxActive} active, ${this.queue.length}/${this.maxQueued} queued)`,
      )
    }

    return new Promise<RunLease>((resolve, reject) => {
      const entry: QueueEntry = {
        id: ++this.nextQueueId,
        resolve,
        reject,
        signal,
        timeout: setTimeout(() => {
          this.removeQueuedEntry(entry.id)
          reject(
            new RunLimiterError(
              'RUN_QUEUE_TIMEOUT',
              `Timed out waiting for an agent run slot after ${this.queueTimeoutMs}ms`,
            ),
          )
        }, this.queueTimeoutMs),
      }

      if (signal) {
        entry.onAbort = () => {
          this.removeQueuedEntry(entry.id)
          reject(new Error('Agent run request aborted while waiting for capacity'))
        }
        signal.addEventListener('abort', entry.onAbort, { once: true })
      }

      this.queue.push(entry)
    })
  }

  stopAccepting(reason = 'Agent runs are temporarily unavailable'): void {
    this.accepting = false
    this.stopReason = reason

    while (this.queue.length > 0) {
      const entry = this.queue.shift()
      if (!entry) continue
      this.disposeQueuedEntry(entry)
      entry.reject(new RunLimiterError('RUN_LIMITER_STOPPED', reason))
    }
  }

  resumeAccepting(): void {
    this.accepting = true
    this.stopReason = undefined
    this.dispatchQueuedRuns()
  }

  getStats(): RunLimiterStats {
    return {
      active: this.active,
      queued: this.queue.length,
      maxActive: this.maxActive,
      maxQueued: this.maxQueued,
      queueTimeoutMs: this.queueTimeoutMs,
      accepting: this.accepting,
      stopReason: this.stopReason,
    }
  }

  private createLease(): RunLease {
    let released = false
    return {
      release: () => {
        if (released) return
        released = true
        this.active = Math.max(0, this.active - 1)
        this.dispatchQueuedRuns()
      },
    }
  }

  private dispatchQueuedRuns(): void {
    while (this.accepting && this.active < this.maxActive && this.queue.length > 0) {
      const entry = this.queue.shift()
      if (!entry) {
        return
      }

      this.disposeQueuedEntry(entry)
      this.active++
      entry.resolve(this.createLease())
    }
  }

  private removeQueuedEntry(id: number): void {
    const index = this.queue.findIndex((entry) => entry.id === id)
    if (index < 0) {
      return
    }

    const [entry] = this.queue.splice(index, 1)
    if (entry) {
      this.disposeQueuedEntry(entry)
    }
  }

  private disposeQueuedEntry(entry: QueueEntry): void {
    clearTimeout(entry.timeout)
    if (entry.signal && entry.onAbort) {
      entry.signal.removeEventListener('abort', entry.onAbort)
    }
  }
}
