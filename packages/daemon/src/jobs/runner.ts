import { setTimeout as delay } from 'node:timers/promises'
import type { JobsRepo } from './repo.js'
import type { JobItem } from './types.js'

interface RunInput {
  jobId: string
  concurrency: number
  execute: (item: JobItem, signal: AbortSignal) => Promise<unknown>
}

/** A failed execution may still have useful, inspectable partial evidence. */
export class JobExecutionError extends Error {
  constructor(message: string, readonly result: unknown) { super(message) }
}

export interface JobRunner {
  run(input: RunInput & { failureMode: 'continue' | 'abort' }): Promise<void>
  runPartitioned(input: RunInput): Promise<void>
  cancel(jobId: string): void
  shutdown(): void
  /** Includes FIFO waiters so callers cannot accidentally submit duplicates. */
  isRunning(jobId: string): boolean
}

interface JobControl {
  controller: AbortController
  canceled: boolean
  aborted: boolean
}

export function createJobRunner(input: { repo: JobsRepo; globalMax: number }): JobRunner {
  const { repo } = input
  const globalMax = Number.isFinite(input.globalMax) ? Math.max(1, Math.floor(input.globalMax)) : 16
  const active = new Map<string, JobControl>()
  const waiting: Array<{ grant: (admitted: boolean) => void }> = []
  let running = 0
  let closed = false

  function acquire(ctl: JobControl): Promise<boolean> {
    if (running < globalMax) {
      running++
      return Promise.resolve(true)
    }
    return new Promise((resolve) => {
      const waiter = { grant: (admitted: boolean) => {
        ctl.controller.signal.removeEventListener('abort', abort)
        resolve(admitted)
      } }
      const abort = () => {
        const index = waiting.indexOf(waiter)
        if (index >= 0) waiting.splice(index, 1)
        waiter.grant(false)
      }
      waiting.push(waiter)
      ctl.controller.signal.addEventListener('abort', abort, { once: true })
    })
  }

  function release() {
    running--
    const next = waiting.shift()
    if (next) {
      running++
      next.grant(true)
    }
  }

  function cancel(jobId: string) {
    const job = repo.get(jobId)
    if (!job || !['pending', 'running'].includes(job.status)) return
    repo.cancelRemainingItems(jobId, 'job canceled')
    repo.updateStatus(jobId, 'canceled')
    const ctl = active.get(jobId)
    if (ctl) {
      ctl.canceled = true
      ctl.controller.abort(new Error('job canceled'))
    }
  }

  async function run({ jobId, concurrency, execute }: RunInput, failureMode: 'continue' | 'abort', partitioned: boolean) {
    if (active.has(jobId)) return
    const job = repo.get(jobId)
    if (!job || !['pending', 'running'].includes(job.status)) return
    if (closed) { cancel(jobId); return }
    const ctl: JobControl = { controller: new AbortController(), canceled: false, aborted: false }
    active.set(jobId, ctl)
    let admitted = false
    try {
      admitted = await acquire(ctl)
      if (!admitted || ctl.canceled) return
      repo.updateStatus(jobId, 'running')
      const items = partitioned ? [] : repo.listItems(jobId, 0).filter((item) => item.status === 'queued')
      let cursor = 0
      const cap = Number.isFinite(concurrency) ? Math.max(1, Math.floor(concurrency)) : 1
      async function worker(): Promise<void> {
        while (!ctl.controller.signal.aborted) {
          const item = partitioned ? repo.claimNextPartitioned(jobId) : items[cursor++]
          if (!item) return
          if (!partitioned) repo.startItem(jobId, item.idx)
          try {
            const result = await execute(item, ctl.controller.signal)
            if (!ctl.controller.signal.aborted) repo.completeItem(jobId, item.idx, { status: 'succeeded', result })
          } catch (err) {
            if (ctl.controller.signal.aborted) return
            const error = err instanceof Error ? err.message : String(err)
            if (partitioned && item.attempts < item.maxAttempts) {
              repo.requeueItem(jobId, item.idx, error)
              await delay(Math.min(5_000, Math.max(250, item.attempts * 500)), undefined, { signal: ctl.controller.signal }).catch(() => {})
            } else {
              repo.completeItem(jobId, item.idx, { status: 'failed', error, ...(err instanceof JobExecutionError ? { result: err.result } : {}) })
              if (failureMode === 'abort') {
                ctl.aborted = true
                repo.cancelRemainingItems(jobId, 'job stopped after an item failed')
                ctl.controller.abort(new Error(error))
              }
            }
          }
        }
      }
      do {
        await Promise.all(Array.from({ length: cap }, () => worker()))
        if (!partitioned || ctl.controller.signal.aborted) break
        const finalJob = repo.finalizePartitionedJob(jobId)
        if (!finalJob || !['pending', 'running'].includes(finalJob.status)) return
        await delay(50, undefined, { signal: ctl.controller.signal }).catch(() => {})
      } while (!ctl.controller.signal.aborted)
      repo.updateStatus(jobId, ctl.canceled ? 'canceled' : ctl.aborted ? 'failed' : 'completed')
    } catch (err) {
      if (!ctl.canceled) {
        const error = err instanceof Error ? err.message : String(err)
        repo.cancelRemainingItems(jobId, error)
        repo.updateStatus(jobId, 'failed', error)
        ctl.controller.abort(err)
      }
    } finally {
      active.delete(jobId)
      if (admitted) release()
    }
  }

  return {
    isRunning: (jobId) => active.has(jobId),
    cancel,
    shutdown() {
      closed = true
      for (const jobId of active.keys()) cancel(jobId)
    },
    run: (args) => run(args, args.failureMode, false),
    runPartitioned: (args) => run(args, 'continue', true),
  }
}
