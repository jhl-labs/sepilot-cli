import type { JobsRepo } from './repo.js'
import type { JobItem } from './types.js'

export interface JobRunner {
  run(input: {
    jobId: string
    concurrency: number
    failureMode: 'continue' | 'abort'
    execute: (item: JobItem) => Promise<unknown>
  }): Promise<void>
  runPartitioned(input: {
    jobId: string
    concurrency: number
    execute: (item: JobItem) => Promise<unknown>
  }): Promise<void>
  cancel(jobId: string): void
  isRunning(jobId: string): boolean
}

interface JobControl {
  canceled: boolean
  aborted: boolean
}

function retryDelayMs(attempts: number): number {
  return Math.min(5_000, Math.max(250, attempts * 500))
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

export function createJobRunner(input: { repo: JobsRepo; globalMax: number }): JobRunner {
  const { repo, globalMax } = input
  const active = new Map<string, JobControl>()

  return {
    isRunning(jobId) {
      return active.has(jobId)
    },
    cancel(jobId) {
      const ctl = active.get(jobId)
      if (ctl) ctl.canceled = true
    },
    async run({ jobId, concurrency, failureMode, execute }) {
      if (active.size >= globalMax) {
        repo.updateStatus(jobId, 'failed', `global concurrency cap (${globalMax}) reached`)
        return
      }
      const ctl: JobControl = { canceled: false, aborted: false }
      active.set(jobId, ctl)
      repo.updateStatus(jobId, 'running')

      const items = repo.listItems(jobId, 0).filter((i) => i.status === 'queued')
      const cap = Math.max(1, concurrency)
      let cursor = 0

      async function worker(): Promise<void> {
        while (true) {
          if (ctl.canceled || ctl.aborted) return
          const i = cursor++
          if (i >= items.length) return
          const item = items[i]!
          repo.startItem(jobId, item.idx)
          try {
            const result = await execute(item)
            repo.completeItem(jobId, item.idx, {
              status: 'succeeded',
              result,
            })
          } catch (err) {
            const msg = err instanceof Error ? err.message : String(err)
            repo.completeItem(jobId, item.idx, {
              status: 'failed',
              error: msg,
            })
            if (failureMode === 'abort') ctl.aborted = true
          }
        }
      }

      try {
        await Promise.all(Array.from({ length: cap }, () => worker()))
      } finally {
        active.delete(jobId)
      }

      const finalStatus = ctl.canceled ? 'canceled' : ctl.aborted ? 'failed' : 'completed'
      repo.updateStatus(jobId, finalStatus)
    },
    async runPartitioned({ jobId, concurrency, execute }) {
      if (active.has(jobId)) return
      if (active.size >= globalMax) {
        repo.updateStatus(jobId, 'failed', `global concurrency cap (${globalMax}) reached`)
        return
      }
      const ctl: JobControl = { canceled: false, aborted: false }
      active.set(jobId, ctl)
      repo.updateStatus(jobId, 'running')

      const cap = Math.max(1, concurrency)
      async function worker(): Promise<void> {
        while (!ctl.canceled && !ctl.aborted) {
          const item = repo.claimNextPartitioned(jobId)
          if (!item) return
          try {
            const result = await execute(item)
            repo.completeItem(jobId, item.idx, {
              status: 'succeeded',
              result,
            })
          } catch (err) {
            const msg = err instanceof Error ? err.message : String(err)
            if (item.attempts < item.maxAttempts) {
              repo.requeueItem(jobId, item.idx, msg)
              await sleep(retryDelayMs(item.attempts))
            } else {
              repo.completeItem(jobId, item.idx, {
                status: 'failed',
                error: msg,
              })
            }
          }
        }
      }

      try {
        while (!ctl.canceled && !ctl.aborted) {
          await Promise.all(Array.from({ length: cap }, () => worker()))
          const finalJob = repo.finalizePartitionedJob(jobId)
          if (
            !finalJob ||
            finalJob.status === 'completed' ||
            finalJob.status === 'failed' ||
            finalJob.status === 'canceled'
          ) {
            return
          }
          await sleep(50)
        }
        if (ctl.canceled) {
          repo.updateStatus(jobId, 'canceled')
          return
        }
        repo.finalizePartitionedJob(jobId)
      } finally {
        active.delete(jobId)
      }
    },
  }
}
