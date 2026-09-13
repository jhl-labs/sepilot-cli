import type { HookPayload } from '@sepilotd/core'
import type { JobsRepo } from './repo.js'
import type { JobRunner } from './runner.js'

/** Post hooks are observations: admission/execution failures are separate job evidence. */
export function createBackgroundHooks(deps: {
  repo: JobsRepo
  runner: JobRunner
  maxPending?: number
  onFinished?: (jobId: string, sessionId: string, status: string) => void
}) {
  const pending = new Set<string>()
  const maxPending = Number.isFinite(deps.maxPending)
    ? Math.max(1, Math.floor(deps.maxPending!))
    : 128
  return (
    handlerId: string,
    payload: HookPayload,
    execute: (signal: AbortSignal) => Promise<unknown>,
  ): void => {
    if (!payload.event.startsWith('post:'))
      throw new Error('Background hooks cannot gate pre events')
    const job = deps.repo.create({ kind: 'hook', total: 1, concurrency: 1 })
    const sessionId = typeof payload.data.sessionId === 'string' ? payload.data.sessionId : null
    // Never persist input: it may contain prompts, tool arguments, or credentials.
    // Restart recovery fails interrupted commands instead of replaying uncertain effects.
    deps.repo.insertItems(job.id, [
      { idx: 0, sessionId, request: { handlerId, event: payload.event } },
    ])
    if (pending.size >= maxPending) {
      const error = `Background hook queue is full (${maxPending}); command was not started`
      deps.repo.completeItem(job.id, 0, { status: 'failed', error })
      deps.repo.updateStatus(job.id, 'failed', error)
      return
    }
    pending.add(job.id)
    void deps.runner
      .run({
        jobId: job.id,
        concurrency: 1,
        failureMode: 'abort',
        execute: (_item, signal) => {
          if (sessionId)
            deps.repo.updateItemProgress(job.id, 0, {
              sessionId,
              phase: 'hook_command',
              updatedAt: Date.now(),
            })
          return execute(signal)
        },
      })
      .finally(() => pending.delete(job.id))
      .then(() => {
        const final = deps.repo.get(job.id)
        if (sessionId && final) deps.onFinished?.(job.id, sessionId, final.status)
      })
      .catch(() => {
        /* runner retains failure */
      })
  }
}
