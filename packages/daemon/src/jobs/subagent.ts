import type { SubagentDispatchInput, SubagentDispatcher } from '../agent/subagent-dispatcher.js'
import type { JobsRepo } from './repo.js'
import type { JobItem } from './types.js'
import { JobExecutionError, type JobRunner } from './runner.js'
import type { Job } from './types.js'

export interface BackgroundSubagents {
  start(input: SubagentDispatchInput & { requestKey?: string }): {
    jobId: string
    status: string
    createdAt: number
  }
  inspect(
    jobId: string,
    parentSessionId: string,
  ): {
    job: Job
    activity: ReturnType<JobsRepo['listActivity']>
    items: ReturnType<JobsRepo['listCompletedItems']>
  } | null
  cancel(jobId: string, parentSessionId: string): boolean
}

/** A detached job owns its cancellation; caller cancellation is checked before admission only. */
export function createBackgroundSubagents(deps: {
  dispatcher: SubagentDispatcher
  repo: JobsRepo
  runner: JobRunner
  onFinished?: (job: Job, parentSessionId: string) => void
  /** Bound active plus queued submissions, not just concurrent engines. */
  maxPending?: number
}): BackgroundSubagents {
  const pending = new Set<string>()
  const maxPending = Number.isFinite(deps.maxPending)
    ? Math.max(1, Math.floor(deps.maxPending!))
    : 128
  const owns = (id: string, parent: string) =>
    Boolean(parent) &&
    deps.repo.get(id)?.kind === 'subagent' &&
    (deps.repo.listItems(id, 0)[0]?.request as SubagentDispatchInput | undefined)
      ?.parentSessionId === parent
  return {
    start(input) {
      input.signal?.throwIfAborted()
      if (!input.parentSessionId || !input.parentExecutionPolicy)
        throw new Error(
          'Background delegation requires a parent session and explicit inherited tool policy',
        )
      const unauthorized = input.tools?.filter(name => !input.parentExecutionPolicy!.allowedToolNames.includes(name)) ?? []
      if (unauthorized.length) {
        const error = new Error(`SUBAGENT_TOOL_ESCALATION: ${unauthorized.join(', ')}`) as Error & { code: string }
        error.code = 'SUBAGENT_TOOL_ESCALATION'
        throw error
      }
      // Persist only data. Never serialize a live signal/callback or recover authority from daemon defaults.
      const { signal: _signal, onSession: _onSession, onEvent: _onEvent, ...data } = input
      const request = JSON.parse(
        JSON.stringify({ ...data, detached: true }),
      ) as SubagentDispatchInput
      if (data.requestKey) {
        const previous = deps.repo.findSubagentRequest(input.parentSessionId, data.requestKey)
        if (previous) {
          if (JSON.stringify(previous.request) !== JSON.stringify(request))
            throw new Error('Background requestKey already belongs to a different request')
          const job = deps.repo.get(previous.jobId)!
          return { jobId: job.id, status: job.status, createdAt: job.createdAt }
        }
      }
      if (pending.size >= maxPending)
        throw new Error(
          `Background subagent queue is full (${maxPending}); no additional job was submitted`,
        )
      const job = deps.repo.create({ kind: 'subagent', total: 1, concurrency: 1 })
      deps.repo.insertItems(job.id, [{ idx: 0, request, sessionId: input.parentSessionId }])
      pending.add(job.id)
      void deps.runner
        .run({
          jobId: job.id,
          concurrency: 1,
          failureMode: 'abort',
          execute: (item, signal) => executeSubagentJob(deps.dispatcher, deps.repo, item, signal),
        })
        .finally(() => pending.delete(job.id))
        .then(() => {
          const final = deps.repo.get(job.id)
          if (final) deps.onFinished?.(final, input.parentSessionId!)
        })
        .catch(() => {
          /* execution evidence remains in the durable job even if notification fails */
        })
      return { jobId: job.id, status: job.status, createdAt: job.createdAt }
    },
    inspect(jobId, parentSessionId) {
      if (!owns(jobId, parentSessionId)) return null
      return {
        job: deps.repo.get(jobId)!,
        activity: deps.repo.listActivity(jobId),
        items: deps.repo.listCompletedItems(jobId, 0),
      }
    },
    cancel(jobId, parentSessionId) {
      if (!owns(jobId, parentSessionId)) return false
      deps.runner.cancel(jobId)
      return true
    },
  }
}

/** Shared execution contract for direct background submissions and work plans. */
export async function executeSubagentJob(
  dispatcher: SubagentDispatcher,
  repo: JobsRepo,
  item: JobItem,
  signal: AbortSignal,
) {
  const result = await dispatcher.dispatch({
    ...(item.request as SubagentDispatchInput),
    signal,
    onSession: (sessionId) =>
      repo.updateItemProgress(item.jobId, item.idx, {
        sessionId,
        phase: 'starting',
        updatedAt: Date.now(),
      }),
    onEvent: (event) => {
      if (event.type !== 'subagent_progress') return
      // Persist bounded structured activity, never prompt, thinking, tool args
      // or raw outputs. This survives disconnects without creating a second log.
      if (event.inner.type === 'thinking' || event.inner.type === 'message') return
      repo.updateItemProgress(item.jobId, item.idx, {
        sessionId: event.subagentId,
        phase: event.inner.type,
        ...(event.inner.type === 'tool_call' ? { toolName: event.inner.toolCall.name } : {}),
        ...(event.inner.type === 'approval_request'
          ? { toolName: event.inner.toolCall.name, approvalRequestId: event.inner.requestId }
          : {}),
        updatedAt: Date.now(),
      })
    },
  })
  if (result.status !== 'completed') {
    throw new JobExecutionError(
      result.error ?? `subagent ${result.status}: execution did not complete`,
      result,
    )
  }
  return result
}
