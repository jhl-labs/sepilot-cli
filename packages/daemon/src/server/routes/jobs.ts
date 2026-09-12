import type { FastifyInstance } from 'fastify'
import type {
  SubagentDispatchInput,
  SubagentDispatcher,
} from '../../agent/subagent-dispatcher.js'
import type { JobsRepo } from '../../jobs/repo.js'
import type { JobRunner } from '../../jobs/runner.js'

export interface JobsBatchExecutor {
  // Batch item executor — production wires this to the chat service.
  executeItem(item: { idx: number; request: unknown }): Promise<unknown>
}

export interface RegisterJobsRoutesDeps {
  repo: JobsRepo
  runner: JobRunner
  executor: JobsBatchExecutor
  subagentDispatcher?: () => SubagentDispatcher | null
  jobMaxConcurrency: number
}

export function registerJobsRoutes(
  app: FastifyInstance,
  deps: RegisterJobsRoutesDeps,
): void {
  app.post<{
    Body: {
      items: unknown[]
      concurrency: number
      failureMode: 'continue' | 'abort'
      preserveOrder: boolean
    }
  }>('/jobs/batch', async (req) => {
    const { items, concurrency, failureMode } = req.body
    if (!Array.isArray(items) || items.length === 0) {
      const err = new Error('items must be a non-empty array') as Error & {
        statusCode?: number
      }
      err.statusCode = 400
      throw err
    }
    const cap = Math.min(
      Math.max(1, concurrency | 0),
      deps.jobMaxConcurrency,
    )
    const job = deps.repo.create({
      kind: 'batch',
      total: items.length,
      concurrency: cap,
    })
    deps.repo.insertItems(
      job.id,
      items.map((request, idx) => ({ idx, request })),
    )

    // Fire and forget — runner persists state via repo.
    void deps.runner
      .run({
        jobId: job.id,
        concurrency: cap,
        failureMode,
        execute: (item) =>
          deps.executor.executeItem({ idx: item.idx, request: item.request }),
      })
      .catch(() => {
        /* runner persists error itself */
      })

    return {
      jobId: job.id,
      status: 'pending',
      total: items.length,
      createdAt: job.createdAt,
    }
  })

  app.post<{ Body: SubagentDispatchInput }>('/jobs/subagent', async (req, reply) => {
    const dispatcher = deps.subagentDispatcher?.() ?? null
    if (!dispatcher) {
      const err = new Error('Subagent dispatcher not initialized') as Error & {
        statusCode?: number
      }
      err.statusCode = 503
      throw err
    }

    const input = req.body
    if (!input || typeof input.prompt !== 'string' || input.prompt.trim().length === 0) {
      const err = new Error('prompt required') as Error & { statusCode?: number }
      err.statusCode = 400
      throw err
    }

    const job = deps.repo.create({
      kind: 'subagent',
      total: 1,
      concurrency: 1,
    })
    deps.repo.insertItems(job.id, [{ idx: 0, request: input }])

    void deps.runner
      .run({
        jobId: job.id,
        concurrency: 1,
        failureMode: 'abort',
        execute: async (item) => {
          const result = await dispatcher.dispatch(item.request as SubagentDispatchInput)
          if (result.status === 'failed') {
            throw new Error(result.error ?? 'subagent failed')
          }
          return result
        },
      })
      .catch(() => {
        /* runner persists error itself */
      })

    return reply.status(202).send({
      jobId: job.id,
      status: 'pending',
      total: 1,
      createdAt: job.createdAt,
    })
  })

  app.get<{ Params: { id: string } }>('/jobs/:id', async (req) => {
    const job = deps.repo.get(req.params.id)
    if (!job) {
      const err = new Error('not found') as Error & { statusCode?: number }
      err.statusCode = 404
      throw err
    }
    return job
  })

  app.get<{
    Params: { id: string }
    Querystring: { since?: string }
  }>('/jobs/:id/items', async (req) => {
    const job = deps.repo.get(req.params.id)
    if (!job) {
      const err = new Error('not found') as Error & { statusCode?: number }
      err.statusCode = 404
      throw err
    }
    const since = Math.max(
      0,
      parseInt(req.query.since ?? '0', 10) || 0,
    )
    const items = deps.repo.listCompletedItems(req.params.id, since)
    return { jobId: req.params.id, status: job.status, items }
  })

  app.delete<{ Params: { id: string } }>('/jobs/:id', async (req) => {
    const job = deps.repo.get(req.params.id)
    if (!job) {
      const err = new Error('not found') as Error & { statusCode?: number }
      err.statusCode = 404
      throw err
    }
    deps.runner.cancel(req.params.id)
    return { ok: true }
  })
}
