import type { FastifyInstance, FastifyReply, FastifyRequest } from 'fastify'
import type { SwarmRun } from '../../agent/swarm/run/swarm-run.js'
import type { SwarmRunRegistry } from '../../agent/swarm/run/run-registry.js'
import type { SwarmRunStore } from '../../agent/swarm/run/run-store.js'
import type { SwarmDriveInput, SwarmDriveResult } from '../../agent/swarm/index.js'
import { attachAgentSchema, createRunSchema, driveAgentSchema, interruptAgentSchema, sendKeysSchema } from './swarm-schema.js'
import { resolveApiActor } from '../request-actor.js'
import '../fastify-types.js'

export interface SwarmRoutesDeps {
  /**
   * Resolver for the registry instance — function (not direct ref) so the
   * route stays valid across runtime swaps and so tests can inject stubs.
   */
  registry: () => SwarmRunRegistry | null
  /** Resolver for the run-jsonl store; required for backfill on GET /events. */
  store: () => SwarmRunStore | null
  /**
   * Hook to actually start the run — at minimum calls run.start(); the real
   * implementation (Task 20) launches the warm pool + the agent engine.
   */
  startRun: (
    run: SwarmRun,
    opts: { warmPool: readonly string[]; autoApproveAgents: boolean; noSupervisor?: boolean },
  ) => Promise<void>
  /** Forward keystrokes / resize to the tmux session backing one agent handle. */
  keysHandler: (
    runId: string,
    handle: string,
    payload: {
      keys?: string
      keyName?: string
      enter?: boolean
      resize?: { cols: number; rows: number }
      principal?: string
    },
  ) => Promise<void>
  /** Capture the current cleaned tmux pane contents for one agent handle. */
  captureHandler: (
    runId: string,
    handle: string,
    options?: { lines?: number; raw?: boolean },
  ) => Promise<string>
  /** Drive one agent through send/wait/observe turns. */
  driveHandler: (
    runId: string,
    handle: string,
    input: Omit<SwarmDriveInput, 'agent_handle'>,
  ) => Promise<SwarmDriveResult>
  /**
   * Stop all warm-pool tmux sessions for the named run, immediately.
   * Called by DELETE so cancel doesn't leave agents running for minutes
   * after the user asked to stop. The startSwarmRun's finally block will
   * still run (no-op for already-stopped agents).
   */
  killRunAgents: (runId: string) => Promise<void>
}

interface RunIdParams {
  id: string
}
interface RunAgentParams {
  id: string
  handle: string
}
interface RunsListQuery {
  status?: string
}
interface EventsQuery {
  since?: string
  stream?: string
  follow?: string
}
interface CaptureQuery {
  lines?: string
  raw?: string
}

function requestOwner(request: FastifyRequest): string {
  return resolveApiActor(request.authContext, 'api:anonymous')
}

function leaseConflict(run: SwarmRun | undefined, handle: string, owner: string): string | null {
  const leaseRun = run as Partial<Pick<SwarmRun, 'isInteractiveHeld' | 'interactiveLeaseOwner'>> | undefined
  if (typeof leaseRun?.isInteractiveHeld !== 'function') return null
  if (!leaseRun.isInteractiveHeld(handle)) return null
  const currentOwner = typeof leaseRun.interactiveLeaseOwner === 'function'
    ? leaseRun.interactiveLeaseOwner(handle)
    : undefined
  return currentOwner && currentOwner !== owner ? currentOwner : null
}

function leaseConflictReply(reply: FastifyReply, owner: string) {
  return reply.status(409).send({
    error: {
      code: 'INTERACTIVE_LEASE_HELD',
      message: `interactive lease is held by ${owner}`,
    },
  })
}

/**
 * HTTP surface for the swarm CLI orchestrator.
 *
 * - `POST /api/v1/swarm/runs`            — create + start a run
 * - `GET  /api/v1/swarm/runs`            — list (optional ?status=)
 * - `GET  /api/v1/swarm/runs/:id`        — snapshot for one run
 * - `DELETE /api/v1/swarm/runs/:id`      — cancel + deregister
 * - `GET  /api/v1/swarm/runs/:id/events` — JSONL backfill or SSE stream
 * - `POST /api/v1/swarm/runs/:id/agents/:handle/keys` — send keys / resize
 * - `GET /api/v1/swarm/runs/:id/agents/:handle/capture` — current pane text
 */
export function registerSwarmRoutes(
  app: FastifyInstance,
  deps: SwarmRoutesDeps,
): void {
  app.post('/api/v1/swarm/runs', async (request, reply) => {
    const parsed = createRunSchema.safeParse(request.body)
    if (!parsed.success) {
      return reply.status(400).send({
        error: { code: 'INVALID_REQUEST', message: parsed.error.message },
      })
    }
    const registry = deps.registry()
    if (!registry) {
      return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE' } })
    }
    const cwd = parsed.data.cwd ?? process.cwd()
    const run = registry.create({
      goal: parsed.data.goal,
      worktreePath: cwd,
      worktreeCreatedByDaemon: false,
    })
    try {
      await deps.startRun(run, {
        warmPool: parsed.data.warmPool,
        autoApproveAgents: parsed.data.autoApproveAgents,
        noSupervisor: parsed.data.noSupervisor,
      })
    } catch (err) {
      registry.deregister(run.id)
      const message = err instanceof Error ? err.message : String(err)
      return reply
        .status(422)
        .send({ error: { code: 'AGENT_UNAVAILABLE', message } })
    }
    return reply.status(200).send({ runId: run.id })
  })

  app.get<{ Querystring: RunsListQuery }>(
    '/api/v1/swarm/runs',
    async (request, reply) => {
      const registry = deps.registry()
      if (!registry) return reply.status(503).send([])
      const query = request.query
      const status = query.status
      const all = registry.list().map((r) => r.snapshot())
      return reply.send(status ? all.filter((r) => r.status === status) : all)
    },
  )

  app.get<{ Querystring: { limit?: string } }>(
    '/api/v1/swarm/runs/history',
    async (request, reply) => {
      const store = deps.store()
      if (!store) return reply.status(503).send([])
      const { limit: rawLimit } = request.query
      const limit = Math.max(1, Math.min(200, Number(rawLimit ?? 50) || 50))
      const runs = await store.listHistory(limit)
      // Newest first.
      runs.sort((a, b) => (b.endedAt ?? b.createdAt) - (a.endedAt ?? a.createdAt))
      return reply.send(runs)
    },
  )

  app.get<{ Params: RunIdParams }>(
    '/api/v1/swarm/runs/:id',
    async (request, reply) => {
      const params = request.params
      const run = deps.registry()?.get(params.id)
      if (run) return reply.send(run.snapshot())
      // Fallback: reconstruct from jsonl for completed runs.
      const store = deps.store()
      if (store) {
        const reconstructed = await store.reconstruct(params.id)
        if (reconstructed) return reply.send(reconstructed)
      }
      return reply.status(404).send({ error: { code: 'NOT_FOUND' } })
    },
  )

  app.delete<{ Params: RunIdParams }>(
    '/api/v1/swarm/runs/:id',
    async (request, reply) => {
      const params = request.params
      const registry = deps.registry()
      const run = registry?.get(params.id)
      if (!run || !registry) {
        return reply.status(404).send({ error: { code: 'NOT_FOUND' } })
      }
      // Tear down tmux sessions FIRST so the engine's next swarm.* tool call
      // fails fast (instead of letting it loop for minutes after cancel).
      try { await deps.killRunAgents(params.id) } catch { /* best-effort */ }
      run.finish('cancelled')
      registry.deregister(params.id)
      return reply.send({ status: 'cancelled' })
    },
  )

  app.get<{ Params: RunIdParams; Querystring: EventsQuery }>(
    '/api/v1/swarm/runs/:id/events',
    async (request, reply) => {
      const params = request.params
      const query = request.query
      const sinceRaw = query.since
      const since = Number(sinceRaw ?? 0) || 0
      const wantsStream =
        query.stream === '1' ||
        (request.headers.accept ?? '').includes('text/event-stream')

      const store = deps.store()
      if (!store) return reply.status(503).send([])

      const backfill = await store.readSince(params.id, since)

      if (!wantsStream) {
        return reply.send(backfill)
      }

      reply.raw.setHeader('Content-Type', 'text/event-stream')
      reply.raw.setHeader('Cache-Control', 'no-cache')
      reply.raw.setHeader('Connection', 'keep-alive')
      reply.raw.flushHeaders?.()
      for (const e of backfill) {
        reply.raw.write(`data: ${JSON.stringify(e)}\n\n`)
      }
      const follow = query.follow !== '0'
      if (!follow) {
        reply.raw.end()
        return reply
      }
      const run = deps.registry()?.get(params.id)
      if (!run) {
        reply.raw.end()
        return reply
      }
      const onEvent = (e: unknown) => {
        reply.raw.write(`data: ${JSON.stringify(e)}\n\n`)
      }
      run.on('event', onEvent)
      const keepaliveMs = Number(process.env.SEPILOTD_SSE_KEEPALIVE_MS ?? 15000)
      const ka = setInterval(() => {
        reply.raw.write(': keep-alive\n\n')
      }, keepaliveMs)
      request.raw.on('close', () => {
        clearInterval(ka)
        run.off('event', onEvent)
        try {
          reply.raw.end()
        } catch {
          /* ignore */
        }
      })
      return reply
    },
  )

  app.post<{ Params: RunAgentParams }>(
    '/api/v1/swarm/runs/:id/agents/:handle/keys',
    async (request, reply) => {
      const parsed = sendKeysSchema.safeParse(request.body)
      if (!parsed.success) {
        return reply
          .status(400)
          .send({ error: { code: 'INVALID_REQUEST', message: parsed.error.message } })
      }
      const params = request.params
      const owner = requestOwner(request)
      const conflictOwner = leaseConflict(deps.registry()?.get(params.id), params.handle, owner)
      if (conflictOwner) return leaseConflictReply(reply, conflictOwner)
      try {
        await deps.keysHandler(params.id, params.handle, { ...parsed.data, principal: owner })
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err)
        if (message.startsWith('SWARM_KEY_DENIED:')) {
          return reply.status(403).send({ error: { code: 'SWARM_KEY_DENIED', message } })
        }
        return reply.status(404).send({ error: { code: 'NOT_FOUND', message } })
      }
      return reply.send({ ok: true })
    },
  )

  app.post<{ Params: RunAgentParams }>(
    '/api/v1/swarm/runs/:id/agents/:handle/interrupt',
    async (request, reply) => {
      const parsed = interruptAgentSchema.safeParse(request.body ?? {})
      if (!parsed.success) {
        return reply
          .status(400)
          .send({ error: { code: 'INVALID_REQUEST', message: parsed.error.message } })
      }
      const params = request.params
      const owner = requestOwner(request)
      const conflictOwner = leaseConflict(deps.registry()?.get(params.id), params.handle, owner)
      if (conflictOwner) return leaseConflictReply(reply, conflictOwner)
      try {
        await deps.keysHandler(params.id, params.handle, {
          keyName: parsed.data.escape ? 'Escape' : 'C-c',
          principal: owner,
        })
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err)
        if (message.startsWith('SWARM_KEY_DENIED:')) {
          return reply.status(403).send({ error: { code: 'SWARM_KEY_DENIED', message } })
        }
        return reply.status(404).send({ error: { code: 'NOT_FOUND', message } })
      }
      return reply.send({ ok: true })
    },
  )

  app.post<{ Params: RunAgentParams }>(
    '/api/v1/swarm/runs/:id/agents/:handle/attach',
    async (request, reply) => {
      const parsed = attachAgentSchema.safeParse(request.body ?? {})
      if (!parsed.success) {
        return reply
          .status(400)
          .send({ error: { code: 'INVALID_REQUEST', message: parsed.error.message } })
      }
      const params = request.params
      const run = deps.registry()?.get(params.id)
      if (!run?.getAgent(params.handle)) {
        return reply.status(404).send({ error: { code: 'NOT_FOUND' } })
      }
      const owner = requestOwner(request)
      const acquired = parsed.data.renew
        ? run.renewInteractiveLease(params.handle, owner)
        : run.acquireInteractiveLease(params.handle, owner)
      if (!acquired) {
        const conflictOwner = run.interactiveLeaseOwner(params.handle) ?? 'another owner'
        return leaseConflictReply(reply, conflictOwner)
      }
      const lease = run.getInteractiveLease(params.handle)
      return reply.send({ ok: true, lease })
    },
  )

  app.post<{ Params: RunAgentParams }>(
    '/api/v1/swarm/runs/:id/agents/:handle/detach',
    async (request, reply) => {
      const params = request.params
      const run = deps.registry()?.get(params.id)
      if (!run?.getAgent(params.handle)) {
        return reply.status(404).send({ error: { code: 'NOT_FOUND' } })
      }
      const owner = requestOwner(request)
      if (!run.releaseInteractiveLease(params.handle, owner)) {
        const conflictOwner = run.interactiveLeaseOwner(params.handle) ?? 'another owner'
        return leaseConflictReply(reply, conflictOwner)
      }
      return reply.send({ ok: true })
    },
  )

  app.get<{ Params: RunAgentParams; Querystring: CaptureQuery }>(
    '/api/v1/swarm/runs/:id/agents/:handle/capture',
    async (request, reply) => {
      const params = request.params
      const query = request.query
      const rawLines = query.lines
      let lines: number | undefined
      if (rawLines !== undefined) {
        lines = Number(rawLines)
        if (!Number.isFinite(lines) || lines < 1) {
          return reply.status(400).send({
            error: {
              code: 'INVALID_REQUEST',
              message: 'lines must be a positive number',
            },
          })
        }
        lines = Math.floor(lines)
      }
      const raw = query.raw === '1' || query.raw === 'true'
      try {
        const text = await deps.captureHandler(params.id, params.handle, { lines, raw })
        return reply.send({ text })
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err)
        return reply
          .status(404)
          .send({ error: { code: 'NOT_FOUND', message } })
      }
    },
  )

  app.post<{ Params: RunAgentParams }>(
    '/api/v1/swarm/runs/:id/agents/:handle/drive',
    async (request, reply) => {
      const parsed = driveAgentSchema.safeParse(request.body)
      if (!parsed.success) {
        return reply
          .status(400)
          .send({ error: { code: 'INVALID_REQUEST', message: parsed.error.message } })
      }
      const params = request.params
      try {
        const result = await deps.driveHandler(params.id, params.handle, parsed.data)
        return reply.send(result)
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err)
        return reply
          .status(404)
          .send({ error: { code: 'NOT_FOUND', message } })
      }
    },
  )
}

export async function swarmRoutes(app: FastifyInstance) {
  registerSwarmRoutes(app, {
    registry: () => app.runtime?.swarmRunRegistry ?? null,
    store: () => app.runtime?.swarmRunStore ?? null,
    startRun: async (run, opts) => {
      const start = app.runtime?.startSwarmRun
      if (!start) throw new Error('swarm runtime not initialized')
      await start(run, opts)
    },
    keysHandler: async (runId, handle, payload) => {
      const fwd = app.runtime?.forwardSwarmKeys
      if (!fwd) throw new Error('swarm runtime not initialized')
      await fwd(runId, handle, payload)
    },
    captureHandler: async (runId, handle, options) => {
      const capture = app.runtime?.captureSwarmAgent
      if (!capture) throw new Error('swarm runtime not initialized')
      return capture(runId, handle, options)
    },
    killRunAgents: async (runId) => {
      const kill = app.runtime?.killSwarmRunAgents
      if (!kill) throw new Error('swarm runtime not initialized')
      await kill(runId)
    },
    driveHandler: async (runId, handle, input) => {
      const drive = app.runtime?.driveSwarmAgent
      if (!drive) throw new Error('swarm runtime not initialized')
      return drive(runId, { ...input, agent_handle: handle })
    },
  })
}
