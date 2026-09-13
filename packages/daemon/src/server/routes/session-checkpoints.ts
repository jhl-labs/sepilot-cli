import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import '../fastify-types.js'
import { zodRequestValidation } from './utils.js'
import { RewindConflictError } from '../../agent/edit-rollback/store.js'

interface IdParams {
  id: string
}

interface RewindBody {
  checkpointId: string
  expectedCheckpointIds?: string[]
}

const idParamsSchema = z.object({ id: z.string().min(1) })
const rewindBodySchema = z.object({ checkpointId: z.string().min(1), expectedCheckpointIds: z.array(z.string().min(1)).max(100).optional() })

/**
 * Durable edit-checkpoint listing and file rewind (time travel). Every
 * committed agent edit checkpoint is persisted by EditSnapshotStore;
 * rewinding restores all touched files to their state before the chosen
 * checkpoint and drops the rewound history, mirroring Claude Code's
 * /rewind file scope. Conversation rewind is served separately by
 * POST /sessions/:id/branch and /undo.
 */
export function registerSessionCheckpointRoutes(app: FastifyInstance): void {
  app.get<{ Params: { id: string; checkpointId: string } }>('/sessions/:id/checkpoints/:checkpointId/preview', {
    preValidation: zodRequestValidation({ params: { schema: idParamsSchema.extend({ checkpointId: z.string().min(1) }), message: 'Invalid checkpoint' } }),
  }, async (request, reply) => {
    const runtime = app.runtime
    const params = request.params
    if (!runtime?.sessions || !runtime.editSnapshotStore) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Edit checkpoints not available' } })
    if (!await runtime.sessions.get(params.id)) return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'Session not found' } })
    try { return { data: await runtime.editSnapshotStore.previewRewind(params.id, params.checkpointId) } }
    catch (error) {
      if (error instanceof Error && error.message.includes('not found')) return reply.status(404).send({ error: { code: 'NOT_FOUND', message: error.message } })
      throw error
    }
  })
  app.get<{ Params: IdParams }>('/sessions/:id/checkpoints', {
    preValidation: zodRequestValidation({
      params: { schema: idParamsSchema, message: 'Invalid session id' },
    }),
  }, async (request, reply) => {
    const runtime = app.runtime
    if (!runtime?.sessions || !runtime.editSnapshotStore) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Edit checkpoints not available' },
      })
    }
    const params = request.params
    const session = await runtime.sessions.get(params.id)
    if (!session) {
      return reply.status(404).send({
        error: { code: 'NOT_FOUND', message: 'Session not found' },
      })
    }
    const checkpoints = await runtime.editSnapshotStore.listCheckpoints(params.id)
    return { data: { checkpoints } }
  })

  app.post<{ Params: IdParams; Body: RewindBody }>('/sessions/:id/rewind', {
    preValidation: zodRequestValidation({
      params: { schema: idParamsSchema, message: 'Invalid session id' },
      body: { schema: rewindBodySchema, message: 'Invalid rewind request body' },
    }),
  }, async (request, reply) => {
    const runtime = app.runtime
    if (!runtime?.sessions || !runtime.editSnapshotStore) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Edit checkpoints not available' },
      })
    }
    const params = request.params
    const body = request.body
    const session = await runtime.sessions.get(params.id)
    if (!session) {
      return reply.status(404).send({
        error: { code: 'NOT_FOUND', message: 'Session not found' },
      })
    }
    try {
      if (runtime.activeRuns?.get(params.id)) {
        return reply.status(409).send({ error: { code: 'REWIND_CONFLICT', message: 'Wait for the active run to finish before rewinding files.' } })
      }
      const result = await runtime.editSnapshotStore.rewindFiles(
        params.id,
        body.checkpointId,
        body.expectedCheckpointIds,
      )
      return { data: result }
    } catch (err) {
      if (err instanceof RewindConflictError) {
        return reply.status(409).send({ error: { code: 'REWIND_CONFLICT', message: err.message, paths: err.paths } })
      }
      const message = err instanceof Error ? err.message : String(err)
      if (message.includes('not found')) {
        return reply.status(404).send({ error: { code: 'NOT_FOUND', message } })
      }
      throw err
    }
  })
}
