import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import '../fastify-types.js'
import { zodRequestValidation } from './utils.js'

interface IdParams {
  id: string
}

interface RewindBody {
  checkpointId: string
}

const idParamsSchema = z.object({ id: z.string().min(1) })
const rewindBodySchema = z.object({ checkpointId: z.string().min(1) })

/**
 * Durable edit-checkpoint listing and file rewind (time travel). Every
 * committed agent edit checkpoint is persisted by EditSnapshotStore;
 * rewinding restores all touched files to their state before the chosen
 * checkpoint and drops the rewound history, mirroring Claude Code's
 * /rewind file scope. Conversation rewind is served separately by
 * POST /sessions/:id/branch and /undo.
 */
export function registerSessionCheckpointRoutes(app: FastifyInstance): void {
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
      const result = await runtime.editSnapshotStore.rewindFiles(
        params.id,
        body.checkpointId,
      )
      return { data: result }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      if (message.includes('not found')) {
        return reply.status(404).send({ error: { code: 'NOT_FOUND', message } })
      }
      throw err
    }
  })
}
