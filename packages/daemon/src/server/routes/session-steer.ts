import { randomUUID } from 'node:crypto'
import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import '../fastify-types.js'
import {
  appendSteeringNote,
  createSteeringNote,
  type AgentSteeringNote,
} from '../../agent/graph/state-board.js'
import { zodRequestValidation } from './utils.js'

interface SessionIdParams {
  id: string
}

const sessionIdParamsSchema = z.object({ id: z.string().min(1) })

const steerRequestSchema = z.object({
  message: z.string().min(1),
  kind: z.enum(['instruction', 'question']).optional(),
})

const cancelSteerRequestSchema = z.union([
  z.object({ selector: z.enum(['latest', 'all']) }),
  z.object({ noteId: z.string().min(1) }),
])

type SteerBody = z.infer<typeof steerRequestSchema>
type CancelSteerBody = z.infer<typeof cancelSteerRequestSchema>

/**
 * `POST /sessions/:id/steer` — queue a mid-run user steering note (an
 * instruction or question) onto the LIVE, in-flight run's graph state so the
 * running loop picks it up on its next turn (see
 * `agent/graph/state-board.ts` `appendSteeringNote` /
 * `takeUnconsumedSteeringNotes`). 409s when the session has no active run;
 * the note is never appended to a session with nothing running.
 */
export function registerSessionSteerRoute(app: FastifyInstance): void {
  app.post<{ Params: SessionIdParams; Body: SteerBody }>(
    '/sessions/:id/steer',
    {
      preValidation: zodRequestValidation({
        params: { schema: sessionIdParamsSchema, message: 'Invalid session id' },
        body: { schema: steerRequestSchema, message: 'Invalid steer request body' },
      }),
    },
    async (request, reply) => {
      const runtime = app.runtime
      if (!runtime?.activeRuns) {
        return reply.status(503).send({
          error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
        })
      }

      const { id: sessionId } = request.params
      const activeRun = runtime.activeRuns.get(sessionId)
      if (!activeRun) {
        return reply.status(409).send({
          error: { code: 'NO_ACTIVE_RUN', message: 'Session has no active run' },
        })
      }

      const { message, kind } = request.body
      const liveState = runtime.activeRuns.getLiveState(sessionId) as {
        steeringNotes?: AgentSteeringNote[]
      } | null
      const note = liveState
        ? appendSteeringNote(liveState, { message, kind })
        : createSteeringNote({ message, kind })
      if (!liveState) {
        runtime.activeRuns.queuePendingSteeringNote(sessionId, note)
      }

      // No checkpoint hook here: `runtime.runCheckpoints.save` requires a full
      // `createGraphRunCheckpoint(state, context, resumeStage, ...)` call, and
      // the `GraphExecutionContext` (active graph node id/iteration/resume
      // stage/pending tool execution) is only known to the in-flight graph
      // engine loop (see `agent/graph/engine.ts` and `agent/graph/nodes.ts`),
      // not to this HTTP route, which only has `runtime.activeRuns.getLiveState`
      // (a bare `AgentState`-shaped object, no execution context). Wiring a
      // route-reachable checkpoint save would mean threading the live
      // `GraphExecutionContext` out of the engine loop into `activeRuns` just
      // for this narrow case — nontrivial new plumbing, so it is intentionally
      // not done here. If the run is admitted but the live state is not
      // registered yet, `ActiveRunRegistry` carries the pending note forward
      // into the first live state. The crash window between the steering-note
      // append/queue above and the next natural engine checkpoint (per graph node, see
      // `engine.ts` `context.saveRunCheckpoint`) is accepted; the durable
      // `steering_ack`/`steering_consumed` session events below and in
      // `agent/graph/nodes.ts` already make the note recoverable from the
      // session journal even if the in-memory `liveState` copy is lost.
      await runtime.sessions.appendEvent(sessionId, {
        type: 'steering_ack',
        id: randomUUID(),
        timestamp: new Date().toISOString(),
        noteId: note.id,
        kind: note.kind,
        message: note.message,
      })
      runtime.sessionWatchBroker?.emit({
        type: 'steering_ack',
        sessionId,
        noteId: note.id,
        kind: note.kind,
        message: note.message,
      })

      const pendingSteeringNoteCount =
        runtime.activeRuns.get(sessionId)?.pendingSteeringNoteCount ?? 1
      return reply.status(202).send({
        data: { noteId: note.id, queued: true, pendingSteeringNoteCount },
      })
    },
  )

  app.post<{ Params: SessionIdParams; Body: CancelSteerBody }>(
    '/sessions/:id/steer/cancel',
    {
      preValidation: zodRequestValidation({
        params: { schema: sessionIdParamsSchema, message: 'Invalid session id' },
        body: { schema: cancelSteerRequestSchema, message: 'Invalid steer cancellation request' },
      }),
    },
    async (request, reply) => {
      const runtime = app.runtime
      if (!runtime?.activeRuns) {
        return reply.status(503).send({
          error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
        })
      }

      const { id: sessionId } = request.params
      if (!runtime.activeRuns.get(sessionId)) {
        return reply.status(409).send({
          error: { code: 'NO_ACTIVE_RUN', message: 'Session has no active run' },
        })
      }

      const result = runtime.activeRuns.cancelPendingSteeringNotes(sessionId, request.body)
      if (result.status !== 'cancelled') {
        const error = {
          already_consumed: {
            status: 409,
            code: 'STEERING_ALREADY_CONSUMED',
            message: 'The steering note was already applied to the active run',
          },
          already_cancelled: {
            status: 409,
            code: 'STEERING_ALREADY_CANCELLED',
            message: 'The steering note was already cancelled',
          },
          no_pending: {
            status: 409,
            code: 'NO_PENDING_STEERING',
            message: 'The active run has no pending steering notes',
          },
          not_found: {
            status: 404,
            code: 'STEERING_NOTE_NOT_FOUND',
            message: 'The steering note was not found on the active run',
          },
        }[result.status]
        return reply.status(error.status).send({
          error: { code: error.code, message: error.message },
        })
      }

      for (const noteId of result.cancelledNoteIds) {
        await runtime.sessions.appendEvent(sessionId, {
          type: 'steering_cancelled',
          id: randomUUID(),
          timestamp: new Date().toISOString(),
          noteId,
        })
        runtime.sessionWatchBroker?.emit({
          type: 'steering_cancelled',
          sessionId,
          noteId,
        })
      }

      return reply.status(200).send({ data: result })
    },
  )
}
