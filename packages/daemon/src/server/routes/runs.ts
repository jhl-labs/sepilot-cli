import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import '../fastify-types.js'
import { buildStateBoard } from '../../agent/graph/state-board.js'
import {
  latestStateBoardAt,
  recoverStateBoard,
} from '../../agent/graph/state-board-recover.js'
import { zodRequestValidation } from './utils.js'

interface SessionIdParams {
  sessionId: string
}

const sessionIdParamsSchema = z.object({ sessionId: z.string().min(1) })

export async function runsRoutes(app: FastifyInstance) {
  app.get('/runs', async (_request, reply) => {
    const activeRuns = app.runtime?.activeRuns
    if (!activeRuns) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
      })
    }
    return activeRuns.list()
  })

  app.get<{ Params: { sessionId: string } }>('/runs/:sessionId', async (request, reply) => {
    const runtime = app.runtime
    if (!runtime?.activeRuns) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
      })
    }

    const { sessionId } = request.params
    const run = runtime.activeRuns.get(sessionId)
    if (!run) {
      return reply.status(404).send({
        error: { code: 'NOT_FOUND', message: 'Run not active' },
      })
    }

    const checkpoint = (await runtime.runCheckpoints?.inspect(sessionId)) ?? {
      status: 'missing' as const,
    }
    return {
      ...run,
      checkpoint,
    }
  })

  app.post<{ Params: SessionIdParams }>(
    '/runs/:sessionId/cancel',
    {
      preValidation: zodRequestValidation({
        params: { schema: sessionIdParamsSchema, message: 'Invalid session id' },
      }),
    },
    async (request, reply) => {
      const runtime = app.runtime
      const activeRuns = runtime?.activeRuns
      if (!activeRuns) {
        return reply.status(503).send({
          error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
        })
      }
      const { sessionId } = request.params
      const cancelled = await activeRuns.cancel(sessionId)
      if (!cancelled && activeRuns.get(sessionId)) {
        return reply.status(409).send({ error: { code: 'RUN_CANCEL_UNAVAILABLE', message: 'This active run has no registered cancellation owner. No cancellation was performed; use its originating task controls.' } })
      }
      // A cancellation acknowledgment is a completion boundary, not merely a
      // signal-delivery receipt. Waiting for the session lease prevents a
      // client that immediately submits its next turn from racing cleanup and
      // receiving BUSY for the run it just cancelled.
      await runtime.sessionBusy?.waitUntilIdle(sessionId)
      return { data: { cancelled } }
    },
  )

  // Canonical state-board inspection endpoint (PLAN_023 D4). Returns the
  // structured board, not rendered text, so each surface renders it itself.
  // The durable journal is the source of truth; `source` reports whether a run
  // is currently active ('live') or only persisted ('journal').
  app.get<{ Params: { sessionId: string } }>(
    '/runs/:sessionId/state',
    async (request, reply) => {
      const runtime = app.runtime
      if (!runtime) {
        return reply.status(503).send({
          error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
        })
      }
      const { sessionId } = request.params
      const active = runtime.activeRuns?.get(sessionId) ?? null
      const events = await runtime.sessions.getEvents(sessionId)
      const board = recoverStateBoard(events)
      if (board) {
        return {
          data: {
            board,
            source: active ? 'live' : 'journal',
            updatedAt: latestStateBoardAt(events) ?? null,
          },
        }
      }
      // Active run that has not journaled a board yet — rebuild from the latest
      // node-boundary checkpoint so a fresh run still surfaces its board.
      if (active) {
        const inspection = await runtime.runCheckpoints?.inspect(sessionId)
        if (inspection?.status === 'available' && inspection.checkpoint.graphState) {
          const updatedAt = Date.parse(inspection.checkpoint.checkpointedAt)
          return {
            data: {
              board: buildStateBoard(inspection.checkpoint.graphState),
              source: 'live' as const,
              updatedAt: Number.isFinite(updatedAt) ? updatedAt : null,
            },
          }
        }
      }
      return { data: { board: null, source: 'none' as const, updatedAt: null } }
    },
  )
}
