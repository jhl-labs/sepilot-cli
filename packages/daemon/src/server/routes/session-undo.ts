import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import '../fastify-types.js'
import {
  trimToPreviousTurn,
  type UndoSnapshot,
} from '../runtime/undo-stack.js'
import { zodRequestValidation } from './utils.js'

interface IdParams {
  id: string
}

const idParamsSchema = z.object({ id: z.string().min(1) })

export function registerSessionUndoRoutes(app: FastifyInstance): void {
  app.post<{ Params: IdParams }>('/sessions/:id/undo', {
    preValidation: zodRequestValidation({
      params: { schema: idParamsSchema, message: 'Invalid session id' },
    }),
  }, async (request, reply) => {
    const runtime = app.runtime
    if (!runtime?.sessions || !runtime.sessionUndoStack) {
      reply.code(503)
      return { error: 'session undo not available' }
    }
    const params = request.params
    const sessionId = params.id
    const session = await runtime.sessions.get(sessionId)
    if (!session) {
      reply.code(404)
      return { error: 'session not found' }
    }
    const events = await runtime.sessions.getEvents(sessionId)
    const trimmed = trimToPreviousTurn(events)
    if (trimmed === null) {
      reply.code(409)
      return { error: 'nothing to undo on this session' }
    }
    if (!runtime.sessions.replaceEvents) {
      reply.code(503)
      return { error: 'session store does not support undo' }
    }
    const snapshot: UndoSnapshot = {
      events,
      capturedAt: new Date().toISOString(),
    }
    runtime.sessionUndoStack.pushRedo(sessionId, snapshot)
    await runtime.sessions.replaceEvents(sessionId, trimmed)
    await runtime.sessionRuntimeSnapshots?.refreshSession(sessionId).catch(() => undefined)
    return {
      data: {
        sessionId,
        eventCount: trimmed.length,
        ...runtime.sessionUndoStack.inspect(sessionId),
      },
    }
  })

  app.post<{ Params: IdParams }>('/sessions/:id/redo', {
    preValidation: zodRequestValidation({
      params: { schema: idParamsSchema, message: 'Invalid session id' },
    }),
  }, async (request, reply) => {
    const runtime = app.runtime
    if (!runtime?.sessions || !runtime.sessionUndoStack) {
      reply.code(503)
      return { error: 'session redo not available' }
    }
    const params = request.params
    const sessionId = params.id
    const next = runtime.sessionUndoStack.popRedo(sessionId)
    if (!next) {
      reply.code(409)
      return { error: 'nothing to redo on this session' }
    }
    if (!runtime.sessions.replaceEvents) {
      reply.code(503)
      return { error: 'session store does not support redo' }
    }
    const current = await runtime.sessions.getEvents(sessionId)
    runtime.sessionUndoStack.pushUndo(sessionId, {
      events: current,
      capturedAt: new Date().toISOString(),
    })
    await runtime.sessions.replaceEvents(sessionId, next.events)
    await runtime.sessionRuntimeSnapshots?.refreshSession(sessionId).catch(() => undefined)
    return {
      data: {
        sessionId,
        eventCount: next.events.length,
        ...runtime.sessionUndoStack.inspect(sessionId),
      },
    }
  })
}
