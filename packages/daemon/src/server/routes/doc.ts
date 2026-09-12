// 글쓰기 모드 (writing canvas) REST + SSE endpoints.
//
// REST:
//   POST   /api/v1/doc/open           { path? , initialContent? } → DocSession
//   GET    /api/v1/doc/:id            → DocSession (latest)
//   POST   /api/v1/doc/:id/user_edit  { changes: DocChange[], expectedVersion? } → { version }
//   POST   /api/v1/doc/:id/save       { path? } → { path, mtimeMs }
//   POST   /api/v1/doc/:id/reload     → { version }
//   DELETE /api/v1/doc/:id            → { ok }
//   POST   /api/v1/doc/:id/diff/:previewId/accept  → { version }
//   POST   /api/v1/doc/:id/diff/:previewId/cancel  → { ok }
//   GET    /api/v1/doc/:id/previews   → DocDiffPreview[]
//
// SSE:
//   GET    /api/v1/doc/events?token=<daemon_token> → text/event-stream
//          event: doc.updated | doc.diff_pending | doc.diff_resolved | doc.closed
//          data:  JSON payload (DocEvent 그대로)

import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import '../fastify-types.js'
import { randomUUID } from 'node:crypto'
import { getDocRegistry } from '../../agent/doc/session.js'
import { zodRequestValidation } from './utils.js'
import { buildSseResponseHeaders } from '../sse-response.js'

const docIdParamsSchema = z.object({
  id: z.string().min(1),
})

const docPreviewParamsSchema = docIdParamsSchema.extend({
  previewId: z.string().min(1),
})

const docOpenBodySchema = z.object({
  path: z.string().nullable().optional(),
  initialContent: z.string().optional(),
}).default({})

const docChangeSchema = z.object({
  start: z.number().int().min(0),
  end: z.number().int().min(0),
  newText: z.string(),
}).refine((change) => change.end >= change.start, {
  message: 'end must be greater than or equal to start',
})

const docUserEditBodySchema = z.object({
  changes: z.array(docChangeSchema),
  expectedVersion: z.number().int().min(1).optional(),
})

const docSaveBodySchema = z.object({
  path: z.string().optional(),
}).default({})

type DocIdParams = z.infer<typeof docIdParamsSchema>
type DocPreviewParams = z.infer<typeof docPreviewParamsSchema>
type DocOpenBody = z.infer<typeof docOpenBodySchema>
type DocUserEditBody = z.infer<typeof docUserEditBodySchema>
type DocSaveBody = z.infer<typeof docSaveBodySchema>

export async function registerDocRoutes(app: FastifyInstance): Promise<void> {
  const registry = getDocRegistry()

  app.post<{ Body: DocOpenBody }>(
    '/doc/open',
    {
      preValidation: zodRequestValidation({
        body: {
          schema: docOpenBodySchema,
          message: 'Invalid doc open request body',
        },
      }),
    },
    async (request, reply) => {
      const body = request.body
      const session = await registry.open({
        path: body.path ?? null,
        initialContent: body.initialContent,
      })
      return reply.send({ data: session })
    },
  )

  app.get<{ Params: DocIdParams }>('/doc/:id', {
    preValidation: zodRequestValidation({
      params: {
        schema: docIdParamsSchema,
        message: 'Invalid doc id',
      },
    }),
  }, async (request, reply) => {
    const params = request.params
    const s = registry.get(params.id)
    if (!s) return reply.status(404).send({ error: { code: 'NOT_FOUND' } })
    return reply.send({ data: s })
  })

  app.post<{
    Params: DocIdParams
    Body: DocUserEditBody
  }>('/doc/:id/user_edit', {
    preValidation: zodRequestValidation({
      params: {
        schema: docIdParamsSchema,
        message: 'Invalid doc id',
      },
      body: {
        schema: docUserEditBodySchema,
        message: 'Invalid doc user edit request body',
      },
    }),
  }, async (request, reply) => {
    const params = request.params
    const body = request.body
    try {
      const result = registry.apply(params.id, body.changes, 'user', {
        toolName: 'user.edit',
        expectedVersion: body.expectedVersion,
      })
      return reply.send({ data: result })
    } catch (err) {
      return reply.status(409).send({
        error: {
          code: 'VERSION_MISMATCH',
          message: err instanceof Error ? err.message : String(err),
        },
      })
    }
  })

  app.post<{ Params: DocIdParams; Body: DocSaveBody }>(
    '/doc/:id/save',
    {
      preValidation: zodRequestValidation({
        params: {
          schema: docIdParamsSchema,
          message: 'Invalid doc id',
        },
        body: {
          schema: docSaveBodySchema,
          message: 'Invalid doc save request body',
        },
      }),
    },
    async (request, reply) => {
      const params = request.params
      const body = request.body
      try {
        const result = await registry.saveToDisk(params.id, body.path)
        return reply.send({ data: result })
      } catch (err) {
        return reply.status(500).send({
          error: { code: 'SAVE_FAILED', message: err instanceof Error ? err.message : String(err) },
        })
      }
    },
  )

  app.post<{ Params: DocIdParams }>('/doc/:id/undo', {
    preValidation: zodRequestValidation({
      params: {
        schema: docIdParamsSchema,
        message: 'Invalid doc id',
      },
    }),
  }, async (request, reply) => {
    const params = request.params
    try {
      const result = registry.undoLastLlmChange(params.id)
      if (!result) {
        return reply.status(404).send({
          error: { code: 'NOTHING_TO_UNDO', message: 'no LLM-authored change in history' },
        })
      }
      return reply.send({ data: result })
    } catch (err) {
      return reply.status(500).send({
        error: { code: 'UNDO_FAILED', message: err instanceof Error ? err.message : String(err) },
      })
    }
  })

  app.post<{ Params: DocIdParams }>('/doc/:id/reload', {
    preValidation: zodRequestValidation({
      params: {
        schema: docIdParamsSchema,
        message: 'Invalid doc id',
      },
    }),
  }, async (request, reply) => {
    const params = request.params
    try {
      const result = await registry.reloadFromDisk(params.id)
      return reply.send({ data: result })
    } catch (err) {
      return reply.status(500).send({
        error: { code: 'RELOAD_FAILED', message: err instanceof Error ? err.message : String(err) },
      })
    }
  })

  app.delete<{ Params: DocIdParams }>('/doc/:id', {
    preValidation: zodRequestValidation({
      params: {
        schema: docIdParamsSchema,
        message: 'Invalid doc id',
      },
    }),
  }, async (request, reply) => {
    const params = request.params
    const ok = registry.close(params.id, 'http delete')
    return reply.send({ data: { ok } })
  })

  app.post<{ Params: DocPreviewParams }>(
    '/doc/:id/diff/:previewId/accept',
    {
      preValidation: zodRequestValidation({
        params: {
          schema: docPreviewParamsSchema,
          message: 'Invalid doc preview id',
        },
      }),
    },
    async (request, reply) => {
      const params = request.params
      try {
        const result = registry.acceptPreview(params.id, params.previewId)
        return reply.send({ data: result })
      } catch (err) {
        return reply.status(409).send({
          error: { code: 'PREVIEW_INVALID', message: err instanceof Error ? err.message : String(err) },
        })
      }
    },
  )

  app.post<{ Params: DocPreviewParams }>(
    '/doc/:id/diff/:previewId/cancel',
    {
      preValidation: zodRequestValidation({
        params: {
          schema: docPreviewParamsSchema,
          message: 'Invalid doc preview id',
        },
      }),
    },
    async (request, reply) => {
      const params = request.params
      const ok = registry.cancelPreview(params.id, params.previewId)
      return reply.send({ data: { ok } })
    },
  )

  app.get<{ Params: DocIdParams }>('/doc/:id/previews', {
    preValidation: zodRequestValidation({
      params: {
        schema: docIdParamsSchema,
        message: 'Invalid doc id',
      },
    }),
  }, async (request, reply) => {
    const params = request.params
    try {
      return reply.send({ data: registry.listPreviews(params.id) })
    } catch (err) {
      return reply.status(404).send({
        error: { code: 'NOT_FOUND', message: err instanceof Error ? err.message : String(err) },
      })
    }
  })

  // SSE event stream — registry의 모든 doc event를 connected client에 fanout.
  // token은 query param으로 받음(EventSource는 custom header 불가).
  app.get('/doc/events', async (request, reply) => {
    reply.hijack()
    reply.raw.writeHead(200, buildSseResponseHeaders(request, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
      'X-Request-ID': request.requestId ?? randomUUID(),
    }))

    const send = (event: string, data: unknown) => {
      try {
        reply.raw.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`)
      } catch {
        // socket closed
      }
    }

    const onUpdated = (e: unknown) => send('doc.updated', e)
    const onDiffPending = (e: unknown) => send('doc.diff_pending', e)
    const onDiffResolved = (e: unknown) => send('doc.diff_resolved', e)
    const onClosed = (e: unknown) => send('doc.closed', e)

    registry.events.on('updated', onUpdated)
    registry.events.on('diffPending', onDiffPending)
    registry.events.on('diffResolved', onDiffResolved)
    registry.events.on('closed', onClosed)

    // keepalive every 15s
    const ka = setInterval(() => {
      try {
        reply.raw.write(`: keepalive\n\n`)
      } catch {
        // noop
      }
    }, 15_000)

    let finishStream!: () => void
    const closed = new Promise<void>((resolve) => { finishStream = resolve })
    const cleanup = () => {
      clearInterval(ka)
      registry.events.off('updated', onUpdated)
      registry.events.off('diffPending', onDiffPending)
      registry.events.off('diffResolved', onDiffResolved)
      registry.events.off('closed', onClosed)
      finishStream()
    }

    reply.raw.once('close', cleanup)
    request.raw.once('aborted', cleanup)
    request.raw.once('error', cleanup)

    // 첫 ping
    send('doc.hello', { ts: Date.now() })

    await closed
  })
}
