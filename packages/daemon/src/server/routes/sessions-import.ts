import { randomUUID } from 'node:crypto'
import type { FastifyInstance } from 'fastify'
import type { Dispatcher } from 'undici'
import { z } from 'zod'
import { buildImportedSessionEvents } from '../../share/import.js'
import { parseImportableSessionEvents } from '../../share/import-schema.js'
import {
  assertPublicUrl,
  createPinnedLookupDispatcher,
  rejectPrivateLiteralUrl,
} from '../../utils/ssrf-guard.js'
import {
  openApiJsonResponseRef,
  openApiSchemaRef,
  type OpenApiComponentOverrides,
  type OpenApiOverrideMap,
} from '../openapi.js'
import { openApiComponentsFromZod } from '../openapi-zod.js'
import { zodRequestValidation } from './utils.js'

const SHARE_IMPORT_TIMEOUT_MS = 10_000
const SHARE_IMPORT_MAX_REDIRECTS = 5
const SHARE_IMPORT_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
const SHARE_IMPORT_MAX_EVENTS = 2_000

type FetchInitWithDispatcher = RequestInit & { dispatcher: Dispatcher }

const sessionImportRequestSchema = z.object({
  shareUrl: z.string().trim().url(),
  title: z.string().trim().min(1).max(200).optional(),
})

const sessionImportRequestInputSchema = z.preprocess(
  (value) => value ?? {},
  sessionImportRequestSchema,
)

const sessionImportResponseSchema = z.object({
  data: z.object({
    sessionId: z.string(),
    importedEvents: z.number().int().nonnegative(),
    title: z.string(),
    sourceTitle: z.string(),
  }),
})

const sharedSessionEnvelopeSchema = z.object({
  data: z.object({
    session: z.object({
      title: z.string().trim().min(1).max(200),
      provider: z.string().min(1).max(200),
      model: z.string().min(1).max(200),
      device: z.string().min(1).max(200),
    }),
    events: z.array(z.unknown()).max(SHARE_IMPORT_MAX_EVENTS),
  }),
})

type SessionImportBody = z.infer<typeof sessionImportRequestSchema>

export const sessionImportOpenApiComponents: OpenApiComponentOverrides = openApiComponentsFromZod({
  schemas: {
    SessionImportRequest: sessionImportRequestSchema,
    SessionImportResponse: sessionImportResponseSchema,
  },
})

export const sessionImportOpenApiOverrides: OpenApiOverrideMap = {
  '/api/v1/sessions/import': {
    post: {
      summary: 'Import shared session into a local session',
      tags: ['Sessions'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('SessionImportRequest'),
          },
        },
      },
      responses: {
        200: openApiJsonResponseRef('SessionImportResponse'),
        400: { description: 'Invalid request' },
        502: { description: 'Share fetch failed' },
      },
    },
  },
}

function buildShareDataUrl(shareUrl: string): string {
  const url = new URL(shareUrl)
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new Error('Share URL must use http or https')
  }
  if (url.username || url.password) {
    throw new Error('Share URL credentials are not allowed')
  }
  try {
    rejectPrivateLiteralUrl(url)
  } catch {
    throw new Error('Share URL host is not allowed')
  }
  url.searchParams.set('format', 'json')
  return url.toString()
}

type SharedSessionFetchResult =
  | { ok: true; payload: unknown }
  | { ok: false; kind: 'http'; status: number }
  | { ok: false; kind: 'invalid-payload' }

function isJsonContentType(value: string | null): boolean {
  const mime = value?.split(';', 1)[0]?.trim().toLowerCase()
  return mime === 'application/json' || Boolean(mime?.endsWith('+json'))
}

async function readBoundedJson(response: Response): Promise<unknown> {
  if (!isJsonContentType(response.headers.get('content-type'))) {
    await discardResponseBody(response)
    throw new Error('shared session response is not JSON')
  }

  const contentLength = response.headers.get('content-length')
  if (contentLength != null) {
    if (!/^\d+$/u.test(contentLength)) {
      await discardResponseBody(response)
      throw new Error('shared session content length is invalid')
    }
    const declaredBytes = Number(contentLength)
    if (!Number.isSafeInteger(declaredBytes) || declaredBytes > SHARE_IMPORT_MAX_RESPONSE_BYTES) {
      await discardResponseBody(response)
      throw new Error('shared session response is too large')
    }
  }

  const reader = response.body?.getReader()
  if (!reader) throw new Error('shared session response is empty')
  const chunks: Uint8Array[] = []
  let totalBytes = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    if (!value) continue
    totalBytes += value.byteLength
    if (totalBytes > SHARE_IMPORT_MAX_RESPONSE_BYTES) {
      await reader.cancel().catch(() => undefined)
      throw new Error('shared session response is too large')
    }
    chunks.push(value)
  }

  const bytes = new Uint8Array(totalBytes)
  let offset = 0
  for (const chunk of chunks) {
    bytes.set(chunk, offset)
    offset += chunk.byteLength
  }
  const text = new TextDecoder('utf-8', { fatal: true }).decode(bytes)
  return JSON.parse(text) as unknown
}

async function fetchSharedSessionPayload(initialUrl: string): Promise<SharedSessionFetchResult> {
  const signal = AbortSignal.timeout(SHARE_IMPORT_TIMEOUT_MS)
  let currentUrl = initialUrl

  for (let hop = 0; ; hop += 1) {
    const resolution = await assertPublicUrl(currentUrl, signal)
    const dispatcher = createPinnedLookupDispatcher(resolution)
    try {
      const response = await fetch(currentUrl, {
        headers: {
          accept: 'application/json',
        },
        redirect: 'manual',
        signal,
        dispatcher,
      } as FetchInitWithDispatcher)

      if (response.status >= 300 && response.status < 400) {
        const location = response.headers.get('location')
        if (!location || hop >= SHARE_IMPORT_MAX_REDIRECTS) {
          await discardResponseBody(response)
          return { ok: false, kind: 'http', status: response.status }
        }
        let nextUrl: URL
        try {
          nextUrl = new URL(location, resolution.url)
        } catch {
          await discardResponseBody(response)
          throw new Error('Invalid shared session redirect URL')
        }
        if (resolution.url.protocol === 'https:' && nextUrl.protocol !== 'https:') {
          await discardResponseBody(response)
          throw new Error('Shared session redirect cannot downgrade HTTPS')
        }
        currentUrl = nextUrl.toString()
        await discardResponseBody(response)
        continue
      }

      if (!response.ok) {
        await discardResponseBody(response)
        return { ok: false, kind: 'http', status: response.status }
      }

      try {
        return { ok: true, payload: await readBoundedJson(response) }
      } catch {
        return { ok: false, kind: 'invalid-payload' }
      }
    } finally {
      await dispatcher.close().catch(() => undefined)
    }
  }
}

async function discardResponseBody(response: Response): Promise<void> {
  await response.body?.cancel().catch(() => undefined)
}

export async function sessionImportRoutes(app: FastifyInstance) {
  app.post<{ Body: SessionImportBody }>(
    '/sessions/import',
    {
      preValidation: zodRequestValidation({
        body: {
          schema: sessionImportRequestInputSchema,
          message: 'Invalid session import request body',
        },
      }),
    },
    async (request, reply) => {
      const runtime = app.runtime
      if (!runtime) {
        return reply.status(503).send({
          error: {
            code: 'SERVICE_UNAVAILABLE',
            message: 'Runtime not initialized',
          },
        })
      }

      const body = request.body
      const sessionCreateTimestamp = new Date().toISOString()
      let shareDataUrl: string
      try {
        shareDataUrl = buildShareDataUrl(body.shareUrl)
      } catch (error) {
        return reply.status(400).send({
          error: {
            code: 'INVALID_REQUEST',
            message: error instanceof Error ? error.message : 'Invalid share URL',
          },
        })
      }

      let fetchResult: SharedSessionFetchResult
      try {
        fetchResult = await fetchSharedSessionPayload(shareDataUrl)
      } catch {
        return reply.status(502).send({
          error: {
            code: 'BAD_GATEWAY',
            message: 'Failed to fetch shared session',
          },
        })
      }

      if (!fetchResult.ok) {
        const message = fetchResult.kind === 'http'
          ? `Shared session fetch failed with ${fetchResult.status}`
          : 'Shared session payload is invalid'
        return reply.status(502).send({
          error: {
            code: 'BAD_GATEWAY',
            message,
          },
        })
      }

      const parsedPayload = sharedSessionEnvelopeSchema.safeParse(fetchResult.payload)
      if (!parsedPayload.success) {
        return reply.status(502).send({
          error: {
            code: 'BAD_GATEWAY',
            message: 'Shared session payload is invalid',
          },
        })
      }

      const sharedSession = parsedPayload.data.data
      const parsedEvents = parseImportableSessionEvents(sharedSession.events)
      if (!parsedEvents.success) {
        return reply.status(502).send({
          error: {
            code: 'BAD_GATEWAY',
            message: 'Shared session payload is invalid',
          },
        })
      }
      const importedEvents = buildImportedSessionEvents(parsedEvents.events)
      if (importedEvents.length === 0) {
        return reply.status(400).send({
          error: {
            code: 'INVALID_REQUEST',
            message: 'Shared session does not contain transferable events',
          },
        })
      }

      const session = await runtime.sessions.create({
        id: randomUUID(),
        title: body.title ?? sharedSession.session.title,
        createdAt: sessionCreateTimestamp,
        updatedAt: sessionCreateTimestamp,
        provider: sharedSession.session.provider,
        model: sharedSession.session.model,
        device: runtime.config.device?.name ?? sharedSession.session.device,
        status: 'active',
        tags: ['imported:share'],
      })

      for (const event of importedEvents) {
        await runtime.sessions.appendEvent(session.id, event)
      }

      return {
        data: {
          sessionId: session.id,
          importedEvents: importedEvents.length,
          title: session.title,
          sourceTitle: sharedSession.session.title,
        },
      }
    },
  )
}
