import type { FastifyInstance, FastifyRequest } from 'fastify'
import { z } from 'zod'
import '../fastify-types.js'
import { createShareToken } from '../../share/token.js'
import { openApiComponentsFromZod } from '../openapi-zod.js'
import {
  openApiJsonResponseRef,
  openApiParameterRef,
  openApiSchemaRef,
  type OpenApiComponentOverrides,
  type OpenApiOverrideMap,
} from '../openapi.js'
import { serializeSessionMarkdown } from './session-export.js'
import { zodRequestValidation } from './utils.js'

const sessionShareModeSchema = z.enum(['knowledge', 'public'])

const sessionShareRequestSchema = z.object({
  mode: sessionShareModeSchema.default('knowledge'),
})

const sessionShareRequestInputSchema = z.preprocess(
  (value) => value ?? {},
  sessionShareRequestSchema,
)

const sessionShareResponseSchema = z.object({
  data: z.object({
    shared: z.boolean(),
    mode: sessionShareModeSchema.optional(),
    path: z.string().optional(),
    shareUrl: z.string().optional(),
    expiresAt: z.string().datetime().optional(),
    sessionId: z.string().optional(),
    reason: z.string().optional(),
  }),
})

const sessionShareParamsSchema = z.object({
  id: z.string().min(1),
})

type SessionShareParams = z.input<typeof sessionShareParamsSchema>
type SessionShareBody = z.output<typeof sessionShareRequestSchema>

export const sessionShareOpenApiComponents: OpenApiComponentOverrides = openApiComponentsFromZod({
  schemas: {
    SessionShareRequest: sessionShareRequestSchema,
    SessionShareResponse: sessionShareResponseSchema,
  },
  parameters: {
    SessionShareIdParam: {
      name: 'id',
      in: 'path',
      required: true,
      schema: sessionShareParamsSchema.shape.id,
    },
  },
})

export const sessionShareOpenApiOverrides: OpenApiOverrideMap = {
  '/api/v1/sessions/{id}/share': {
    post: {
      summary: 'Share session to knowledge store or public viewer',
      tags: ['Sessions'],
      parameters: [openApiParameterRef('SessionShareIdParam')],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('SessionShareRequest'),
          },
        },
      },
      responses: { 200: openApiJsonResponseRef('SessionShareResponse'), 404: { description: 'Not found' } },
    },
  },
}

function firstHeader(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value
}

function configuredPublicOrigin(baseUrl: string | undefined): string | undefined {
  const value = baseUrl?.trim()
  if (!value) {
    return undefined
  }
  try {
    const parsed = new URL(value)
    return parsed.origin
  } catch {
    return undefined
  }
}

function requestOrigin(request: FastifyRequest): string | undefined {
  const host = firstHeader(request.headers.host)?.trim()
  if (!host) {
    return undefined
  }

  const protocol = request.protocol === 'https' ? 'https' : 'http'
  try {
    const parsed = new URL(`${protocol}://${host}`)
    if (
      !parsed.hostname
      || parsed.username
      || parsed.password
      || parsed.pathname !== '/'
      || parsed.search
      || parsed.hash
    ) {
      return undefined
    }
    return parsed.origin
  } catch {
    return undefined
  }
}

function buildPublicShareUrl(
  request: FastifyRequest,
  token: string,
  baseUrl?: string,
): string {
  const pathname = `/share/${encodeURIComponent(token)}`
  const origin = configuredPublicOrigin(baseUrl) ?? requestOrigin(request)
  if (!origin) {
    return pathname
  }
  return new URL(pathname, origin).toString()
}

function getPublicShareConfig(app: FastifyInstance): {
  enabled: boolean
  secret: string
  ttlSeconds: number
  baseUrl?: string
} {
  const config = app.runtime?.config.share?.public
  return {
    enabled: config?.enabled === true,
    secret: config?.secret ?? '',
    ttlSeconds: Math.max(1, config?.ttlSeconds ?? 600),
    baseUrl: config?.baseUrl,
  }
}

export async function sessionShareRoutes(app: FastifyInstance) {
  app.post<{ Params: SessionShareParams; Body: SessionShareBody }>('/sessions/:id/share', {
    preValidation: zodRequestValidation({
      params: {
        schema: sessionShareParamsSchema,
        message: 'Invalid session id',
      },
      body: {
        schema: sessionShareRequestInputSchema,
        message: 'Invalid session share request body',
      },
    }),
  }, async (request, reply) => {
    const runtime = app.runtime
    if (!runtime) {
      return reply.status(503).send({
        error: {
          code: 'SERVICE_UNAVAILABLE',
          message: 'Runtime not initialized',
        },
      })
    }

    const params = request.params
    const session = await runtime.sessions.get(params.id)
    if (!session) {
      return reply.status(404).send({
        error: {
          code: 'NOT_FOUND',
          message: 'Session not found',
        },
      })
    }

    const body = request.body
    const mode = body.mode
    if (mode === 'public') {
      const publicShare = getPublicShareConfig(app)
      if (!publicShare.enabled || !publicShare.secret) {
        return {
          data: {
            shared: false,
            mode: 'public',
            sessionId: params.id,
            reason: 'Public share is not configured',
          },
        }
      }

      const token = createShareToken(
        {
          sessionId: params.id,
          ttlSeconds: publicShare.ttlSeconds,
        },
        publicShare.secret,
      )
      return {
        data: {
          shared: true,
          mode: 'public',
          sessionId: params.id,
          shareUrl: buildPublicShareUrl(request, token, publicShare.baseUrl),
          expiresAt: new Date(Date.now() + publicShare.ttlSeconds * 1000).toISOString(),
        },
      }
    }

    const events = await runtime.sessions.getEvents(params.id)
    const markdown = serializeSessionMarkdown(session, events, null)

    try {
      const path = `sessions/${session.device}/${params.id}.md`
      await runtime.gatewayClient.syncKnowledge('push').catch(() => {})

      const result = await runtime.gatewayClient.updateDocument(path, markdown)
      if (result) {
        return {
          data: {
            shared: true,
            mode: 'knowledge',
            path,
            sessionId: params.id,
          },
        }
      }
      return {
        data: {
          shared: false,
          mode: 'knowledge',
          sessionId: params.id,
          reason: 'Gateway upload failed',
        },
      }
    } catch {
      return {
        data: {
          shared: false,
          mode: 'knowledge',
          sessionId: params.id,
          reason: 'Gateway not available',
        },
      }
    }
  })
}
