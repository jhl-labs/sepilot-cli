import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import '../fastify-types.js'
import {
  EXTENSION_ACCESS_TOKEN_SCOPES,
} from '../runtime/extension-tokens.js'
import { zodRequestValidation } from './utils.js'
import { openApiComponentsFromZod } from '../openapi-zod.js'
import {
  openApiJsonResponseRef,
  openApiParameterRef,
  openApiSchemaRef,
  type OpenApiComponentOverrides,
  type OpenApiOverrideMap,
} from '../openapi.js'

const extensionAccessTokenScopeSchema = z.enum(EXTENSION_ACCESS_TOKEN_SCOPES)

const extensionAccessTokenSummarySchema = z.object({
  id: z.string(),
  label: z.string(),
  scopes: z.array(extensionAccessTokenScopeSchema),
  createdAt: z.string().datetime(),
  expiresAt: z.string().datetime().optional(),
  revokedAt: z.string().datetime().optional(),
  active: z.boolean(),
})

const issueExtensionAccessTokenRequestSchema = z.object({
  label: z.string().trim().min(1).max(120),
  scopes: z.array(extensionAccessTokenScopeSchema).min(1),
  expiresAt: z.string().datetime().optional(),
}).superRefine((value, context) => {
  if (value.expiresAt && Date.parse(value.expiresAt) <= Date.now()) {
    context.addIssue({
      code: z.ZodIssueCode.custom,
      path: ['expiresAt'],
      message: 'expiresAt must be in the future',
    })
  }
})

const issuedExtensionAccessTokenSchema = extensionAccessTokenSummarySchema.extend({
  token: z.string(),
})

const extensionAccessTokenListResponseSchema = z.object({
  data: z.array(extensionAccessTokenSummarySchema),
})

const issuedExtensionAccessTokenResponseSchema = z.object({
  data: issuedExtensionAccessTokenSchema,
})

const revokedExtensionAccessTokenResponseSchema = z.object({
  data: extensionAccessTokenSummarySchema,
})

const extensionAccessTokenIdParamsSchema = z.object({
  id: z.string().min(1),
})

type IssueExtensionAccessTokenBody =
  z.infer<typeof issueExtensionAccessTokenRequestSchema>
type ExtensionAccessTokenIdParams =
  z.infer<typeof extensionAccessTokenIdParamsSchema>

export const authTokenOpenApiComponents: OpenApiComponentOverrides =
  openApiComponentsFromZod({
    schemas: {
      ExtensionAccessTokenScope: extensionAccessTokenScopeSchema,
      ExtensionAccessTokenSummary: extensionAccessTokenSummarySchema,
      IssueExtensionAccessTokenRequest: issueExtensionAccessTokenRequestSchema,
      IssuedExtensionAccessToken: issuedExtensionAccessTokenSchema,
      ExtensionAccessTokenListResponse: extensionAccessTokenListResponseSchema,
      IssuedExtensionAccessTokenResponse: issuedExtensionAccessTokenResponseSchema,
      RevokedExtensionAccessTokenResponse: revokedExtensionAccessTokenResponseSchema,
    },
    parameters: {
      ExtensionAccessTokenIdParam: {
        name: 'id',
        in: 'path',
        required: true,
        schema: extensionAccessTokenIdParamsSchema.shape.id,
      },
    },
  })

export const authTokenOpenApiOverrides: OpenApiOverrideMap = {
  '/api/v1/auth/tokens': {
    get: {
      summary: 'List daemon-issued extension access tokens',
      tags: ['Auth'],
      responses: {
        200: openApiJsonResponseRef('ExtensionAccessTokenListResponse'),
      },
    },
    post: {
      summary: 'Issue a scoped daemon extension access token',
      tags: ['Auth'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('IssueExtensionAccessTokenRequest'),
          },
        },
      },
      responses: {
        200: openApiJsonResponseRef('IssuedExtensionAccessTokenResponse'),
      },
    },
  },
  '/api/v1/auth/tokens/{id}': {
    delete: {
      summary: 'Revoke a daemon-issued extension access token',
      tags: ['Auth'],
      parameters: [openApiParameterRef('ExtensionAccessTokenIdParam')],
      responses: {
        200: openApiJsonResponseRef('RevokedExtensionAccessTokenResponse'),
        404: { description: 'Not found' },
      },
    },
  },
}

export async function authTokenRoutes(app: FastifyInstance) {
  app.get('/auth/tokens', async (_request, reply) => {
    const runtime = app.runtime
    if (!runtime) {
      return reply.status(503).send({
        error: {
          code: 'SERVICE_UNAVAILABLE',
          message: 'Runtime not initialized',
        },
      })
    }

    return {
      data: runtime.extensionTokenStore.list(),
    }
  })

  app.post<{ Body: IssueExtensionAccessTokenBody }>('/auth/tokens', {
    preValidation: zodRequestValidation({
      body: {
        schema: issueExtensionAccessTokenRequestSchema,
        message: 'Invalid extension token request body',
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

    return {
      data: await runtime.extensionTokenStore.issue(request.body),
    }
  })

  app.delete<{ Params: ExtensionAccessTokenIdParams }>(
    '/auth/tokens/:id',
    {
      preValidation: zodRequestValidation({
        params: {
          schema: extensionAccessTokenIdParamsSchema,
          message: 'Invalid extension token params',
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

      const params = request.params
      const revoked = await runtime.extensionTokenStore.revoke(
        params.id,
      )
      if (!revoked) {
        return reply.status(404).send({
          error: {
            code: 'NOT_FOUND',
            message: `Extension token not found: ${params.id}`,
          },
        })
      }

      return { data: revoked }
    },
  )
}
