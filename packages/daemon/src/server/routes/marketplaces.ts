import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import '../fastify-types.js'
import { zodRequestValidation } from './utils.js'
import type { SkillSourceUrlNotAllowedError } from '../../skills/errors.js'
import { MARKETPLACE_NAME_PATTERN } from '../../skills/marketplace-catalog.js'

const marketplaceNameSchema = z
  .string()
  .trim()
  .regex(MARKETPLACE_NAME_PATTERN, 'name must be a lowercase slug')

const marketplaceCreateBodySchema = z.object({
  name: marketplaceNameSchema,
  url: z.string().trim().url().refine((value) => /^https:\/\//i.test(value), {
    message: 'url must be https',
  }),
})

const marketplaceParamsSchema = z.object({
  name: marketplaceNameSchema,
})

type MarketplaceCreateBody = z.infer<typeof marketplaceCreateBodySchema>
type MarketplaceParams = z.infer<typeof marketplaceParamsSchema>

function getErrorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message
    ? error.message
    : fallback
}

function isSourceUrlNotAllowed(
  error: unknown,
): error is SkillSourceUrlNotAllowedError {
  return error instanceof Error && error.name === 'SkillSourceUrlNotAllowedError'
}

export async function marketplacesRoutes(app: FastifyInstance) {
  const runtime = app.runtime

  app.get('/marketplaces', async (_req, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const data = await runtime.marketplaceCatalog.list()
    return { data }
  })

  app.post<{ Body: MarketplaceCreateBody }>('/marketplaces', {
    preValidation: zodRequestValidation({
      body: {
        schema: marketplaceCreateBodySchema,
        message: 'Invalid marketplace request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const body = request.body
    const name = body.name
    const url = body.url
    try {
      runtime.skillSourceUrlPolicy?.assertAllowed(url, 'marketplace')
      const entry = await runtime.marketplaceCatalog.add(name, url)
      return { data: entry }
    } catch (error: unknown) {
      if (isSourceUrlNotAllowed(error)) {
        return reply.status(403).send({
          error: {
            code: 'SOURCE_NOT_ALLOWED',
            message: error.message,
            url: error.url,
            reason: error.reason,
          },
        })
      }
      return reply.status(409).send({
        error: {
          code: 'CONFLICT',
          message: getErrorMessage(error, 'marketplace already exists'),
        },
      })
    }
  })

  app.delete<{ Params: MarketplaceParams }>('/marketplaces/:name', {
    preValidation: zodRequestValidation({
      params: {
        schema: marketplaceParamsSchema,
        message: 'Invalid marketplace name',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const params = request.params
    const removed = await runtime.marketplaceCatalog.remove(params.name)
    return { data: { removed } }
  })
}
