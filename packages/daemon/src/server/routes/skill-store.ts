import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import '../fastify-types.js'
import { openApiComponentsFromZod } from '../openapi-zod.js'
import {
  openApiJsonResponseRef,
  openApiParameterRef,
  openApiSchemaRef,
  type OpenApiComponentOverrides,
  type OpenApiOverrideMap,
} from '../openapi.js'
import { skillMetadataSchema } from './skill-schema.js'
import { fastifySchemaFromZod, zodRequestValidation } from './utils.js'

const skillStoreSearchQuerySchema = z.object({
  q: z.string().optional(),
  limit: z.coerce.number().int().min(1).optional(),
})

const skillStoreListQuerySchema = z.object({
  page: z.coerce.number().int().min(1).optional(),
})

const skillStoreIdParamsSchema = z.object({
  id: z.string().min(1),
})

const publishSkillBodySchema = z.object({
  metadata: skillMetadataSchema,
  content: z.string().min(1),
})

const rateSkillBodySchema = z.object({
  score: z.number().min(1).max(5),
})

const storedSkillSchema = z.object({
  metadata: skillMetadataSchema,
  content: z.string(),
  downloads: z.number().int(),
  rating: z.number(),
  publishedBy: z.string(),
  publishedAt: z.string(),
})

const skillStoreMutationResponseSchema = z.object({
  success: z.boolean(),
})

type SkillStoreSearchQuery = z.infer<typeof skillStoreSearchQuerySchema>
type SkillStoreListQuery = z.infer<typeof skillStoreListQuerySchema>
type SkillStoreIdParams = z.infer<typeof skillStoreIdParamsSchema>
type PublishSkillBody = z.infer<typeof publishSkillBodySchema>
type RateSkillBody = z.infer<typeof rateSkillBodySchema>

export const skillStoreOpenApiComponents: OpenApiComponentOverrides = openApiComponentsFromZod({
  schemas: {
    SkillStoreMetadata: skillMetadataSchema,
    StoredSkill: storedSkillSchema,
    SkillStoreSearchResponse: z.object({ data: z.array(skillMetadataSchema) }),
    SkillStoreListResponse: z.object({
      skills: z.array(skillMetadataSchema),
      total: z.number().int(),
    }),
    SkillStoreResponse: z.object({ data: storedSkillSchema }),
    PublishSkillRequest: publishSkillBodySchema,
    PublishSkillResponse: z.object({
      data: z.object({
        id: z.string(),
      }),
    }),
    RateSkillRequest: rateSkillBodySchema,
    SkillStoreMutationResponse: skillStoreMutationResponseSchema,
  },
  parameters: {
    SkillStoreSearchQueryParam: {
      name: 'q',
      in: 'query',
      schema: z.string(),
    },
    SkillStoreSearchLimitParam: {
      name: 'limit',
      in: 'query',
      schema: z.number().int().min(1),
    },
    SkillStorePageParam: {
      name: 'page',
      in: 'query',
      schema: z.number().int().min(1),
    },
    SkillStoreIdParam: {
      name: 'id',
      in: 'path',
      required: true,
      schema: skillStoreIdParamsSchema.shape.id,
    },
  },
})

export const skillStoreOpenApiOverrides: OpenApiOverrideMap = {
  '/api/v1/skill-store/search': {
    get: {
      summary: 'Search published skills',
      tags: ['Skill Store'],
      parameters: [
        openApiParameterRef('SkillStoreSearchQueryParam'),
        openApiParameterRef('SkillStoreSearchLimitParam'),
      ],
      responses: { 200: openApiJsonResponseRef('SkillStoreSearchResponse') },
    },
  },
  '/api/v1/skill-store': {
    get: {
      summary: 'List published skills',
      tags: ['Skill Store'],
      parameters: [openApiParameterRef('SkillStorePageParam')],
      responses: { 200: openApiJsonResponseRef('SkillStoreListResponse') },
    },
    post: {
      summary: 'Publish skill',
      tags: ['Skill Store'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('PublishSkillRequest'),
          },
        },
      },
      responses: { 200: openApiJsonResponseRef('PublishSkillResponse') },
    },
  },
  '/api/v1/skill-store/{id}': {
    get: {
      summary: 'Get published skill',
      tags: ['Skill Store'],
      parameters: [openApiParameterRef('SkillStoreIdParam')],
      responses: { 200: openApiJsonResponseRef('SkillStoreResponse'), 404: { description: 'Not found' } },
    },
    delete: {
      summary: 'Delete published skill',
      tags: ['Skill Store'],
      parameters: [openApiParameterRef('SkillStoreIdParam')],
      responses: { 200: openApiJsonResponseRef('SkillStoreMutationResponse'), 404: { description: 'Not found' } },
    },
  },
  '/api/v1/skill-store/{id}/rate': {
    post: {
      summary: 'Rate published skill',
      tags: ['Skill Store'],
      parameters: [openApiParameterRef('SkillStoreIdParam')],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('RateSkillRequest'),
          },
        },
      },
      responses: { 200: openApiJsonResponseRef('SkillStoreMutationResponse'), 404: { description: 'Not found' } },
    },
  },
}

/**
 * Skill Marketplace API routes.
 * Matches the SkillStoreClient API contract.
 */
export async function skillStoreRoutes(app: FastifyInstance) {
  const runtime = app.runtime

  // Search skills
  app.get<{ Querystring: SkillStoreSearchQuery }>('/skill-store/search', {
    schema: fastifySchemaFromZod({ querystring: skillStoreSearchQuerySchema }),
  }, async (request, reply) => {
    if (!runtime?.skillStore) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Skill store not available' } })
    const queryInput = request.query
    const query = queryInput.q ?? ''
    const limit = queryInput.limit ?? 20
    const results = runtime.skillStore.search(query, limit)
    return { data: results }
  })

  // List skills (paginated)
  app.get<{ Querystring: SkillStoreListQuery }>('/skill-store', {
    schema: fastifySchemaFromZod({ querystring: skillStoreListQuerySchema }),
  }, async (request, reply) => {
    if (!runtime?.skillStore) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Skill store not available' } })
    const queryInput = request.query
    const page = queryInput.page ?? 1
    const result = runtime.skillStore.list(page)
    return result
  })

  // Get single skill
  app.get<{ Params: SkillStoreIdParams }>('/skill-store/:id', {
    schema: fastifySchemaFromZod({ params: skillStoreIdParamsSchema }),
  }, async (request, reply) => {
    if (!runtime?.skillStore) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Skill store not available' } })
    const params = request.params
    const skill = runtime.skillStore.get(params.id)
    if (!skill) return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'Skill not found' } })
    return { data: skill }
  })

  // Publish skill
  app.post<{ Body: PublishSkillBody }>('/skill-store', {
    preValidation: zodRequestValidation({
      body: {
        schema: publishSkillBodySchema,
        message: 'Invalid skill publish request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime?.skillStore) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Skill store not available' } })
    const body = request.body
    const { metadata, content } = body
    const id = runtime.skillStore.publish(metadata, content)
    return { data: { id } }
  })

  // Rate skill
  app.post<{ Params: SkillStoreIdParams; Body: RateSkillBody }>('/skill-store/:id/rate', {
    schema: fastifySchemaFromZod({
      params: skillStoreIdParamsSchema,
    }),
    preValidation: zodRequestValidation({
      body: {
        schema: rateSkillBodySchema,
        message: 'Invalid skill rating request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime?.skillStore) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Skill store not available' } })
    const params = request.params
    const body = request.body
    const { score } = body
    runtime.skillStore.rate(params.id, score)
    return { success: true }
  })

  // Delete skill
  app.delete<{ Params: SkillStoreIdParams }>('/skill-store/:id', {
    schema: fastifySchemaFromZod({ params: skillStoreIdParamsSchema }),
  }, async (request, reply) => {
    if (!runtime?.skillStore) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Skill store not available' } })
    const params = request.params
    const deleted = runtime.skillStore.delete(params.id)
    if (!deleted) return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'Skill not found' } })
    return { success: true }
  })
}
