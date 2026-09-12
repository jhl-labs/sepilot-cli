import type { FastifyInstance } from 'fastify'
import type { AuditEvent } from '@sepilotd/core'
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
import {
  skillMetadataSchema,
  skillsSearchQuerySchema,
  skillValidationRequestSchema,
} from './skill-schema.js'
import { MarketplaceSource } from '../../skills/sources/marketplace.js'
import type { SkillSourceUrlNotAllowedError } from '../../skills/errors.js'
import { parseSkillSource } from '../../skills/source-parser.js'
import { fastifySchemaFromZod, zodRequestValidation } from './utils.js'
import {
  InvalidCwdError,
  invalidCwdResponse,
  resolveRequestWorkspace,
} from './request-cwd.js'

const skillIdParamsSchema = z.object({
  id: z.string().min(1),
})

const skillsListQuerySchema = z.object({
  includeDisabled: z.enum(['true', 'false']).optional(),
  includeBuiltins: z.enum(['true', 'false']).optional(),
  cwd: z.string().trim().min(1).optional(),
  workspaceRoot: z.string().trim().min(1).optional(),
})

const skillGetQuerySchema = z.object({
  cwd: z.string().trim().min(1).optional(),
  workspaceRoot: z.string().trim().min(1).optional(),
})

const skillContentSchema = z.object({
  metadata: skillMetadataSchema,
  content: z.string(),
})

const skillValidationResultSchema = z.object({
  valid: z.boolean(),
  errors: z.array(z.string()),
  warnings: z.array(z.string()),
})

const skillInstallRequestSchema = z.object({
  source: z.string().trim().min(1),
  force: z.boolean().optional(),
  expectedDigest: z.string().trim().min(1).optional(),
})

const skillInstallPreviewRequestSchema = z.object({
  source: z.string().trim().min(1),
})

const marketplaceSkillSearchQuerySchema = z.object({
  query: z.string().trim().min(1),
  marketplace: z.string().trim().min(1).optional(),
  limit: z.string().regex(/^\d+$/).optional(),
})

const skillCreateRequestSchema = z.object({
  metadata: skillMetadataSchema,
  content: z.string().min(1),
  force: z.boolean().optional(),
})

const skillUpdateRequestSchema = z.object({
  expectedDigest: z.string().trim().min(1).optional(),
}).default({})

type SkillsSearchQuery = z.infer<typeof skillsSearchQuerySchema>
type SkillGetQuery = z.infer<typeof skillGetQuerySchema>
type MarketplaceSkillSearchQuery = z.infer<typeof marketplaceSkillSearchQuerySchema>
type SkillIdParams = z.infer<typeof skillIdParamsSchema>
type ValidateSkillBody = z.infer<typeof skillValidationRequestSchema>
type SkillInstallBody = z.infer<typeof skillInstallRequestSchema>
type SkillInstallPreviewBody = z.infer<typeof skillInstallPreviewRequestSchema>
type SkillCreateBody = z.infer<typeof skillCreateRequestSchema>
type SkillUpdateBody = z.infer<typeof skillUpdateRequestSchema>

type SkillValidationFailure = Error & {
  result: unknown
}

function isSkillValidationError(error: unknown): error is SkillValidationFailure {
  return error instanceof Error
    && error.name === 'SkillValidationError'
    && 'result' in error
}

function isSkillAlreadyExistsError(error: unknown): error is Error & { skillId?: string } {
  return error instanceof Error && error.name === 'SkillAlreadyExistsError'
}

function isSkillDigestMismatchError(
  error: unknown,
): error is Error & { expectedDigest?: string; actualDigest?: string } {
  return error instanceof Error && error.name === 'SkillDigestMismatchError'
}

function isSkillDigestRequiredError(
  error: unknown,
): error is Error & { actualDigest?: string } {
  return error instanceof Error && error.name === 'SkillDigestRequiredError'
}

function isSourceUrlNotAllowed(
  error: unknown,
): error is SkillSourceUrlNotAllowedError {
  return error instanceof Error && error.name === 'SkillSourceUrlNotAllowedError'
}

function sourceUrlNotAllowedError(error: SkillSourceUrlNotAllowedError) {
  return {
    code: 'SOURCE_NOT_ALLOWED',
    message: error.message,
    url: error.url,
    reason: error.reason,
  }
}

function getRouteErrorMessage(_error: unknown, fallback: string): string {
  // Never surface a raw fetch/exec error to clients: git and network stderr can
  // embed internal temp paths and source URLs. Callers log the raw error
  // server-side; the client only gets a stable, generic message. Domain errors
  // with safe, user-actionable messages (validation, digest, path traversal,
  // source-url) are handled explicitly before this fallback is reached.
  return fallback
}

function isBuiltinSkill(skill: { author?: string; tags?: string[]; provenance?: { verification?: string } }): boolean {
  return skill.provenance?.verification === 'builtin'
    || (skill.author === 'sepilotd' && skill.tags?.includes('builtin') === true)
}

function filterBuiltins<T extends { author?: string; tags?: string[]; provenance?: { verification?: string } }>(
  skills: T[],
  includeBuiltins?: string,
): T[] {
  return includeBuiltins === 'false' ? skills.filter((skill) => !isBuiltinSkill(skill)) : skills
}

async function resolveSkillQueryWorkspace(
  query: { cwd?: string; workspaceRoot?: string },
): Promise<{ cwd?: string; workspaceRoot?: string }> {
  if (!query.cwd && !query.workspaceRoot) return {}
  return resolveRequestWorkspace(
    query.cwd ?? query.workspaceRoot,
    query.workspaceRoot,
  )
}

export const skillsOpenApiComponents: OpenApiComponentOverrides = openApiComponentsFromZod({
  schemas: {
    SkillMetadata: skillMetadataSchema,
    SkillContent: skillContentSchema,
    SkillListResponse: z.object({ data: z.array(skillMetadataSchema) }),
    SkillResponse: z.object({ data: skillContentSchema }),
    ValidateSkillRequest: skillValidationRequestSchema,
    SkillValidationResult: skillValidationResultSchema,
    SkillValidationResponse: z.object({ data: skillValidationResultSchema }),
  },
  parameters: {
    SkillsSearchQueryParam: {
      name: 'query',
      in: 'query',
      required: true,
      schema: skillsSearchQuerySchema.shape.query,
    },
    SkillIdParam: {
      name: 'id',
      in: 'path',
      required: true,
      schema: skillIdParamsSchema.shape.id,
    },
  },
})

export const skillsOpenApiOverrides: OpenApiOverrideMap = {
  '/api/v1/skills': {
    get: {
      summary: 'List skills',
      tags: ['Skills'],
      responses: { 200: openApiJsonResponseRef('SkillListResponse') },
    },
  },
  '/api/v1/skills/search': {
    get: {
      summary: 'Search skills',
      tags: ['Skills'],
      parameters: [openApiParameterRef('SkillsSearchQueryParam')],
      responses: { 200: openApiJsonResponseRef('SkillListResponse') },
    },
  },
  '/api/v1/skills/{id}': {
    get: {
      summary: 'Get skill',
      tags: ['Skills'],
      parameters: [openApiParameterRef('SkillIdParam')],
      responses: { 200: openApiJsonResponseRef('SkillResponse'), 404: { description: 'Not found' } },
    },
  },
  '/api/v1/skills/validate': {
    post: {
      summary: 'Validate skill',
      tags: ['Skills'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('ValidateSkillRequest'),
          },
        },
      },
      responses: { 200: openApiJsonResponseRef('SkillValidationResponse') },
    },
  },
}

export async function skillsRoutes(app: FastifyInstance) {
  const runtime = app.runtime

  app.get<{ Querystring: z.infer<typeof skillsListQuerySchema> }>('/skills', {
    schema: fastifySchemaFromZod({ querystring: skillsListQuerySchema }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const skillsQuery = request.query
    let context: { cwd?: string; workspaceRoot?: string }
    try {
      context = await resolveSkillQueryWorkspace(skillsQuery)
    } catch (error) {
      if (error instanceof InvalidCwdError) {
        return reply.status(400).send(invalidCwdResponse(error))
      }
      throw error
    }
    const skills = skillsQuery.includeDisabled === 'true'
      ? await runtime.skillRegistry.listAllForCwd(context.cwd, context.workspaceRoot)
      : await runtime.skillRegistry.listForCwd(context.cwd, context.workspaceRoot)
    return { data: filterBuiltins(skills, skillsQuery.includeBuiltins) }
  })

  app.get<{ Querystring: SkillsSearchQuery }>('/skills/search', {
    schema: fastifySchemaFromZod({ querystring: skillsSearchQuerySchema }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const query = request.query
    let context: { cwd?: string; workspaceRoot?: string }
    try {
      context = await resolveSkillQueryWorkspace(query)
    } catch (error) {
      if (error instanceof InvalidCwdError) {
        return reply.status(400).send(invalidCwdResponse(error))
      }
      throw error
    }
    const results = await runtime.skillRegistry.searchForCwd(
      query.query,
      context.cwd,
      context.workspaceRoot,
    )
    return { data: filterBuiltins(results, query.includeBuiltins) }
  })

  app.get<{ Querystring: MarketplaceSkillSearchQuery }>('/skills/marketplace/search', {
    schema: fastifySchemaFromZod({ querystring: marketplaceSkillSearchQuerySchema }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const query = request.query
    try {
      const source = new MarketplaceSource({
        catalog: runtime.marketplaceCatalog,
        urlPolicy: runtime.skillSourceUrlPolicy,
      })
      const results = await source.search(query.query, {
        marketplace: query.marketplace,
        limit: query.limit ? Number.parseInt(query.limit, 10) : undefined,
      })
      const data = await Promise.all(results.map(async (result) => ({
        ...result,
        installed: await runtime.skillRegistry.get(result.metadata.id) !== null,
      })))
      return { data }
    } catch (error: unknown) {
      if (isSourceUrlNotAllowed(error)) {
        return reply.status(403).send({ error: sourceUrlNotAllowedError(error) })
      }
      request.log.warn({ err: error }, 'skills marketplace search failed')
      return reply.status(502).send({
        error: {
          code: 'FETCH_FAILED',
          message: getRouteErrorMessage(error, 'marketplace search failed'),
        },
      })
    }
  })

  app.get<{ Params: SkillIdParams; Querystring: SkillGetQuery }>('/skills/:id', {
    schema: fastifySchemaFromZod({ params: skillIdParamsSchema, querystring: skillGetQuerySchema }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const params = request.params
    let context: { cwd?: string; workspaceRoot?: string }
    try {
      context = await resolveSkillQueryWorkspace(request.query)
    } catch (error) {
      if (error instanceof InvalidCwdError) {
        return reply.status(400).send(invalidCwdResponse(error))
      }
      throw error
    }
    const skill = await runtime.skillRegistry.getForCwd(
      params.id,
      context.cwd,
      context.workspaceRoot,
    )
    if (!skill) return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'Skill not found' } })
    return { data: skill }
  })

  app.post<{ Body: ValidateSkillBody }>('/skills/validate', {
    preValidation: zodRequestValidation({
      body: {
        schema: skillValidationRequestSchema,
        message: 'Invalid skill validation request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const body = request.body
    const { metadata, content } = body

    const { validateSkill } = await import('../../skills/validator.js')
    const result = validateSkill(metadata, content, runtime.toolRegistry, runtime.policyEngine, runtime.autonomy)
    return { data: result }
  })

  app.post<{ Body: SkillInstallPreviewBody }>('/skills/install/preview', {
    preValidation: zodRequestValidation({
      body: {
        schema: skillInstallPreviewRequestSchema,
        message: 'Invalid skill install preview request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const body = request.body
    try {
      const preview = await runtime.installPipeline.preview({ source: body.source })
      return {
        data: {
          digest: preview.digest,
          candidates: preview.fetched.map((item) => ({
            metadata: { ...item.metadata, source: item.source },
            source: item.source,
          })),
        },
      }
    } catch (error: unknown) {
      if (error instanceof Error && error.name === 'SkillPathTraversalError') {
        return reply.status(400).send({ error: { code: 'PATH_TRAVERSAL', message: error.message } })
      }
      if (isSourceUrlNotAllowed(error)) {
        return reply.status(403).send({ error: sourceUrlNotAllowedError(error) })
      }
      request.log.warn({ err: error }, 'skill fetch failed')
      return reply.status(502).send({
        error: {
          code: 'FETCH_FAILED',
          message: getRouteErrorMessage(error, 'fetch failed'),
        },
      })
    }
  })

  app.post<{ Body: SkillInstallBody }>('/skills/install', {
    preValidation: zodRequestValidation({
      body: {
        schema: skillInstallRequestSchema,
        message: 'Invalid skill install request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const body = request.body
    const source = body.source
    const force = body.force
    try {
      const result = await runtime.installPipeline.install({
        source,
        force,
        expectedDigest: body.expectedDigest,
        requireExpectedDigest: true,
      })
      if (body.force === true && runtime.auditLogger) {
        for (const skill of result.installed) {
          await runtime.auditLogger.log({
            timestamp: new Date().toISOString(),
            event: 'skill.install.force',
            device: runtime.config.device.name,
            skill: skill.id,
            source,
          } satisfies AuditEvent)
        }
      }
      return { data: { installed: result.installed } }
    } catch (error: unknown) {
      if (isSkillValidationError(error)) {
        return reply.status(400).send({ error: { code: 'VALIDATION_FAILED', validation: error.result } })
      }
      if (isSkillAlreadyExistsError(error)) {
        return reply.status(409).send({
          error: {
            code: 'ALREADY_EXISTS',
            message: error.message,
            skillId: error.skillId,
          },
        })
      }
      if (isSkillDigestMismatchError(error)) {
        return reply.status(409).send({
          error: {
            code: 'DIGEST_MISMATCH',
            message: error.message,
            expectedDigest: error.expectedDigest,
            actualDigest: error.actualDigest,
          },
        })
      }
      if (isSkillDigestRequiredError(error)) {
        return reply.status(409).send({
          error: {
            code: 'DIGEST_REQUIRED',
            message: error.message,
            actualDigest: error.actualDigest,
          },
        })
      }
      if (error instanceof Error && error.name === 'SkillPathTraversalError') {
        return reply.status(400).send({ error: { code: 'PATH_TRAVERSAL', message: error.message } })
      }
      if (isSourceUrlNotAllowed(error)) {
        return reply.status(403).send({ error: sourceUrlNotAllowedError(error) })
      }
      request.log.warn({ err: error }, 'skill fetch failed')
      return reply.status(502).send({
        error: {
          code: 'FETCH_FAILED',
          message: getRouteErrorMessage(error, 'fetch failed'),
        },
      })
    }
  })

  app.post<{ Body: SkillCreateBody }>('/skills/create', {
    preValidation: zodRequestValidation({
      body: {
        schema: skillCreateRequestSchema,
        message: 'Invalid skill create request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const body = request.body
    const existing = await runtime.skillRegistry.get(body.metadata.id)
    if (existing && body.force !== true) {
      return reply.status(409).send({
        error: {
          code: 'ALREADY_EXISTS',
          message: `Skill already exists: ${body.metadata.id}`,
        },
      })
    }
    try {
      await runtime.skillRegistry.register(body.metadata, body.content, {
        force: false,
      })
      return { data: { id: body.metadata.id, version: body.metadata.version } }
    } catch (error: unknown) {
      if (isSkillValidationError(error)) {
        return reply.status(400).send({ error: { code: 'VALIDATION_FAILED', validation: error.result } })
      }
      request.log.warn({ err: error }, 'skill create failed')
      return reply.status(500).send({
        error: {
          code: 'INTERNAL_ERROR',
          message: getRouteErrorMessage(error, 'skill create failed'),
        },
      })
    }
  })

  app.delete<{ Params: SkillIdParams }>('/skills/:id', {
    schema: fastifySchemaFromZod({ params: skillIdParamsSchema }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const params = request.params
    const existing = await runtime.skillRegistry.get(params.id)
    if (!existing) return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'Skill not found' } })
    await runtime.skillRegistry.remove(params.id)
    return { data: { removed: true } }
  })

  app.post<{ Params: SkillIdParams }>('/skills/:id/enable', {
    schema: fastifySchemaFromZod({ params: skillIdParamsSchema }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const { id } = request.params
    if (!(await runtime.skillRegistry.get(id))) {
      return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'Skill not found' } })
    }
    await runtime.skillRegistry.setEnabled(id, true)
    return { data: { id, enabled: true } }
  })

  app.post<{ Params: SkillIdParams }>('/skills/:id/disable', {
    schema: fastifySchemaFromZod({ params: skillIdParamsSchema }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const { id } = request.params
    if (!(await runtime.skillRegistry.get(id))) {
      return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'Skill not found' } })
    }
    await runtime.skillRegistry.setEnabled(id, false)
    return { data: { id, enabled: false } }
  })

  app.post<{ Params: SkillIdParams; Body: SkillUpdateBody }>('/skills/:id/update', {
    preValidation: zodRequestValidation({
      params: {
        schema: skillIdParamsSchema,
        message: 'Invalid skill id',
      },
      body: {
        schema: skillUpdateRequestSchema,
        message: 'Invalid skill update request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const params = request.params
    const body = request.body
    const existing = await runtime.skillRegistry.get(params.id)
    if (!existing) return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'Skill not found' } })
    const src = existing.metadata.source
    if (!src) return { data: { changed: false, reason: 'no source recorded' } }

    try {
      const ref =
        src.type === 'url'
          ? parseSkillSource(src.ref)
          : src.type === 'git'
            ? parseSkillSource(src.ref)
            : (() => {
                const [mp, name] = src.ref.split('/')
                if (!mp || !name) {
                  throw new Error('malformed marketplace source ref')
                }
                return { type: 'marketplace' as const, marketplace: mp, name }
              })()
      const preview = await runtime.installPipeline.previewRef({ ref })
      if (!body.expectedDigest) {
        return reply.status(409).send({
          error: {
            code: 'DIGEST_REQUIRED',
            message: 'Skill update requires approving the preview digest',
            actualDigest: preview.digest,
          },
        })
      }
      if (body.expectedDigest !== preview.digest) {
        return reply.status(409).send({
          error: {
            code: 'DIGEST_MISMATCH',
            message: `skill source changed between preview and update: expected ${body.expectedDigest}, got ${preview.digest}`,
            expectedDigest: body.expectedDigest,
            actualDigest: preview.digest,
          },
        })
      }
      const result = await runtime.installPipeline.installRef({
        ref,
        force: false,
        allowOverwrite: true,
        expectedDigest: body.expectedDigest,
        requireExpectedDigest: true,
      })
      const fresh = result.installed[0]
      if (!fresh) return { data: { changed: false, reason: 'no skills returned' } }
      const changed = fresh.version !== existing.metadata.version
      return { data: { changed, from: existing.metadata.version, to: fresh.version } }
    } catch (error: unknown) {
      if (isSkillValidationError(error)) {
        return reply.status(400).send({ error: { code: 'VALIDATION_FAILED', validation: error.result } })
      }
      if (isSkillDigestMismatchError(error)) {
        return reply.status(409).send({
          error: {
            code: 'DIGEST_MISMATCH',
            message: error.message,
            expectedDigest: error.expectedDigest,
            actualDigest: error.actualDigest,
          },
        })
      }
      if (isSkillDigestRequiredError(error)) {
        return reply.status(409).send({
          error: {
            code: 'DIGEST_REQUIRED',
            message: error.message,
            actualDigest: error.actualDigest,
          },
        })
      }
      if (error instanceof Error && error.name === 'SkillPathTraversalError') {
        return reply.status(400).send({ error: { code: 'PATH_TRAVERSAL', message: error.message } })
      }
      if (error instanceof Error && error.message === 'malformed marketplace source ref') {
        return reply.status(400).send({ error: { code: 'BAD_REQUEST', message: error.message } })
      }
      if (error instanceof Error && /Only https|Unrecognised source|empty source/i.test(error.message)) {
        return reply.status(400).send({ error: { code: 'BAD_REQUEST', message: error.message } })
      }
      if (isSourceUrlNotAllowed(error)) {
        return reply.status(403).send({ error: sourceUrlNotAllowedError(error) })
      }
      request.log.warn({ err: error }, 'skill fetch failed')
      return reply.status(502).send({
        error: {
          code: 'FETCH_FAILED',
          message: getRouteErrorMessage(error, 'fetch failed'),
        },
      })
    }
  })
}
