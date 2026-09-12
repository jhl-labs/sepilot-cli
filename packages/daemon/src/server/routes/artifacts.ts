import { join } from 'node:path'
import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import '../fastify-types.js'
import { ArtifactStore } from '../../memory/artifact-store.js'
import { openApiComponentsFromZod } from '../openapi-zod.js'
import {
  openApiJsonResponseRef,
  openApiParameterRef,
  type OpenApiComponentOverrides,
  type OpenApiOverrideMap,
} from '../openapi.js'
import { fastifySchemaFromZod, getRuntimeDataDir } from './utils.js'

const artifactSchema = z.object({
  id: z.string(),
  key: z.string().optional(),
  version: z.number().int().positive().optional(),
  type: z.enum(['code', 'html', 'document', 'mermaid', 'svg', 'image']),
  title: z.string().optional(),
  language: z.string().optional(),
  content: z.string(),
})

const recentArtifactSchema = z.object({
  sessionId: z.string(),
  artifact: artifactSchema,
  modifiedAt: z.number(),
})

const sessionArtifactParamsSchema = z.object({
  sessionId: z.string().min(1),
})

const artifactListQuerySchema = z.object({
  limit: z.coerce.number().int().min(1).max(50).optional(),
})

type SessionArtifactParams = z.input<typeof sessionArtifactParamsSchema>
type ArtifactListQuery = z.input<typeof artifactListQuerySchema>

export const artifactOpenApiComponents: OpenApiComponentOverrides = openApiComponentsFromZod({
  schemas: {
    Artifact: artifactSchema,
    ArtifactListResponse: z.object({ data: z.array(artifactSchema) }),
    RecentArtifact: recentArtifactSchema,
    RecentArtifactListResponse: z.object({ data: z.array(recentArtifactSchema) }),
  },
  parameters: {
    SessionArtifactSessionIdParam: {
      name: 'sessionId',
      in: 'path',
      required: true,
      schema: sessionArtifactParamsSchema.shape.sessionId,
    },
  },
})

export const artifactOpenApiOverrides: OpenApiOverrideMap = {
  '/api/v1/artifacts': {
    get: {
      summary: 'List recent artifacts across sessions',
      tags: ['Artifacts'],
      responses: { 200: openApiJsonResponseRef('RecentArtifactListResponse') },
    },
  },
  '/api/v1/sessions/{sessionId}/artifacts': {
    get: {
      summary: 'List session artifacts',
      tags: ['Sessions'],
      parameters: [openApiParameterRef('SessionArtifactSessionIdParam')],
      responses: { 200: openApiJsonResponseRef('ArtifactListResponse') },
    },
  },
}

export async function artifactRoutes(app: FastifyInstance) {
  app.get<{ Querystring: ArtifactListQuery }>(
    '/artifacts',
    {
      schema: fastifySchemaFromZod({ querystring: artifactListQuerySchema }),
    },
    async (request) => {
      const runtime = app.runtime
      if (!runtime) return { data: [] }
      const store = new ArtifactStore(
        join(getRuntimeDataDir(runtime), 'artifacts'),
      )
      const query = request.query
      return { data: await store.listRecent(query.limit ?? 20) }
    },
  )

  app.get<{ Params: SessionArtifactParams }>(
    '/sessions/:sessionId/artifacts',
    {
      schema: fastifySchemaFromZod({ params: sessionArtifactParamsSchema }),
    },
    async (request) => {
      const runtime = app.runtime
      if (!runtime) return { data: [] }
      const params = request.params

      const store = new ArtifactStore(
        join(getRuntimeDataDir(runtime), 'artifacts'),
      )
      const artifacts = await store.listBySession(params.sessionId)
      return { data: artifacts }
    },
  )
}
