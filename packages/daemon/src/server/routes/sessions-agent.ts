import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import '../fastify-types.js'
import { openApiComponentsFromZod } from '../openapi-zod.js'
import {
  openApiJsonResponseRef,
  openApiParameterRef,
  type OpenApiComponentOverrides,
  type OpenApiOverrideMap,
} from '../openapi.js'
import { fastifySchemaFromZod } from './utils.js'

const agentBodySchema = z.object({
  agentId: z.string().min(1),
})

const RESERVED_AGENT_IDS = new Set(['plan', 'build'])

const agentResponseSchema = z.object({
  data: z.object({
    sessionId: z.string(),
    primaryAgentId: z.string(),
  }),
})

const sessionParamsSchema = z.object({
  id: z.string().min(1),
})

type SessionParams = z.input<typeof sessionParamsSchema>
type AgentBody = z.input<typeof agentBodySchema>

export const sessionAgentOpenApiComponents: OpenApiComponentOverrides = openApiComponentsFromZod({
  schemas: {
    SessionAgentBody: agentBodySchema,
    SessionAgentResponse: agentResponseSchema,
  },
  parameters: {
    SessionAgentIdParam: {
      name: 'id',
      in: 'path',
      required: true,
      schema: sessionParamsSchema.shape.id,
    },
  },
})

export const sessionAgentOpenApiOverrides: OpenApiOverrideMap = {
  '/api/v1/sessions/{id}/agent': {
    post: {
      summary: 'Set the primary agent id for a session (plan/build/custom)',
      tags: ['Sessions'],
      parameters: [openApiParameterRef('SessionAgentIdParam')],
      responses: { 200: openApiJsonResponseRef('SessionAgentResponse'), 503: { description: 'Service unavailable' } },
    },
  },
}

export async function sessionAgentRoutes(app: FastifyInstance) {
  app.post<{ Params: SessionParams; Body: AgentBody }>('/sessions/:id/agent', {
    schema: fastifySchemaFromZod({ params: sessionParamsSchema, body: agentBodySchema }),
  }, async (request, reply) => {
    const runtime = app.runtime
    if (!runtime) {
      return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    }
    const { id } = request.params
    const { agentId } = request.body
    // Reject case/whitespace variants of reserved agent ids: the policy
    // engine plan-mode block matches the normalized id, so storing 'Plan'
    // or 'plan ' would leave a session looking like plan mode while a
    // surface that does an exact compare disagrees. Exact reserved ids and
    // arbitrary custom-agent ids pass through unchanged.
    const normalizedAgentId = agentId.trim().toLowerCase()
    if (RESERVED_AGENT_IDS.has(normalizedAgentId) && !RESERVED_AGENT_IDS.has(agentId)) {
      return reply.status(400).send({
        error: {
          code: 'INVALID_AGENT_ID',
          message: `Ambiguous agent id '${agentId}'. Use the exact reserved id '${normalizedAgentId}'.`,
        },
      })
    }
    await runtime.primaryAgents.set(id, agentId)
    runtime.sessionWatchBroker?.emit({
      type: 'primary_agent_updated',
      sessionId: id,
      agentId,
    })
    return reply.send({
      data: {
        sessionId: id,
        primaryAgentId: agentId,
      },
    })
  })
}
