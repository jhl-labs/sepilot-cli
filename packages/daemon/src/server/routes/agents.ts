import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import '../fastify-types.js'
import { openApiComponentsFromZod } from '../openapi-zod.js'
import {
  openApiJsonResponseRef,
  type OpenApiComponentOverrides,
  type OpenApiOverrideMap,
} from '../openapi.js'
import {
  registerUserAgents,
  type BaseGraphBuilders,
  type UserAgentSpec,
} from '../../agent/user-agents/loader.js'
import { builtinGraphBuilders } from '../../agent/graph/presets/index.js'
import {
  createAgentGraphAnalysisSnapshots,
  type GraphNodeModelOverrides,
} from '../../agent/graph/analysis.js'
import type { ILLMProvider } from '@sepilotd/core'

const agentSummarySchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string(),
  source: z.string().optional(),
})

const graphNodeModelOverrideSchema = z.object({
  model: z.string().optional(),
})

const graphNodeSchema = z.object({
  id: z.string(),
  label: z.string(),
  summary: z.string(),
  lifecycleState: z.string(),
  resumeStage: z.string(),
  promptKind: z.string(),
  prompt: z.string(),
  modelConfigurable: z.boolean(),
  recommendedModel: z.string(),
  activeModel: z.string().optional(),
  notes: z.array(z.string()),
})

const graphEdgeSchema = z.object({
  from: z.string(),
  to: z.string(),
  type: z.enum(['direct', 'conditional']),
  label: z.string().optional(),
})

const graphSnapshotSchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string(),
  source: z.string().optional(),
  startNode: z.string(),
  nodeCount: z.number(),
  edgeCount: z.number(),
  nodes: z.array(graphNodeSchema),
  edges: z.array(graphEdgeSchema),
  modelOverrides: z.record(graphNodeModelOverrideSchema),
})

const userAgentRequestSchema = z.object({
  id: z.string().regex(/^[a-z0-9][a-z0-9-_]*$/i),
  name: z.string().min(1).optional(),
  description: z.string().min(1),
  base: z.string().min(1).optional(),
  model: z.string().min(1).optional(),
  temperature: z.number().min(0).max(2).optional(),
  topP: z.number().min(0).max(1).optional(),
  maxTokens: z.number().int().positive().optional(),
  maxIterations: z.number().int().positive().optional(),
  systemPrompt: z.string().min(1),
})

export const agentOpenApiComponents: OpenApiComponentOverrides = openApiComponentsFromZod({
  schemas: {
    AgentSummary: agentSummarySchema,
    AgentListResponse: z.object({
      data: z.array(agentSummarySchema),
    }),
    AgentGraphListResponse: z.object({
      data: z.array(graphSnapshotSchema),
    }),
    UserAgentRequest: userAgentRequestSchema,
  },
})

export const agentOpenApiOverrides: OpenApiOverrideMap = {
  '/api/v1/agents': {
    get: {
      summary: 'List agents',
      tags: ['Agents'],
      responses: { 200: openApiJsonResponseRef('AgentListResponse') },
    },
    post: {
      summary: 'Create or replace a user-defined agent',
      tags: ['Agents'],
      requestBody: {
        content: {
          'application/json': { schema: { $ref: '#/components/schemas/UserAgentRequest' } },
        },
      },
      responses: { 200: openApiJsonResponseRef('AgentSummary') },
    },
  },
  '/api/v1/agents/graphs': {
    get: {
      summary: 'Analyze registered agent graphs',
      tags: ['Agents'],
      responses: { 200: openApiJsonResponseRef('AgentGraphListResponse') },
    },
  },
  '/api/v1/agents/{id}': {
    delete: {
      summary: 'Delete a user-defined agent',
      tags: ['Agents'],
      parameters: [
        { in: 'path', name: 'id', required: true, schema: { type: 'string' } },
      ],
      responses: { 204: { description: 'Agent removed' } },
    },
  },
}

const baseBuilderAdapter: BaseGraphBuilders = {
  get: (id) => builtinGraphBuilders[id as keyof typeof builtinGraphBuilders],
  list: () => Object.keys(builtinGraphBuilders),
}

function createGraphAnalysisProvider(runtime: NonNullable<FastifyInstance['runtime']>): ILLMProvider {
  const provider = runtime.providerRegistry.getDefault()
    ?? runtime.providerRegistry.list()[0]
  if (provider) return provider

  return {
    id: 'unconfigured',
    name: 'Unconfigured Provider',
    models: [],
    chat: async () => {
      throw new Error('No provider configured')
    },
    stream: async function* () {
      throw new Error('No provider configured')
    },
  } as unknown as ILLMProvider
}

export async function agentRoutes(app: FastifyInstance) {
  app.get('/agents', async (_request, _reply) => {
    const runtime = app.runtime
    if (!runtime?.graphRegistry) {
      return { data: [{ id: 'react', name: 'React', description: 'Simple ReAct loop', source: 'builtin' }] }
    }

    const agents = [
      { id: 'react', name: 'React Loop', description: 'Simple think-act-observe loop', source: 'builtin' },
      {
        id: 'instant',
        name: 'Instant',
        description:
          'React loop with completion verification forced off (no ANSWER: protocol, no toolless-final judge) — lowest latency',
        source: 'builtin',
      },
      { id: 'auto', name: 'Auto', description: 'LLM auto-selects the best agent', source: 'builtin' },
      ...runtime.graphRegistry.list().map((g) => ({
        id: g.id,
        name: g.name,
        description: g.description,
        source: g.source ?? 'builtin',
      })),
    ]
    return { data: agents }
  })

  app.get('/agents/graphs', async (_request, reply) => {
    const runtime = app.runtime
    if (!runtime?.graphRegistry) {
      return { data: [] }
    }

    try {
      const provider = createGraphAnalysisProvider(runtime)
      const data = createAgentGraphAnalysisSnapshots(
        runtime.graphRegistry.list(),
        {
          provider,
          tools: runtime.toolRegistry,
          policy: runtime.policyEngine,
          autonomy: runtime.autonomy,
        },
        runtime.config.agent.graphNodeModelOverrides as GraphNodeModelOverrides,
      )
      return { data }
    } catch (error) {
      reply.code(500)
      return {
        error: error instanceof Error ? error.message : 'failed to analyze graphs',
      }
    }
  })

  app.post('/agents', async (request, reply) => {
    const runtime = app.runtime
    if (!runtime?.userAgentLoader || !runtime.graphRegistry) {
      reply.code(503)
      return { error: 'user agent loader not available' }
    }
    const parsed = userAgentRequestSchema.safeParse(request.body)
    if (!parsed.success) {
      reply.code(400)
      return { error: 'invalid agent payload', details: parsed.error.flatten() }
    }
    const body = parsed.data
    const spec: UserAgentSpec = {
      id: body.id,
      name: body.name ?? body.id,
      description: body.description,
      base: body.base ?? 'enhanced',
      model: body.model,
      temperature: body.temperature,
      topP: body.topP,
      maxTokens: body.maxTokens,
      maxIterations: body.maxIterations,
      systemPrompt: body.systemPrompt,
    }
    const existing = runtime.graphRegistry.get(spec.id)
    if (existing && existing.source !== 'user') {
      reply.code(409)
      return { error: `agent id "${spec.id}" is reserved by ${existing.source ?? 'builtin'}` }
    }
    const record = await runtime.userAgentLoader.writeAgent(spec)
    runtime.graphRegistry.unregister(spec.id)
    registerUserAgents(runtime.graphRegistry, [record], baseBuilderAdapter)
    return {
      data: {
        id: spec.id,
        name: spec.name,
        description: spec.description,
        source: 'user',
      },
    }
  })

  app.delete<{ Params: { id: string } }>('/agents/:id', async (request, reply) => {
    const runtime = app.runtime
    if (!runtime?.userAgentLoader || !runtime.graphRegistry) {
      reply.code(503)
      return { error: 'user agent loader not available' }
    }
    const { id } = request.params
    const existing = runtime.graphRegistry.get(id)
    if (existing && existing.source !== 'user') {
      reply.code(409)
      return { error: `agent id "${id}" is not user-defined` }
    }
    const removed = await runtime.userAgentLoader.deleteAgent(id)
    if (!removed) {
      reply.code(404)
      return { error: 'agent not found' }
    }
    runtime.graphRegistry.unregister(id)
    reply.code(204)
    return null
  })
}
