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
  resolveCommand,
  type UserCommandSpec,
} from '../../agent/user-commands/loader.js'
import { zodRequestValidation } from './utils.js'

const idParamsSchema = z.object({ id: z.string().min(1) })

const commandSummarySchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string(),
  args: z.enum(['none', 'optional', 'required']),
  agent: z.string().optional(),
  model: z.string().optional(),
})

const userCommandRequestSchema = z.object({
  id: z.string().regex(/^[a-z0-9][a-z0-9-_]*$/i),
  name: z.string().min(1).optional(),
  description: z.string().min(1),
  args: z.enum(['none', 'optional', 'required']).optional(),
  agent: z.string().min(1).optional(),
  model: z.string().min(1).optional(),
  body: z.string().min(1),
})

const resolveRequestSchema = z.object({
  args: z.string().optional(),
})

const resolveResponseSchema = z.object({
  id: z.string(),
  prompt: z.string(),
  agent: z.string().optional(),
  model: z.string().optional(),
})

export const commandOpenApiComponents: OpenApiComponentOverrides = openApiComponentsFromZod({
  schemas: {
    UserCommandSummary: commandSummarySchema,
    UserCommandListResponse: z.object({ data: z.array(commandSummarySchema) }),
    UserCommandRequest: userCommandRequestSchema,
    UserCommandResolveRequest: resolveRequestSchema,
    UserCommandResolveResponse: resolveResponseSchema,
  },
})

export const commandOpenApiOverrides: OpenApiOverrideMap = {
  '/api/v1/commands': {
    get: {
      summary: 'List custom commands',
      tags: ['Commands'],
      responses: { 200: openApiJsonResponseRef('UserCommandListResponse') },
    },
    post: {
      summary: 'Create or replace a custom command',
      tags: ['Commands'],
      requestBody: {
        content: {
          'application/json': { schema: { $ref: '#/components/schemas/UserCommandRequest' } },
        },
      },
      responses: { 200: openApiJsonResponseRef('UserCommandSummary') },
    },
  },
  '/api/v1/commands/{id}': {
    delete: {
      summary: 'Delete a custom command',
      tags: ['Commands'],
      parameters: [{ in: 'path', name: 'id', required: true, schema: { type: 'string' } }],
      responses: { 204: { description: 'Command removed' } },
    },
  },
  '/api/v1/commands/{id}/resolve': {
    post: {
      summary: 'Resolve a command into a chat prompt with substituted args',
      tags: ['Commands'],
      parameters: [{ in: 'path', name: 'id', required: true, schema: { type: 'string' } }],
      requestBody: {
        content: {
          'application/json': { schema: { $ref: '#/components/schemas/UserCommandResolveRequest' } },
        },
      },
      responses: { 200: openApiJsonResponseRef('UserCommandResolveResponse') },
    },
  },
}

function summarize(spec: UserCommandSpec) {
  return {
    id: spec.id,
    name: spec.name,
    description: spec.description,
    args: spec.args,
    agent: spec.agent,
    model: spec.model,
  }
}

export async function commandRoutes(app: FastifyInstance) {
  app.get('/commands', async (_request, reply) => {
    const runtime = app.runtime
    if (!runtime?.userCommandStore) {
      reply.code(503)
      return { error: 'user command store not available' }
    }
    return { data: runtime.userCommandStore.list().map((r) => summarize(r.spec)) }
  })

  app.post('/commands', async (request, reply) => {
    const runtime = app.runtime
    if (!runtime?.userCommandStore) {
      reply.code(503)
      return { error: 'user command store not available' }
    }
    const parsed = userCommandRequestSchema.safeParse(request.body)
    if (!parsed.success) {
      reply.code(400)
      return { error: 'invalid command payload', details: parsed.error.flatten() }
    }
    const body = parsed.data
    const spec: UserCommandSpec = {
      id: body.id,
      name: body.name ?? body.id,
      description: body.description,
      args: body.args ?? 'optional',
      agent: body.agent,
      model: body.model,
      body: body.body,
    }
    const record = await runtime.userCommandStore.writeCommand(spec)
    return { data: summarize(record.spec) }
  })

  app.delete<{ Params: { id: string } }>('/commands/:id', {
    preValidation: zodRequestValidation({
      params: { schema: idParamsSchema, message: 'Invalid command id' },
    }),
  }, async (request, reply) => {
    const runtime = app.runtime
    if (!runtime?.userCommandStore) {
      reply.code(503)
      return { error: 'user command store not available' }
    }
    const params = request.params
    const removed = await runtime.userCommandStore.deleteCommand(params.id)
    if (!removed) {
      reply.code(404)
      return { error: 'command not found' }
    }
    reply.code(204)
    return null
  })

  app.post<{ Params: { id: string } }>('/commands/:id/resolve', {
    preValidation: zodRequestValidation({
      params: { schema: idParamsSchema, message: 'Invalid command id' },
    }),
  }, async (request, reply) => {
    const runtime = app.runtime
    if (!runtime?.userCommandStore) {
      reply.code(503)
      return { error: 'user command store not available' }
    }
    const parsed = resolveRequestSchema.safeParse(request.body ?? {})
    if (!parsed.success) {
      reply.code(400)
      return { error: 'invalid resolve payload', details: parsed.error.flatten() }
    }
    const params = request.params
    const record = runtime.userCommandStore.get(params.id)
    if (!record) {
      reply.code(404)
      return { error: 'command not found' }
    }
    try {
      const resolved = resolveCommand(record.spec, parsed.data.args ?? '')
      return { data: resolved }
    } catch (err) {
      reply.code(400)
      return { error: (err as Error).message }
    }
  })
}
