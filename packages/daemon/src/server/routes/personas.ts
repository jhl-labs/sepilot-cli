import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import { listPersonas, getPersona } from '../../agent/personas.js'
import { openApiComponentsFromZod } from '../openapi-zod.js'
import {
  openApiJsonResponseRef,
  openApiParameterRef,
  type OpenApiComponentOverrides,
  type OpenApiOverrideMap,
} from '../openapi.js'
import { fastifySchemaFromZod } from './utils.js'

const personaSchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string(),
  systemPromptAddition: z.string(),
})

const personaIdParamsSchema = z.object({
  id: z.string().min(1),
})

type PersonaIdParams = z.input<typeof personaIdParamsSchema>

export const personaOpenApiComponents: OpenApiComponentOverrides = openApiComponentsFromZod({
  schemas: {
    Persona: personaSchema,
    PersonaListResponse: z.object({ data: z.array(personaSchema) }),
    PersonaResponse: z.object({ data: personaSchema }),
  },
  parameters: {
    PersonaIdParam: {
      name: 'id',
      in: 'path',
      required: true,
      schema: personaIdParamsSchema.shape.id,
    },
  },
})

export const personaOpenApiOverrides: OpenApiOverrideMap = {
  '/api/v1/personas': {
    get: {
      summary: 'List personas',
      tags: ['Personas'],
      responses: { 200: openApiJsonResponseRef('PersonaListResponse') },
    },
  },
  '/api/v1/personas/{id}': {
    get: {
      summary: 'Get persona',
      tags: ['Personas'],
      parameters: [openApiParameterRef('PersonaIdParam')],
      responses: { 200: openApiJsonResponseRef('PersonaResponse'), 404: { description: 'Not found' } },
    },
  },
}

export async function personaRoutes(app: FastifyInstance) {
  app.get('/personas', async () => {
    return { data: listPersonas() }
  })

  app.get<{ Params: PersonaIdParams }>('/personas/:id', {
    schema: fastifySchemaFromZod({ params: personaIdParamsSchema }),
  }, async (request, reply) => {
    const params = request.params
    const persona = getPersona(params.id)
    if (!persona) return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'Persona not found' } })
    return { data: persona }
  })
}
