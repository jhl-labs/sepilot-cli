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

const pendingQuestionDtoSchema = z.object({
  id: z.string(),
  sessionId: z.string(),
  prompt: z.string(),
  choices: z.array(z.string()).optional(),
})

const questionListResponseSchema = z.object({
  data: z.array(pendingQuestionDtoSchema),
})

const questionAnswerResponseSchema = z.object({
  data: z.object({
    answered: z.boolean(),
  }),
})

const sessionParamsSchema = z.object({
  id: z.string().min(1),
})

const answerParamsSchema = z.object({
  id: z.string().min(1),
  qid: z.string().min(1),
})

const answerBodySchema = z.object({
  answer: z.string(),
})

type SessionParams = z.input<typeof sessionParamsSchema>
type AnswerParams = z.input<typeof answerParamsSchema>
type AnswerBody = z.input<typeof answerBodySchema>

export const sessionQuestionsOpenApiComponents: OpenApiComponentOverrides = openApiComponentsFromZod({
  schemas: {
    PendingQuestion: pendingQuestionDtoSchema,
    QuestionListResponse: questionListResponseSchema,
    QuestionAnswerResponse: questionAnswerResponseSchema,
    QuestionAnswerBody: answerBodySchema,
  },
  parameters: {
    SessionQuestionIdParam: {
      name: 'id',
      in: 'path',
      required: true,
      schema: sessionParamsSchema.shape.id,
    },
    SessionQuestionQidParam: {
      name: 'qid',
      in: 'path',
      required: true,
      schema: answerParamsSchema.shape.qid,
    },
  },
})

export const sessionQuestionsOpenApiOverrides: OpenApiOverrideMap = {
  '/api/v1/sessions/{id}/questions': {
    get: {
      summary: 'List pending questions for a session',
      tags: ['Sessions'],
      parameters: [openApiParameterRef('SessionQuestionIdParam')],
      responses: { 200: openApiJsonResponseRef('QuestionListResponse') },
    },
  },
  '/api/v1/sessions/{id}/questions/{qid}': {
    post: {
      summary: 'Answer a pending question',
      tags: ['Sessions'],
      parameters: [
        openApiParameterRef('SessionQuestionIdParam'),
        openApiParameterRef('SessionQuestionQidParam'),
      ],
      responses: {
        200: openApiJsonResponseRef('QuestionAnswerResponse'),
        404: { description: 'Question not found' },
      },
    },
  },
}

export async function sessionQuestionsRoutes(app: FastifyInstance) {
  app.get<{ Params: SessionParams }>('/sessions/:id/questions', {
    schema: fastifySchemaFromZod({ params: sessionParamsSchema }),
  }, async (request, reply) => {
    const runtime = app.runtime
    if (!runtime) {
      return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    }
    const { id } = request.params
    const pending = runtime.questions.list(id)
    return reply.send({
      data: pending.map(q => ({
        id: q.id,
        sessionId: q.sessionId,
        prompt: q.prompt,
        choices: q.choices,
      })),
    })
  })

  app.post<{ Params: AnswerParams; Body: AnswerBody }>('/sessions/:id/questions/:qid', {
    schema: fastifySchemaFromZod({ params: answerParamsSchema, body: answerBodySchema }),
  }, async (request, reply) => {
    const runtime = app.runtime
    if (!runtime) {
      return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    }
    const { qid } = request.params
    const { answer } = request.body
    const answered = runtime.questions.answer(qid, answer)
    if (!answered) {
      return reply.status(404).send({ error: { code: 'NOT_FOUND', message: 'Question not found' } })
    }
    return reply.send({ data: { answered: true } })
  })
}
