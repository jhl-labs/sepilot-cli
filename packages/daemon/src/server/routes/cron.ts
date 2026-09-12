// DEPRECATED SURFACE — `/api/v1/cron` predates the unified scheduler model and
// is kept only for backwards compatibility with older clients. New code should
// use `/api/v1/scheduled-tasks` (REST, natural-language `when`, timezone,
// retries, run history) or the `scheduler` capability (`/scheduler/jobs*`).
// All routes here are backed by the same job-store used by those surfaces.
import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import '../fastify-types.js'
import { openApiComponentsFromZod } from '../openapi-zod.js'
import {
  openApiJsonResponseRef,
  openApiSchemaRef,
  type OpenApiComponentOverrides,
  type OpenApiOverrideMap,
} from '../openapi.js'
import type { ScheduledJob } from '../../scheduler/job-store.js'
import { SchedulerParseError } from '../../scheduler/time-parser.js'
import {
  isSchedulerSurfaceEnabled,
  sendSchedulerSurfaceDisabled,
} from '../../scheduler/surface-access.js'

const createCronTaskSchema = z.object({
  name: z.string().min(1),
  schedule: z.string().min(1),
  instruction: z.string().min(1),
  enabled: z.boolean().optional(),
})

const cronTaskIdParamsSchema = z.object({
  id: z.string().min(1),
})

const cronTaskSchema = z.object({
  id: z.string(),
  name: z.string(),
  schedule: z.string(),
  instruction: z.string(),
  enabled: z.boolean(),
  lastRun: z.string().optional(),
  nextRun: z.string().optional(),
})

const cronTaskListResponseSchema = z.object({
  data: z.array(cronTaskSchema),
})

const cronTaskResponseSchema = z.object({
  data: cronTaskSchema,
})

type CronTaskIdParams = z.input<typeof cronTaskIdParamsSchema>

export const cronOpenApiComponents: OpenApiComponentOverrides = openApiComponentsFromZod({
  schemas: {
    CreateCronTaskRequest: createCronTaskSchema,
    CronTask: cronTaskSchema,
    CronTaskListResponse: cronTaskListResponseSchema,
    CronTaskResponse: cronTaskResponseSchema,
  },
  parameters: {
    CronTaskIdParam: {
      name: 'id',
      in: 'path',
      required: true,
      schema: cronTaskIdParamsSchema.shape.id,
    },
  },
})

export const cronOpenApiOverrides: OpenApiOverrideMap = {
  '/api/v1/cron': {
    get: {
      summary: 'List cron tasks (deprecated — use /api/v1/scheduled-tasks)',
      tags: ['Cron'],
      responses: { 200: openApiJsonResponseRef('CronTaskListResponse') },
    },
    post: {
      summary: 'Create cron task (deprecated — use /api/v1/scheduled-tasks)',
      tags: ['Cron'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('CreateCronTaskRequest'),
          },
        },
      },
      responses: { 200: openApiJsonResponseRef('CronTaskResponse') },
    },
  },
  '/api/v1/cron/{id}': {
    delete: {
      summary: 'Delete cron task (deprecated — use /api/v1/scheduled-tasks)',
      tags: ['Cron'],
      parameters: [{ $ref: '#/components/parameters/CronTaskIdParam' }],
      responses: { 204: { description: 'Deleted' } },
    },
  },
}

/** Shape legacy CronTask response from a ScheduledJob */
const toLegacy = (j: ScheduledJob) => ({
  id: j.id,
  name: j.name,
  schedule: j.cron ?? '',
  instruction: j.instruction,
  enabled: j.enabled,
  lastRun: j.lastRunAt != null ? new Date(j.lastRunAt).toISOString() : undefined,
  nextRun: new Date(j.nextRunAt).toISOString(),
})

export async function cronRoutes(app: FastifyInstance) {
  // Every /cron response carries a Deprecation header
  app.addHook('onSend', async (_req, reply) => {
    reply.header('Deprecation', 'true')
  })

  const requireCliSurface = (reply: import('fastify').FastifyReply) => {
    if (isSchedulerSurfaceEnabled(app.runtime?.config, 'cli')) return true
    sendSchedulerSurfaceDisabled(reply, 'cli')
    return false
  }

  app.get('/cron', async (_request, reply) => {
    if (!requireCliSurface(reply)) return reply
    const store = app.runtime?.jobStore
    if (!store) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
      })
    }
    return { data: store.list().map(toLegacy) }
  })

  app.post('/cron', async (request, reply) => {
    if (!requireCliSurface(reply)) return reply
    const store = app.runtime?.jobStore
    const parser = app.runtime?.parseWhen

    if (!store || !parser) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
      })
    }

    let body: z.infer<typeof createCronTaskSchema>
    try {
      body = createCronTaskSchema.parse(request.body)
    } catch {
      return reply.status(400).send({
        error: { code: 'INVALID_BODY', message: 'Invalid request body' },
      })
    }

    let parsed
    try {
      parsed = parser(body.schedule)
    } catch (err) {
      if (err instanceof SchedulerParseError) {
        return reply.status(400).send({
          error: { code: 'PARSE_FAILED', message: err.message },
        })
      }
      return reply.status(400).send({
        error: { code: 'PARSE_FAILED', message: (err as Error).message },
      })
    }

    const job = store.create({
      name: body.name,
      kind: parsed.kind,
      cron: parsed.kind === 'recurring' ? parsed.cron : null,
      runAt: parsed.kind === 'oneshot' ? parsed.runAt : null,
      nextRunAt: parsed.kind === 'oneshot' ? parsed.runAt : parsed.nextRunAt,
      instruction: body.instruction,
      channelType: null,
      channelTarget: null,
      replyToMessageId: null,
      parentSessionId: null,
      enabled: body.enabled ?? true,
      createdBy: 'rest',
    })
    return { data: toLegacy(job) }
  })

  app.delete<{ Params: CronTaskIdParams }>('/cron/:id', async (request, reply) => {
    if (!requireCliSurface(reply)) return reply
    const store = app.runtime?.jobStore

    if (!store) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
      })
    }

    // If job not found, still return 204 (idempotent delete)
    const { id } = request.params
    const existing = store.get(id)
    if (existing) {
      store.cancel(id)
    }
    return reply.status(204).send()
  })
}
