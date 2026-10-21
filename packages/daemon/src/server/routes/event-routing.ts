import type { FastifyInstance, FastifyReply } from 'fastify'
import { z } from 'zod'
import '../fastify-types.js'
import {
  destinationSchema,
  eventRoutingConfigSchema,
  redactDestination,
  restoreDestinationSecrets,
  ruleSchema,
  DEFAULT_EVENT_ROUTING_CONFIG,
  type EventDestinationInput,
  type EventRoutingConfig,
} from '../../event-routing/config-schema.js'
import type { EventLogRecord, EventLogStatus } from '../../event-routing/event-store.js'
import type { DeliveryStatus } from '../../event-routing/delivery-outbox.js'
import { eventView, ruleMatches } from '../../event-routing/match.js'
import { buildLlmTurnMessage } from '../../event-routing/llm-prompt.js'
import { DEFAULT_PASSTHROUGH_TEMPLATE, renderTemplate } from '../../event-routing/template.js'
import { applyAndPersistRuntimeUpdate } from '../runtime/config-runtime.js'

const idParams = z.object({ id: z.string().min(1).max(64) })
const eventStatus = z.enum(['received', 'processing', 'processed', 'unmatched', 'failed'])
const deliveryStatus = z.enum(['pending', 'delivering', 'delivered', 'dead'])
const listQuery = z.object({ limit: z.coerce.number().int().min(1).max(500).optional() })
const dryRunBody = z.object({ source: z.string().min(1), type: z.string().min(1), payload: z.record(z.unknown()).default({}) })

function zodIssuesMessage(error: z.ZodError): string {
  return error.issues.map((issue) => `${issue.path.join('.')}: ${issue.message}`).join('; ')
}

function badRequest(reply: FastifyReply, error: z.ZodError | string) {
  const message = typeof error === 'string' ? error : zodIssuesMessage(error)
  return reply.status(400).send({ error: { code: 'INVALID_REQUEST', message } })
}

interface CommitFailure {
  status: number
  error: { code: string; message: string }
}
interface CommitSuccess<T> {
  next: EventRoutingConfig
  data: T
}
type CommitOutcome<T> = CommitSuccess<T> | CommitFailure

function isCommitFailure<T>(outcome: CommitOutcome<T>): outcome is CommitFailure {
  return 'status' in outcome
}

function withoutPayload(record: EventLogRecord): Omit<EventLogRecord, 'payload'> {
  const { payload: _payload, ...rest } = record
  return rest
}

export async function eventRoutingRoutes(app: FastifyInstance): Promise<void> {
  const runtime = app.runtime
  if (!runtime) return

  const current = (): EventRoutingConfig => runtime.config.eventRouting ?? DEFAULT_EVENT_ROUTING_CONFIG

  const commit = async <T>(
    mutate: (cfg: EventRoutingConfig) => CommitOutcome<T>,
  ): Promise<CommitOutcome<T>> => {
    let outcome: CommitOutcome<T> | undefined
    await runtime.configMutationService.apply('config.eventRouting.update', async () => {
      const cfg = runtime.config.eventRouting ?? DEFAULT_EVENT_ROUTING_CONFIG
      outcome = mutate(cfg)
      if (isCommitFailure(outcome)) return
      runtime.config.eventRouting = outcome.next
      await applyAndPersistRuntimeUpdate(runtime, new Set(['eventRouting']))
    })
    return outcome as CommitOutcome<T>
  }

  const sendCommitFailure = (reply: FastifyReply, failure: CommitFailure) =>
    reply.status(failure.status).send({ error: failure.error })

  const routing = (reply: FastifyReply) => {
    if (!runtime.eventRouting) {
      void reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Event routing not started' } })
      return null
    }
    return runtime.eventRouting
  }

  app.get('/event-routing/destinations', async () => ({ data: current().destinations.map(redactDestination) }))

  app.post('/event-routing/destinations', async (request, reply) => {
    const parsed = destinationSchema.safeParse(request.body)
    if (!parsed.success) return badRequest(reply, parsed.error)
    const outcome = await commit((cfg) => {
      if (cfg.destinations.some((d) => d.id === parsed.data.id)) {
        return { status: 409, error: { code: 'CONFLICT', message: `destination exists: ${parsed.data.id}` } }
      }
      const nextConfig = eventRoutingConfigSchema.safeParse({ ...cfg, destinations: [...cfg.destinations, parsed.data] })
      if (!nextConfig.success) {
        return { status: 400, error: { code: 'INVALID_REQUEST', message: zodIssuesMessage(nextConfig.error) } }
      }
      return { next: nextConfig.data, data: redactDestination(parsed.data) }
    })
    if (isCommitFailure(outcome)) return sendCommitFailure(reply, outcome)
    return reply.status(201).send({ data: outcome.data })
  })

  app.put('/event-routing/destinations/:id', async (request, reply) => {
    const params = idParams.safeParse(request.params)
    if (!params.success) return badRequest(reply, params.error)
    const body = request.body
    const raw = { ...(body as Record<string, unknown>), id: params.data.id }
    const outcome = await commit((cfg) => {
      const existing = cfg.destinations.find((d) => d.id === params.data.id)
      if (!existing) return { status: 404, error: { code: 'NOT_FOUND', message: `destination not found: ${params.data.id}` } }
      const restored = restoreDestinationSecrets(existing, raw as EventDestinationInput)
      const parsedDestination = destinationSchema.safeParse(restored)
      if (!parsedDestination.success) {
        return { status: 400, error: { code: 'INVALID_REQUEST', message: zodIssuesMessage(parsedDestination.error) } }
      }
      const nextConfig = eventRoutingConfigSchema.safeParse({
        ...cfg,
        destinations: cfg.destinations.map((d) => (d.id === existing.id ? parsedDestination.data : d)),
      })
      if (!nextConfig.success) {
        return { status: 400, error: { code: 'INVALID_REQUEST', message: zodIssuesMessage(nextConfig.error) } }
      }
      return { next: nextConfig.data, data: redactDestination(parsedDestination.data) }
    })
    if (isCommitFailure(outcome)) return sendCommitFailure(reply, outcome)
    return { data: outcome.data }
  })

  app.delete('/event-routing/destinations/:id', async (request, reply) => {
    const params = idParams.safeParse(request.params)
    if (!params.success) return badRequest(reply, params.error)
    const outcome = await commit((cfg) => {
      if (!cfg.destinations.some((d) => d.id === params.data.id)) {
        return { status: 404, error: { code: 'NOT_FOUND', message: `destination not found: ${params.data.id}` } }
      }
      const users = cfg.rules.filter((rule) => rule.deliver.includes(params.data.id)).map((rule) => rule.id)
      if (users.length) return { status: 409, error: { code: 'CONFLICT', message: `destination used by rules: ${users.join(', ')}` } }
      const nextConfig = eventRoutingConfigSchema.safeParse({ ...cfg, destinations: cfg.destinations.filter((d) => d.id !== params.data.id) })
      if (!nextConfig.success) {
        return { status: 400, error: { code: 'INVALID_REQUEST', message: zodIssuesMessage(nextConfig.error) } }
      }
      return { next: nextConfig.data, data: undefined }
    })
    if (isCommitFailure(outcome)) return sendCommitFailure(reply, outcome)
    return reply.status(204).send()
  })

  app.post('/event-routing/destinations/:id/test', async (request, reply) => {
    const params = idParams.safeParse(request.params)
    if (!params.success) return badRequest(reply, params.error)
    const destination = current().destinations.find((d) => d.id === params.data.id)
    if (!destination) return reply.status(404).send({ error: { code: 'NOT_FOUND', message: `destination not found: ${params.data.id}` } })
    const svc = routing(reply)
    if (!svc) return reply
    return { data: await svc.testDestination(destination) }
  })

  app.get('/event-routing/rules', async () => ({ data: current().rules }))

  app.post('/event-routing/rules', async (request, reply) => {
    const parsed = ruleSchema.safeParse(request.body)
    if (!parsed.success) return badRequest(reply, parsed.error)
    const outcome = await commit((cfg) => {
      if (cfg.rules.some((r) => r.id === parsed.data.id)) {
        return { status: 409, error: { code: 'CONFLICT', message: `rule exists: ${parsed.data.id}` } }
      }
      const nextConfig = eventRoutingConfigSchema.safeParse({ ...cfg, rules: [...cfg.rules, parsed.data] })
      if (!nextConfig.success) {
        return { status: 400, error: { code: 'INVALID_REQUEST', message: zodIssuesMessage(nextConfig.error) } }
      }
      return { next: nextConfig.data, data: parsed.data }
    })
    if (isCommitFailure(outcome)) return sendCommitFailure(reply, outcome)
    return reply.status(201).send({ data: outcome.data })
  })

  app.put('/event-routing/rules/:id', async (request, reply) => {
    const params = idParams.safeParse(request.params)
    if (!params.success) return badRequest(reply, params.error)
    const body = request.body
    const outcome = await commit((cfg) => {
      if (!cfg.rules.some((r) => r.id === params.data.id)) {
        return { status: 404, error: { code: 'NOT_FOUND', message: `rule not found: ${params.data.id}` } }
      }
      const parsedRule = ruleSchema.safeParse({ ...(body as Record<string, unknown>), id: params.data.id })
      if (!parsedRule.success) {
        return { status: 400, error: { code: 'INVALID_REQUEST', message: zodIssuesMessage(parsedRule.error) } }
      }
      const nextConfig = eventRoutingConfigSchema.safeParse({
        ...cfg,
        rules: cfg.rules.map((r) => (r.id === params.data.id ? parsedRule.data : r)),
      })
      if (!nextConfig.success) {
        return { status: 400, error: { code: 'INVALID_REQUEST', message: zodIssuesMessage(nextConfig.error) } }
      }
      return { next: nextConfig.data, data: parsedRule.data }
    })
    if (isCommitFailure(outcome)) return sendCommitFailure(reply, outcome)
    return { data: outcome.data }
  })

  app.delete('/event-routing/rules/:id', async (request, reply) => {
    const params = idParams.safeParse(request.params)
    if (!params.success) return badRequest(reply, params.error)
    const outcome = await commit((cfg) => {
      if (!cfg.rules.some((r) => r.id === params.data.id)) {
        return { status: 404, error: { code: 'NOT_FOUND', message: `rule not found: ${params.data.id}` } }
      }
      const nextConfig = eventRoutingConfigSchema.safeParse({ ...cfg, rules: cfg.rules.filter((r) => r.id !== params.data.id) })
      if (!nextConfig.success) {
        return { status: 400, error: { code: 'INVALID_REQUEST', message: zodIssuesMessage(nextConfig.error) } }
      }
      return { next: nextConfig.data, data: undefined }
    })
    if (isCommitFailure(outcome)) return sendCommitFailure(reply, outcome)
    return reply.status(204).send()
  })

  app.post('/event-routing/rules/:id/dry-run', async (request, reply) => {
    const params = idParams.safeParse(request.params)
    if (!params.success) return badRequest(reply, params.error)
    const body = dryRunBody.safeParse(request.body)
    if (!body.success) return badRequest(reply, body.error)
    const rule = current().rules.find((r) => r.id === params.data.id)
    if (!rule) return reply.status(404).send({ error: { code: 'NOT_FOUND', message: `rule not found: ${params.data.id}` } })
    const event = { id: 'dry-run', subject: null, occurredAt: Date.now(), dedupeKey: 'dry-run', ...body.data }
    const matched = ruleMatches({ ...rule, enabled: true }, event)
    const process = rule.process
    if (process.mode === 'llm') {
      return { data: { matched, text: null, llmMessage: matched ? buildLlmTurnMessage(process.prompt, event) : null } }
    }
    const text = matched ? renderTemplate(rule.template ?? DEFAULT_PASSTHROUGH_TEMPLATE, { ...eventView(event), rule: { id: rule.id } }) : null
    return { data: { matched, text } }
  })

  app.get('/event-routing/events', async (request, reply) => {
    const query = listQuery.extend({ source: z.string().optional(), status: eventStatus.optional() }).safeParse(request.query)
    if (!query.success) return badRequest(reply, query.error)
    const svc = routing(reply)
    if (!svc) return reply
    const status = query.data.status ? [query.data.status as EventLogStatus] : undefined
    return { data: svc.store.list({ source: query.data.source, status, limit: query.data.limit }).map(withoutPayload) }
  })

  app.get('/event-routing/events/:id', async (request, reply) => {
    const params = idParams.safeParse(request.params)
    if (!params.success) return badRequest(reply, params.error)
    const svc = routing(reply)
    if (!svc) return reply
    const event = svc.store.get(params.data.id)
    if (!event) return reply.status(404).send({ error: { code: 'NOT_FOUND', message: `event not found: ${params.data.id}` } })
    return { data: { event, ruleRuns: svc.ruleRuns.listByEvent(event.id), deliveries: svc.outbox.list({ eventId: event.id }) } }
  })

  app.get('/event-routing/deliveries', async (request, reply) => {
    const query = listQuery.extend({ status: deliveryStatus.optional() }).safeParse(request.query)
    if (!query.success) return badRequest(reply, query.error)
    const svc = routing(reply)
    if (!svc) return reply
    const status = query.data.status ? [query.data.status as DeliveryStatus] : undefined
    return { data: svc.outbox.list({ status, limit: query.data.limit }) }
  })

  app.post('/event-routing/deliveries/:id/replay', async (request, reply) => {
    const params = idParams.safeParse(request.params)
    if (!params.success) return badRequest(reply, params.error)
    const svc = routing(reply)
    if (!svc) return reply
    const original = svc.outbox.get(params.data.id)
    if (!original) return reply.status(404).send({ error: { code: 'NOT_FOUND', message: `delivery not found: ${params.data.id}` } })
    if (original.status !== 'dead' && original.status !== 'delivered') {
      return reply.status(409).send({ error: { code: 'CONFLICT', message: 'only dead or delivered deliveries can be replayed' } })
    }
    const replay = svc.outbox.replay(params.data.id)
    if (!replay) return reply.status(404).send({ error: { code: 'NOT_FOUND', message: `delivery not found: ${params.data.id}` } })
    return { data: replay }
  })

  app.post('/event-routing/deliveries/:id/ack', async (request, reply) => {
    const params = idParams.safeParse(request.params)
    if (!params.success) return badRequest(reply, params.error)
    const svc = routing(reply)
    if (!svc) return reply
    if (!svc.outbox.ack(params.data.id)) return reply.status(409).send({ error: { code: 'CONFLICT', message: 'only dead deliveries can be acknowledged' } })
    return { data: { acked: true } }
  })
}
