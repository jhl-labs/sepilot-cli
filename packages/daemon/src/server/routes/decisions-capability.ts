import type { FastifyInstance, FastifyReply } from 'fastify'
import { z } from 'zod'
import type { DecisionRequest } from '@sepilotd/core'
import '../fastify-types.js'
import { bindCapability } from '../capabilities/bind.js'
import { DecisionProviderError } from '../../decisions/shared.js'
import type { DecisionProviderRegistry } from '../../decisions/registry.js'

const decisionChoiceSchema = z.object({
  value: z.string().min(1),
  description: z.string().optional(),
})
const decisionLevelSchema = z.object({
  label: z.string().min(1),
  description: z.string().optional(),
})
const decisionQuestionSchema = z.discriminatedUnion('type', [
  z.object({ type: z.literal('predicate'), name: z.string().min(1), instructions: z.string().min(1) }),
  z.object({
    type: z.literal('choice'),
    name: z.string().min(1),
    instructions: z.string().min(1),
    choices: z.array(decisionChoiceSchema).min(2),
  }),
  z.object({
    type: z.literal('score'),
    name: z.string().min(1),
    instructions: z.string().min(1),
    levels: z.array(decisionLevelSchema).min(2),
  }),
])

export const decideRequestSchema = z.object({
  provider: z.string().min(1).optional(),
  model: z.string().min(1).optional(),
  context: z.union([z.string(), z.record(z.unknown())]),
  questions: z.array(decisionQuestionSchema).min(1).max(400),
  timeoutMs: z.number().int().min(1000).max(120000).optional(),
})

export type DecideRequestBody = z.infer<typeof decideRequestSchema>

const testConnectionSchema = z.object({
  provider: z.string().min(1).optional(),
  model: z.string().min(1).optional(),
})

function sendError(reply: FastifyReply, status: number, code: string, message: string): FastifyReply {
  void reply.status(status).send({ error: { code, message } })
  return reply
}

function resolveProvider(
  registry: DecisionProviderRegistry | undefined,
  providerId: string | undefined,
  reply: FastifyReply,
) {
  if (!registry || registry.size === 0) {
    sendError(reply, 404, 'NO_DECISION_PROVIDER_CONFIGURED', 'No decision provider is configured. Add one under `decisionProviders` in config.')
    return null
  }
  const provider = providerId ? registry.get(providerId) : registry.getDefault()
  if (!provider) {
    sendError(reply, 404, 'DECISION_PROVIDER_NOT_FOUND', `Decision provider '${providerId ?? ''}' is not configured`)
    return null
  }
  return provider
}

function statusForDecisionError(error: DecisionProviderError): number {
  switch (error.code) {
    case 'INVALID_DECISION_REQUEST':
      return 400
    case 'DECISION_PROVIDER_TIMEOUT':
      return 504
    default:
      return 502
  }
}

export async function registerDecisionsCapabilityRoutes(
  app: FastifyInstance,
  registryOverride?: DecisionProviderRegistry,
): Promise<void> {
  const registryOf = () => registryOverride ?? app.runtime?.decisionProviderRegistry

  await bindCapability(
    app,
    {
      name: 'decisions',
      version: '1',
      methods: [
        { method: 'GET', path: '/decisions/providers' },
        { method: 'POST', path: '/decisions' },
        { method: 'POST', path: '/decisions/test-connection' },
      ],
    },
    async (a) => {
      a.get('/decisions/providers', async () => {
        const registry = registryOf()
        const defaultId = registry?.getDefault()?.id
        return (registry?.list() ?? []).map((provider) => ({
          id: provider.id,
          label: provider.name,
          kind: provider.kind,
          models: provider.models,
          default: provider.id === defaultId,
        }))
      })

      a.post('/decisions', async (req, reply) => {
        const parsed = decideRequestSchema.safeParse(req.body)
        if (!parsed.success) {
          return sendError(reply, 400, 'INVALID_REQUEST', parsed.error.message)
        }
        const provider = resolveProvider(registryOf(), parsed.data.provider, reply)
        if (!provider) return reply
        const request: DecisionRequest = {
          model: parsed.data.model,
          context: parsed.data.context,
          questions: parsed.data.questions,
        }
        try {
          return await provider.decide(request, { timeoutMs: parsed.data.timeoutMs })
        } catch (error) {
          if (error instanceof DecisionProviderError) {
            return sendError(reply, statusForDecisionError(error), error.code, error.message)
          }
          throw error
        }
      })

      a.post('/decisions/test-connection', async (req, reply) => {
        const parsed = testConnectionSchema.safeParse(req.body ?? {})
        if (!parsed.success) {
          return sendError(reply, 400, 'INVALID_REQUEST', parsed.error.message)
        }
        const provider = resolveProvider(registryOf(), parsed.data.provider, reply)
        if (!provider) return reply
        try {
          const result = await provider.decide({
            model: parsed.data.model,
            context: 'Connection check.',
            questions: [{ type: 'predicate', name: 'reachable', instructions: 'Is this text in English?' }],
          })
          return { ok: true, latencyMs: result.latencyMs, model: result.model }
        } catch (error) {
          const reason = error instanceof Error ? error.message : String(error)
          return { ok: false, reason }
        }
      })
    },
  )
}
