import type { FastifyInstance } from 'fastify'
import '../fastify-types.js'
import { z } from 'zod'
import {
  availableProviderModels,
  currentModelTarget,
  formatAvailableModels,
  pullOllamaModel,
  resolveModelTarget,
  switchDefaultModelWithSelfTest,
} from '../runtime/model-control.js'
import { MODEL_PROBE_TIMEOUT_MS } from '../../providers/model-probe.js'
import { zodRequestValidation } from './utils.js'

const modelSwitchRequestSchema = z.object({
  target: z.string().trim().min(1),
  timeoutMs: z.number().int().min(1_000).max(60_000).optional().default(MODEL_PROBE_TIMEOUT_MS),
})

const modelPullRequestSchema = z.object({
  model: z.string().trim().min(1),
  providerId: z.string().trim().min(1).optional(),
  timeoutMs: z.number().int().min(1_000).max(10 * 60_000).optional(),
  switch: z.boolean().optional().default(false),
})

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

export function registerConfigModelRoutes(app: FastifyInstance): void {
  const runtime = app.runtime

  app.get('/config/model', async (_request, reply) => {
    if (!runtime) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
      })
    }

    const current = currentModelTarget(runtime)
    return {
      data: {
        current,
        providers: availableProviderModels(runtime),
        lines: formatAvailableModels(runtime),
      },
    }
  })

  app.post<{ Body: z.infer<typeof modelSwitchRequestSchema> }>('/config/model/switch', {
    preValidation: zodRequestValidation({
      body: {
        schema: modelSwitchRequestSchema,
        message: 'Invalid model switch request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
      })
    }
    const body = request.body

    const resolution = resolveModelTarget(runtime, body.target)
    if (resolution.status === 'not_found') {
      return reply.status(404).send({
        error: {
          code: 'MODEL_NOT_FOUND',
          message: `Unknown model: "${resolution.query}".`,
        },
      })
    }
    if (resolution.status === 'ambiguous') {
      return reply.status(409).send({
        error: {
          code: 'MODEL_AMBIGUOUS',
          message: `Ambiguous model "${resolution.query}". Use provider/model.`,
          matches: resolution.matches,
        },
      })
    }

    try {
      const result = await switchDefaultModelWithSelfTest(runtime, resolution.target, {
        timeoutMs: body.timeoutMs,
        reason: 'model.switch.api',
      })
      return { data: result }
    } catch (error) {
      return reply.status(400).send({
        error: { code: 'MODEL_SWITCH_FAILED', message: errorMessage(error) },
      })
    }
  })

  app.post<{ Body: z.infer<typeof modelPullRequestSchema> }>('/config/model/pull', {
    preValidation: zodRequestValidation({
      body: {
        schema: modelPullRequestSchema,
        message: 'Invalid model pull request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
      })
    }
    const body = request.body

    try {
      const pull = await pullOllamaModel(runtime, body.model, {
        providerId: body.providerId,
        timeoutMs: body.timeoutMs,
      })
      if (!body.switch) {
        return { data: { pull } }
      }

      const switched = await switchDefaultModelWithSelfTest(
        runtime,
        { providerId: pull.providerId, modelId: pull.model },
        {
          timeoutMs: Math.min(body.timeoutMs ?? MODEL_PROBE_TIMEOUT_MS, 60_000),
          reason: 'model.pull.switch',
        },
      )
      return { data: { pull, switch: switched } }
    } catch (error) {
      return reply.status(400).send({
        error: { code: 'MODEL_PULL_FAILED', message: errorMessage(error) },
      })
    }
  })
}
