import type { SessionEvent } from '@sepilotd/core'
import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import '../fastify-types.js'
import { openApiComponentsFromZod } from '../openapi-zod.js'
import {
  openApiJsonResponseRef,
  openApiSchemaRef,
  type OpenApiOverrideMap,
} from '../openapi.js'
import { zodRequestValidation } from './utils.js'
import { getRatePer1k } from '../../providers/pricing.js'

// Fallback per-1K rate for models not in the maintained price table.
const FALLBACK_RATE = { input: 0.001, output: 0.002 }

const estimateRequestSchema = z.object({
  message: z.string().min(1),
  model: z.string().optional(),
  sessionId: z.string().optional(),
})

const estimatePricingSchema = z.object({
  inputPer1k: z.number(),
  outputPer1k: z.number(),
})

const estimateResponseSchema = z.object({
  data: z.object({
    model: z.string(),
    estimatedInputTokens: z.number().int(),
    estimatedOutputTokens: z.number().int(),
    estimatedCost: z.number(),
    pricing: estimatePricingSchema,
  }),
})

type EstimateBody = z.infer<typeof estimateRequestSchema>

export const estimateOpenApiComponents = openApiComponentsFromZod({
  schemas: {
    EstimateRequest: estimateRequestSchema,
    EstimatePricing: estimatePricingSchema,
    EstimateResponse: estimateResponseSchema,
  },
})

export const estimateOpenApiOverrides: OpenApiOverrideMap = {
  '/api/v1/estimate': {
    post: {
      summary: 'Estimate cost',
      tags: ['Chat'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('EstimateRequest'),
          },
        },
      },
      responses: {
        200: openApiJsonResponseRef('EstimateResponse'),
        400: { description: 'Invalid request' },
      },
    },
  },
}

function getEventContentLength(event: SessionEvent): number {
  if ('content' in event && typeof event.content === 'string') {
    return event.content.length
  }
  return 0
}

export async function estimateRoutes(app: FastifyInstance) {
  app.post<{ Body: EstimateBody }>('/estimate', {
    preValidation: zodRequestValidation({
      body: {
        schema: estimateRequestSchema,
        message: 'Invalid estimate request body',
      },
    }),
  }, async (request, reply) => {
    const runtime = app.runtime
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    const body = request.body
    const { message, model, sessionId } = body

    // Estimate input tokens
    let inputChars = message.length
    if (sessionId) {
      try {
        const events = await runtime.sessions.getEvents(sessionId)
        for (const event of events) {
          inputChars += getEventContentLength(event)
        }
      } catch { /* session not found */ }
    }

    const estimatedInputTokens = Math.ceil(inputChars / 4)
    const estimatedOutputTokens = Math.ceil(estimatedInputTokens * 0.5)

    const resolvedModel = model ?? runtime.config.agent.defaultModel ?? 'unknown'
    const pricing = getRatePer1k('', resolvedModel) ?? FALLBACK_RATE

    const estimatedCost = (estimatedInputTokens / 1000) * pricing.input + (estimatedOutputTokens / 1000) * pricing.output

    return {
      data: {
        model: resolvedModel,
        estimatedInputTokens,
        estimatedOutputTokens,
        estimatedCost: Math.round(estimatedCost * 10000) / 10000,
        pricing: { inputPer1k: pricing.input, outputPer1k: pricing.output },
      },
    }
  })
}
