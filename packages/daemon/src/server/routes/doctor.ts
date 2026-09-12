import type { FastifyInstance } from 'fastify'
import { z } from 'zod'
import { buildDoctorReport } from '../../diagnostics/doctor.js'
import '../fastify-types.js'
import { openApiComponentsFromZod } from '../openapi-zod.js'
import {
  openApiJsonResponseRef,
  type OpenApiComponentOverrides,
  type OpenApiOverrideMap,
} from '../openapi.js'

const doctorCheckSchema = z.object({
  name: z.string(),
  category: z.string(),
  status: z.enum(['PASS', 'WARN', 'FAIL']),
  message: z.string(),
  recommendation: z.string().optional(),
})

const doctorSummarySchema = z.object({
  score: z.number(),
  grade: z.enum(['Excellent', 'Good', 'Fair', 'Poor', 'Critical']),
  warnings: z.number().int().nonnegative(),
  errors: z.number().int().nonnegative(),
  generatedAt: z.string().datetime(),
})

const doctorReportSchema = z.object({
  data: z.array(doctorCheckSchema),
  summary: doctorSummarySchema,
})

export const doctorOpenApiComponents: OpenApiComponentOverrides = openApiComponentsFromZod({
  schemas: {
    DoctorCheck: doctorCheckSchema,
    DoctorSummary: doctorSummarySchema,
    DoctorReport: doctorReportSchema,
  },
})

export const doctorOpenApiOverrides: OpenApiOverrideMap = {
  '/api/v1/doctor': {
    get: {
      summary: 'Run daemon doctor diagnostics',
      tags: ['System'],
      responses: { 200: openApiJsonResponseRef('DoctorReport') },
    },
  },
}

export async function doctorRoutes(app: FastifyInstance) {
  app.get('/doctor', async (_request, reply) => {
    const runtime = app.runtime
    if (!runtime) {
      return reply.status(503).send({
        error: {
          code: 'SERVICE_UNAVAILABLE',
          message: 'Runtime not initialized',
        },
      })
    }

    return buildDoctorReport({
      runtime,
      authToken: app.authToken ?? null,
    })
  })
}
