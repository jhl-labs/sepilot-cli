import { homedir } from 'node:os'
import type { IncomingHttpHeaders } from 'node:http'
import type { FastifyReply, FastifyRequest } from 'fastify'
import type { z, ZodTypeAny } from 'zod'
import { openApiSchemaFromZod } from '../openapi-zod.js'

const DEFAULT_DATA_DIR = `${homedir()}/.sepilotd`

export function getRuntimeDataDir(runtime: { dataDir?: string }): string {
  return runtime.dataDir ?? DEFAULT_DATA_DIR
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

export function normalizeHeaders(headers: IncomingHttpHeaders): Record<string, string> {
  const normalized: Record<string, string> = {}
  for (const [key, value] of Object.entries(headers)) {
    if (typeof value === 'string') {
      normalized[key] = value
      continue
    }
    if (Array.isArray(value) && value.length > 0) {
      normalized[key] = value[0] ?? ''
    }
  }
  return normalized
}

interface FastifySchemaFromZodInput {
  body?: ZodTypeAny
  params?: ZodTypeAny
  querystring?: ZodTypeAny
}

interface ZodValidationTarget {
  schema: ZodTypeAny
  message: string
}

interface ZodRequestValidationInput {
  body?: ZodValidationTarget
  params?: ZodValidationTarget
  query?: ZodValidationTarget
}

export function fastifySchemaFromZod(input: FastifySchemaFromZodInput): Record<string, unknown> {
  const schema: Record<string, unknown> = {}
  if (input.body) {
    schema.body = openApiSchemaFromZod(input.body)
  }
  if (input.params) {
    schema.params = openApiSchemaFromZod(input.params)
  }
  if (input.querystring) {
    schema.querystring = openApiSchemaFromZod(input.querystring)
  }
  return schema
}

function assignParsedRequestValue(
  request: FastifyRequest,
  key: 'body' | 'params' | 'query',
  value: unknown,
): void {
  const mutableRequest = request as FastifyRequest & Record<'body' | 'params' | 'query', unknown>
  mutableRequest[key] = value
}

export function zodRequestValidation(input: ZodRequestValidationInput) {
  return async (request: FastifyRequest, reply: FastifyReply) => {
    if (input.params) {
      const parsed = parseRequestInput(
        reply,
        input.params.schema,
        request.params,
        input.params.message,
      )
      if (!parsed) return reply
      assignParsedRequestValue(request, 'params', parsed)
    }

    if (input.query) {
      const parsed = parseRequestInput(
        reply,
        input.query.schema,
        request.query,
        input.query.message,
      )
      if (!parsed) return reply
      assignParsedRequestValue(request, 'query', parsed)
    }

    if (input.body) {
      const parsed = parseRequestInput(
        reply,
        input.body.schema,
        request.body,
        input.body.message,
      )
      if (!parsed) return reply
      assignParsedRequestValue(request, 'body', parsed)
    }
  }
}

export function parseRequestInput<T extends ZodTypeAny>(
  reply: FastifyReply,
  schema: T,
  value: unknown,
  message: string,
): z.infer<T> | null {
  const parsed = schema.safeParse(value)
  if (parsed.success) {
    return parsed.data
  }

  void reply.status(400).send({
    error: {
      code: 'INVALID_REQUEST',
      message,
    },
  })
  return null
}
