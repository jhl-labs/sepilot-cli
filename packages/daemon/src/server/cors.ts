import type { FastifyInstance, FastifyRequest } from 'fastify'
import { skipOverride } from './skip-override.js'

const CORS_ALLOWED_METHODS = 'GET, POST, PUT, PATCH, DELETE, OPTIONS'
// X-Memory-Scope-*와 X-Sepilotd-Surface는 browser-backed surfaces가 daemon에
// 전달하는 헤더다. preflight에서 허용하지 않으면 브라우저가 actual request를
// 차단한다.
const CORS_ALLOWED_HEADERS =
  'Authorization, Content-Type, X-Request-ID, X-Sepilotd-Surface, X-Memory-Scope-User-Id, X-Memory-Scope-Channel-Type, X-Memory-Scope-Channel-Id, X-Memory-Scope-Session-Id, X-Memory-Scope-Groups'
const TRUSTED_APP_ORIGINS = new Set(['app://index'])
const LOOPBACK_HOSTNAMES = new Set([
  '127.0.0.1',
  'localhost',
  '::1',
  '[::1]',
])

function mergeVary(existing: string | undefined, value: string): string {
  if (!existing) return value
  const parts = new Set(existing.split(',').map((part) => part.trim()).filter(Boolean))
  parts.add(value)
  return Array.from(parts).join(', ')
}

function parseConfiguredOrigins(): Set<string> {
  return new Set(
    (process.env.SEPILOTD_CORS_ALLOWED_ORIGINS ?? '')
      .split(/[,\n;]+/)
      .map((origin) => origin.trim())
      .filter(Boolean),
  )
}

export function isLoopbackCorsOrigin(origin: string): boolean {
  let url: URL
  try {
    url = new URL(origin)
  } catch {
    return false
  }

  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    return false
  }

  const hostname = url.hostname.toLowerCase()
  return LOOPBACK_HOSTNAMES.has(hostname) || hostname.endsWith('.localhost')
}

export function isDaemonCorsOriginAllowed(origin: string): boolean {
  const normalized = origin.trim()
  return TRUSTED_APP_ORIGINS.has(normalized)
    || isLoopbackCorsOrigin(normalized)
    || parseConfiguredOrigins().has(normalized)
}

export function buildDaemonCorsHeaders(
  request: FastifyRequest,
  headers: Record<string, string> = {},
): Record<string, string> {
  const origin = request.headers.origin
  if (typeof origin !== 'string' || !isDaemonCorsOriginAllowed(origin)) {
    return headers
  }

  return {
    ...headers,
    'Access-Control-Allow-Origin': origin,
    'Access-Control-Allow-Methods': CORS_ALLOWED_METHODS,
    'Access-Control-Allow-Headers': CORS_ALLOWED_HEADERS,
    Vary: mergeVary(headers.Vary, 'Origin'),
  }
}

export async function corsPlugin(app: FastifyInstance) {
  // Handle OPTIONS preflight for any path
  app.options('/*', async (request, reply) => {
    if (
      typeof request.headers.origin === 'string'
      && !isDaemonCorsOriginAllowed(request.headers.origin)
    ) {
      reply.header('Vary', 'Origin')
      return reply.status(403).send({
        error: {
          code: 'CORS_ORIGIN_DENIED',
          message: 'Origin is not allowed by daemon CORS policy',
        },
      })
    }

    for (const [key, value] of Object.entries(
      buildDaemonCorsHeaders(request, { 'Access-Control-Max-Age': '3600' }),
    )) {
      reply.header(key, value)
    }
    return reply.status(204).send()
  })

  // Add CORS headers to all responses
  app.addHook('onSend', async (request, reply) => {
    for (const [key, value] of Object.entries(buildDaemonCorsHeaders(request))) {
      reply.header(key, value)
    }
  })
}

skipOverride(corsPlugin)
