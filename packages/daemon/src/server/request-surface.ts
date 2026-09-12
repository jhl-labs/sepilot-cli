import type { FastifyRequest } from 'fastify'
import type { RequestAuthContext } from './auth.js'

const MAX_SURFACE_LABEL_LENGTH = 80

function firstHeaderValue(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value
}

export function normalizeSurfaceLabel(value: unknown): string | null {
  if (typeof value !== 'string') return null
  const normalized = value.trim().toLowerCase()
  if (!normalized) return null
  const safe = normalized.replace(/[^a-z0-9._:-]/g, '-').slice(0, MAX_SURFACE_LABEL_LENGTH)
  return safe || null
}

export function resolveRequestSurface(request: FastifyRequest): string | null {
  return normalizeSurfaceLabel(firstHeaderValue(request.headers['x-sepilotd-surface']))
}

export function resolveClientLabel(
  authContext: RequestAuthContext | undefined,
  surface: string | null,
): string | null {
  if (authContext?.kind === 'extension') {
    return surface ? `${surface}:${authContext.label}` : authContext.label
  }
  if (surface) return surface
  return authContext?.kind === 'master' ? 'master' : null
}
