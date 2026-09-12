import { z } from 'zod'

const SAFE_ID_REGEX = /^[A-Za-z0-9][A-Za-z0-9_.\-]{0,199}$/

export function isSafeId(value: unknown): value is string {
  if (typeof value !== 'string') return false
  if (value.length === 0 || value.length > 200) return false
  if (value.includes('..')) return false
  return SAFE_ID_REGEX.test(value)
}

export function assertSafeId(value: unknown, kind: string): string {
  if (!isSafeId(value)) {
    throw new Error(
      `invalid ${kind}: must be 1-200 chars matching [A-Za-z0-9_.-], no traversal`,
    )
  }
  return value
}

export const safeIdSchema = z
  .string()
  .min(1)
  .max(200)
  .refine(
    (v) => !v.includes('..') && SAFE_ID_REGEX.test(v),
    'invalid id: must be 1-200 chars matching [A-Za-z0-9_.-], no traversal',
  )

const UUID_REGEX =
  /^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$/

export function isUuid(value: unknown): value is string {
  return typeof value === 'string' && UUID_REGEX.test(value)
}

export const uuidSchema = z.string().regex(UUID_REGEX, 'invalid uuid')
