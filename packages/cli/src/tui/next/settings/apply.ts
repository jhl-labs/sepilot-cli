import type { SettingsItem } from './schema.js'

export interface SettingsApplyDeps {
  setSessionValue(key: string, value: unknown): void
  /** PUT /api/v1/config accepts a dotted-key update document. */
  putConfig(updates: Record<string, unknown>): Promise<void>
}

const UNSAFE_PATH_SEGMENTS = new Set(['__proto__', 'prototype', 'constructor'])

/** Immutable nested set for local view-model updates. */
export function setAtPath(
  target: Record<string, unknown>,
  path: string,
  value: unknown,
): Record<string, unknown> {
  const segments = path.split('.')
  if (segments.some((segment) => segment.length === 0)) {
    throw new Error(`settings: invalid config path "${path}"`)
  }
  if (segments.some((segment) => UNSAFE_PATH_SEGMENTS.has(segment))) {
    throw new Error(`settings: unsafe config path "${path}"`)
  }

  const [head, ...rest] = segments
  const next = { ...target }
  if (rest.length === 0) {
    next[head] = value
    return next
  }

  const child = target[head]
  const childRecord = isPlainRecord(child) ? child : {}
  next[head] = setAtPath(childRecord, rest.join('.'), value)
  return next
}

export async function applySetting(
  item: SettingsItem,
  value: unknown,
  deps: SettingsApplyDeps,
): Promise<{ ok: boolean; error?: string }> {
  try {
    if (item.scope === 'session') {
      if (!item.sessionKey) return { ok: false, error: `settings: ${item.id} has no sessionKey` }
      deps.setSessionValue(item.sessionKey, value)
      return { ok: true }
    }

    if (!item.configPath) {
      const suffix = item.dialogId ? ` opens dialog "${item.dialogId}"` : ' has no configPath'
      return { ok: false, error: `settings: ${item.id}${suffix}` }
    }

    // The daemon config API accepts a partial dotted-key update document, not
    // a nested replacement of the complete config snapshot.
    await deps.putConfig({ [item.configPath]: value })
    return { ok: true }
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error) }
  }
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
