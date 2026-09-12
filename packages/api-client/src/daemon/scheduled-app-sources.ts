/** Durable identities only: app contents and credentials never belong in a job link. */
export interface ScheduledAppSource {
  appId: string
  collection?: string
  itemId?: string
}

export function normalizeScheduledAppSources(value: unknown): ScheduledAppSource[] {
  if (!Array.isArray(value) || value.length > 16) {
    throw new Error('source_refs must be an array of at most 16 app references')
  }
  const refs: ScheduledAppSource[] = []
  for (const entry of value) {
    if (!entry || typeof entry !== 'object' || Array.isArray(entry)) {
      throw new Error('each source reference must be an object')
    }
    const raw = entry as Record<string, unknown>
    if (Object.keys(raw).some(key => !['appId', 'collection', 'itemId'].includes(key))) {
      throw new Error('source references accept only appId, collection, and itemId')
    }
    const ref: ScheduledAppSource = { appId: '' }
    for (const key of ['appId', 'collection', 'itemId'] as const) {
      if (key !== 'appId' && raw[key] === undefined) continue
      const text = raw[key]
      if (typeof text !== 'string' || !text.trim() || text.length > 256 || /[\u0000-\u001f]/.test(text)) {
        throw new Error(`${key} must be a non-empty identifier of at most 256 characters`)
      }
      ref[key] = text.trim()
    }
    if (ref.itemId && !ref.collection) throw new Error('itemId requires collection')
    if (!refs.some(existing => JSON.stringify(existing) === JSON.stringify(ref))) refs.push(ref)
  }
  return refs
}

export function scheduledAppSourcesFromMetadata(
  metadata: Record<string, unknown> | null | undefined,
): ScheduledAppSource[] {
  const profile = metadata?.scheduledAppSources
  if (profile === undefined) {
    // Existing Desktop-created reminders already carry these canonical ids.
    // Project them into the common query contract without duplicating jobs.
    const source = metadata?.source
    if (!source || typeof source !== 'object' || Array.isArray(source)) return []
    const legacy = source as Record<string, unknown>
    if (typeof legacy.appId !== 'string') return []
    if (legacy.type === 'app.todo' && typeof legacy.todoId === 'string') {
      return normalizeScheduledAppSources([{ appId: legacy.appId, collection: 'tasks', itemId: legacy.todoId }])
    }
    if (legacy.type === 'app.notification') return normalizeScheduledAppSources([{ appId: legacy.appId }])
    return []
  }
  if (!profile || typeof profile !== 'object' || Array.isArray(profile)
    || (profile as Record<string, unknown>).version !== 1) {
    throw new Error('Unsupported scheduledAppSources metadata')
  }
  return normalizeScheduledAppSources((profile as Record<string, unknown>).refs)
}

export function withScheduledAppSources(
  metadata: Record<string, unknown> | null | undefined,
  value: unknown,
): Record<string, unknown> {
  return { ...metadata, scheduledAppSources: { version: 1, refs: normalizeScheduledAppSources(value) } }
}

/** A broad app/collection query includes linked descendants, never title matches. */
export function scheduledAppSourceMatches(ref: ScheduledAppSource, query: ScheduledAppSource): boolean {
  return ref.appId === query.appId
    && (query.collection === undefined || ref.collection === query.collection)
    && (query.itemId === undefined || ref.itemId === query.itemId)
}

export const SCHEDULED_APP_SOURCE_SCHEMA = {
  type: 'object',
  properties: {
    appId: { type: 'string', minLength: 1, maxLength: 256, description: 'Real app id from apps.list/read.' },
    collection: { type: 'string', minLength: 1, maxLength: 256, description: 'Collection name from the app schema; omit for an entire app.' },
    itemId: { type: 'string', minLength: 1, maxLength: 256, description: 'Real record id from apps.read; requires collection.' },
  },
  required: ['appId'],
  additionalProperties: false,
} as const
