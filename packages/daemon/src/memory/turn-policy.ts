import type { SessionEvent } from '@sepilotd/core'

export interface TurnMemoryPolicy { read: boolean; write: boolean; automaticRead?: boolean; automaticWrite?: boolean }

export function parseTurnMemoryPolicy(value: unknown): Partial<TurnMemoryPolicy> {
  if (!value || typeof value !== 'object') return {}
  const raw = value as Record<string, unknown>
  return { ...(typeof raw.read === 'boolean' ? { read: raw.read } : {}),
    ...(typeof raw.write === 'boolean' ? { write: raw.write } : {}) }
}

export function resolveTurnMemoryPolicy(explicit: unknown, semantic: unknown, fallback = false): TurnMemoryPolicy {
  if (explicit !== undefined && (!explicit || typeof explicit !== 'object' || Array.isArray(explicit)
    || Object.entries(explicit).some(([key, value]) => !['read', 'write'].includes(key) || typeof value !== 'boolean'))) {
    throw new Error('memoryPolicy must contain only boolean read/write fields')
  }
  const caller = parseTurnMemoryPolicy(explicit)
  const intent = parseTurnMemoryPolicy(semantic)
  const uncertain = fallback || (semantic !== undefined && (intent.read === undefined || intent.write === undefined))
  return { read: caller.read !== false && intent.read !== false,
    write: caller.write !== false && intent.write !== false,
    ...(uncertain ? { automaticRead: caller.read === true, automaticWrite: caller.write === true } : {}) }

}

export function memoryToolAllowed(name: string, policy: TurnMemoryPolicy): boolean {
  if (!name.startsWith('memory.') && !name.startsWith('knowledge.')) return true
  if (['memory.search', 'memory.list', 'memory.graph.search', 'memory.graph.page', 'memory.graph.neighbors', 'knowledge.search', 'knowledge.read', 'knowledge.review'].includes(name)) return policy.read
  return policy.write && (name === 'memory.remember' || policy.read)
}

export function eventsAllowedForMemory(events: SessionEvent[]): SessionEvent[] {
  const result: SessionEvent[] = []
  let turn: SessionEvent[] = []
  const flush = () => {
    if (!turn.some(e => e.type === 'router_decision' && (e.decision.memoryPolicy?.write === false || e.decision.memoryPolicy?.automaticWrite === false))) result.push(...turn)
    turn = []
  }
  for (const event of events) {
    if (event.type === 'user_message') flush()
    turn.push(event)
  }
  flush()
  return result
}
