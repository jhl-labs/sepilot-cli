import { createHash } from 'node:crypto'
import type { Persona } from '../persona/schema.js'
import { memoryResetOwner } from './reset.js'

export const PERSONA_MEMORY_PREFIX = 'scope:persona:'

export function personaMemoryTags(namespace: string): string[] {
  if (!/^[a-f0-9]{64}$/.test(namespace)) throw new Error('Invalid persona memory namespace')
  return [`scope:user:persona-${namespace}`, `${PERSONA_MEMORY_PREFIX}${namespace}`]
}

/** A persona owns a separate memory identity for each durable caller. */
export function resolvePersonaMemoryScope(
  baseTags: string[],
  selection: { persona?: string; personaIds?: readonly string[] },
  catalog: readonly Pick<Persona, 'id' | 'memoryScope'>[],
  session?: { memoryNamespace?: string; messageCount?: number } | null,
): { scopeTags: string[]; memoryNamespace?: string } {
  const ids = [...new Set([selection.persona, ...(selection.personaIds ?? [])].filter((id): id is string => Boolean(id)))]
  const isolated = ids.filter(id => catalog.some(persona => persona.id === id && persona.memoryScope === 'isolated'))
  if (isolated.length && ids.length !== 1) throw new Error('An isolated persona needs its own conversation; it cannot share a persona panel')
  const memoryNamespace = isolated.length
    ? createHash('sha256').update(JSON.stringify([memoryResetOwner(baseTags), isolated[0]])).digest('hex')
    : undefined
  if (session && (session.memoryNamespace || (session.messageCount ?? 0) > 0) && session.memoryNamespace !== memoryNamespace) {
    throw new Error('This conversation belongs to a different memory space. Start a new conversation for this persona')
  }
  return memoryNamespace
    ? { memoryNamespace, scopeTags: personaMemoryTags(memoryNamespace) }
    : { scopeTags: baseTags }
}

/** Background runs inherit the persisted boundary, never a transport's default owner. */
export function inheritedPersonaMemory(session: { memoryNamespace?: string; personaIds?: readonly string[] } | null | undefined, catalog: readonly Persona[]) {
  if (!session?.memoryNamespace) return undefined
  const persona = session.personaIds?.length === 1 ? catalog.find(item => item.id === session.personaIds![0] && item.memoryScope === 'isolated') : undefined
  if (!persona) throw new Error('The scheduled conversation no longer has its isolated persona')
  return { persona, memoryNamespace: session.memoryNamespace, scopeTags: personaMemoryTags(session.memoryNamespace) }
}
