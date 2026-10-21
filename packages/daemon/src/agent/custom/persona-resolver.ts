import { assertCustomAgentUsable, type CustomAgentRecord } from './agents.js'
import type { Persona } from '../personas.js'
import { getPersona, listPersonas } from '../personas.js'

export interface PersistedPersonaRecord {
  id: string
  name: string
  description?: string
  systemPrompt: string
  memoryScope?: 'shared' | 'isolated'
  allowedTools?: readonly string[]
  deniedTools?: readonly string[]
  routable?: boolean
}

export function customAgentToPersona(record: CustomAgentRecord): Persona {
  assertCustomAgentUsable(record)
  if (record.isolation)
    throw new Error(`Agent "${record.id}" requires worktree isolation; use subagent dispatch, not a primary persona`)
  return {
    id: record.id,
    name: record.description ?? record.id,
    description: record.description ?? '',
    systemPromptAddition: record.systemPrompt,
    ...(record.allowedTools ? { allowedTools: record.allowedTools } : {}),
    ...(record.deniedTools ? { deniedTools: record.deniedTools } : {}),
  }
}

export function persistedPersonaToPersona(record: PersistedPersonaRecord): Persona {
  return {
    id: record.id,
    name: record.name,
    description: record.description ?? record.name,
    ...(record.allowedTools ? { allowedTools: [...record.allowedTools] } : {}),
    ...(record.deniedTools ? { deniedTools: [...record.deniedTools] } : {}),
    systemPromptAddition: record.systemPrompt + (record.memoryScope === 'isolated' ? '\n\nYou have an isolated persona memory space. Memory tools and Activity Journal here belong only to this character and this user; ordinary chat memories are separate. Recall relevant character history before claiming continuity. Record user-approved character traits, preferences and relationship conventions as fictional character identity, separately from observed user facts and actual shared events. Use memory.remember with subject=persona and reality=fictional for agreed character settings, subject=user and reality=real for actual user facts, and subject=relationship with the appropriate reality for shared conventions. Preserve these evidence fields with memory.update for durable agreed changes, and memory.daily.append for significant episodes. Do not turn invented backstory into evidence of real events. Current explicit character settings and user corrections override older character memories. Do not silently rewrite the base persona or store secrets. Use memory.forget when asked to forget; only claim persistence or deletion after successful tool results.' : ''),
  }
}

export function resolvePersona(
  id: string | undefined,
  customAgents: readonly CustomAgentRecord[] = [],
  persistedPersonas: readonly PersistedPersonaRecord[] = [],
): Persona | undefined {
  if (!id) return undefined
  const custom = customAgents.find(a => a.id === id)
  if (custom) return customAgentToPersona(custom)
  const persisted = persistedPersonas.find(persona => persona.id === id)
  if (persisted) return persistedPersonaToPersona(persisted)
  return getPersona(id)
}

export function resolvePersonaCatalog(
  customAgents: readonly CustomAgentRecord[] = [],
  persistedPersonas: readonly PersistedPersonaRecord[] = [],
): Persona[] {
  const catalog = new Map(listPersonas().map((persona) => [persona.id, persona]))
  for (const persisted of persistedPersonas) {
    catalog.set(persisted.id, persistedPersonaToPersona(persisted))
  }
  for (const custom of customAgents) {
    if (custom.issues?.length || custom.isolation) catalog.delete(custom.id)
    else catalog.set(custom.id, customAgentToPersona(custom))
  }
  const excludedIds = new Set(persistedPersonas
    .filter(persona => persona.memoryScope === 'isolated' || persona.routable === false)
    .map(persona => persona.id))
  return [...catalog.values()].filter(persona => !excludedIds.has(persona.id))
}

export function resolvePersonas(
  ids: readonly string[] | undefined,
  customAgents: readonly CustomAgentRecord[] = [],
  persistedPersonas: readonly PersistedPersonaRecord[] = [],
): Persona[] {
  if (!ids || ids.length === 0) return []
  const seen = new Set<string>()
  const result: Persona[] = []
  for (const id of ids) {
    if (seen.has(id)) continue
    seen.add(id)
    const persona = resolvePersona(id, customAgents, persistedPersonas)
    if (persona) result.push(persona)
  }
  return result
}
