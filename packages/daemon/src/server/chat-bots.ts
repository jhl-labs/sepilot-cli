import type { RoleBotProfile } from '../agent/bots/types.js'
import type { RoleBotRegistry } from '../agent/bots/registry.js'
import type { PersistedPersonaRecord } from '../agent/custom/persona-resolver.js'
import type { ModeRouterOptions } from '../agent/mode-router.js'
import type { IntentRoutingResult } from './routes/chat-intent.js'
import type { ToolExposureGroup } from '../tools/role-filter.js'
import { BUILT_IN_PERSONAS, type Persona } from '../agent/personas.js'

export interface RoleBotRequestFields {
  bot?: string
  bots?: readonly string[]
  persona?: string
  personaIds?: readonly string[]
  mode?: string
  skillRefs?: ReadonlyArray<{ name: string }>
  maxIterations?: number
  ragEnabled?: boolean
  thinkingLevel?: string
}

export interface RoleBotRequestError {
  code: 'BOT_NOT_FOUND' | 'BOT_EXECUTOR_ONLY'
  message: string
  unknownIds: string[]
}

export const MAX_MULTICAST_BOTS = 6

export function applyRoleBotRequestDefaults<T extends RoleBotRequestFields>(
  body: T,
  registry: Pick<RoleBotRegistry, 'get'> | undefined,
): { body: T; bot?: RoleBotProfile; error?: RoleBotRequestError } {
  if (!registry) return { body }
  const multicast = (body.bots ?? []).map((id) => id.trim()).filter(Boolean)
  if (multicast.length > 0) {
    const unknownIds = multicast.filter((id) => !registry.get(id))
    if (unknownIds.length > 0) return { body, error: notFound(unknownIds) }
    if (multicast.length === 1) return applyRoleBotRequestDefaults({ ...body, bots: undefined, bot: multicast[0] }, registry)
    return {
      body: {
        ...body,
        bots: undefined,
        mode: 'persona-panel',
        personaIds: [...new Set(multicast)].slice(0, MAX_MULTICAST_BOTS),
        panelStrategy: (body as { panelStrategy?: string }).panelStrategy ?? 'sequential',
      },
    }
  }
  const id = body.bot?.trim()
  if (!id) return { body }
  const bot = registry.get(id)
  if (!bot) return { body, error: notFound([id]) }
  if (bot.engine.startsWith('external:')) {
    return {
      body,
      error: {
        code: 'BOT_EXECUTOR_ONLY',
        message: `Bot "${bot.id}" is an executor and cannot own a chat turn. Use bot "orchestrator" and name it in the request, or delegate with subagent.dispatch(bot="${bot.id}").`,
        unknownIds: [],
      },
    }
  }
  const callerMode = body.mode?.trim()
  return {
    bot,
    body: {
      ...body,
      bot: undefined,
      persona: bot.id,
      mode: callerMode && callerMode !== 'auto' ? callerMode : bot.engine,
      maxIterations: body.maxIterations ?? bot.maxIterations,
      ragEnabled: body.ragEnabled ?? bot.documents,
      thinkingLevel: body.thinkingLevel ?? bot.thinkingLevel,
    },
  }
}

function notFound(unknownIds: string[]): RoleBotRequestError {
  return {
    code: 'BOT_NOT_FOUND',
    message: `Unknown bot${unknownIds.length > 1 ? 's' : ''}: ${unknownIds.join(', ')}. List available bots with GET /api/v1/bots.`,
    unknownIds,
  }
}

export function keepContinuingRoleBot(
  routing: IntentRoutingResult,
  previousBotId: string | undefined,
  registry: Pick<RoleBotRegistry, 'get'> | undefined,
): IntentRoutingResult {
  if (routing.decision.contractRelation !== 'continue' || !previousBotId) return routing
  if (!routing.decision.fallback && (routing.decision.confidence === 'medium' || routing.decision.confidence === 'high')) return routing
  if (routing.effectivePersona === previousBotId || !registry?.get(previousBotId)) return routing
  return {
    ...routing,
    effectivePersona: previousBotId,
    decision: { ...routing.decision, persona: previousBotId },
  }
}

export function applyRoutedRoleBot(
  routing: IntentRoutingResult,
  registry: Pick<RoleBotRegistry, 'get'> | undefined,
): { routing: IntentRoutingResult; bot?: RoleBotProfile } {
  const bot = registry?.get(routing.effectivePersona)
  if (!bot) return { routing }
  if (routing.effectiveMode !== undefined && routing.effectiveMode !== 'auto') return { routing, bot }
  return {
    bot,
    routing: {
      ...routing,
      effectiveMode: bot.engine,
      decision: {
        ...routing.decision,
        mode: bot.engine,
        confidence: 'high',
        fallback: false,
        ...(bot.toolGroups ? { toolGroups: [...bot.toolGroups] as ToolExposureGroup[] } : {}),
      },
    },
  }
}

export function roleBotSkillRefs(
  bot: RoleBotProfile | undefined,
  existing: ReadonlyArray<{ name: string }>,
): Array<{ name: string }> {
  const present = new Set(existing.map((ref) => ref.name))
  return (bot?.skills ?? []).filter((name) => !present.has(name)).map((name) => ({ name }))
}

export function roleBotModeRouterOptions(
  bot: RoleBotProfile | undefined,
): Pick<ModeRouterOptions, 'stagePolicy' | 'memoryRetrieval'> {
  if (!bot) return {}
  return {
    stagePolicy: bot.stages,
    memoryRetrieval: bot.memory !== 'off',
  }
}

export function roleBotLongTermInjectKb(bot: RoleBotProfile | undefined): number | undefined {
  if (!bot) return undefined
  if (bot.memory === 'off') return 0
  if (bot.memory === 'ambient') return 2
  return 8
}

export function roleBotToolGroups(
  bot: RoleBotProfile | undefined,
  routed: readonly ToolExposureGroup[] | undefined,
): ToolExposureGroup[] | undefined {
  if (bot?.toolGroups) return [...bot.toolGroups]
  return routed ? [...routed] : undefined
}

export function withRoleBotPersonas(
  persisted: readonly PersistedPersonaRecord[],
  registry: Pick<RoleBotRegistry, 'list'> | undefined,
): PersistedPersonaRecord[] {
  if (!registry) return [...persisted]
  const persistedIds = new Set(persisted.map((record) => record.id))
  const bots = registry.list()
    .filter((bot) => !persistedIds.has(bot.id))
    .map((bot): PersistedPersonaRecord => ({
      id: bot.id,
      name: bot.name,
      description: bot.description,
      systemPrompt: bot.systemPrompt,
      allowedTools: bot.allowedTools,
      deniedTools: bot.deniedTools,
      routable: bot.routable,
    }))
  return [...persisted, ...bots]
}

export function routerPersonaCandidates(
  catalog: readonly Persona[],
  registry: Pick<RoleBotRegistry, 'get'> | undefined,
  userDefinedIds: ReadonlySet<string>,
): Persona[] {
  if (!registry) return [...catalog]
  const catchAll = Boolean(registry.get(CATCH_ALL_BOT_ID))
  return catalog.filter((persona) =>
    (persona.id === 'default' && !catchAll)
    || Boolean(registry.get(persona.id))
    || userDefinedIds.has(persona.id)
    || !BUILTIN_PERSONA_IDS.has(persona.id))
}

export const CATCH_ALL_BOT_ID = 'general-assistant'

const BUILTIN_PERSONA_IDS: ReadonlySet<string> = new Set(BUILT_IN_PERSONAS.map((persona) => persona.id))
