import { isTaskInvocation } from '../../agent/task-invocation.js'
import type { Message } from '@sepilotd/core'
import type { GraphAgentRegistry } from '../../agent/graph/registry.js'
import type { IntentDecision, IntentHints } from '../../agent/intent-router.js'
import type { IntentRouter } from '../../agent/intent-router.js'
import { makeDefaultDecision } from '../../agent/intent-router.js'
import { resolveAutoFallbackMode } from '../../agent/mode-catalog.js'
import { ProviderCircuitBreaker } from '../../providers/circuit-breaker.js'

export const INTENT_ROUTER_CIRCUIT_BREAKER = new ProviderCircuitBreaker({
  failureThreshold: 5,
  openDurationMs: 5 * 60_000,
})

export interface MinimalLogger {
  info: (msg: string, data?: Record<string, unknown>) => void
}

export interface MinimalWarnLogger extends MinimalLogger {
  warn: (msg: string, data?: Record<string, unknown>) => void
}

const routerWarnState = {
  noProvider: false,
  noModel: false,
  activeModel: false,
}

export function resolveIntentRouterModel(args: {
  routerProvider:
    | {
        id: string
        models: ReadonlyArray<{
          id: string
          capabilities?: { embedding?: boolean; toolUse?: boolean }
        }>
      }
    | undefined
  configuredModel: string | undefined
  configuredProviderId: string | undefined
  activeModel?: string | undefined
  logger: MinimalWarnLogger
}): string | undefined {
  if (!args.routerProvider) {
    if (!routerWarnState.noProvider) {
      args.logger.warn('intent_router.config_missing_provider', {
        configuredProvider: args.configuredProviderId ?? '(default)',
        note: 'agent.intentRouter.enabled=true but no matching provider was registered; routing will be skipped',
      })
      routerWarnState.noProvider = true
    }
    return undefined
  }
  if (args.configuredModel) return args.configuredModel
  const activeModel = args.activeModel?.trim()
  const activeModelInfo = activeModel
    ? args.routerProvider.models.find((model) => model.id === activeModel)
    : undefined
  if (
    activeModelInfo
    && !(
      activeModelInfo.capabilities?.embedding === true
      && activeModelInfo.capabilities?.toolUse !== true
    )
  ) {
    if (!routerWarnState.activeModel) {
      args.logger.warn('intent_router.config_using_active_model', {
        provider: args.routerProvider.id,
        model: activeModelInfo.id,
        note: 'No dedicated agent.intentRouter.model is configured; the active turn model will perform the LLM routing decision.',
      })
      routerWarnState.activeModel = true
    }
    return activeModelInfo.id
  }
  if (!routerWarnState.noModel) {
    args.logger.warn('intent_router.config_missing_model', {
      provider: args.routerProvider.id,
      activeModel: activeModel || '(none)',
      note: 'agent.intentRouter.enabled=true but neither a dedicated router model nor a compatible active turn model is available; LLM routing will be skipped',
    })
    routerWarnState.noModel = true
  }
  return undefined
}

export function __resetIntentRouterWarnState(): void {
  routerWarnState.noProvider = false
  routerWarnState.noModel = false
  routerWarnState.activeModel = false
}

export interface ApplyIntentRoutingDeps {
  defaultMode?: string
  router: IntentRouter | null
  graphRegistry?: GraphAgentRegistry
  message: string
  prevMessages: ReadonlyArray<Message>
  body: {
    mode?: string
    persona?: string
    skillRefs?: ReadonlyArray<{ name: string }>
    intentRouting?: { enabled?: boolean }
  }
  isMultimodal?: boolean
}

export interface IntentRoutingResult {
  effectiveMode: string | undefined
  effectivePersona: string | undefined
  effectiveSkillRefs: Array<{ name: string }>
  decision: IntentDecision
}

export function resolveInitialExecutionMode(
  routing: IntentRoutingResult,
  defaultMode: string,
  activeRemoteBrowser = false,
  surface?: string,
): string {
  const selected = routing.effectiveMode ?? defaultMode
  if (selected !== 'auto') return selected
  return !routing.decision.fallback && routing.decision.confidence === 'high'
    ? routing.decision.mode : resolveAutoFallbackMode({ surface, activeRemoteBrowser })
}

export async function applyIntentRouting(deps: ApplyIntentRoutingDeps): Promise<IntentRoutingResult> {
  const hints: IntentHints = {
    mode: deps.body.mode ?? deps.defaultMode ?? 'instant',
    persona: deps.body.persona,
    skillRefs: deps.body.skillRefs,
  }

  const taskSelected = isTaskInvocation(deps.message) || hints.skillRefs?.some(ref => ref.name === 'task')
  if (taskSelected && (!hints.mode || hints.mode === 'instant' || hints.mode === 'auto')) {
    const decision = makeDefaultDecision({ ...hints, mode: 'react' }, 'explicit task management command')
    decision.toolGroups = ['scheduler-advanced']
    decision.fallback = false
    decision.skipped = true
    return {
      effectiveMode: 'react',
      effectivePersona: hints.persona,
      effectiveSkillRefs: [...(hints.skillRefs ?? [])],
      decision,
    }
  }

  if (deps.body.intentRouting?.enabled === false || !deps.router) {
    return disabledPath(hints, 'router disabled')
  }

  const explicitMode = hints.mode?.trim()

  if (explicitMode && explicitMode !== 'auto') {
    const decision = makeDefaultDecision(
      { ...hints, mode: explicitMode },
      'explicit caller mode selected',
    )
    decision.fallback = false
    decision.skipped = true
    return {
      effectiveMode: explicitMode,
      effectivePersona: hints.persona,
      effectiveSkillRefs: decision.skillIds.map((name) => ({ name })),
      decision,
    }
  }

  if (deps.isMultimodal) {
    const effectiveMode = 'react'
    return {
      effectiveMode,
      effectivePersona: deps.body.persona,
      effectiveSkillRefs: [...(deps.body.skillRefs ?? [])],
      decision: {
        ...makeDefaultDecision(
          { ...hints, mode: effectiveMode },
          'multimodal turn routed to react',
        ),
        mode: effectiveMode,
      },
    }
  }

  const decision = await deps.router.decide(deps.message, deps.prevMessages, hints)
  const hintedMode = hints.mode?.trim()
  const explicitlySelectedSkills = hints.skillRefs?.length
    ? [...hints.skillRefs]
    : undefined
  const effectiveDecision = explicitlySelectedSkills
    ? { ...decision, skillIds: explicitlySelectedSkills.map(({ name }) => name) }
    : decision
  const effectiveMode =
    !hintedMode || hintedMode === 'auto'
      ? 'auto'
      : hintedMode
  return {
    effectiveMode,
    effectivePersona: effectiveDecision.persona,
    effectiveSkillRefs: explicitlySelectedSkills
      ?? effectiveDecision.skillIds.map(name => ({ name })),
    decision: effectiveDecision,
  }
}

export function logIntentRouterDecision(
  logger: MinimalLogger,
  routing: IntentRoutingResult,
  hints: { mode?: string; persona?: string; skillCount: number },
  messageLength: number,
  sessionId: string,
  routerModel: string | undefined,
): void {
  if (routing.decision.skipped || (routing.decision.fallback && ROUTER_NOT_INVOKED_REASONS.has(routing.decision.reason))) {
    logger.info('intent_router.skipped', {
      event: 'intent_router.skipped',
      sessionId,
      messageLength,
      mode: routing.effectiveMode,
      reason: routing.decision.reason,
    })
    return
  }
  logger.info('intent_router.decision', {
    event: 'intent_router.decision',
    sessionId,
    messageLength,
    hints,
    decision: {
      mode: routing.decision.mode,
      persona: routing.decision.persona,
      skillIds: routing.decision.skillIds,
      toolGroups: routing.decision.toolGroups,
      confidence: routing.decision.confidence,
      fallback: routing.decision.fallback,
    },
    latencyMs: routing.decision.latencyMs,
    routerModel,
    routerTokens: routing.decision.routerTokens,
  })
}

const ROUTER_NOT_INVOKED_REASONS: ReadonlySet<string> = new Set([
  'router disabled',
])

export function routingLeavesMemoryIntentUnresolved(decision: Pick<IntentDecision, 'fallback' | 'skipped' | 'reason'>): boolean {
  return decision.fallback === true
    && decision.skipped !== true
    && !ROUTER_NOT_INVOKED_REASONS.has(decision.reason)
}

function disabledPath(hints: IntentHints, reason: string): IntentRoutingResult {
  const decision = makeDefaultDecision(hints, reason)
  return {
    effectiveMode: hints.mode ?? 'auto',
    effectivePersona: hints.persona,
    effectiveSkillRefs: decision.skillIds.map(name => ({ name })),
    decision,
  }
}
