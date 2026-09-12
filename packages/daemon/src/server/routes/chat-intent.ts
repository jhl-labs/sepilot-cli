import { isTaskInvocation } from '../../agent/task-invocation.js'
import type { Message } from '@sepilotd/core'
import type { GraphAgentRegistry } from '../../agent/graph/registry.js'
import type { IntentDecision, IntentHints } from '../../agent/intent-router.js'
import type { IntentRouter } from '../../agent/intent-router.js'
import { makeDefaultDecision } from '../../agent/intent-router.js'
import { ProviderCircuitBreaker } from '../../providers/circuit-breaker.js'

/**
 * Shared circuit breaker for the intent router provider. Imported by both
 * chat.ts and chat-stream.ts so failure state accumulates across all routing
 * turns in the daemon process — not per-handler. Spec §7.2: dedicated
 * breaker, 5 consecutive failures trip a 5-minute cool-down.
 */
export const INTENT_ROUTER_CIRCUIT_BREAKER = new ProviderCircuitBreaker({
  failureThreshold: 5,
  openDurationMs: 5 * 60_000,
})

/** Minimal structural logger shape used by the helpers in this module. */
export interface MinimalLogger {
  info: (msg: string, data?: Record<string, unknown>) => void
}

/** Logger shape with warn support, used by router-init helpers. */
export interface MinimalWarnLogger extends MinimalLogger {
  warn: (msg: string, data?: Record<string, unknown>) => void
}

// One-shot warn flags so misconfiguration surfaces once per daemon process
// instead of on every chat turn.
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
  /** Active turn model, used only when no dedicated router model is configured. */
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

/** Test-only: reset the one-shot warn state so a fresh process can be simulated. */
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
  /**
   * True only when the turn carries provider-native image/audio/binary
   * document parts and the router must not classify on the plain-text portion.
   * Extracted text attachments remain eligible for graph-backed routing.
   */
  isMultimodal?: boolean
}

export interface IntentRoutingResult {
  /** Resolved agent mode. `undefined` means "use the runtime default". */
  effectiveMode: string | undefined
  effectivePersona: string | undefined
  effectiveSkillRefs: Array<{ name: string }>
  decision: IntentDecision
}

/** Match the dispatcher without reclassifying natural-language input. */
export function resolveInitialExecutionMode(routing: IntentRoutingResult, defaultMode: string, activeRemoteBrowser = false): string {
  const selected = routing.effectiveMode ?? defaultMode
  if (selected !== 'auto') return selected
  return !routing.decision.fallback && routing.decision.confidence === 'high'
    ? routing.decision.mode : activeRemoteBrowser ? 'react' : 'instant'
}

export async function applyIntentRouting(deps: ApplyIntentRoutingDeps): Promise<IntentRoutingResult> {
  const hints: IntentHints = {
    mode: deps.body.mode ?? deps.defaultMode ?? 'instant',
    persona: deps.body.persona,
    skillRefs: deps.body.skillRefs,
  }

  // An explicit command selects a capability surface without paying for an
  // intent classifier. The main agent still judges read/create/update/clarify.
  // Preserve concrete specialist mode choices and all caller tool boundaries.
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

  // A concrete mode selects the starting strategy. Skip the classifier so
  // lightweight turns do not
  // pay for an irrelevant routing-model call.
  if (explicitMode && explicitMode !== 'auto') {
    // Not a fallback: the caller chose. Labelling it low-confidence/fallback
    // widened the tool surface as if classification had failed and showed
    // operators a "router failed" row for every explicit-mode turn.
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
    // The intent router is intentionally cheap and does not see the durable
    // run contract. For default/auto turns, let AgentModeRouter perform the
    // contract-aware graph selection; keep the intent decision for persona and
    // skills. A concrete caller mode is an execution boundary, not a hint the
    // cheap classifier may replace.
    effectiveMode,
    effectivePersona: effectiveDecision.persona,
    effectiveSkillRefs: explicitlySelectedSkills
      ?? effectiveDecision.skillIds.map(name => ({ name })),
    decision: effectiveDecision,
  }
}

/**
 * Emit the structured `intent_router.decision` log entry.
 *
 * Extracted to avoid the ~26-line duplication between chat.ts and
 * chat-stream.ts — both handlers call this with slightly different `hints`
 * (body.persona vs. the destructured `persona` variable).
 */
export function logIntentRouterDecision(
  logger: MinimalLogger,
  routing: IntentRoutingResult,
  hints: { mode?: string; persona?: string; skillCount: number },
  messageLength: number,
  sessionId: string,
  routerModel: string | undefined,
): void {
  // An explicit caller mode or a disabled router never invoked the classifier.
  // Logging that as a `fallback: true` decision inflates router-failure
  // metrics with turns the router was never asked about.
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

/** Synthetic decision reasons produced in this module when the router is not consulted. */
const ROUTER_NOT_INVOKED_REASONS: ReadonlySet<string> = new Set([
  'router disabled',
])

function disabledPath(hints: IntentHints, reason: string): IntentRoutingResult {
  const decision = makeDefaultDecision(hints, reason)
  return {
    // Preserve concrete caller mode hints, but keep unhinted/default turns in
    // AgentModeRouter's contract-aware auto path instead of silently falling
    // back to a static runtime default.
    effectiveMode: hints.mode ?? 'auto',
    effectivePersona: hints.persona,
    effectiveSkillRefs: decision.skillIds.map(name => ({ name })),
    decision,
  }
}
