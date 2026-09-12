import type { DaemonChatStreamPayload, DaemonMemoryContextItem } from './types.js'
import type { RunStopReason } from '@sepilotd/core'
import type {
  AgentState,
  ActivityItem,
  Message,
  StreamEventControllerBindings,
  StreamEventControllerOptions,
} from './chat-surface-types.js'
import {
  applyApprovalResolution,
  createApprovalResolutionActivity,
  formatAcceptanceCriteriaCount,
  formatMemoryContextMessage,
  formatRunContractDetail,
  formatToolCall,
  formatToolInput,
  normalizeUsage,
  summarizeText,
  summarizeMemoryContextItems,
  toolMetaFromResultStatus,
} from './chat-surface-utils.js'

const INTERIM_PROGRESS_PATTERNS = [
  /\b(?:i['’]ll|i will|let me|i am going to|i'm going to)\s+(?:check|collect|gather|fetch|look|inspect|review|read|run|verify|describe|analyze|diagnose)\b/i,
  /\b(?:retry|retrying|rerun(?:ning)?|trying again)\b/i,
  /\bnow\s+(?:checking|collecting|gathering|fetching|looking|inspecting|reviewing|reading|running|verifying|describing|analyzing|diagnosing)\b/i,
  /(?:먼저|이제|다음(?:으로)?|나머지(?:를|도)?|다시|재차)\s*.*(?:확인|수집|조회|살펴|검토|가져오|읽어보|실행|재실행|시도|정리|파악|분석|describe|작성|저장)\s*(?:하겠|해보겠|해 보겠)/i,
  /(?:다른|별도|대체)\s*(?:방법|명령어|커맨드|경로)?(?:으로|들을|를)?\s*.*(?:확인|수집|조회|살펴|검토|가져오|읽어보|실행|재실행|시도|정리|파악|분석|진단|describe|작성|저장)\s*(?:하겠|해보겠|해 보겠)/i,
]

function hasAnswerProtocolStem(content: string): boolean {
  return /^\s*(?:ANSWER|INCOMPLETE):/i.test(content)
}

function isLikelyInterimProgressText(content: string): boolean {
  const normalized = content.trim().replace(/\s+/g, ' ')
  if (!normalized || normalized.length > 900) {
    return false
  }

  return INTERIM_PROGRESS_PATTERNS.some((pattern) => pattern.test(normalized))
}

function stateChangeActivityCopy(state: AgentState): {
  label: string
  detail: string
} {
  switch (state) {
    case 'thinking':
      return {
        label: 'Thinking',
        detail: 'Preparing the next step and waiting for model output.',
      }
    case 'acting':
      return {
        label: 'Acting',
        detail: 'Running a tool or applying a requested action.',
      }
    case 'observing':
      return {
        label: 'Observing',
        detail: 'Reading tool output and deciding the next step.',
      }
    case 'done':
      return {
        label: 'Done',
        detail: 'Run finished.',
      }
    case 'error':
      return {
        label: 'Error',
        detail: 'Run failed.',
      }
    default:
      return {
        label: 'Idle',
        detail: 'Waiting for input.',
      }
  }
}

function formatPanelRosterSummary(
  personas: Array<{ id: string; name: string; description?: string }>,
): string {
  if (personas.length === 0) return 'No resolved panelists.'
  return personas
    .map((persona, index) =>
      `${index + 1}. ${persona.name}${persona.description ? ` - ${persona.description}` : ''}`,
    )
    .join('\n')
}

function formatDebateRoundContent(
  round: Extract<DaemonChatStreamPayload, { type: 'debate_round' }>['round'],
): string {
  return [
    `Debate round: ${summarizeText(round.topic, 120)}`,
    `Decision: ${round.finalDecision}`,
    round.rationale ? `Rationale: ${summarizeText(round.rationale, 240)}` : '',
  ]
    .filter(Boolean)
    .join('\n\n')
}

function formatSkillMeta(skillIds: string[] | undefined): string | undefined {
  return skillIds && skillIds.length > 0 ? `skills: ${skillIds.join(', ')}` : undefined
}

function formatModeRouteMeta(event: Extract<DaemonChatStreamPayload, { type: 'mode_route_decision' }>): string {
  const confidence = event.confidence === undefined ? 'confidence n/a' : `confidence ${event.confidence.toFixed(2)}`
  const fallback = event.fallback ? ' · fallback' : ''
  const candidates = event.candidates?.length ? ` · candidates: ${event.candidates.join(', ')}` : ''
  return `${confidence}${fallback}${candidates}`
}

function formatCheckpointDetail(files: Array<{ path: string }>): string {
  if (files.length === 0) return 'No files captured'
  const visible = files.slice(0, 3).map((file) => file.path).join(', ')
  const more = files.length > 3 ? `, +${files.length - 3} more` : ''
  return `${files.length} file${files.length === 1 ? '' : 's'} captured: ${visible}${more}`
}

export function createStreamEventController(
  bindings: StreamEventControllerBindings,
  options: StreamEventControllerOptions,
): {
  getAssistantContent: () => string
  /** Structured stop cause of the last terminal event, if the daemon sent one. */
  getStopReason: () => RunStopReason | null
  updateAssistant: (content: string, usage?: Message['usage']) => void
  handleEvent: (event: DaemonChatStreamPayload) => void
} {
  const maxActivityItems = options.maxActivityItems ?? 24
  let assistantContent = ''
  let stopReason: RunStopReason | null = null
  const recordStopReason = (reason: RunStopReason | undefined) => {
    if (!reason) return
    stopReason = reason
    bindings.setStopReason?.(reason)
  }
  let assistantCitations: Message['citations']
  let lastSupersededProgress: string | null = null
  let lastToolOutcome: { toolName: string; status: 'success' | 'error'; detail: string } | null =
    null
  const successfulToolOutcomes: Array<{ toolName: string; detail: string }> = []
  const toolNamesById = new Map<string, string>()
  const seenToolCallIds = new Set<string>()
  const toolResultSignatureById = new Map<string, string>()
  let completionHandled = false
  const now = options.now ?? Date.now
  const thinkingUiIntervalMs = Math.max(0, options.thinkingUiIntervalMs ?? 100)
  let thinkingSegment = ''
  let thinkingUiDirty = false
  let lastThinkingUiAt = Number.NEGATIVE_INFINITY

    const buildCompletionAssistantContent = (): string => {
      const trimmedAssistantContent = assistantContent.trim()
      const hasProtocolStem = hasAnswerProtocolStem(trimmedAssistantContent)
      const progressSummary =
        trimmedAssistantContent
        && !hasProtocolStem
        && isLikelyInterimProgressText(trimmedAssistantContent)
          ? summarizeText(trimmedAssistantContent)
          : lastSupersededProgress

      if (
        trimmedAssistantContent
        && (hasProtocolStem || !isLikelyInterimProgressText(trimmedAssistantContent))
      ) {
        return assistantContent
      }

    if (lastToolOutcome) {
      if (successfulToolOutcomes.length > 0) {
        const evidence = successfulToolOutcomes
          .slice(-3)
          .map((outcome) => `- ${outcome.toolName}: ${outcome.detail}`)
          .join('\n')
        const failedFollowUp = lastToolOutcome.status === 'error'
          ? [
              `Failed follow-up check: ${lastToolOutcome.toolName}.`,
              lastToolOutcome.detail === 'No details' ? '' : lastToolOutcome.detail,
            ].filter(Boolean).join(' ')
          : ''
        const progress = progressSummary ? `\nLast progress update: ${progressSummary}.` : ''
        return [
          'Final reply missing after the run completed.',
          'Successful tool evidence:',
          evidence,
          failedFollowUp,
          progress,
        ].filter(Boolean).join('\n')
      }
      const outcomeLabel =
        lastToolOutcome.status === 'success' ? 'completed successfully' : 'completed with an error'
      const detail =
        lastToolOutcome.detail && lastToolOutcome.detail !== 'No details'
          ? ` Last tool output: ${lastToolOutcome.detail}`
          : ''
      const progress = progressSummary ? ` Last progress update: ${progressSummary}.` : ''
      return `Final reply missing after the run completed. ${lastToolOutcome.toolName} ${outcomeLabel}.${progress}${detail}`
    }

    if (progressSummary) {
      return `Final reply missing after the run completed. Last progress update: ${progressSummary}.`
    }

    return options.doneFallbackText
  }

  const clearSupersededProgress = () => {
    if (hasAnswerProtocolStem(assistantContent) || !isLikelyInterimProgressText(assistantContent)) {
      return
    }
    lastSupersededProgress = summarizeText(assistantContent)
    pushActivity({
      id: options.createId(),
      kind: 'state',
      label: 'Progress update',
      detail: lastSupersededProgress,
      status: 'neutral',
      meta: 'Superseded',
    })
    options.onSupersededProgress?.(lastSupersededProgress)
    assistantContent = ''
    updateAssistant('')
  }

  const pushActivity = (nextItem: ActivityItem) => {
    bindings.setActivities((prev) => [...prev.slice(-(maxActivityItems - 1)), nextItem])
  }

  // Tool/state/phase activities are pushed with a `running` (or `pending`)
  // status and never get a terminal update — `tool_result` etc. push a fresh
  // `result` item rather than reconciling the original. Once the run reaches a
  // terminal state nothing is actually running anymore, so leaving those items
  // as `running`/`pending` makes the activity summary report nonsense like
  // "14 running · 2 waiting" long after the run finished. Settle them to a
  // neutral status so the footer reflects reality.
  const settleLingeringActivities = () => {
    bindings.setActivities((prev) => {
      let changed = false
      const next = prev.map((item) => {
        if (item.status === 'running' || item.status === 'pending') {
          changed = true
          return { ...item, status: 'neutral' as const }
        }
        return item
      })
      return changed ? next : prev
    })
  }

  const settleToolActivity = (
    toolName: string,
    status: 'success' | 'error',
  ) => {
    bindings.setActivities((prev) => {
      const index = prev.findLastIndex((item) =>
        item.kind === 'tool'
        && item.label === toolName
        && item.status === 'running'
      )
      if (index < 0) return prev
      const next = [...prev]
      next[index] = { ...next[index]!, status }
      return next
    })
  }

  const updateThinkingActivity = (content: string) => {
    const detail = summarizeText(content)
    bindings.setActivities((prev) => {
      const trimmed = prev.slice(-(maxActivityItems - 1))
      const lastItem = trimmed[trimmed.length - 1]
      if (lastItem && lastItem.kind === 'thinking' && lastItem.status === 'running') {
        const next = [...trimmed]
        next[next.length - 1] = {
          ...lastItem,
          detail,
        }
        return next
      }

      return [
        ...trimmed,
        {
          id: options.createId(),
          kind: 'thinking',
          label: 'Reasoning',
          detail,
          status: 'running',
        },
      ]
    })
  }

  const flushThinkingUi = (force = false): void => {
    if (!thinkingUiDirty) return
    const currentTime = now()
    if (!force && currentTime - lastThinkingUiAt < thinkingUiIntervalMs) return
    const visibleThinking = thinkingSegment.slice(-2_000)
    bindings.setAgentState('thinking')
    bindings.setStatus('Reasoning…')
    if (visibleThinking) bindings.setThinking?.(visibleThinking)
    updateThinkingActivity(visibleThinking || 'Reasoning…')
    thinkingUiDirty = false
    lastThinkingUiAt = currentTime
  }

  const upsertToolMessage = (
    toolId: string,
    fallback: { toolName: string; content: string },
    updates: Partial<Message>,
  ) => {
    bindings.setMessages((prev) => {
      const toolIndex = prev.findIndex(
        (message) => message.role === 'tool' && message.id === toolId,
      )

      if (toolIndex === -1) {
        return [
          ...prev,
          {
            id: toolId,
            role: 'tool',
            toolName: fallback.toolName,
            content: fallback.content,
            toolStatus: 'running',
            ...updates,
          },
        ]
      }

      const current = prev[toolIndex]
      const next = [...prev]
      next[toolIndex] = {
        ...current,
        ...updates,
        toolName: current.toolName || fallback.toolName,
        content: current.content || fallback.content,
      }
      return next
    })
  }

  const upsertContextMessage = (contextId: string, items: DaemonMemoryContextItem[]) => {
    const content = formatMemoryContextMessage(items)

    bindings.setMessages((prev) => {
      const contextIndex = prev.findIndex(
        (message) => message.role === 'context' && message.id === contextId,
      )

      if (contextIndex !== -1) {
        const next = [...prev]
        next[contextIndex] = {
          ...next[contextIndex],
          content,
          contextItems: items,
        }
        return next
      }

      const assistantIndex = prev.findIndex(
        (message) => message.role === 'assistant' && message.id === options.assistantId,
      )
      const insertIndex = assistantIndex === -1 ? prev.length : assistantIndex
      return [
        ...prev.slice(0, insertIndex),
        {
          id: contextId,
          role: 'context',
          content,
          contextItems: items,
        },
        ...prev.slice(insertIndex),
      ]
    })
  }

  const appendSurfaceMessage = (
    role: Message['role'],
    content: string,
    metadata: Pick<
      Message,
      | 'personaId'
      | 'personaName'
      | 'questionRequestId'
      | 'questionPrompt'
      | 'questionChoices'
      | 'questionState'
      | 'sessionId'
    > = {},
  ) => {
    bindings.setMessages((prev) => {
      const assistantIndex = prev.findIndex((message) => message.id === options.assistantId)
      const insertIndex = assistantIndex === -1 ? prev.length : assistantIndex
      const message = {
        id: options.createId(),
        role,
        content,
        ...metadata,
      }
      return [
        ...prev.slice(0, insertIndex),
        message,
        ...prev.slice(insertIndex),
      ]
    })
  }

  const updateAssistant = (
    content: string,
    usage?: Message['usage'],
    sessionId?: string,
  ) => {
    assistantContent = content
    bindings.setMessages((prev) =>
      prev.map((message) =>
        message.id === options.assistantId
          ? {
              ...message,
              content,
              citations: assistantCitations ?? message.citations,
              usage: usage ?? message.usage,
              sessionId: sessionId ?? message.sessionId,
            }
          : message,
      ),
    )
  }

  const handleEvent = (event: DaemonChatStreamPayload) => {
    if (!('type' in event)) {
      return
    }

    if (event.type === 'thinking') {
      bindings.setProviderWait?.(null)
      // Reasoning providers commonly stream one token (sometimes one
      // character) per SSE frame. Updating Ink/React state three times for
      // every frame can saturate the terminal renderer while the model/GPU is
      // healthy. Preserve every delta in the transport and trace, but render a
      // bounded rolling segment at most once per interval. The first delta is
      // immediate and the last pending delta is flushed before the next
      // non-thinking event, so sparse streams and final text are never lost.
      thinkingSegment = `${thinkingSegment}${event.content ?? ''}`.slice(-2_000)
      thinkingUiDirty = true
      flushThinkingUi(false)
      return
    }

    flushThinkingUi(true)
    thinkingSegment = ''

    if (
      event.type === 'text_delta'
      || event.type === 'message'
      || event.type === 'tool_call'
      || event.type === 'tool_result'
      || event.type === 'node_trace'
      || event.type === 'done'
      || event.type === 'error'
    ) {
      bindings.setProviderWait?.(null)
    }

    switch (event.type) {
      case 'execution_policy':
        bindings.setExecutionPolicy?.(event.policy)
        bindings.setStatus(
          event.policy.clamped
            ? `Execution policy clamped to ${event.policy.effectiveAutonomy}`
            : `Execution policy: ${event.policy.effectiveAutonomy}`,
        )
        pushActivity({
          id: options.createId(),
          kind: event.policy.clamped ? 'error' : 'state',
          label: 'Execution policy',
          detail: event.policy.clamped
            ? event.policy.clampReason ?? 'The requested autonomy was restricted by daemon policy.'
            : `${event.policy.agentMode} · ${event.policy.workspaceBoundary} workspace`,
          status: event.policy.clamped ? 'error' : 'neutral',
          meta: `${event.policy.configuredAutonomy} → ${event.policy.effectiveAutonomy}`,
        })
        break
      case 'state_change':
        if (event.state === 'thinking' || event.state === 'acting') {
          clearSupersededProgress()
        }
        bindings.setAgentState(event.state)
        if (event.state === 'done' || event.state === 'error') {
          settleLingeringActivities()
        }
        const copy = stateChangeActivityCopy(event.state)
        pushActivity({
          id: options.createId(),
          kind: 'state',
          label: copy.label,
          detail: copy.detail,
          status:
            event.state === 'error' ? 'error' : event.state === 'done' ? 'success' : 'running',
        })
        break
      case 'memory_context': {
        const items = event.items ?? []
        if (items.length === 0) {
          break
        }
        assistantCitations = items
        const summary = summarizeMemoryContextItems(items)
        upsertContextMessage(event.id, items)
        bindings.setMessages((prev) =>
          prev.map((message) =>
            message.id === options.assistantId
              ? {
                  ...message,
                  citations: items,
                }
              : message,
          ),
        )
        pushActivity({
          id: options.createId(),
          kind: 'context',
          label: 'Relevant context',
          detail: summary.detail,
          status: 'neutral',
          meta: summary.meta,
        })
        break
      }
      case 'context_usage':
        bindings.setContextUsage?.(event.context)
        break
      case 'llm_request': {
        const digest = event.requestDigest
        const target = [digest.providerId, digest.model].filter(Boolean).join('/')
        const phase = digest.source ? ` · ${digest.source}` : ''
        const role = digest.auxiliary ? 'Auxiliary model' : 'Main model'
        bindings.setProviderWait?.({
          providerId: digest.providerId,
          model: digest.model,
          source: digest.source,
          auxiliary: digest.auxiliary ?? false,
          startedAt: digest.startedAt ?? Date.now(),
          timeoutMs: digest.timeoutMs,
        })
        bindings.setAgentState('thinking')
        bindings.setStatus(digest.auxiliary ? 'Preparing next step…' : 'Waiting for model response…')
        pushActivity({
          id: options.createId(),
          kind: 'thinking',
          label: `${role} request`,
          detail: `Waiting for ${target || digest.model}${phase}`,
          status: 'running',
          meta: typeof digest.timeoutMs === 'number'
            ? `timeout ${Math.ceil(digest.timeoutMs / 1000)}s`
            : undefined,
        })
        break
      }
      case 'reasoning_step':
        bindings.setAgentState('thinking')
        bindings.setStatus('Reasoning…')
        pushActivity({
          id: options.createId(),
          kind: 'thinking',
          label: `Reasoning: ${event.label}`,
          detail: summarizeText(event.detail ?? event.label),
          status: 'running',
        })
        break
      case 'action_progress': {
        const detail = `${event.summary} → ${event.nextStep}`
        bindings.setAgentState('thinking')
        bindings.setStatus(summarizeText(event.summary, 120))
        pushActivity({
          id: options.createId(),
          kind: 'thinking',
          label: 'Next action',
          detail: summarizeText(detail, 320),
          status: 'running',
          meta: event.toolNames.length > 0 ? event.toolNames.join(' · ') : undefined,
        })
        break
      }
      case 'router_decision':
        bindings.setStatus(`Router: ${event.decision.mode}/${event.decision.persona}`)
        pushActivity({
          id: options.createId(),
          kind: 'state',
          label: `Router: ${event.decision.mode}/${event.decision.persona}`,
          detail: summarizeText(event.decision.reason || 'No router rationale provided.'),
          status: event.decision.fallback ? 'error' : 'neutral',
          meta: [
            event.decision.confidence,
            event.decision.fallback ? 'fallback' : '',
            formatSkillMeta(event.decision.skillIds) ?? '',
          ].filter(Boolean).join(' · '),
        })
        break
      case 'mode_route_decision':
        bindings.setStatus(`Mode route: ${event.chosen}${event.persona ? `/${event.persona}` : ''}`)
        pushActivity({
          id: options.createId(),
          kind: 'state',
          label: `Mode route: ${event.chosen}${event.persona ? `/${event.persona}` : ''}`,
          detail: summarizeText(event.reason || 'No route rationale provided.'),
          status: event.fallback ? 'error' : 'neutral',
          meta: formatModeRouteMeta(event),
        })
        break
      case 'quality_gate_verdict':
        bindings.setStatus(
          event.decision === 'retry'
            ? `Quality gate retry: ${event.phase}`
            : event.decision === 'incomplete'
              ? `Quality gate incomplete: ${event.phase}`
              : `Quality gate passed: ${event.phase}`,
        )
        pushActivity({
          id: options.createId(),
          kind: event.decision === 'pass' ? 'state' : 'error',
          label: `Quality gate: ${event.phase}`,
          detail: summarizeText(event.blockingReason ?? event.decision),
          status: event.decision === 'pass' ? 'success' : 'error',
          meta: `${event.decision} · backtracks ${event.backtrackCount}`,
        })
        break
      case 'backtrack':
        bindings.setStatus(`Backtracking ${event.phase} attempt ${event.attempt}`)
        pushActivity({
          id: options.createId(),
          kind: 'error',
          label: `Backtrack: ${event.phase}`,
          detail: summarizeText(event.reason),
          status: 'error',
          meta: `attempt ${event.attempt}`,
        })
        break
      case 'recovery':
        bindings.setStatus(`Recovery: ${event.kind}`)
        pushActivity({
          id: options.createId(),
          kind: event.recoverable ? 'state' : 'error',
          label: `Recovery: ${event.kind}`,
          detail: summarizeText(event.message),
          status: event.recoverable ? 'neutral' : 'error',
          meta: `${event.scope} · ${event.action}`,
        })
        break
      case 'state_board':
        bindings.onStateBoard?.({
          criteriaTotal: event.criteriaTotal,
          planTotal: event.planTotal,
          planDone: event.planDone,
          todosTotal: event.todosTotal,
          todosDone: event.todosDone,
          ...(event.currentNode ? { currentNode: event.currentNode } : {}),
          ...(event.nodeState ? { nodeState: event.nodeState } : {}),
          ...(event.iteration !== undefined ? { iteration: event.iteration } : {}),
          ...(event.todos ? { todos: event.todos } : {}),
          ...(event.plan ? { plan: event.plan } : {}),
        })
        break
      case 'node_trace':
        bindings.onNodeTrace?.({
          node: event.node,
          durationMs: event.durationMs,
          ...(event.nextEdge ? { nextEdge: event.nextEdge } : {}),
        })
        break
      case 'steering_ack':
        bindings.onSteeringEvent?.({
          kind: 'ack',
          noteId: event.noteId,
          message: event.message,
          noteKind: event.kind,
        })
        bindings.setStatus(`Steering queued: ${event.kind}`)
        pushActivity({
          id: options.createId(),
          kind: 'state',
          label: 'Steering queued',
          detail: summarizeText(event.message),
          status: 'neutral',
          meta: `${event.kind} · ${event.noteId}`,
        })
        break
      case 'steering_consumed':
        bindings.onSteeringEvent?.({ kind: 'consumed', noteId: event.noteId })
        bindings.setStatus(`Steering consumed: ${event.noteId}`)
        pushActivity({
          id: options.createId(),
          kind: 'state',
          label: 'Steering consumed',
          detail: 'Queued steering note was surfaced to the model.',
          status: 'success',
          meta: event.noteId,
        })
        break
      case 'edit_checkpoint_opened':
        bindings.setStatus(`Edit checkpoint opened: ${event.checkpoint.label}`)
        pushActivity({
          id: options.createId(),
          kind: 'state',
          label: 'Edit checkpoint opened',
          detail: formatCheckpointDetail(event.checkpoint.files),
          status: 'running',
          meta: event.checkpoint.checkpointId,
        })
        break
      case 'edit_checkpoint_resolved': {
        const status = event.checkpoint.status
        bindings.setStatus(`Edit checkpoint ${status}: ${event.checkpoint.label}`)
        pushActivity({
          id: options.createId(),
          kind: status === 'reverted' ? 'error' : 'state',
          label: `Edit checkpoint ${status}`,
          detail: formatCheckpointDetail(event.checkpoint.files),
          status: status === 'reverted' ? 'error' : 'success',
          meta: event.checkpoint.checkpointId,
        })
        break
      }
      case 'panel_open': {
        const personas = event.personas ?? []
        bindings.setStatus(`Persona panel started (${personas.length} panelists)`)
        appendSurfaceMessage('system', `Persona panel opened\n\n${formatPanelRosterSummary(personas)}`)
        pushActivity({
          id: options.createId(),
          kind: 'state',
          label: 'Persona panel',
          detail: summarizeText(personas.map((persona) => persona.name).join(', ') || 'No panelists'),
          status: 'neutral',
          meta: `${personas.length} panelists`,
        })
        break
      }
      case 'panel_turn_start':
        bindings.setAgentState('thinking')
        bindings.setStatus(`${event.personaName} is responding…`)
        pushActivity({
          id: options.createId(),
          kind: 'thinking',
          label: `${event.personaName} responding`,
          detail: 'Panelist turn started.',
          status: 'running',
        })
        break
      case 'panel_turn_complete':
        bindings.setStatus(`${event.personaName} answered`)
        appendSurfaceMessage('assistant', `### ${event.personaName}\n\n${event.text}`, {
          personaId: event.personaId,
          personaName: event.personaName,
        })
        pushActivity({
          id: options.createId(),
          kind: 'result',
          label: `${event.personaName} answered`,
          detail: summarizeText(event.text),
          status: 'success',
        })
        break
      case 'panel_turn_failed':
        bindings.setStatus(`${event.personaName} failed`)
        appendSurfaceMessage('system', `Panelist ${event.personaName} failed\n\n${event.error}`)
        pushActivity({
          id: options.createId(),
          kind: 'error',
          label: `${event.personaName} failed`,
          detail: summarizeText(event.error),
          status: 'error',
        })
        break
      case 'panel_synthesizing':
        bindings.setAgentState('thinking')
        bindings.setStatus('Synthesizing persona panel…')
        pushActivity({
          id: options.createId(),
          kind: 'thinking',
          label: 'Synthesizing persona panel',
          detail: `${event.panelists} panelist replies`,
          status: 'running',
        })
        break
      case 'subagent_progress': {
        // Nested activity from a dispatched subagent. Surface it as its own
        // activity row so a long (or parallel) subagent isn't a silent gap.
        const inner = event.inner
        const label = event.label ? `subagent · ${event.label}` : 'subagent'
        let detail: string = inner.type
        if (inner.type === 'reasoning_step') detail = inner.label
        else if (inner.type === 'action_progress') detail = `${inner.summary} → ${inner.nextStep}`
        else if (inner.type === 'thinking') detail = inner.content ?? 'reasoning'
        else if (inner.type === 'tool_call') detail = `→ ${inner.toolCall?.name ?? 'tool'}`
        else if (inner.type === 'tool_result') detail = `${inner.status === 'success' ? 'ok' : 'error'}: ${summarizeText(inner.output ?? '')}`
        else if (inner.type === 'message') detail = summarizeText(inner.content ?? '')
        pushActivity({
          id: options.createId(),
          kind: 'subagent',
          label,
          detail: summarizeText(detail),
          status: inner.type === 'tool_result' && inner.status !== 'success' ? 'error' : 'running',
          meta: event.subagentId,
        })
        break
      }
      case 'debate_round':
        appendSurfaceMessage('system', formatDebateRoundContent(event.round))
        pushActivity({
          id: options.createId(),
          kind: event.round.finalDecision === 'reject' ? 'error' : 'state',
          label: `Debate: ${event.round.finalDecision}`,
          detail: summarizeText(event.round.rationale || event.round.topic),
          status: event.round.finalDecision === 'reject' ? 'error' : 'neutral',
          meta: event.round.topic,
        })
        break
      case 'text_delta':
        assistantContent += event.text ?? ''
        bindings.setStatus('Streaming response…')
        updateAssistant(assistantContent, undefined, event.sessionId)
        break
      case 'tool_call': {
        const toolCall = event.toolCall
        const toolId = toolCall?.id ?? options.createId()
        const toolName = toolCall?.name ?? 'tool'
        toolNamesById.set(toolId, toolName)
        if (seenToolCallIds.has(toolId)) break
        seenToolCallIds.add(toolId)
        bindings.setAgentState('acting')
        bindings.setStatus(`Running ${toolName}…`)
        clearSupersededProgress()
        upsertToolMessage(
          toolId,
          {
            toolName,
            content: formatToolInput(toolCall?.arguments ?? {}),
          },
          {
            toolInput: (toolCall?.arguments ?? {}) as Record<string, unknown>,
            toolStatus: 'running',
            toolMeta: 'Running now',
            toolNeedsApproval: false,
            approvalRequestId: undefined,
            approvalState: undefined,
            resumeAvailable: undefined,
            sessionId: event.sessionId,
          },
        )
        pushActivity({
          id: options.createId(),
          kind: 'tool',
          label: toolName,
          detail: summarizeText(formatToolCall(toolCall ?? {})),
          status: 'running',
        })
        break
      }
      case 'approval_request': {
        const toolCall = event.toolCall
        const toolId = toolCall?.id ?? options.createId()
        const toolName = toolCall?.name ?? 'tool'
        toolNamesById.set(toolId, toolName)
        bindings.setStatus(`Approval required for ${toolName}`)
        clearSupersededProgress()
        upsertToolMessage(
          toolId,
          {
            toolName,
            content: formatToolInput(toolCall?.arguments ?? {}),
          },
          {
            toolInput: (toolCall?.arguments ?? {}) as Record<string, unknown>,
            toolStatus: 'pending',
            toolMeta: 'Approval required to continue',
            toolNeedsApproval: true,
            approvalRequestId: event.requestId,
            approvalState: undefined,
            resumeAvailable: options.approvalResumeAvailable || undefined,
            toolDiff: typeof event.previewDiff === 'string' ? event.previewDiff : undefined,
            sessionId: event.sessionId,
          },
        )
        pushActivity({
          id: options.createId(),
          kind: 'approval',
          label: toolName,
          detail: 'Execution paused pending approval',
          status: 'pending',
        })
        break
      }
      case 'auto_approval': {
        // Surface a positive activity item so the operator sees *why* a
        // tool ran without a consent prompt — without it accumulated
        // session/always rules look like the daemon silently skipped
        // consent.
        const toolName = event.toolCall?.name ?? 'tool'
        const verdict = event.decision === 'approved' ? 'auto-approved' : 'auto-denied'
        pushActivity({
          id: options.createId(),
          kind: 'approval',
          label: `${toolName} ${verdict}`,
          detail: `${event.scope} rule '${event.rule.pattern}'`,
          status: event.decision === 'approved' ? 'success' : 'error',
        })
        break
      }
      case 'approval_response':
        bindings.setStatus(null)
        bindings.setMessages((prev) => {
          const target = prev.find(
            (message) => message.role === 'tool' && message.approvalRequestId === event.requestId,
          )
          return target
            ? applyApprovalResolution(prev, target.id, event.decision, event.note)
            : prev
        })
        pushActivity(
          createApprovalResolutionActivity({
            id: options.createId(),
            approved: event.decision,
            detail: event.note ?? 'Approval response recorded.',
          }),
        )
        break
      case 'tool_result': {
        const resultSignature = JSON.stringify([
          event.status,
          event.output,
          event.recovery ?? null,
        ])
        if (toolResultSignatureById.get(event.toolCallId) === resultSignature) break
        toolResultSignatureById.set(event.toolCallId, resultSignature)
        bindings.setAgentState('observing')
        lastToolOutcome = {
          toolName: toolNamesById.get(event.toolCallId) ?? 'Tool',
          status: event.status === 'success' ? 'success' : 'error',
          detail: summarizeText(event.output),
        }
        // A completed tool must immediately cease to be the live activity.
        // Otherwise the footer keeps saying "Running fs.glob…" while the
        // graph is actually waiting on its next model/planner node.
        settleToolActivity(lastToolOutcome.toolName, lastToolOutcome.status)
        bindings.setStatus(
          lastToolOutcome.status === 'success'
            ? `${lastToolOutcome.toolName} completed; continuing…`
            : `${lastToolOutcome.toolName} failed; recovering…`,
        )
        if (lastToolOutcome.status === 'success') {
          successfulToolOutcomes.push({
            toolName: lastToolOutcome.toolName,
            detail: lastToolOutcome.detail,
          })
        }
        upsertToolMessage(
          event.toolCallId,
          { toolName: 'Tool result', content: '' },
          {
            toolStatus: event.status === 'success' ? 'success' : 'error',
            toolNeedsApproval: false,
            approvalRequestId: undefined,
            approvalState: undefined,
            resumeAvailable: undefined,
            toolMeta: toolMetaFromResultStatus(
              event.status,
              event.recovery,
              event.output,
              event.executionPosture,
            ),
            toolResult: event.output,
            sessionId: event.sessionId,
            toolDiff:
              typeof event.metadata?.editDiff === 'string' ? event.metadata.editDiff : undefined,
          },
        )
        pushActivity({
          id: options.createId(),
          kind: 'result',
          label: event.recovery ? 'Recovered tool result' : 'Tool result',
          detail: summarizeText(event.output),
          status: event.status === 'success' ? 'success' : 'error',
          meta:
            event.recovery === 'journal'
              ? 'Saved execution'
              : event.recovery === 'probe'
                ? 'Verified recovery'
                : undefined,
        })
        break
      }
      case 'cowork_plan':
        bindings.setStatus('Cowork plan ready')
        pushActivity({
          id: options.createId(),
          kind: 'state',
          label: 'Cowork plan',
          detail: summarizeText(
            event.plan.map((step) => `${step.role}: ${step.instruction}`).join('\n'),
          ),
          status: 'neutral',
          meta: `${event.plan.length} tasks`,
        })
        break
      case 'cowork_task_start':
        bindings.setAgentState('thinking')
        bindings.setStatus(`${event.role} is working…`)
        pushActivity({
          id: options.createId(),
          kind: 'state',
          label: `${event.role} started`,
          detail: summarizeText(event.instruction),
          status: 'running',
        })
        break
      case 'cowork_task_complete':
        bindings.setStatus(`${event.role} completed`)
        pushActivity({
          id: options.createId(),
          kind: 'result',
          label: `${event.role} completed`,
          detail: summarizeText(event.result),
          status: 'success',
          meta: summarizeText(event.instruction),
        })
        break
      case 'cowork_task_failed':
        bindings.setStatus(`${event.role} failed`)
        pushActivity({
          id: options.createId(),
          kind: 'error',
          label: `${event.role} failed`,
          detail: summarizeText(event.error),
          status: 'error',
          meta: summarizeText(event.instruction),
        })
        break
      case 'cowork_synthesizing':
        bindings.setAgentState('thinking')
        bindings.setStatus('Synthesizing cowork output…')
        pushActivity({
          id: options.createId(),
          kind: 'thinking',
          label: 'Synthesizing cowork output',
          detail: summarizeText(event.summary),
          status: 'running',
        })
        break
      case 'cowork_discuss_request':
        bindings.setStatus('Waiting for cowork input…')
        pushActivity({
          id: options.createId(),
          kind: 'state',
          label: 'Cowork question',
          detail: summarizeText(event.prompt),
          status: 'running',
          meta: event.choices?.length ? event.choices.join(' · ') : undefined,
        })
        break
      case 'question_request': {
        const choiceText = event.choices?.length
          ? `\n\nChoices: ${event.choices.join(' · ')}`
          : ''
        bindings.setAgentState('thinking')
        bindings.setStatus('Waiting for your answer…')
        appendSurfaceMessage(
          'system',
          `Question requested\n\n${event.prompt}${choiceText}\n\nQuestion ID: ${event.questionId}`,
          {
            questionRequestId: event.questionId,
            questionPrompt: event.prompt,
            questionChoices: event.choices,
            questionState: 'pending',
            sessionId: event.sessionId,
          },
        )
        pushActivity({
          id: options.createId(),
          kind: 'state',
          label: 'User answer requested',
          detail: summarizeText(event.prompt),
          status: 'pending',
          meta: event.questionId,
        })
        break
      }
      case 'cowork_discuss_response':
        bindings.setStatus('Cowork input received')
        pushActivity({
          id: options.createId(),
          kind: 'result',
          label: 'Cowork answer',
          detail: summarizeText(event.response),
          status: 'success',
          meta: summarizeText(event.prompt),
        })
        break
      case 'run_contract':
        bindings.setStatus(
          `Run contract: ${formatAcceptanceCriteriaCount(event.contract.acceptanceCriteria.length, 'short')}`,
        )
        pushActivity({
          id: options.createId(),
          kind: 'state',
          label: 'Run contract',
          detail: formatRunContractDetail(event.contract),
          status: 'running',
          meta: formatAcceptanceCriteriaCount(event.contract.acceptanceCriteria.length),
        })
        break
      case 'phase_change': {
        const entered = event.enteredPhase ?? 'finalize'
        const closed = event.closedPhase
        const closedTokens = closed ? closed.usage.inputTokens + closed.usage.outputTokens : 0
        const closedSummary = closed ? ` (closed ${closed.phase}: ${closedTokens} tokens)` : ''
        bindings.setStatus(`Phase: ${entered}${closedSummary}`)
        pushActivity({
          id: options.createId(),
          kind: 'state',
          label: `Phase: ${entered}`,
          detail: closed
            ? `Closed ${closed.phase}: ${closedTokens.toLocaleString()} tokens`
            : 'Phase started',
          status: 'running',
        })
        break
      }
      case 'post_edit_findings': {
        // Splits diagnostics into `[caller]`-prefixed (downstream files
        // the edit broke) and the rest (the edited file's own LSP
        // issues). Activity panels highlight broken callers because
        // those are the strongest "you broke something" signal —
        // without surfacing it the operator only sees the eventual
        // model self-correction one turn later.
        const findings = event as {
          editedFiles?: string[]
          reverseCallers?: string[]
          diagnostics?: Array<{ file: string; summary: string }>
        }
        const diagnostics = findings.diagnostics ?? []
        const broken = diagnostics.filter((d) => d.summary.startsWith('[caller]'))
        const own = diagnostics.filter((d) => !d.summary.startsWith('[caller]'))
        const editedSummary = (findings.editedFiles ?? []).slice(0, 3).join(', ')
        const detail =
          broken.length > 0
            ? `${broken.length} caller(s) broken: ${broken
                .slice(0, 2)
                .map((c) => c.file)
                .join(', ')}`
            : own.length > 0
              ? `${own.length} diagnostic(s) on edited files`
              : (findings.reverseCallers ?? []).length > 0
                ? `Likely callers: ${(findings.reverseCallers ?? []).slice(0, 3).join(', ')}`
                : 'No outstanding issues'
        pushActivity({
          id: options.createId(),
          kind: broken.length > 0 ? 'error' : 'state',
          label: editedSummary ? `Post-edit: ${editedSummary}` : 'Post-edit findings',
          detail,
          status: broken.length > 0 ? 'error' : 'neutral',
        })
        break
      }
      case 'message':
        assistantContent = event.content ?? assistantContent
        bindings.setStatus(null)
        updateAssistant(assistantContent, undefined, event.sessionId)
        break
      case 'done': {
        if (completionHandled) break
        completionHandled = true
        recordStopReason(event.stopReason)
        // Bare-empty run: mirrors the final branch of
        // buildCompletionAssistantContent (nothing streamed, no tool outcome,
        // no superseded progress) — the placeholder fallback text is all the
        // user would get, so let the surface escalate to an error/retry state.
        const bareEmpty =
          !assistantContent.trim() && !lastToolOutcome && !lastSupersededProgress
        bindings.setAgentState('done')
        bindings.setStatus(null)
        updateAssistant(
          buildCompletionAssistantContent(),
          normalizeUsage(event.usage),
          event.sessionId,
        )
        if (bareEmpty) {
          options.onEmptyCompletion?.()
        }
        settleLingeringActivities()
        pushActivity({
          id: options.createId(),
          kind: 'usage',
          label: 'Run completed',
          detail: `${event.usage.inputTokens.toLocaleString()}→${event.usage.outputTokens.toLocaleString()} tokens`,
          status: 'success',
          meta: event.usage.costUsd ? `$${event.usage.costUsd.toFixed(4)}` : undefined,
        })
        break
      }
      case 'error': {
        const detail = event.error?.message ?? options.errorFallbackText
        recordStopReason(event.error?.stopReason)
        bindings.setAgentState('error')
        bindings.setStatus(null)
        bindings.setError(options.formatErrorMessage?.(detail) ?? detail)
        updateAssistant(assistantContent || options.errorFallbackText)
        settleLingeringActivities()
        pushActivity({
          id: options.createId(),
          kind: 'error',
          label: options.errorLabel,
          detail,
          status: 'error',
        })
        break
      }
      default:
        break
    }
  }

  return {
    getAssistantContent: () => assistantContent,
    getStopReason: () => stopReason,
    updateAssistant,
    handleEvent,
  }
}
