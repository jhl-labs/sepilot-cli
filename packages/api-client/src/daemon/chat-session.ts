import type { DaemonSessionDetail } from './types.js'
import type { ActivityItem, Message, StoredSessionState } from './chat-surface-types.js'
import {
  applyApprovalResolution,
  createApprovalResolutionActivity,
  formatAcceptanceCriteriaCount,
  formatMemoryContextMessage,
  formatToolCall,
  formatToolInput,
  mapStoredToolStatus,
  summarizeMemoryContextItems,
  summarizeText,
  toolMetaFromResultStatus,
  toolMetaFromStoredStatus,
} from './chat-surface-utils.js'

export const SESSION_HISTORY_RENDER_EVENT_LIMIT = 100
export const SESSION_HISTORY_RENDER_CHAR_BUDGET = 240_000
export const SESSION_HISTORY_WINDOW_MESSAGE_ID_PREFIX = 'history-window:'

type SessionDetailEvent = DaemonSessionDetail['events'][number]

interface SurfaceSessionEventEntry {
  index: number
  event: SessionDetailEvent
}

interface SurfaceSessionEventSelection {
  entries: SurfaceSessionEventEntry[]
  hiddenEventCount: number
  visibleEventCount: number
  totalEventCount: number
}

interface RenderCharEstimateState {
  visited: WeakSet<object>
  nodes: number
}

const SESSION_HISTORY_RENDER_ESTIMATE_MAX_DEPTH = 8
const SESSION_HISTORY_RENDER_ESTIMATE_MAX_NODES = 400

function addCapped(total: number, amount: number, limit: number): number {
  return Math.min(limit, total + Math.max(0, amount))
}

function estimateRenderChars(
  value: unknown,
  limit: number,
  state: RenderCharEstimateState,
  depth = 0,
): number {
  if (limit <= 0) {
    return 0
  }
  if (typeof value === 'string') {
    return Math.min(value.length, limit)
  }
  if (typeof value === 'number' || typeof value === 'boolean' || typeof value === 'bigint') {
    return Math.min(String(value).length, limit)
  }
  if (!value || typeof value !== 'object') {
    return 0
  }
  if (
    depth >= SESSION_HISTORY_RENDER_ESTIMATE_MAX_DEPTH ||
    state.nodes >= SESSION_HISTORY_RENDER_ESTIMATE_MAX_NODES ||
    state.visited.has(value)
  ) {
    return 0
  }

  state.visited.add(value)
  state.nodes += 1

  if (Array.isArray(value)) {
    let chars = 0
    for (const item of value) {
      chars = addCapped(chars, estimateRenderChars(item, limit - chars, state, depth + 1), limit)
      if (chars >= limit) {
        break
      }
    }
    return chars
  }

  let chars = 0
  for (const item of Object.values(value)) {
    chars = addCapped(chars, estimateRenderChars(item, limit - chars, state, depth + 1), limit)
    if (chars >= limit) {
      break
    }
  }
  return chars
}

function estimateSessionEventRenderChars(event: SessionDetailEvent, limit: number): number {
  return Math.max(
    1,
    estimateRenderChars(event, limit, {
      visited: new WeakSet<object>(),
      nodes: 0,
    }),
  )
}

function selectSurfaceSessionEventEntries(
  events: readonly SessionDetailEvent[],
): SurfaceSessionEventSelection {
  let visibleEventCount = 0
  let renderChars = 0
  let startIndex = events.length

  for (let index = events.length - 1; index >= 0; index -= 1) {
    if (visibleEventCount >= SESSION_HISTORY_RENDER_EVENT_LIMIT) {
      break
    }

    const event = events[index]
    if (!event) {
      continue
    }

    const remainingBudget = Math.max(1, SESSION_HISTORY_RENDER_CHAR_BUDGET - renderChars + 1)
    const eventChars = estimateSessionEventRenderChars(event, remainingBudget)
    if (visibleEventCount > 0 && renderChars + eventChars > SESSION_HISTORY_RENDER_CHAR_BUDGET) {
      break
    }

    renderChars += eventChars
    visibleEventCount += 1
    startIndex = index
  }

  return {
    entries: events
      .slice(startIndex)
      .map((event, offset) => ({ event, index: startIndex + offset })),
    hiddenEventCount: startIndex,
    visibleEventCount,
    totalEventCount: events.length,
  }
}

function pluralizeEvent(count: number): string {
  return count === 1 ? 'event' : 'events'
}

function formatHistoryWindowNotice(selection: SurfaceSessionEventSelection): string {
  return [
    'Session replay — the events below are historical and are not running again.',
    `Showing the latest ${selection.visibleEventCount} of ${selection.totalEventCount} session ${pluralizeEvent(selection.totalEventCount)}.`,
    `${selection.hiddenEventCount} earlier ${pluralizeEvent(selection.hiddenEventCount)} hidden from this surface view; export the session for the complete transcript.`,
  ].join(' ')
}

function formatTodoSummary(
  items: DaemonSessionDetail['events'][number] extends infer Event
    ? Event extends { type: 'todo_list'; items: infer Items }
      ? Items
      : never
    : never,
): string {
  return items
    .map((item) => {
      const status =
        item.status === 'completed'
          ? '[done]'
          : item.status === 'in_progress'
            ? '[active]'
            : item.status === 'blocked'
              ? '[blocked]'
              : item.status === 'cancelled'
                ? '[cancelled]'
                : '[todo]'
      return `${status} ${item.content}`
    })
    .join('\n')
}

function formatCoworkPlanSummary(plan: Array<{ role: string; instruction: string }>): string {
  return plan.map((step) => `${step.role}: ${step.instruction}`).join('\n')
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

function formatDebateRoundMessage(
  round: Extract<SessionDetailEvent, { type: 'debate_round' }>['round'],
): string {
  return [
    `Debate round: ${summarizeText(round.topic ?? round.roundId ?? 'untitled', 120)}`,
    round.finalDecision ? `Decision: ${round.finalDecision}` : '',
    round.rationale ? `Rationale: ${summarizeText(round.rationale, 240)}` : '',
  ]
    .filter(Boolean)
    .join('\n\n')
}

function formatProviderAttemptTarget(
  event: Extract<SessionDetailEvent, { type: 'provider_attempt' }>,
): string {
  return `${event.provider}/${event.model}`
}

function formatProviderAttemptMessage(
  event: Extract<SessionDetailEvent, { type: 'provider_attempt' }>,
): string {
  const lines = [
    `Provider attempt failed: ${formatProviderAttemptTarget(event)}`,
    event.errorMessage,
    event.nextProvider && event.nextModel
      ? `Retrying with ${event.nextProvider}/${event.nextModel}`
      : null,
  ].filter((line): line is string => Boolean(line))
  return lines.join('\n')
}

function formatProviderAttemptMeta(
  event: Extract<SessionDetailEvent, { type: 'provider_attempt' }>,
): string {
  return `attempt ${event.attempt}${event.retryable ? ' · retryable' : ''}`
}

function formatRecoveryMessage(event: Extract<SessionDetailEvent, { type: 'recovery' }>): string {
  return [
    `Recovery: ${event.scope}/${event.kind} via ${event.action}`,
    event.message,
    event.recoverable ? null : 'Not recoverable',
  ]
    .filter((line): line is string => Boolean(line))
    .join('\n')
}

function formatSkillMeta(skillIds: string[] | undefined): string | undefined {
  return skillIds && skillIds.length > 0 ? `skills: ${skillIds.join(', ')}` : undefined
}

function formatModeRouteMeta(event: Extract<SessionDetailEvent, { type: 'mode_route_decision' }>): string {
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

function settleRestoredActivities(
  activities: ActivityItem[],
  sessionStatus: DaemonSessionDetail['status'],
): ActivityItem[] {
  if (sessionStatus === 'active') return activities
  let changed = false
  const next = activities.map((activity) => {
    if (activity.status === 'running' || activity.status === 'pending') {
      changed = true
      return { ...activity, status: 'neutral' as const }
    }
    return activity
  })
  return changed ? next : activities
}

export function buildStoredSessionState(session: DaemonSessionDetail): StoredSessionState {
  const messages: Message[] = []
  const activities: ActivityItem[] = []
  const eventSelection = selectSurfaceSessionEventEntries(session.events ?? [])
  let assistantCitations: Message['citations']
  const toolMessageIndexes = new Map<string, number>()
  const requestToToolCallId = new Map<string, string>()
  const pendingApprovalByToolCallId = new Map(
    (session.pendingApprovals ?? []).map((approval) => [approval.toolCallId, approval]),
  )

  if (eventSelection.hiddenEventCount > 0) {
    messages.push({
      id: `${SESSION_HISTORY_WINDOW_MESSAGE_ID_PREFIX}${session.id}`,
      role: 'system',
      content: formatHistoryWindowNotice(eventSelection),
      timestamp: eventSelection.entries[0]?.event.timestamp ?? session.updatedAt,
    })
    activities.push({
      id: `${SESSION_HISTORY_WINDOW_MESSAGE_ID_PREFIX}${session.id}`,
      kind: 'state',
      label: 'History window',
      detail: `${eventSelection.hiddenEventCount} earlier ${pluralizeEvent(eventSelection.hiddenEventCount)} hidden from this surface view`,
      status: 'neutral',
      meta: `${eventSelection.visibleEventCount}/${eventSelection.totalEventCount} events`,
    })
  }

  for (const { index, event } of eventSelection.entries) {
    const branchFromEventIndex = index + 1

    switch (event.type) {
      case 'user_message':
        messages.push({
          id: event.id,
          role: 'user',
          content: event.content,
          timestamp: event.timestamp,
          branchFromEventIndex,
        })
        break
      case 'memory_context': {
        const summary = summarizeMemoryContextItems(event.items)
        assistantCitations = event.items
        messages.push({
          id: event.id,
          role: 'context',
          content: formatMemoryContextMessage(event.items),
          timestamp: event.timestamp,
          branchFromEventIndex,
          contextItems: event.items,
        })
        activities.push({
          id: `${event.id}-context`,
          kind: 'context',
          label: 'Relevant context',
          detail: summary.detail,
          status: 'neutral',
          meta: summary.meta,
        })
        break
      }
      case 'assistant_message':
        messages.push({
          id: event.id,
          role: 'assistant',
          content: event.content,
          timestamp: event.timestamp,
          branchFromEventIndex,
          citations: assistantCitations,
        })
        if (event.thinking) {
          activities.push({
            id: `${event.id}-thinking`,
            kind: 'thinking',
            label: 'Reasoning',
            detail: summarizeText(event.thinking),
            status: 'running',
          })
        }
        break
      case 'router_decision':
        activities.push({
          id: `${event.id}-router`,
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
        activities.push({
          id: `${event.id}-mode-route`,
          kind: 'state',
          label: `Mode route: ${event.chosen}${event.persona ? `/${event.persona}` : ''}`,
          detail: summarizeText(event.reason || 'No route rationale provided.'),
          status: event.fallback ? 'error' : 'neutral',
          meta: formatModeRouteMeta(event),
        })
        break
      case 'quality_gate_verdict':
        activities.push({
          id: `${event.id}-quality-gate`,
          kind: event.decision === 'pass' ? 'state' : 'error',
          label: `Quality gate: ${event.phase}`,
          detail: summarizeText(event.blockingReason ?? event.decision),
          status: event.decision === 'pass' ? 'success' : 'error',
          meta: `${event.decision} · backtracks ${event.backtrackCount}`,
        })
        break
      case 'backtrack':
        activities.push({
          id: `${event.id}-backtrack`,
          kind: 'error',
          label: `Backtrack: ${event.phase}`,
          detail: summarizeText(event.reason),
          status: 'error',
          meta: `attempt ${event.attempt}`,
        })
        break
      case 'cowork_plan':
        activities.push({
          id: `${event.id}-cowork-plan`,
          kind: 'state',
          label: 'Cowork plan',
          detail: summarizeText(formatCoworkPlanSummary(event.plan)),
          status: 'neutral',
          meta: `${event.plan.length} tasks`,
        })
        break
      case 'cowork_task_start':
        activities.push({
          id: `${event.id}-cowork-task-start`,
          kind: 'state',
          label: `${event.role} started`,
          detail: summarizeText(event.instruction),
          status: 'running',
        })
        break
      case 'cowork_task_complete':
        activities.push({
          id: `${event.id}-cowork-task-complete`,
          kind: 'result',
          label: `${event.role} completed`,
          detail: summarizeText(event.result),
          status: 'success',
          meta: summarizeText(event.instruction),
        })
        break
      case 'cowork_task_failed':
        activities.push({
          id: `${event.id}-cowork-task-failed`,
          kind: 'error',
          label: `${event.role} failed`,
          detail: summarizeText(event.error),
          status: 'error',
          meta: summarizeText(event.instruction),
        })
        break
      case 'cowork_synthesizing':
        activities.push({
          id: `${event.id}-cowork-synthesizing`,
          kind: 'thinking',
          label: 'Synthesizing cowork output',
          detail: summarizeText(event.summary),
          status: 'running',
        })
        break
      case 'cowork_discuss_request':
        activities.push({
          id: `${event.id}-cowork-discuss-request`,
          kind: 'state',
          label: 'Cowork discussion requested',
          detail: summarizeText(event.prompt),
          status: 'running',
          meta: event.choices?.length ? event.choices.join(' · ') : undefined,
        })
        break
      case 'cowork_discuss_response':
        activities.push({
          id: `${event.id}-cowork-discuss-response`,
          kind: 'result',
          label: 'Cowork discussion answered',
          detail: summarizeText(event.response),
          status: 'success',
          meta: summarizeText(event.prompt),
        })
        break
      case 'panel_open':
        messages.push({
          id: event.id,
          role: 'system',
          content: `Persona panel opened\n\n${formatPanelRosterSummary(event.personas)}`,
          timestamp: event.timestamp,
          branchFromEventIndex,
        })
        activities.push({
          id: `${event.id}-panel-open`,
          kind: 'state',
          label: 'Persona panel',
          detail: summarizeText(
            event.personas.map((persona) => persona.name).join(', ') || 'No panelists',
          ),
          status: 'neutral',
          meta: `${event.personas.length} panelists`,
        })
        break
      case 'panel_turn_complete':
        messages.push({
          id: event.id,
          role: 'assistant',
          content: event.text,
          timestamp: event.timestamp,
          branchFromEventIndex,
          personaId: event.personaId,
          personaName: event.personaName,
        })
        activities.push({
          id: `${event.id}-panel-turn-complete`,
          kind: 'result',
          label: `${event.personaName} answered`,
          detail: summarizeText(event.text),
          status: 'success',
        })
        break
      case 'panel_turn_failed':
        messages.push({
          id: event.id,
          role: 'system',
          content: `Panelist ${event.personaName} failed\n\n${event.error}`,
          timestamp: event.timestamp,
          branchFromEventIndex,
        })
        activities.push({
          id: `${event.id}-panel-turn-failed`,
          kind: 'error',
          label: `${event.personaName} failed`,
          detail: summarizeText(event.error),
          status: 'error',
        })
        break
      case 'panel_synthesizing':
        activities.push({
          id: `${event.id}-panel-synthesizing`,
          kind: 'thinking',
          label: 'Synthesizing persona panel',
          detail: `${event.panelists} panelist replies`,
          status: 'running',
        })
        break
      case 'debate_round':
        messages.push({
          id: event.id,
          role: 'system',
          content: formatDebateRoundMessage(event.round),
          timestamp: event.timestamp,
          branchFromEventIndex,
        })
        activities.push({
          id: `${event.id}-debate-round`,
          kind: event.round.finalDecision === 'reject' ? 'error' : 'state',
          label: `Debate: ${event.round.finalDecision ?? 'recorded'}`,
          detail: summarizeText(event.round.rationale ?? event.round.topic ?? 'Debate round recorded'),
          status: event.round.finalDecision === 'reject' ? 'error' : 'neutral',
          meta: event.round.topic,
        })
        break
      case 'edit_checkpoint_opened':
        activities.push({
          id: `${event.id}-edit-checkpoint-opened`,
          kind: 'state',
          label: 'Edit checkpoint opened',
          detail: formatCheckpointDetail(event.checkpoint.files),
          status: 'running',
          meta: event.checkpoint.checkpointId,
        })
        break
      case 'edit_checkpoint_resolved':
        activities.push({
          id: `${event.id}-edit-checkpoint-resolved`,
          kind: event.checkpoint.status === 'reverted' ? 'error' : 'state',
          label: `Edit checkpoint ${event.checkpoint.status}`,
          detail: formatCheckpointDetail(event.checkpoint.files),
          status: event.checkpoint.status === 'reverted' ? 'error' : 'success',
          meta: event.checkpoint.checkpointId,
        })
        break
      case 'tool_call': {
        messages.push({
          id: event.id,
          role: 'tool',
          content: formatToolInput(event.input),
          timestamp: event.timestamp,
          branchFromEventIndex,
          toolName: event.tool,
          toolInput: event.input,
          toolStatus: mapStoredToolStatus(event.status),
          toolMeta: toolMetaFromStoredStatus(event.status),
          toolNeedsApproval:
            event.status === 'pending' || pendingApprovalByToolCallId.has(event.id),
          approvalRequestId: pendingApprovalByToolCallId.get(event.id)?.requestId,
          approvalState: pendingApprovalByToolCallId.get(event.id)?.state,
          resumeAvailable: pendingApprovalByToolCallId.get(event.id)?.resumeAvailable,
        })
        toolMessageIndexes.set(event.id, messages.length - 1)
        activities.push({
          id: `${event.id}-tool`,
          kind: 'tool',
          label: event.tool,
          detail: summarizeText(
            formatToolCall({
              id: event.id,
              name: event.tool,
              arguments: event.input,
            }),
          ),
          status: mapStoredToolStatus(event.status),
          meta: toolMetaFromStoredStatus(event.status),
        })
        break
      }
      case 'approval_request': {
        const pendingApproval = pendingApprovalByToolCallId.get(event.toolCallId)
        const approvalRequestId = pendingApproval?.requestId
        requestToToolCallId.set(event.id, event.toolCallId)
        if (approvalRequestId) {
          requestToToolCallId.set(approvalRequestId, event.toolCallId)
        }
        const toolIndex = toolMessageIndexes.get(event.toolCallId)
        const toolMeta =
          pendingApproval?.state === 'stale'
            ? pendingApproval?.resumeAvailable
              ? 'Saved checkpoint available for resume'
              : 'Reconnect required before this run can continue'
            : approvalRequestId
              ? 'Approval required'
              : 'Awaiting recorded approval'

        if (toolIndex !== undefined) {
          messages[toolIndex] = {
            ...messages[toolIndex],
            branchFromEventIndex,
            toolStatus: 'pending',
            toolNeedsApproval: true,
            approvalState: pendingApproval?.state,
            approvalRequestId,
            resumeAvailable: pendingApproval?.resumeAvailable,
            toolMeta,
          }
        } else {
          messages.push({
            id: event.toolCallId,
            role: 'tool',
            content: formatToolInput(event.input),
            timestamp: event.timestamp,
            branchFromEventIndex,
            toolName: event.tool,
            toolInput: event.input,
            toolStatus: 'pending',
            toolNeedsApproval: true,
            approvalState: pendingApproval?.state,
            approvalRequestId,
            resumeAvailable: pendingApproval?.resumeAvailable,
            toolMeta,
          })
          toolMessageIndexes.set(event.toolCallId, messages.length - 1)
        }
        activities.push({
          id: `${event.id}-approval`,
          kind: 'approval',
          label: event.tool,
          detail: 'Execution paused pending approval',
          status: 'pending',
        })
        break
      }
      case 'approval_response': {
        // Defensive ?? on event.approved kept around for older
        // imported sessions where the legacy schema omitted decision;
        // (event as any) sidesteps TS's exhaustive narrowing — the
        // typed schema makes decision required, but old jsonl rows
        // can still arrive without it.
        const fallbackApproved = (event as { approved?: boolean }).approved
        const decision = event.decision ?? (fallbackApproved ? 'approved' : 'denied')
        const toolCallId = requestToToolCallId.get(event.requestId)
        const toolIndex = toolCallId ? toolMessageIndexes.get(toolCallId) : undefined
        if (toolIndex !== undefined) {
          messages[toolIndex] = {
            ...applyApprovalResolution(
              [messages[toolIndex]],
              messages[toolIndex].id,
              decision,
              event.note,
            )[0],
            branchFromEventIndex,
          }
        }
        activities.push({
          ...createApprovalResolutionActivity({
            id: `${event.id}-approval-response`,
            approved: decision,
            detail: `Handled by ${event.approvedBy}`,
          }),
        })
        break
      }
      case 'auto_approval': {
        // Stored counterpart of the live AgentEvent — replay it so a
        // session reconstructed from history shows the same positive
        // signal a live operator saw. Without this case, opening an
        // older session in desktop/web would show tool_call/tool_result
        // pairs with no consent trail and the operator would have to
        // open the json view to figure out *why* each tool ran.
        activities.push({
          id: `${event.id}-auto-approval`,
          kind: 'approval',
          label: `${event.tool} auto-${event.decision}`,
          detail: `${event.scope} rule '${event.rule.pattern}'`,
          status: event.decision === 'approved' ? 'success' : 'error',
        })
        break
      }
      case 'tool_result': {
        const toolStatus = event.status === 'success' ? 'success' : 'error'
        const toolIndex = toolMessageIndexes.get(event.toolCallId)
        if (toolIndex !== undefined) {
          messages[toolIndex] = {
            ...messages[toolIndex],
            branchFromEventIndex,
            toolStatus,
            toolNeedsApproval: false,
            approvalRequestId: undefined,
            approvalState: undefined,
            toolMeta: toolMetaFromResultStatus(
              event.status,
              event.recovery,
              event.output,
              event.executionPosture,
            ),
            toolResult: event.output,
            toolDiff:
              typeof event.metadata?.editDiff === 'string' ? event.metadata.editDiff : undefined,
          }
        } else {
          messages.push({
            id: event.toolCallId,
            role: 'tool',
            content: '',
            timestamp: event.timestamp,
            branchFromEventIndex,
            toolName: 'Tool result',
            toolStatus,
            toolResult: event.output,
            toolDiff:
              typeof event.metadata?.editDiff === 'string' ? event.metadata.editDiff : undefined,
            toolMeta: toolMetaFromResultStatus(
              event.status,
              event.recovery,
              event.output,
              event.executionPosture,
            ),
          })
        }
        activities.push({
          id: `${event.id}-result`,
          kind: 'result',
          label: event.recovery ? 'Recovered tool result' : 'Tool result',
          detail: summarizeText(event.output),
          status: toolStatus,
          meta:
            event.recovery === 'journal'
              ? 'Saved execution'
              : event.recovery === 'probe'
                ? 'Verified recovery'
                : `${event.duration_ms}ms`,
        })
        break
      }
      case 'context_compact':
        activities.push({
          id: `${event.id}-compact`,
          kind: 'compact',
          label: 'Context compacted',
          detail: summarizeText(event.summary),
          status: 'neutral',
          meta: `${event.beforeTokens}→${event.afterTokens} tokens`,
        })
        break
      case 'provider_attempt':
        if (event.status === 'failed') {
          messages.push({
            id: event.id,
            role: 'system',
            content: formatProviderAttemptMessage(event),
            timestamp: event.timestamp,
            branchFromEventIndex,
          })
        }
        activities.push({
          id: `${event.id}-provider`,
          kind: event.status === 'failed' ? 'error' : 'state',
          label: `Provider attempt ${event.status}`,
          detail: summarizeText(
            event.status === 'failed'
              ? (event.errorMessage ?? formatProviderAttemptTarget(event))
              : formatProviderAttemptTarget(event),
          ),
          status:
            event.status === 'failed'
              ? 'error'
              : event.status === 'succeeded'
                ? 'success'
                : 'running',
          meta:
            event.nextProvider && event.nextModel
              ? `next ${event.nextProvider}/${event.nextModel}`
              : formatProviderAttemptMeta(event),
        })
        break
      case 'recovery':
        messages.push({
          id: event.id,
          role: 'system',
          content: formatRecoveryMessage(event),
          timestamp: event.timestamp,
          branchFromEventIndex,
        })
        activities.push({
          id: `${event.id}-recovery`,
          kind: event.recoverable ? 'state' : 'error',
          label: 'Recovery',
          detail: summarizeText(event.message),
          status: event.recoverable ? 'neutral' : 'error',
          meta: `${event.scope}/${event.kind} · ${event.action}`,
        })
        break
      case 'steering_ack':
        activities.push({
          id: `${event.id}-steering-ack`,
          kind: 'state',
          label: 'Steering queued',
          detail: summarizeText(event.message),
          status: 'neutral',
          meta: `${event.kind} · ${event.noteId}`,
        })
        break
      case 'steering_consumed':
        activities.push({
          id: `${event.id}-steering-consumed`,
          kind: 'state',
          label: 'Steering consumed',
          detail: 'Queued steering note was surfaced to the model.',
          status: 'success',
          meta: event.noteId,
        })
        break
      case 'steering_cancelled':
        activities.push({
          id: `${event.id}-steering-cancelled`,
          kind: 'state',
          label: 'Steering cancelled',
          detail: 'Queued steering note was retracted before model consumption.',
          status: 'neutral',
          meta: event.noteId,
        })
        break
      case 'run_contract':
        activities.push({
          id: `${event.id}-contract`,
          kind: 'state',
          label: 'Run contract',
          detail: summarizeText(event.contract.summary),
          status: 'running',
          meta: formatAcceptanceCriteriaCount(event.contract.acceptanceCriteria.length),
        })
        break
      case 'phase_change': {
        const entered = event.enteredPhase ?? 'finalize'
        const closed = event.closedPhase
        const closedTokens = closed ? closed.usage.inputTokens + closed.usage.outputTokens : 0
        activities.push({
          id: `${event.id}-phase`,
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
        const diagnostics = event.diagnostics ?? []
        const broken = diagnostics.filter((d) => d.summary.startsWith('[caller]'))
        const own = diagnostics.filter((d) => !d.summary.startsWith('[caller]'))
        const editedSummary = event.editedFiles.slice(0, 3).join(', ')
        const detail =
          broken.length > 0
            ? `${broken.length} caller(s) broken: ${broken
                .slice(0, 2)
                .map((c) => c.file)
                .join(', ')}`
            : own.length > 0
              ? `${own.length} diagnostic(s) on edited files`
              : event.reverseCallers.length > 0
                ? `Likely callers: ${event.reverseCallers.slice(0, 3).join(', ')}`
                : 'No outstanding issues'
        activities.push({
          id: `${event.id}-post-edit`,
          kind: broken.length > 0 ? 'error' : 'state',
          label: editedSummary ? `Post-edit: ${editedSummary}` : 'Post-edit findings',
          detail,
          status: broken.length > 0 ? 'error' : 'neutral',
        })
        break
      }
      case 'todo_list': {
        const completedCount = event.items.filter((item) => item.status === 'completed').length
        const inProgressCount = event.items.filter((item) => item.status === 'in_progress').length
        const blockedCount = event.items.filter((item) => item.status === 'blocked').length
        const cancelledCount = event.items.filter((item) => item.status === 'cancelled').length
        messages.push({
          id: event.id,
          role: 'todo',
          content: formatTodoSummary(event.items),
          timestamp: event.timestamp,
          branchFromEventIndex,
          todoItems: event.items,
        })
        activities.push({
          id: `${event.id}-todo`,
          kind: 'state',
          label: 'Plan updated',
          detail: summarizeText(
            event.items[0]?.content
              ? `${event.items[0].content}${event.items.length > 1 ? ` +${event.items.length - 1} more` : ''}`
              : 'No plan items yet',
          ),
          status: 'neutral',
          meta: [
            `${completedCount}/${event.items.length} done`,
            `${inProgressCount} active`,
            blockedCount > 0 ? `${blockedCount} blocked` : '',
            cancelledCount > 0 ? `${cancelledCount} cancelled` : '',
          ].filter(Boolean).join(' · '),
        })
        break
      }
      case 'delegation_state':
        activities.push({
          id: `${event.id}-delegation`,
          kind: 'state',
          label: 'Delegation lease',
          detail: summarizeText(event.detail),
          status:
            event.claimHealth === 'lost'
              ? 'error'
              : event.claimHealth === 'degraded'
                ? 'pending'
                : 'neutral',
          meta: `${event.targetDevice}${event.source ? ` · ${event.source}` : ''}`,
        })
        break
      case 'delegation_result':
        activities.push({
          id: `${event.id}-delegation-result`,
          kind: 'state',
          label: 'Delegation result',
          detail: summarizeText(event.result ?? event.status),
          status: event.status === 'completed' ? 'success' : 'error',
          meta: [event.targetDevice, ...(event.artifactHandles ?? [])].join(' · '),
        })
        break
      case 'session_end':
        activities.push({
          id: `${event.id}-end`,
          kind: 'usage',
          label: 'Session completed',
          detail: `${event.totalTokens.input}→${event.totalTokens.output} tokens`,
          status: 'success',
          meta: `$${event.totalCost.toFixed(4)}`,
        })
        break
      default:
        break
    }
  }

  // Restored event records are session-scoped, so carry that source session on
  // each surface message for approval/question actions after shell state changes.
  for (const message of messages) {
    message.sessionId ??= session.id
  }

  return {
    messages,
    activities: settleRestoredActivities(activities, session.status),
    usage: {
      inputTokens: session.totalTokens.input,
      outputTokens: session.totalTokens.output,
      estimatedCost: session.totalCost,
    },
    pendingApproval: session.pendingApprovals?.[0] ?? null,
    provider: session.provider,
    model: session.model,
  }
}

export function buildSessionData(session: DaemonSessionDetail): {
  messages: Message[]
  activities: ActivityItem[]
} {
  const { messages, activities } = buildStoredSessionState(session)
  return { messages, activities }
}
