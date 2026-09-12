import type {
  ActivityItem,
  DaemonArtifact,
  DaemonPendingApproval,
  DaemonPendingQuestion,
  DaemonSessionDetail,
  Message as SurfaceMessage,
} from '@sepilotd/api-client'
import {
  applyApprovalResolution,
  buildStoredSessionState,
  SESSION_HISTORY_WINDOW_MESSAGE_ID_PREFIX,
  summarizeMemoryContextItems,
} from '@sepilotd/api-client'
import type {
  ApprovalRequest,
  Message,
  GraphTraceEntry,
  RunWorkProgress,
  ToolCallState,
} from '../types.js'
import {
  buildStoredToolCallState,
} from './tooling.js'

type ContextCompactEvent = Extract<
  DaemonSessionDetail['events'][number],
  { type: 'context_compact' }
>

export interface HydratedCliSessionState {
  sessionId: string
  messages: Message[]
  artifacts: DaemonArtifact[]
  activities: ActivityItem[]
  usage: { input: number; output: number; cost: number }
  pendingApproval: ApprovalRequest | null
  pendingQuestions: DaemonPendingQuestion[]
  model: string
  provider: string
  plannerWorkingMemory: DaemonSessionDetail['plannerWorkingMemory']
  graphTrace: GraphTraceEntry[]
  runWorkProgress: RunWorkProgress | null
}

export function collectToolCalls(messages: Message[]): ToolCallState[] {
  const entries = messages.flatMap((message, index) => {
    if (message.toolCalls?.length) {
      return message.toolCalls.map((toolCall) => ({ index, toolCall }))
    }
    return message.toolCall ? [{ index, toolCall: message.toolCall }] : []
  })

  return entries.map(({ index, toolCall }) => {
    const superseded = (
      toolCall.status === 'error'
      && messages.slice(index + 1).some((message) => (
        message.role === 'tool'
        || message.role === 'user'
        || (message.role === 'assistant' && message.content.trim().length > 0)
      ))
    )

    return {
      ...toolCall,
      superseded,
    }
  })
}

export function cliMessageToSurfaceMessage(message: Message): SurfaceMessage | null {
  switch (message.role) {
    case 'system':
      return null
    case 'user':
    case 'assistant':
      return {
        id: message.id,
        role: message.role,
        content: message.content,
        citations: message.citations,
      }
    case 'tool':
      return {
        id: message.id,
        role: 'tool',
        content: message.toolCall?.arguments ?? message.content,
        toolName: message.toolCall?.name,
        toolInput: message.toolCall?.input,
        toolStatus: message.toolCall?.status,
        toolResult: message.toolCall?.output,
        toolMeta: message.toolCall?.meta,
        toolNeedsApproval:
          message.toolCall?.status === 'pending'
          || Boolean(message.toolCall?.approvalRequestId),
        approvalRequestId: message.toolCall?.approvalRequestId,
        approvalState: message.toolCall?.approvalState,
        resumeAvailable: message.toolCall?.resumeAvailable,
        toolDiff: message.toolCall?.editDiff,
      }
  }
}

export function surfaceMessageToCliMessage(
  message: SurfaceMessage,
  options?: {
    previousMessage?: Message
    timestamp?: number
  },
): Message {
  const timestamp =
    options?.timestamp ?? options?.previousMessage?.timestamp ?? Date.now()

  // Switching on role lets the compiler narrow the surface Message variants
  // exhaustively. The cli Message has no 'todo' or 'context' role of its own
  // so those branches must be matched explicitly before the user/assistant
  // fallthrough; otherwise their roles leak into cli Message['role'].
  switch (message.role) {
    case 'tool':
      return {
        id: message.id,
        role: 'tool',
        content: '',
        timestamp,
        toolCall: buildStoredToolCallState({
          id: message.id,
          name: message.toolName ?? 'tool',
          input: message.toolInput,
          status: message.toolStatus ?? 'running',
          output: message.toolResult,
          meta: message.toolMeta,
          approvalRequestId: message.approvalRequestId,
          approvalState: message.approvalState,
          resumeAvailable: message.resumeAvailable,
          editDiff: message.toolDiff,
        }),
      }
    case 'context':
      // Full snippets remain available through citations and the memory UI.
      // Keep the inline terminal transcript compact and avoid leaving profile
      // or document bodies in scrollback on every redraw.
      return {
        id: message.id,
        role: 'system',
        content: `Relevant context · ${summarizeMemoryContextItems(message.contextItems ?? []).detail}`,
        timestamp,
      }
    case 'todo':
      return {
        id: message.id,
        role: 'system',
        content: message.content,
        timestamp,
      }
    case 'system':
      return {
        id: message.id,
        role: 'system',
        content: message.content,
        timestamp,
      }
    case 'user':
    case 'assistant':
      return {
        id: message.id,
        role: message.role,
        content: message.content,
        citations:
          message.role === 'assistant' ? message.citations : undefined,
        attachments:
          message.role === 'user'
          && options?.previousMessage?.role === 'user'
            ? options.previousMessage.attachments
            : undefined,
        timestamp,
      }
  }
}

function toTimestamp(value: string | undefined): number {
  const parsed = value ? Date.parse(value) : Number.NaN
  return Number.isFinite(parsed) ? parsed : Date.now()
}

export function buildContextCompactCliMessage(
  event: ContextCompactEvent,
): Message {
  const detail = [
    `Context compacted: ${event.beforeTokens} -> ${event.afterTokens} tokens`,
    event.removedMessageCount || event.preservedMessageCount
      ? [
        event.removedMessageCount
          ? `compacted ${event.removedMessageCount} earlier messages`
          : null,
        event.preservedMessageCount
          ? `preserved ${event.preservedMessageCount} recent messages`
          : null,
      ].filter(Boolean).join(', ')
      : null,
    event.summary,
  ].filter(Boolean).join('\n')

  return {
    id: event.id,
    role: 'system',
    content: detail,
    timestamp: toTimestamp(event.timestamp),
  }
}

export function buildHydratedCliMessages(
  session: Pick<DaemonSessionDetail, 'events'>,
  messages: SurfaceMessage[],
): Message[] {
  const eventMetaById = new Map(
    (session.events ?? []).map((event, index) => [
      event.id,
      {
        index,
        timestamp: toTimestamp(event.timestamp),
      },
    ]),
  )

  return [
    ...messages.map((message, order) => {
      const meta = eventMetaById.get(message.id)
      const syntheticHistoryNoticeIndex = message.id.startsWith(
        SESSION_HISTORY_WINDOW_MESSAGE_ID_PREFIX,
      )
        ? -1
        : Number.MAX_SAFE_INTEGER
      return {
        index: meta?.index ?? syntheticHistoryNoticeIndex,
        order,
        message: surfaceMessageToCliMessage(message, {
          timestamp: meta?.timestamp ?? toTimestamp(message.timestamp),
        }),
      }
    }),
    ...(session.events ?? [])
      .flatMap((event, order) => (
        event.type === 'context_compact'
          ? [{
              index: eventMetaById.get(event.id)?.index ?? Number.MAX_SAFE_INTEGER,
              order: messages.length + order,
              message: buildContextCompactCliMessage(event),
            }]
          : []
      )),
  ]
    .sort((left, right) => (
      left.index - right.index || left.order - right.order
    ))
    .map((entry) => entry.message)
}

export function applyCliApprovalResolution(
  messages: Message[],
  toolCallId: string,
  approved: boolean,
): Message[] {
  const resolvedById = new Map(
    applyApprovalResolution(
      messages
        .map((message) => cliMessageToSurfaceMessage(message))
        .filter((message): message is SurfaceMessage => message !== null),
      toolCallId,
      approved,
    ).map((message) => [message.id, message]),
  )

  return messages.map((message) => {
    const resolved = resolvedById.get(message.id)
    return resolved
      ? surfaceMessageToCliMessage(resolved, { previousMessage: message })
      : message
  })
}

export function toCliPendingApproval(
  pendingApproval?: DaemonPendingApproval | null,
): ApprovalRequest | null {
  if (!pendingApproval) {
    return null
  }

  return {
    requestId: pendingApproval.requestId,
    sessionId: pendingApproval.sessionId,
    toolCallId: pendingApproval.toolCallId,
    toolName: pendingApproval.tool,
    input: pendingApproval.input,
    requestedAt: pendingApproval.requestedAt,
    expiresAt: pendingApproval.expiresAt,
    state: pendingApproval.state,
    resumeAvailable: pendingApproval.resumeAvailable,
  }
}

export function buildHydratedCliSessionState(
  session: DaemonSessionDetail,
  artifacts: DaemonArtifact[] = [],
): HydratedCliSessionState {
  const state = buildStoredSessionState(session)
  let phase: string | null = null
  const graphTrace: GraphTraceEntry[] = []
  let runWorkProgress: RunWorkProgress | null = null
  for (const event of session.events ?? []) {
    if (event.type === 'phase_change') {
      phase = event.enteredPhase ?? 'finalize'
    } else if (event.type === 'node_trace') {
      graphTrace.push({
        node: event.node,
        phase,
        status: 'completed',
        durationMs: event.durationMs,
        nextEdge: event.nextEdge,
      })
    } else if (event.type === 'state_board') {
      runWorkProgress = {
        criteriaTotal: event.board.completionCriteria.length,
        planTotal: event.board.plan.length,
        planDone: event.board.plan.filter((step) => step.status === 'done').length,
        todosTotal: event.board.todos.length,
        todosDone: event.board.todos.filter((todo) => todo.status === 'completed').length,
        plan: event.board.plan.map(({ title, status, depth }) => ({ title, status, depth })),
        todos: event.board.todos.map(({ content, status }) => ({ content, status })),
      }
    }
  }

  return {
    sessionId: session.id,
    messages: buildHydratedCliMessages(session, state.messages),
    artifacts,
    activities: state.activities,
    usage: {
      input: state.usage?.inputTokens ?? 0,
      output: state.usage?.outputTokens ?? 0,
      cost: state.usage?.estimatedCost ?? 0,
    },
    pendingApproval: toCliPendingApproval(state.pendingApproval),
    pendingQuestions: session.pendingQuestions ?? [],
    model: state.model,
    provider: state.provider,
    plannerWorkingMemory: session.plannerWorkingMemory ?? null,
    graphTrace: graphTrace.slice(-120),
    runWorkProgress,
  }
}
