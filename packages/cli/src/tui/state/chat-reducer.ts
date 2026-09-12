import { DEFAULT_CLI_AGENT_MODE } from '../../chat-options.js'
// Pure reducer for the cli chat hook.
//
// Lives outside useChat.ts because nothing inside it touches the
// websocket, http client, or any other runtime. It is referentially
// transparent given (state, action) — every dependency is either
// imported from helper modules or part of the action payload — which
// makes it trivial to unit-test the state transitions without
// rendering the hook.

import type {
  ActivityItem,
  DaemonAgentMode,
  DaemonArtifact,
  DebateRoundSummary,
  EditCheckpointSummary,
  PlannerWorkingMemory,
  AgentState as SurfaceAgentState,
  Message as SurfaceMessage,
  DaemonPendingQuestion,
} from '@sepilotd/api-client'
import { mergeArtifacts } from '@sepilotd/api-client'
import type {
  ApprovalRequest,
  ChatState,
  Message,
  MessageAttachment,
} from '../types.js'
import {
  applyCliApprovalResolution,
  collectToolCalls,
  surfaceMessageToCliMessage,
} from '../utils/surface.js'
import type { HydratedSessionState } from '../hooks/useSession.js'

export type ChatAction =
  | {
      type: 'SEND_MESSAGE'
      id?: string
      content: string
      attachments?: MessageAttachment[]
    }
  | { type: 'START_STREAM' }
  | { type: 'SESSION_SET'; sessionId: string }
  | { type: 'LOAD_SESSION'; session: HydratedSessionState }
  | { type: 'SET_HYDRATING_SESSION'; value: boolean }
  | {
      type: 'APPROVAL_RESOLVED'
      requestId: string
      toolCallId: string
      approved: boolean
    }
  | { type: 'SET_PENDING_APPROVAL'; approval: ApprovalRequest | null }
  | { type: 'ADD_PENDING_QUESTION'; question: DaemonPendingQuestion }
  | { type: 'QUESTION_ANSWERED'; questionId: string }
  | { type: 'MERGE_ARTIFACTS'; artifacts: DaemonArtifact[] }
  | { type: 'SET_STREAM_STATUS'; status: string | null }
  | { type: 'SET_THINKING'; content: string }
  | { type: 'SET_PROVIDER_WAIT'; wait: ChatState['providerWait'] }
  | { type: 'SET_CONTEXT_USAGE'; context: NonNullable<ChatState['contextUsage']> }
  | {
      type: 'SYNC_SURFACE_MESSAGES'
      messages: SurfaceMessage[]
      assistantId: string
      agentState: SurfaceAgentState
    }
  | {
      type: 'SET_ACTIVITIES'
      activities: ActivityItem[] | ((prev: ActivityItem[]) => ActivityItem[])
    }
  | { type: 'ERROR'; message: string }
  | { type: 'STREAM_FINALIZED' }
  | { type: 'SYSTEM_MESSAGE'; content: string }
  | { type: 'APPEND_MESSAGE'; message: Message }
  | { type: 'UPSERT_MESSAGE'; message: Message }
  | { type: 'NEW_SESSION' }
  | { type: 'SET_MODEL'; model: string }
  | { type: 'SET_PROVIDER'; provider: string }
  | { type: 'SET_PROJECT'; projectId: string | null; projectName: string | null }
  | { type: 'SET_MODE'; mode: DaemonAgentMode }
  | {
      type: 'SET_THINKING_LEVEL'
      thinkingLevel: 'off' | 'low' | 'medium' | 'high' | 'max'
    }
  | { type: 'SET_MAX_TOKENS'; maxTokens: number | null }
  | {
      type: 'SET_AUTONOMY'
      autonomy: 'readonly' | 'accept-edits' | 'workspace-write' | 'supervised' | 'autonomous'
    }
  | { type: 'CLEAR_ERROR' }
  | {
      type: 'SET_PLANNER_WORKING_MEMORY'
      workingMemory: PlannerWorkingMemory | null
    }
  | {
      type: 'APPEND_EDIT_ROLLBACK'
      checkpoint: EditCheckpointSummary
    }
  | {
      type: 'APPEND_DEBATE_ROUND'
      round: DebateRoundSummary
    }
  | { type: 'SET_PHASE'; phase: string | null }
  | {
      type: 'ENTER_GRAPH_NODE'
      node: string
      phase?: string | null
      lifecycleState?: string
      iteration?: number
    }
  | {
      type: 'COMPLETE_GRAPH_NODE'
      node: string
      durationMs: number
      nextEdge?: string
    }
  | {
      type: 'SET_STATE_BOARD_COUNTS'
      counts: ChatState['stateBoardCounts']
    }

export const initialState: ChatState = {
  messages: [],
  artifacts: [],
  activities: [],
  currentMessage: '',
  toolCalls: [],
  isStreaming: false,
  streamStatus: null,
  streamStartedAt: null,
  providerWait: null,
  isThinking: false,
  thinkingText: '',
  isHydratingSession: false,
  sessionId: null,
  projectId: null,
  projectName: null,
  model: 'default',
  provider: 'default',
  mode: DEFAULT_CLI_AGENT_MODE,
  thinkingLevel: 'medium',
  maxTokens: null,
  autonomy: 'supervised',
  usage: { input: 0, output: 0, cost: 0 },
  contextUsage: null,
  pendingApproval: null,
  pendingQuestions: [],
  denialFollowup: null,
  plannerWorkingMemory: null,
  error: null,
  currentPhase: null,
  graphTrace: [],
  runWorkProgress: null,
  stateBoardCounts: null,
}

function syncSurfaceMessages(
  state: ChatState,
  messages: SurfaceMessage[],
  assistantId: string,
  agentState: SurfaceAgentState,
): ChatState {
  const previousById = new Map(state.messages.map((message) => [message.id, message]))
  const activeAssistant = messages.find(
    (message) => message.id === assistantId && message.role === 'assistant',
  )
  const finalizeAssistant =
    Boolean(activeAssistant) && (agentState === 'done' || agentState === 'error')
  const nextSurfaceMessages = messages.filter(
    (message) => message.id !== assistantId || finalizeAssistant,
  )

  let nextMessages = state.messages.filter(
    (message) => message.id !== assistantId,
  )

  for (const message of nextSurfaceMessages) {
    const nextMessage = surfaceMessageToCliMessage(message, {
      previousMessage: previousById.get(message.id),
    })
    const existingIndex = nextMessages.findIndex(
      (candidate) => candidate.id === message.id,
    )

    if (existingIndex === -1) {
      nextMessages = [...nextMessages, nextMessage]
      continue
    }

    nextMessages = nextMessages.map((candidate, index) => (
      index === existingIndex ? nextMessage : candidate
    ))
  }

  const usageMessage = [...nextSurfaceMessages]
    .reverse()
    .find((message) => message.usage)

  return {
    ...state,
    messages: nextMessages,
    toolCalls: collectToolCalls(nextMessages),
    currentMessage:
      activeAssistant && !finalizeAssistant
        ? activeAssistant.content
        : '',
    isStreaming: agentState !== 'done' && agentState !== 'error',
    streamStatus:
      agentState === 'done' || agentState === 'error'
        ? null
        : state.streamStatus,
    streamStartedAt:
      agentState === 'done' || agentState === 'error'
        ? null
        : (state.streamStartedAt ?? Date.now()),
    providerWait:
      agentState === 'done' || agentState === 'error' ? null : state.providerWait,
    isThinking: agentState === 'thinking' ? state.isThinking : false,
    // Preserve the latest reasoning segment until the next turn starts so a
    // brief thinking burst does not disappear as soon as a tool or answer
    // event follows it.
    thinkingText: state.thinkingText,
    // Clear the StatusBar's phase/state-board badges once the run ends so a
    // finished run doesn't keep showing stale phase/criteria/todo counts
    // from the previous run on the next turn's status line.
    currentPhase:
      agentState === 'done' || agentState === 'error' ? null : state.currentPhase,
    stateBoardCounts:
      agentState === 'done' || agentState === 'error' ? null : state.stateBoardCounts,
    graphTrace:
      agentState === 'done' || agentState === 'error'
        ? (state.graphTrace ?? []).map((entry) => entry.status === 'running'
            ? {
                ...entry,
                status: agentState === 'error' ? 'error' as const : 'interrupted' as const,
              }
            : entry)
        : state.graphTrace,
    usage: usageMessage?.usage
      ? {
          input: usageMessage.usage.inputTokens,
          output: usageMessage.usage.outputTokens,
          cost: usageMessage.usage.estimatedCost ?? 0,
        }
      : state.usage,
  }
}

export function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case 'SEND_MESSAGE':
      return {
        ...state,
        messages: [
          ...state.messages,
          {
            id: action.id ?? crypto.randomUUID(),
            role: 'user',
            content: action.content,
            attachments: action.attachments,
            timestamp: Date.now(),
          },
        ],
        isStreaming: true,
        streamStatus: 'Connecting to daemon…',
        streamStartedAt: Date.now(),
        providerWait: null,
        isThinking: false,
        thinkingText: '',
        currentMessage: '',
        usage: { input: 0, output: 0, cost: 0 },
        contextUsage: null,
        error: null,
        denialFollowup: null,
        plannerWorkingMemory: null,
        graphTrace: [],
        runWorkProgress: null,
        currentPhase: null,
        stateBoardCounts: null,
      }
    case 'START_STREAM':
      return {
        ...state,
        isStreaming: true,
        streamStatus: 'Connecting to daemon…',
        streamStartedAt: Date.now(),
        providerWait: null,
        isThinking: false,
        thinkingText: '',
        currentMessage: '',
        usage: { input: 0, output: 0, cost: 0 },
        contextUsage: null,
        error: null,
        plannerWorkingMemory: null,
        graphTrace: [],
        runWorkProgress: null,
        currentPhase: null,
        stateBoardCounts: null,
      }
    case 'SESSION_SET':
      return { ...state, sessionId: action.sessionId }
    case 'LOAD_SESSION':
      return {
        ...state,
        sessionId: action.session.sessionId,
        projectId: state.projectId,
        projectName: state.projectName,
        messages: action.session.messages,
        artifacts: action.session.artifacts,
        activities: action.session.activities,
        toolCalls: collectToolCalls(action.session.messages),
        usage: action.session.usage,
        contextUsage: null,
        pendingApproval: action.session.pendingApproval,
        pendingQuestions: action.session.pendingQuestions,
        plannerWorkingMemory: action.session.plannerWorkingMemory,
        graphTrace: action.session.graphTrace,
        runWorkProgress: action.session.runWorkProgress,
        currentPhase: null,
        stateBoardCounts: null,
        provider: action.session.provider,
        model: action.session.model,
        currentMessage: '',
        isStreaming: false,
        streamStatus: null,
        streamStartedAt: null,
        providerWait: null,
        isThinking: false,
        thinkingText: '',
        isHydratingSession: false,
        error: null,
      }
    case 'SET_HYDRATING_SESSION':
      return {
        ...state,
        isHydratingSession: action.value,
        error: action.value ? null : state.error,
      }
    case 'APPROVAL_RESOLVED':
      {
        const messages = applyCliApprovalResolution(
          state.messages,
          action.toolCallId,
          action.approved,
        )
        const resolvedMatchesActive = state.pendingApproval?.requestId === action.requestId

        return {
          ...state,
          messages,
          toolCalls: collectToolCalls(messages),
          pendingApproval: resolvedMatchesActive ? null : state.pendingApproval,
          denialFollowup: !action.approved && resolvedMatchesActive
            ? {
                toolName: state.pendingApproval?.toolName ?? 'tool',
                at: Date.now(),
              }
            : state.denialFollowup,
        }
      }
    case 'SET_PENDING_APPROVAL':
      return {
        ...state,
        pendingApproval: action.approval,
      }
    case 'ADD_PENDING_QUESTION':
      return {
        ...state,
        pendingQuestions: [
          ...state.pendingQuestions.filter((question) => question.id !== action.question.id),
          action.question,
        ],
      }
    case 'QUESTION_ANSWERED':
      return {
        ...state,
        pendingQuestions: state.pendingQuestions.filter((question) => question.id !== action.questionId),
      }
    case 'MERGE_ARTIFACTS':
      return {
        ...state,
        artifacts: mergeArtifacts(state.artifacts, action.artifacts),
      }
    case 'SET_STREAM_STATUS':
      return {
        ...state,
        streamStatus: action.status,
      }
    case 'SET_THINKING':
      return {
        ...state,
        isThinking: true,
        thinkingText: action.content,
      }
    case 'SET_PROVIDER_WAIT':
      return { ...state, providerWait: action.wait }
    case 'SET_CONTEXT_USAGE':
      return { ...state, contextUsage: action.context }
    case 'SYNC_SURFACE_MESSAGES':
      return syncSurfaceMessages(
        state,
        action.messages,
        action.assistantId,
        action.agentState,
      )
    case 'SET_ACTIVITIES':
      return {
        ...state,
        activities: typeof action.activities === 'function'
          ? action.activities(state.activities)
          : action.activities,
      }
    // Terminal fallback when a done event arrives after the live-stream
    // binding was torn down (e.g. an approval-response race nulled it).
    // Without this the elapsed timer and "in progress" status never clear.
    case 'STREAM_FINALIZED':
      return {
        ...state,
        currentMessage: '',
        isStreaming: false,
        streamStatus: null,
        streamStartedAt: null,
        providerWait: null,
        isThinking: false,
        currentPhase: null,
        stateBoardCounts: null,
        graphTrace: (state.graphTrace ?? []).map((entry) => entry.status === 'running'
          ? { ...entry, status: 'interrupted' as const }
          : entry),
      }
    case 'ERROR':
      return {
        ...state,
        error: action.message,
        isStreaming: false,
        streamStatus: null,
        streamStartedAt: null,
        providerWait: null,
        isThinking: false,
        isHydratingSession: false,
        // Clear stale StatusBar badges from the failed run.
        currentPhase: null,
        stateBoardCounts: null,
        graphTrace: (state.graphTrace ?? []).map((entry, index, entries) => (
          entry.status === 'running' && index === entries.length - 1
            ? { ...entry, status: 'error' as const }
            : entry
        )),
      }
    case 'SYSTEM_MESSAGE':
      return {
        ...state,
        messages: [
          ...state.messages,
          {
            id: crypto.randomUUID(),
            role: 'system',
            content: action.content,
            timestamp: Date.now(),
          },
        ],
      }
    case 'APPEND_MESSAGE': {
      const messages = [...state.messages, action.message]
      return {
        ...state,
        messages,
        toolCalls: collectToolCalls(messages),
      }
    }
    case 'UPSERT_MESSAGE': {
      const existingIndex = state.messages.findIndex((message) => (
        message.id === action.message.id
      ))
      const messages = existingIndex === -1
        ? [...state.messages, action.message]
        : state.messages.map((message, index) => (
            index === existingIndex ? action.message : message
          ))
      return {
        ...state,
        messages,
        toolCalls: collectToolCalls(messages),
      }
    }
    case 'NEW_SESSION':
      return {
        ...initialState,
        model: state.model,
        provider: state.provider,
        projectId: state.projectId,
        projectName: state.projectName,
        mode: state.mode,
        thinkingLevel: state.thinkingLevel,
        maxTokens: state.maxTokens,
        autonomy: state.autonomy,
      }
    case 'SET_MODEL':
      return { ...state, model: action.model, contextUsage: null }
    case 'SET_PROVIDER':
      return { ...state, provider: action.provider, contextUsage: null }
    case 'SET_PROJECT':
      return {
        ...state,
        projectId: action.projectId,
        projectName: action.projectName,
      }
    case 'SET_MODE':
      return { ...state, mode: action.mode }
    case 'SET_THINKING_LEVEL':
      return { ...state, thinkingLevel: action.thinkingLevel }
    case 'SET_MAX_TOKENS':
      return { ...state, maxTokens: action.maxTokens }
    case 'SET_AUTONOMY':
      return { ...state, autonomy: action.autonomy }
    case 'CLEAR_ERROR':
      return { ...state, error: null }
    case 'SET_PLANNER_WORKING_MEMORY':
      return { ...state, plannerWorkingMemory: action.workingMemory }
    case 'APPEND_EDIT_ROLLBACK': {
      const current = state.editRollbacks ?? []
      const existingIndex = current.findIndex(
        (checkpoint) => checkpoint.checkpointId === action.checkpoint.checkpointId,
      )
      const next =
        existingIndex === -1
          ? [...current, action.checkpoint]
          : current.map((checkpoint, index) =>
              index === existingIndex ? action.checkpoint : checkpoint,
            )
      return {
        ...state,
        editRollbacks: next.slice(-8),
      }
    }
    case 'APPEND_DEBATE_ROUND':
      return {
        ...state,
        debateRounds: [...(state.debateRounds ?? []), action.round].slice(-8),
      }
    case 'SET_PHASE':
      return { ...state, currentPhase: action.phase }
    case 'ENTER_GRAPH_NODE': {
      const current = state.graphTrace ?? []
      const latest = current.at(-1)
      if (latest?.status === 'running' && latest.node === action.node) {
        return state
      }
      const settled = current.map((entry) => entry.status === 'running'
        ? { ...entry, status: 'interrupted' as const }
        : entry)
      return {
        ...state,
        graphTrace: [
          ...settled,
          {
            node: action.node,
            phase: action.phase,
            lifecycleState: action.lifecycleState,
            iteration: action.iteration,
            status: 'running' as const,
          },
        ].slice(-120),
      }
    }
    case 'COMPLETE_GRAPH_NODE': {
      const current = state.graphTrace ?? []
      const targetIndex = [...current]
        .map((entry, index) => ({ entry, index }))
        .reverse()
        .find(({ entry }) => entry.node === action.node && entry.status === 'running')
        ?.index
      if (targetIndex === undefined) {
        return {
          ...state,
          graphTrace: [
            ...current,
            {
              node: action.node,
              phase: state.currentPhase,
              status: 'completed' as const,
              durationMs: action.durationMs,
              nextEdge: action.nextEdge,
            },
          ].slice(-120),
        }
      }
      return {
        ...state,
        graphTrace: current.map((entry, index) => index === targetIndex
          ? {
              ...entry,
              status: 'completed' as const,
              durationMs: action.durationMs,
              nextEdge: action.nextEdge,
            }
          : entry),
      }
    }
    case 'SET_STATE_BOARD_COUNTS':
      return {
        ...state,
        stateBoardCounts: action.counts,
        runWorkProgress: action.counts ?? state.runWorkProgress,
      }
    default:
      return state
  }
}
