import { useCallback, useEffect, useReducer, useRef } from 'react'
import {
  ApiHttpError,
  buildRecoveredApprovalPrompt,
  createTimedOutChatRunError,
  createTokenSpeedTracker,
  createStreamEventController,
  forwardDaemonStream,
  isTerminalDaemonChatPayload,
  resolveDaemonChatRunAbortReason,
  runDaemonWsClientChat,
  type DaemonArtifact,
  type DaemonChatStreamPayload,
  type DaemonChatRunController,
  type DaemonSessionDetail,
  type ChatSkillRef,
  type AgentState as SurfaceAgentState,
} from '@sepilotd/api-client'
import type { ApprovalRequest, MessageAttachment } from '../types.js'
import { buildCliChatOptions } from '../../chat-options.js'
import type { DaemonClient } from '../../client/http.js'
import type { DaemonWsClient } from '../../client/ws.js'
import {
  formatApiError,
  formatStreamError,
  friendlyErrorMessage,
} from '../../utils/error-message.js'
import { hydrateSession } from './useSession.js'
import { prepareAttachmentUploads } from '../utils/attachments.js'
import { buildRunProgressActivity, upsertRunProgressActivity } from '../utils/run-progress.js'
import { chatReducer, initialState } from '../state/chat-reducer.js'
import {
  buildInitialLiveStreamMessages,
  createLiveStreamBindings,
  handleAbortOutcome as handleAbortOutcomeHelper,
  type LiveStreamSlot,
} from '../state/live-stream.js'
import { applySlashCommandPrefix } from '../state/slash-prefix.js'
import {
  approvalRequestFromPayload,
  buildAlreadyHandledApprovalNotice,
  buildApprovalRepeatKey,
  buildMissingApprovalNotice,
  buildStaleApprovalNotice,
} from '../state/approval-helpers.js'
import { formatJudgmentFeedLine } from '../state/judgment-feed.js'
import { isTuiDebugLevel, logPreview, tuiLog, tuiLogError } from '../utils/debug-log.js'
import { useActiveRunController } from './useActiveRunController.js'
import { sanitizeTuiAgentResponse } from '../utils/agent-response-safety.js'
import {
  openChatStreamWithConnectTimeout,
  resolveCliStreamConnectMs,
} from '../../utils/stream-connect.js'

export { chatReducer, initialState }

const RUN_PROGRESS_INITIAL_DELAY_MS = 10_000
const RUN_PROGRESS_INTERVAL_MS = 15_000
const PREFLIGHT_ROUTING_STATUS_DELAY_MS = 1_500
const PREFLIGHT_SLOW_STATUS_DELAY_MS = 10_000

export interface PendingAttachmentInput {
  path: string
}

const AGENT_MENTION_REGEX = /^@([a-z0-9][a-z0-9-_]*)[ \t]+([\s\S]*)$/i
const SLASH_COMMAND_REGEX = /^\/([a-z0-9][a-z0-9-_]*)\b\s*([\s\S]*)$/i

function normalizeMentionAttachmentPath(path: string): string {
  return path
    .trim()
    .replace(/^@/, '')
    .replace(/[\\/]+$/, '')
}

function attachmentMentionAliases(path: string): string[] {
  const normalized = normalizeMentionAttachmentPath(path)
  const basename = normalized.split(/[\\/]/).filter(Boolean).at(-1)
  return basename && basename !== normalized ? [normalized, basename] : [normalized]
}

export function parseAgentMention(
  content: string,
  options: { attachmentPaths?: Iterable<string> } = {},
): { agentId: string; rest: string } | null {
  const match = content.match(AGENT_MENTION_REGEX)
  if (!match) return null
  const rest = match[2].trim()
  if (!rest) return null
  const agentId = match[1]
  const normalizedAgentId = normalizeMentionAttachmentPath(agentId)
  for (const path of options.attachmentPaths ?? []) {
    if (attachmentMentionAliases(path).includes(normalizedAgentId)) {
      return null
    }
  }
  return { agentId, rest }
}

export function parseSlashCommand(content: string): { commandId: string; args: string } | null {
  const match = content.match(SLASH_COMMAND_REGEX)
  if (!match) return null
  return { commandId: match[1], args: match[2].trim() }
}

export function normalizeError(error: unknown, opts: { sessionId?: string } = {}): string {
  // Prefer the same friendly stream / http copy the cli surfaces use,
  // so a `terminated` SSE drop or a 503 from the daemon shows up in the
  // TUI as a multi-line guided message instead of a raw stack-trace
  // snippet. Falls back to friendlyErrorMessage (json-unwrapped Error
  // .message) for anything we don't classify.
  const message = (
    formatStreamError(error, { sessionId: opts.sessionId }) ??
    formatApiError(error) ??
    friendlyErrorMessage(error)
  )
  return formatDaemonEventError(message)
}

export function formatDaemonEventError(message: string): string {
  let code = ''
  let detail = message
  try {
    const parsed = JSON.parse(message) as {
      error?: { code?: unknown; message?: unknown } | string
    }
    if (typeof parsed.error === 'string') detail = parsed.error
    else if (parsed.error && typeof parsed.error === 'object') {
      if (typeof parsed.error.code === 'string') code = parsed.error.code
      if (typeof parsed.error.message === 'string') detail = parsed.error.message
    }
  } catch { /* keep non-JSON daemon messages unchanged */ }

  if (code === 'UNAUTHORIZED' || /invalid token/i.test(detail)) {
    return [
      'Daemon authentication failed.',
      'Make sure SEPILOTD_DATA_DIR or SEPILOT_DAEMON_TOKEN_FILE points to the token used by the running daemon.',
    ].join('\n')
  }
  return sanitizeTuiAgentResponse(detail).text
}

function buildPersistentErrorLine(message: string): string {
  return `Run failed: ${message}`
}

function buildTransientErrorStateMessage(message: string): string {
  if (message.startsWith('Connection to the daemon was lost mid-run.')) {
    return 'Daemon connection lost mid-run.'
  }
  return message
}

function buildUnsavedFreshSessionNotice(): string {
  return "This message isn't saved to a session yet — retry (↑ + Enter) or start over with /new."
}

function isAlreadyHandledApprovalError(error: unknown): boolean {
  const status =
    error instanceof ApiHttpError
      ? error.status
      : error instanceof Error && /^404(?:\b|:)/.test(error.message)
        ? 404
        : null
  if (status !== 404) return false

  const detail =
    error instanceof ApiHttpError
      ? `${error.code ?? ''}\n${error.message}\n${error.rawBody}`
      : error instanceof Error
        ? error.message
        : String(error)
  return (
    /approval request/i.test(detail) && /(not found|no longer exists|already handled)/i.test(detail)
  )
}

async function assertStreamingResponse(response: Response): Promise<Response> {
  if (response.ok && response.body) {
    return response
  }

  const body = await response.text()
  throw new Error(body || `Request failed with status ${response.status}`)
}

async function uploadAttachments(
  httpClient: Pick<DaemonClient, 'uploadFiles'>,
  attachments: PendingAttachmentInput[],
): Promise<{
  fileIds: string[]
  messageAttachments: MessageAttachment[]
  notices: string[]
}> {
  const prepared = await prepareAttachmentUploads(
    Array.from(new Map(attachments.map((attachment) => [attachment.path, attachment])).values()),
  )

  const uploaded = await httpClient.uploadFiles(
    prepared.uploads.map((file) => ({
      filename: file.filename,
      content: file.content,
    })),
  )

  return {
    fileIds: uploaded.files.map((file) => file.id),
    messageAttachments: prepared.uploads.map((file, index) => ({
      id: uploaded.files[index]?.id,
      path: file.path,
      filename: file.filename,
    })),
    notices: prepared.notices,
  }
}

// Narrow the http surface that useChat depends on so the hook can be
// unit-tested with a hand-rolled stub instead of a full DaemonClient. The
// type is derived via Pick<> so any signature drift in DaemonClient still
// flows through here at compile time. The ws client must remain the full
// type because it is forwarded to runDaemonWsClientChat() which expects
// DaemonWsClient.connect() / .chat() in addition to .close().
export type ChatHttpClient = Pick<
  DaemonClient,
  | 'chatStream'
  | 'cancelActiveRun'
  | 'answerSessionQuestion'
  | 'redoSession'
  | 'resolveUserCommand'
  | 'respondApproval'
  | 'resumeApprovalStream'
  | 'resumeSessionStream'
  | 'undoSession'
  | 'uploadFiles'
>

export function useChat(wsClient: DaemonWsClient | null, httpClient: ChatHttpClient) {
  const [state, dispatch] = useReducer(chatReducer, initialState)
  const cancellationBarrierRef = useRef<Promise<void>>(Promise.resolve())
  const pendingApprovalRef = useRef<ApprovalRequest | null>(null)
  const liveStreamRef = useRef<LiveStreamSlot | null>(null)
  const tokenSpeedTrackerRef = useRef(createTokenSpeedTracker())
  const activeTokenSpeedStartedAtRef = useRef<number | null>(null)
  const tokenSpeedSessionIdRef = useRef<string | null>(null)
  const runController = useActiveRunController()
  const runProgressActivityRef = useRef<{
    id: string
    startedAt: number
  } | null>(null)
  const approvalRepeatCountsRef = useRef(new Map<string, number>())
  const alreadyHandledNoticeRef = useRef(new Set<string>())
  // Guards against double-submitting one approval: a single keypress can
  // reach resolveApproval from both the global keybinding and the modal's
  // onResolve in the same tick (pendingApproval clears asynchronously), which
  // used to fire two POSTs — one 404, one 200 — and the loser's error path
  // tore down the live stream mid-run.
  const approvalRespondInFlightRef = useRef(new Set<string>())
  const currentRunPhaseRef = useRef<string | null>(null)
  const currentRunSessionIdRef = useRef<string | null>(null)

  useEffect(() => {
    pendingApprovalRef.current = state.pendingApproval
  }, [state.pendingApproval])

  useEffect(() => {
    const previousSessionId = tokenSpeedSessionIdRef.current
    if (previousSessionId && previousSessionId !== state.sessionId) {
      activeTokenSpeedStartedAtRef.current = null
      tokenSpeedTrackerRef.current.reset()
    }
    tokenSpeedSessionIdRef.current = state.sessionId
  }, [state.sessionId])

  useEffect(() => {
    const startedAt = state.streamStartedAt
    if (!state.isStreaming || startedAt == null) {
      runProgressActivityRef.current = null
      return
    }

    if (runProgressActivityRef.current?.startedAt !== startedAt) {
      runProgressActivityRef.current = {
        id: crypto.randomUUID(),
        startedAt,
      }
    }

    let interval: ReturnType<typeof setInterval> | null = null
    const emitProgress = () => {
      const current = runProgressActivityRef.current
      if (!current) {
        return
      }

      const activity = buildRunProgressActivity({
        id: current.id,
        startedAt: current.startedAt,
        now: Date.now(),
        streamStatus: state.streamStatus,
        currentMessage: state.currentMessage,
        pendingApprovalToolName: state.pendingApproval?.toolName ?? null,
        pendingQuestionPrompt: state.pendingQuestions[0]?.prompt ?? null,
      })

      dispatch({
        type: 'SET_ACTIVITIES',
        activities: (prev) => upsertRunProgressActivity(prev, activity),
      })
    }

    const timeout = setTimeout(() => {
      emitProgress()
      interval = setInterval(emitProgress, RUN_PROGRESS_INTERVAL_MS)
    }, RUN_PROGRESS_INITIAL_DELAY_MS)

    return () => {
      clearTimeout(timeout)
      if (interval) {
        clearInterval(interval)
      }
    }
  }, [
    state.currentMessage,
    state.isStreaming,
    state.pendingApproval?.toolName,
    state.pendingQuestions,
    state.streamStartedAt,
    state.streamStatus,
  ])

  const beginRunController = runController.begin
  const beginDetachedRunController = runController.beginDetached
  const clearRunController = runController.clear

  const openStreamResponse = useCallback(
    (responsePromise: Promise<Response>, controller: DaemonChatRunController) => {
      const timeoutMs = resolveCliStreamConnectMs()
      const timeoutError = createTimedOutChatRunError(timeoutMs)
      return openChatStreamWithConnectTimeout(responsePromise, {
        timeoutMs,
        timeoutError,
        abort: () => controller.cancel(timeoutError),
      })
    },
    [],
  )

  const beginTokenSpeedRun = useCallback(() => {
    const startedAt = Date.now()
    activeTokenSpeedStartedAtRef.current = startedAt
    return startedAt
  }, [])

  const clearTokenSpeedRun = useCallback((startedAt: number) => {
    if (activeTokenSpeedStartedAtRef.current === startedAt) {
      activeTokenSpeedStartedAtRef.current = null
    }
  }, [])

  const getTokenSpeedStats = useCallback(() => tokenSpeedTrackerRef.current.snapshot(), [])

  const resetTokenSpeedStats = useCallback(() => {
    activeTokenSpeedStartedAtRef.current = null
    tokenSpeedTrackerRef.current.reset()
  }, [])

  const handleAbortOutcome = useCallback(
    (error: unknown): boolean => handleAbortOutcomeHelper({ error, liveStreamRef, dispatch }),
    [],
  )

  const beginLiveStream = useCallback(
    (params?: {
      userMessage?: { id: string; content: string }
      doneFallbackText?: string
      errorFallbackText?: string
      errorLabel?: string
      approvalResumeAvailable?: boolean
    }) => {
      const assistantId = crypto.randomUUID()
      const messages = buildInitialLiveStreamMessages({
        storedMessages: state.messages,
        assistantId,
        userMessage: params?.userMessage,
      })

      const liveStream: LiveStreamSlot = {
        assistantId,
        agentState: 'thinking' as SurfaceAgentState,
        messages,
        controller: createStreamEventController(
          createLiveStreamBindings({
            assistantId,
            liveStreamRef,
            dispatch,
            sanitizeAssistantContent: (content) => sanitizeTuiAgentResponse(content).text,
          }),
          {
            assistantId,
            createId: () => crypto.randomUUID(),
            doneFallbackText: params?.doneFallbackText ?? 'No response',
            errorFallbackText: params?.errorFallbackText ?? 'Request failed',
            errorLabel: params?.errorLabel ?? 'Run failed',
            approvalResumeAvailable: params?.approvalResumeAvailable,
            formatErrorMessage: formatDaemonEventError,
            onSupersededProgress: (content) => {
              dispatch({
                type: 'SYSTEM_MESSAGE',
                content: `Progress snapshot: ${content}`,
              })
            },
          },
        ),
      }

      liveStreamRef.current = liveStream
      return liveStream
    },
    [state.messages],
  )

  const handlePayload = useCallback(
    async (payload: DaemonChatStreamPayload) => {
      if ('sessionId' in payload && !('type' in payload)) {
        if (typeof payload.sessionId === 'string') {
          currentRunSessionIdRef.current = payload.sessionId
          dispatch({ type: 'SESSION_SET', sessionId: payload.sessionId })
        }
        return
      }

      if ('artifacts' in payload) {
        if ('sessionId' in payload && payload.sessionId) {
          currentRunSessionIdRef.current = payload.sessionId
          dispatch({ type: 'SESSION_SET', sessionId: payload.sessionId })
        }
        dispatch({ type: 'MERGE_ARTIFACTS', artifacts: payload.artifacts })
        return
      }

      if (!('type' in payload)) {
        return
      }

      const payloadSessionId =
        typeof (payload as { sessionId?: unknown }).sessionId === 'string'
          ? (payload as { sessionId: string }).sessionId
          : undefined
      if (payloadSessionId && payloadSessionId !== currentRunSessionIdRef.current) {
        currentRunSessionIdRef.current = payloadSessionId
        dispatch({ type: 'SESSION_SET', sessionId: payloadSessionId })
      }

      // With SEPILOT_DEBUG=1, trace key transitions and, at
      // SEPILOT_CLI_LOG_LEVEL=debug, every payload type. This reconstructs a
      // "why did the TUI do that?" report from ~/.sepilotd/logs.
      if (
        isTuiDebugLevel() ||
        payload.type === 'state_change' ||
        payload.type === 'error' ||
        payload.type === 'done' ||
        payload.type === 'approval_request' ||
        payload.type === 'approval_response' ||
        payload.type === 'question_request' ||
        payload.type === 'phase_change'
      ) {
        tuiLog('stream.event', {
          sessionId: currentRunSessionIdRef.current ?? state.sessionId ?? undefined,
          type: payload.type,
          ...(payload.type === 'state_change' ? { state: payload.state } : {}),
          ...(payload.type === 'error'
            ? { code: payload.error?.code, message: logPreview(payload.error?.message) }
            : {}),
          ...(payload.type === 'approval_request'
            ? { requestId: payload.requestId, tool: payload.toolCall?.name }
            : {}),
          ...(payload.type === 'question_request'
            ? { questionId: payload.questionId, prompt: logPreview(payload.prompt) }
            : {}),
        })
      }

      if (payload.type === 'done') {
        const startedAt = activeTokenSpeedStartedAtRef.current
        if (startedAt !== null) {
          tokenSpeedTrackerRef.current.recordRun({
            startedAt,
            finishedAt: Date.now(),
            usage: payload.usage,
          })
          activeTokenSpeedStartedAtRef.current = null
        }
      } else if (payload.type === 'error') {
        activeTokenSpeedStartedAtRef.current = null
      }

      if (payload.type === 'planner_working_memory_updated') {
        dispatch({
          type: 'SET_PLANNER_WORKING_MEMORY',
          workingMemory: payload.workingMemory,
        })
      } else if (
        payload.type === 'edit_checkpoint_opened'
        || payload.type === 'edit_checkpoint_resolved'
      ) {
        dispatch({
          type: 'APPEND_EDIT_ROLLBACK',
          checkpoint: payload.checkpoint,
        })
      } else if (payload.type === 'debate_round') {
        dispatch({
          type: 'APPEND_DEBATE_ROUND',
          round: payload.round,
        })
      } else if (payload.type === 'node_trace') {
        dispatch({
          type: 'COMPLETE_GRAPH_NODE',
          node: payload.node,
          durationMs: payload.durationMs,
          nextEdge: payload.nextEdge,
        })
      } else if (payload.type === 'phase_change') {
        // Surface phase transitions on the status line so operators can
        // see which named stage the run is in (implementation /
        // validation / review / finalize) and how many tokens the
        // previous phase ended up burning.
        const entered = payload.enteredPhase ?? 'finalize'
        currentRunPhaseRef.current = entered
        dispatch({ type: 'SET_PHASE', phase: entered })
        const closedTokens = payload.closedPhase
          ? payload.closedPhase.usage.inputTokens + payload.closedPhase.usage.outputTokens
          : null
        const closedSummary = payload.closedPhase
          ? ` (closed ${payload.closedPhase.phase}: ${closedTokens} tokens)`
          : ''
        dispatch({
          type: 'SET_STREAM_STATUS',
          status: `Phase: ${entered}${closedSummary}`,
        })
      } else if (payload.type === 'state_board') {
        if (payload.currentNode) {
          dispatch({
            type: 'ENTER_GRAPH_NODE',
            node: payload.currentNode,
            phase: currentRunPhaseRef.current,
            lifecycleState: payload.nodeState,
            iteration: payload.iteration,
          })
        }
        // Counts-only progress snapshot from the agent state board (Task 6
        // frame). StatusBar renders these structured numbers directly — no
        // text parsing, no dataset-specific inference.
        dispatch({
          type: 'SET_STATE_BOARD_COUNTS',
          counts: {
            criteriaTotal: payload.criteriaTotal,
            planTotal: payload.planTotal,
            planDone: payload.planDone,
            todosTotal: payload.todosTotal,
            todosDone: payload.todosDone,
            todos: payload.todos,
            plan: payload.plan,
          },
        })
      } else if (payload.type === 'post_edit_findings') {
        // Status surfaces blast radius from the post-edit pass —
        // broken callers in particular are a near-failure signal the
        // user should see immediately rather than waiting for the
        // next reasoning turn to mention them.
        const broken = payload.diagnostics.filter((d) => d.summary.startsWith('[caller]'))
        const editedSummary = payload.editedFiles.slice(0, 2).join(', ')
        const status =
          broken.length > 0
            ? `Post-edit: ${broken.length} caller(s) broken` +
              (editedSummary ? ` after editing ${editedSummary}` : '')
            : `Post-edit findings on ${editedSummary || 'edited files'}`
        dispatch({
          type: 'SET_STREAM_STATUS',
          status,
        })
      }

      // Judgment feed: surface the daemon's decision events (reasoning
      // steps, routing choices, quality-gate verdicts, backtracks) as dim
      // one-line transcript entries so the operator can see *why* the run
      // is doing what it does instead of a silent spinner.
      const judgmentLine = formatJudgmentFeedLine(
        payload as { type: string } & Record<string, unknown>,
      )
      if (judgmentLine) {
        dispatch({ type: 'SYSTEM_MESSAGE', content: judgmentLine })
      }

      if (payload.type === 'execution_policy') {
        dispatch({
          type: 'SET_AUTONOMY',
          autonomy: payload.policy.effectiveAutonomy,
        })
        if (payload.policy.clamped) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: [
              `Permission clamp: requested ${payload.policy.requestedAutonomy ?? 'default'}`,
              `but daemon effective autonomy is ${payload.policy.effectiveAutonomy}.`,
              payload.policy.clampReason,
            ].filter(Boolean).join(' '),
          })
        }
      }

      if (payload.type === 'approval_request') {
        const approvalSessionId =
          payloadSessionId ?? currentRunSessionIdRef.current ?? state.sessionId
        const approval = approvalRequestFromPayload({
          payload,
          sessionId: approvalSessionId,
          phase: currentRunPhaseRef.current,
        })
        const repeatKey = buildApprovalRepeatKey(approval)
        const repeatCount = (approvalRepeatCountsRef.current.get(repeatKey) ?? 0) + 1
        approvalRepeatCountsRef.current.set(repeatKey, repeatCount)
        dispatch({
          type: 'SET_PENDING_APPROVAL',
          approval: {
            ...approval,
            repeatCount: repeatCount > 1 ? repeatCount : undefined,
          },
        })
      } else if (payload.type === 'question_request') {
        const questionSessionId =
          payload.sessionId ?? currentRunSessionIdRef.current ?? state.sessionId ?? ''
        dispatch({
          type: 'ADD_PENDING_QUESTION',
          question: {
            id: payload.questionId,
            sessionId: questionSessionId,
            prompt: payload.prompt,
            choices: payload.choices,
          },
        })
      } else if (
        payload.type === 'tool_result' &&
        pendingApprovalRef.current?.toolCallId === payload.toolCallId
      ) {
        dispatch({ type: 'SET_PENDING_APPROVAL', approval: null })
      }

      const liveStream = liveStreamRef.current
      if (liveStream) {
        const safePayload =
          payload.type === 'message'
            ? { ...payload, content: sanitizeTuiAgentResponse(payload.content ?? '').text }
            : payload
        liveStream.controller.handleEvent(safePayload)
        if (payload.type === 'done' || payload.type === 'error') {
          liveStreamRef.current = null
        }
        return
      }

      switch (payload.type) {
        case 'thinking':
          if (payload.content) {
            dispatch({ type: 'SET_THINKING', content: payload.content })
          }
          dispatch({
            type: 'SET_STREAM_STATUS',
            status: 'Reasoning…',
          })
          break
        // The live-stream binding can be torn down before the terminal event
        // arrives (approval-response races null liveStreamRef). Without this
        // fallback the done event is dropped and isStreaming stays true
        // forever — the "in progress • thinking • 23m" stuck-timer bug.
        case 'done':
          dispatch({ type: 'STREAM_FINALIZED' })
          break
        case 'error':
          dispatch({
            type: 'ERROR',
            message: payload.error?.message ?? 'Unknown error',
          })
          break
        default:
          break
      }
    },
    [state.sessionId],
  )

  const consumeSseResponse = useCallback(
    async (response: Response, signal?: AbortSignal, onActivity?: () => void) => {
      try {
        const stream = await assertStreamingResponse(response)
        await forwardDaemonStream<DaemonChatStreamPayload>(
          stream,
          async (payload) => {
            onActivity?.()
            await handlePayload(payload)
          },
          {
            isTerminalEvent: isTerminalDaemonChatPayload,
          },
        )
      } catch (error) {
        throw resolveDaemonChatRunAbortReason(signal, error) ?? error
      }
    },
    [handlePayload],
  )

  const consumeWsResponse = useCallback(
    async (
      content: string,
      fileIds?: string[],
      skillRefs?: ChatSkillRef[],
      signal?: AbortSignal,
      onActivity?: () => void,
      targetSessionId?: string,
    ) => {
      if (!wsClient) {
        throw new Error('WebSocket client unavailable')
      }

      const chatOptions = buildCliChatOptions({
        model: state.model,
        provider: state.provider,
        mode: state.mode,
        thinkingLevel: state.thinkingLevel,
        autonomy: state.autonomy,
        maxTokens: state.maxTokens,
        projectId: state.projectId,
        fileIds,
        skillRefs,
      })

      try {
        await runDaemonWsClientChat({
          wsClient,
          message: content,
          sessionId: targetSessionId,
          options: chatOptions,
          onEvent: async (payload) => {
            onActivity?.()
            await handlePayload(payload)
          },
        })
      } catch (error) {
        throw resolveDaemonChatRunAbortReason(signal, error) ?? error
      }

      const abortReason = resolveDaemonChatRunAbortReason(signal)
      if (abortReason) {
        throw abortReason
      }
    },
    [
      handlePayload,
      state.autonomy,
      state.maxTokens,
      state.mode,
      state.model,
      state.projectId,
      state.provider,
      state.thinkingLevel,
      wsClient,
    ],
  )

  const startStream = useCallback(
    async (
      content: string,
      recordUserMessage: boolean,
      options?: {
        attachments?: MessageAttachment[]
        displayContent?: string
        fileIds?: string[]
        skillRefs?: ChatSkillRef[]
        targetSessionId?: string
      },
    ): Promise<boolean> => {
      // Local transport abort is immediate, while daemon-side provider and
      // lease cleanup is asynchronous. Serialize the next turn behind the
      // remote cancellation acknowledgment so users can cancel and retype
      // without racing the previous run's session lease.
      await cancellationBarrierRef.current
      const streamSessionId = options?.targetSessionId ?? state.sessionId ?? undefined
      const prefix = await applySlashCommandPrefix({
        content,
        sessionId: streamSessionId ?? null,
        httpClient,
        dispatch,
      })
      if (prefix.applied && prefix.outcome === 'short-circuit') {
        return false
      }
      const workingContent =
        prefix.applied && prefix.outcome === 'rewritten' ? prefix.workingContent : content
      const resolvedAgent =
        prefix.applied && prefix.outcome === 'rewritten' ? prefix.resolvedAgent : undefined
      const attachmentPaths = options?.attachments?.map((attachment) => attachment.path) ?? []
      const mention = parseAgentMention(workingContent, { attachmentPaths })
      const effectiveContent = mention?.rest ?? workingContent
      const workingDisplayContent = options?.displayContent ?? workingContent
      const displayMention = parseAgentMention(workingDisplayContent, { attachmentPaths })
      const effectiveDisplayContent = options?.displayContent
        ? (displayMention?.rest ?? workingDisplayContent)
        : effectiveContent
      const effectiveMode = mention?.agentId ?? resolvedAgent ?? state.mode
      const userMessageId = recordUserMessage ? crypto.randomUUID() : undefined
      const runController = beginRunController()
      const tokenSpeedStartedAt = beginTokenSpeedRun()
      approvalRepeatCountsRef.current.clear()
      currentRunPhaseRef.current = null
      currentRunSessionIdRef.current = streamSessionId ?? null

      tuiLog('run.start', {
        sessionId: streamSessionId,
        mode: effectiveMode,
        recordUserMessage,
        content: logPreview(effectiveContent),
      })

      dispatch(
        recordUserMessage
          ? {
              type: 'SEND_MESSAGE',
              id: userMessageId,
              content: effectiveDisplayContent,
              attachments: options?.attachments,
            }
          : { type: 'START_STREAM' },
      )
      beginLiveStream({
        userMessage:
          recordUserMessage && userMessageId
            ? { id: userMessageId, content: effectiveDisplayContent }
            : undefined,
      })

      let preflightSettled = false
      const routingStatusTimer = setTimeout(() => {
        if (!preflightSettled && !runController.signal.aborted) {
          dispatch({ type: 'SET_STREAM_STATUS', status: 'Daemon preflight: routing request…' })
        }
      }, PREFLIGHT_ROUTING_STATUS_DELAY_MS)
      const slowStatusTimer = setTimeout(() => {
        if (!preflightSettled && !runController.signal.aborted) {
          dispatch({
            type: 'SET_STREAM_STATUS',
            status: 'Daemon preflight is still running; it will time out automatically…',
          })
        }
      }, PREFLIGHT_SLOW_STATUS_DELAY_MS)
      const clearPreflightStatusTimers = () => {
        preflightSettled = true
        clearTimeout(routingStatusTimer)
        clearTimeout(slowStatusTimer)
      }

      try {
        try {
          const chatOptions = buildCliChatOptions({
            model: state.model,
            provider: state.provider,
            mode: effectiveMode as typeof state.mode,
            thinkingLevel: state.thinkingLevel,
            autonomy: state.autonomy,
            maxTokens: state.maxTokens,
            projectId: state.projectId,
            fileIds: options?.fileIds,
            skillRefs: options?.skillRefs,
          })
          const response = await openStreamResponse(
            httpClient.chatStream(
              effectiveContent,
              streamSessionId,
              chatOptions,
              { signal: runController.signal },
            ),
            runController,
          )
          clearPreflightStatusTimers()
          dispatch({ type: 'SET_STREAM_STATUS', status: 'Daemon connected; starting agent…' })
          await consumeSseResponse(response, runController.signal, runController.markActivity)
          // A successfully consumed transport is itself a terminal boundary.
          // The done/error handler normally finalizes the reducer, but keep a
          // second idempotent transition here so a detached live binding or a
          // provider-specific event ordering cannot leave the TUI spinner and
          // elapsed timer alive after the connection has ended.
          dispatch({ type: 'STREAM_FINALIZED' })
          return true
        } catch (streamError) {
          const abortReason = resolveDaemonChatRunAbortReason(runController.signal, streamError)
          if (abortReason) {
            throw abortReason
          }

          clearPreflightStatusTimers()
          if (!wsClient) {
            throw streamError
          }
          dispatch({
            type: 'SET_STREAM_STATUS',
            status: 'SSE unavailable; switching to WebSocket…',
          })
          tuiLogError('run.sse-failed-falling-back-to-ws', streamError, {
            sessionId: streamSessionId,
          })
        }

        await consumeWsResponse(
          effectiveContent,
          options?.fileIds,
          options?.skillRefs,
          runController.signal,
          runController.markActivity,
          streamSessionId,
        )
        dispatch({ type: 'STREAM_FINALIZED' })
        return true
      } catch (error) {
        if (handleAbortOutcome(error)) {
          tuiLog('run.aborted', { sessionId: streamSessionId })
          return true
        }

        tuiLogError('run.failed', error, { sessionId: streamSessionId })
        const message = normalizeError(error, { sessionId: streamSessionId })
        // Keep a persistent error transcript line under the failed user turn;
        // the transient error slot alone disappears on later state changes.
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content: buildPersistentErrorLine(message),
        })
        if (recordUserMessage && !currentRunSessionIdRef.current) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: buildUnsavedFreshSessionNotice(),
          })
        }
        dispatch({
          type: 'ERROR',
          message: buildTransientErrorStateMessage(message),
        })
        liveStreamRef.current = null
        return false
      } finally {
        clearPreflightStatusTimers()
        clearTokenSpeedRun(tokenSpeedStartedAt)
        clearRunController(runController)
      }
    },
    [
      beginLiveStream,
      beginRunController,
      beginTokenSpeedRun,
      clearRunController,
      clearTokenSpeedRun,
      consumeSseResponse,
      consumeWsResponse,
      handleAbortOutcome,
      httpClient,
      openStreamResponse,
      state.autonomy,
      state.maxTokens,
      state.mode,
      state.model,
      state.projectId,
      state.provider,
      state.sessionId,
      state.thinkingLevel,
      wsClient,
    ],
  )

  const sendMessage = useCallback(
    async (
      content: string,
      attachments: PendingAttachmentInput[] = [],
      options: { displayContent?: string; skillRefs?: ChatSkillRef[] } = {},
    ): Promise<boolean> => {
      try {
        const uploaded =
          attachments.length > 0
            ? await uploadAttachments(httpClient, attachments)
            : {
                fileIds: [] as string[],
                messageAttachments: [] as MessageAttachment[],
                notices: [] as string[],
              }

        for (const notice of uploaded.notices) {
          dispatch({ type: 'SYSTEM_MESSAGE', content: notice })
        }

        return await startStream(content, true, {
          attachments: uploaded.messageAttachments,
          displayContent: options.displayContent,
          fileIds: uploaded.fileIds,
          skillRefs: options.skillRefs,
        })
      } catch (error) {
        dispatch({
          type: 'ERROR',
          message: normalizeError(error, { sessionId: state.sessionId ?? undefined }),
        })
        return false
      }
    },
    [httpClient, startStream],
  )

  const loadSession = useCallback(
    (session: DaemonSessionDetail, artifacts: DaemonArtifact[] = []) => {
      dispatch({
        type: 'LOAD_SESSION',
        session: hydrateSession(session, artifacts),
      })
    },
    [],
  )

  const resolveApproval = useCallback(
    async (
      approved: boolean,
      scope: 'once' | 'session' | 'always' | 'run' | 'session-all' = 'once',
      note?: string,
    ) => {
      const approval = state.pendingApproval
      const approvalSessionId = approval?.sessionId || state.sessionId
      if (!approval || !approvalSessionId) {
        return
      }
      if (approvalRespondInFlightRef.current.has(approval.requestId)) {
        return
      }
      approvalRespondInFlightRef.current.add(approval.requestId)

      tuiLog('approval.resolve', {
        sessionId: approvalSessionId,
        requestId: approval.requestId,
        tool: approval.toolName,
        approved,
        scope,
        hasNote: Boolean(note?.trim()),
        approvalState: approval.state,
      })
      // A trimmed note means "change course": send the feedback decision so
      // the agent receives [approval:needs-changes] + the user's instruction
      // instead of a bare denial.
      const trimmedNote = note?.trim() || undefined
      const decision: boolean | 'feedback' = !approved && trimmedNote ? 'feedback' : approved

      try {
        if (approval.state === 'stale' && approval.resumeAvailable) {
          const runController = beginRunController()
          const tokenSpeedStartedAt = beginTokenSpeedRun()
          approvalRepeatCountsRef.current.clear()
          currentRunPhaseRef.current = null
          dispatch({
            type: 'APPROVAL_RESOLVED',
            requestId: approval.requestId,
            toolCallId: approval.toolCallId,
            approved,
          })
          dispatch({ type: 'START_STREAM' })
          beginLiveStream({
            doneFallbackText: 'Run resumed',
            errorFallbackText: 'Resume failed',
            errorLabel: 'Resume failed',
            approvalResumeAvailable: true,
          })
          try {
            const response = await openStreamResponse(
              httpClient.resumeApprovalStream(
                approval.requestId,
                decision,
                approvalSessionId,
                trimmedNote,
                { signal: runController.signal },
              ),
              runController,
            )
            await consumeSseResponse(response, runController.signal, runController.markActivity)
          } finally {
            clearTokenSpeedRun(tokenSpeedStartedAt)
            clearRunController(runController)
          }
          return
        }

        const responseController = beginDetachedRunController()
        const result = await (async () => {
          try {
            return await httpClient.respondApproval(approval.requestId, decision, {
              sessionId: approvalSessionId,
              scope,
              note: trimmedNote,
              signal: responseController.signal,
            })
          } finally {
            responseController.dispose()
          }
        })()

        dispatch({
          type: 'APPROVAL_RESOLVED',
          requestId: approval.requestId,
          toolCallId: approval.toolCallId,
          approved,
        })

        if (trimmedNote) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: `Feedback sent to the agent: "${trimmedNote}" — it will revise the ${approval.toolName} call.`,
          })
        }

        if (result.state === 'stale') {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: buildStaleApprovalNotice({ approved }),
          })
          await startStream(
            buildRecoveredApprovalPrompt(approval.toolName, decision, trimmedNote),
            false,
            { targetSessionId: approvalSessionId },
          )
        }
      } catch (error) {
        if (handleAbortOutcome(error)) {
          return
        }

        tuiLogError('approval.resolve-failed', error, {
          sessionId: approvalSessionId,
          requestId: approval.requestId,
        })
        dispatch({ type: 'SET_PENDING_APPROVAL', approval: null })
        if (isAlreadyHandledApprovalError(error)) {
          // Only notice once per request — reconnect races can surface the
          // same already-handled error several times and repeating the same
          // system line reads as the TUI being stuck.
          if (!alreadyHandledNoticeRef.current.has(approval.requestId)) {
            alreadyHandledNoticeRef.current.add(approval.requestId)
            dispatch({
              type: 'SYSTEM_MESSAGE',
              content: buildAlreadyHandledApprovalNotice(),
            })
          }
          dispatch({ type: 'CLEAR_ERROR' })
          liveStreamRef.current = null
          return
        }
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content: buildMissingApprovalNotice(),
        })
        dispatch({
          type: 'ERROR',
          message: normalizeError(error, { sessionId: approvalSessionId }),
        })
        liveStreamRef.current = null
      } finally {
        approvalRespondInFlightRef.current.delete(approval.requestId)
      }
    },
    [
      beginDetachedRunController,
      beginLiveStream,
      beginRunController,
      beginTokenSpeedRun,
      clearRunController,
      clearTokenSpeedRun,
      consumeSseResponse,
      handleAbortOutcome,
      httpClient,
      openStreamResponse,
      startStream,
      state.pendingApproval,
      state.sessionId,
    ],
  )

  const resumeSession = useCallback(
    async (force = false) => {
      if (!state.sessionId) {
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content: 'No active session to resume.',
        })
        return
      }

      const runController = beginRunController()
      const tokenSpeedStartedAt = beginTokenSpeedRun()

      dispatch({ type: 'START_STREAM' })
      beginLiveStream({
        doneFallbackText: 'Run resumed',
        errorFallbackText: 'Resume failed',
        errorLabel: 'Resume failed',
        approvalResumeAvailable: true,
      })

      tuiLog('resume.start', { sessionId: state.sessionId, force })
      try {
        const response = await openStreamResponse(
          httpClient.resumeSessionStream(
            state.sessionId,
            force ? { force: true } : undefined,
            { signal: runController.signal },
          ),
          runController,
        )
        await consumeSseResponse(response, runController.signal, runController.markActivity)
        tuiLog('resume.done', { sessionId: state.sessionId })
      } catch (error) {
        if (handleAbortOutcome(error)) {
          return
        }

        tuiLogError('resume.failed', error, { sessionId: state.sessionId })
        dispatch({
          type: 'ERROR',
          message: normalizeError(error, { sessionId: state.sessionId ?? undefined }),
        })
        liveStreamRef.current = null
      } finally {
        clearTokenSpeedRun(tokenSpeedStartedAt)
        clearRunController(runController)
      }
    },
    [
      beginLiveStream,
      beginRunController,
      beginTokenSpeedRun,
      clearRunController,
      clearTokenSpeedRun,
      consumeSseResponse,
      handleAbortOutcome,
      httpClient,
      openStreamResponse,
      state.sessionId,
    ],
  )

  const answerQuestion = useCallback(
    async (questionId: string, answer: string) => {
      const targetQuestion = state.pendingQuestions.find((question) => question.id === questionId)
      const targetSessionId =
        targetQuestion?.sessionId || currentRunSessionIdRef.current || state.sessionId
      if (!targetSessionId) {
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content: 'No active session to answer.',
        })
        return
      }

      try {
        const result = await httpClient.answerSessionQuestion(targetSessionId, questionId, answer)
        if (!result.answered) {
          dispatch({
            type: 'SYSTEM_MESSAGE',
            content: `Question ${questionId} was not found or was already answered.`,
          })
          return
        }
        dispatch({ type: 'QUESTION_ANSWERED', questionId })
        dispatch({
          type: 'SYSTEM_MESSAGE',
          content: `Answered question ${questionId}.`,
        })
      } catch (error) {
        dispatch({
          type: 'ERROR',
          message: normalizeError(error, { sessionId: targetSessionId }),
        })
      }
    },
    [httpClient, state.pendingQuestions, state.sessionId],
  )

  const cancelStream = useCallback(() => {
    const cancelledController = runController.cancel()
    // Normally isStreaming and the active controller move together. If a
    // terminal-event race cleared the controller first, retain an escape hatch:
    // Ctrl+C still repairs local state and asks the daemon to cancel any run it
    // may still know about. When both are idle this remains a true no-op.
    if (!cancelledController && !state.isStreaming) return
    const sessionId = currentRunSessionIdRef.current ?? state.sessionId ?? undefined
    tuiLog(cancelledController ? 'run.cancel' : 'run.cancel-stale-ui', { sessionId })
    if (sessionId) {
      const cancellation = httpClient.cancelActiveRun(sessionId).then(() => undefined).catch((error) => {
        tuiLogError('run.remote-cancel-failed', error, { sessionId })
      })
      cancellationBarrierRef.current = cancellation
    }
    if (cancelledController) {
      dispatch({ type: 'SET_STREAM_STATUS', status: 'Cancelling…' })
    } else {
      dispatch({ type: 'STREAM_FINALIZED' })
    }
    // Finalize any in-flight tool activities so the Activity panel doesn't keep
    // showing "running <tool> · waiting for the next stream event" forever after
    // the server run has already stopped (CLI_BACKLOG.md B8).
    dispatch({
      type: 'SET_ACTIVITIES',
      activities: (prev) =>
        prev.map((activity) =>
          activity.status === 'running'
            ? {
                ...activity,
                status: 'error',
                detail: activity.detail ? `${activity.detail} (interrupted)` : 'interrupted',
              }
            : activity,
        ),
    })
    wsClient?.close()
  }, [dispatch, httpClient, runController, state.isStreaming, state.sessionId, wsClient])

  return {
    state,
    dispatch,
    sendMessage,
    loadSession,
    resolveApproval,
    answerQuestion,
    resumeSession,
    cancelStream,
    getTokenSpeedStats,
    resetTokenSpeedStats,
  }
}
