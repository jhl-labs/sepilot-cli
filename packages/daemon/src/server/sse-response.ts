import type { RunStopReason } from '@sepilotd/core'
import { stopReasonApprovalTimeout, stopReasonInactivity } from '../agent/stop-reason.js'
import type { AgentEvent } from '@sepilotd/core'
import type { FastifyReply, FastifyRequest } from 'fastify'
import type { FastifyInstance } from 'fastify'
import { buildDaemonCorsHeaders } from './cors.js'
import { resolveClientLabel, resolveRequestSurface } from './request-surface.js'
import { writeSseComment } from './sse-write.js'
import './fastify-types.js'

const DEFAULT_KEEPALIVE_MS = 15000
const DEFAULT_INACTIVITY_MS = 5 * 60_000
const DEFAULT_PENDING_DECISION_TIMEOUT_MS = 30 * 60_000
export const MODEL_STREAM_WAITING_THINKING = 'Still waiting for the model stream...'

export interface SseLifecycleOptions {
  reply: FastifyReply
  /** Override keepalive cadence. Defaults to SEPILOTD_SSE_KEEPALIVE_MS or 15000. */
  keepaliveMs?: number
  /** If set, fires `onInactivity` once when no `recordEvent` call has happened in the window. */
  inactivityMs?: number
  /** Called once when the inactivity window elapses. */
  onInactivity?: () => void
  /**
   * Reports whether the run is currently blocked on a human decision
   * (approval / question). A blocked run is NOT a stalled run: while this
   * returns a descriptor the inactivity watchdog is suppressed and only the
   * separate decision timeout applies.
   */
  pendingDecision?: () => PendingDecision | null
  /**
   * Upper bound on how long a run may stay blocked on a human decision.
   * `0` disables the bound (wait indefinitely). Defaults to
   * `resolvePendingDecisionTimeoutMs()` when `pendingDecision` is supplied.
   */
  decisionTimeoutMs?: number
  /** Called once when the decision timeout elapses with a decision outstanding. */
  onDecisionTimeout?: (pending: PendingDecision) => void
}

/**
 * A run blocked on a human decision is a first-class run state, distinct from
 * a stalled run. `since` is the moment the block began, so the decision bound
 * is measured from the request rather than from the last stream event.
 */
export interface PendingDecision {
  kind: 'approval' | 'question'
  /** Approval requestId or question questionId — what the operator resolves. */
  id: string
  /** Tool name for an approval, question prompt for a question. */
  label?: string
  since: number
}

export interface SseLifecycle {
  /** Reset the inactivity counter — call on every meaningful agent event. */
  recordEvent(): void
  /** True if the inactivity watchdog (or the decision timeout) has already fired. */
  isInactivityTripped(): boolean
  /** True if the run was aborted because a human decision never arrived. */
  isDecisionTimedOut(): boolean
  /** Tear down all timers. Idempotent. */
  dispose(): void
}

/**
 * Standardised SSE keepalive + inactivity watchdog used by every long-lived
 * agent stream (chat-stream, cowork, session-resume). Keepalive comment frames
 * stop intermediaries from idling the connection; the inactivity timer detects
 * a genuinely stuck agent loop so the run lease and socket can be released.
 */
export function createSseLifecycle(opts: SseLifecycleOptions): SseLifecycle {
  const requestedKeepaliveMs = opts.keepaliveMs
    ?? Number.parseInt(process.env.SEPILOTD_SSE_KEEPALIVE_MS ?? String(DEFAULT_KEEPALIVE_MS), 10)
  const keepaliveMs = Number.isFinite(requestedKeepaliveMs) && requestedKeepaliveMs > 0
    ? Math.max(1000, requestedKeepaliveMs)
    : DEFAULT_KEEPALIVE_MS
  const inactivityMs = opts.inactivityMs
  const decisionTimeoutMs = opts.pendingDecision
    ? opts.decisionTimeoutMs ?? resolvePendingDecisionTimeoutMs()
    : 0

  let lastEventAt = Date.now()
  let inactivityTripped = false
  let decisionTimedOut = false
  let keepalivePending = false

  const keepaliveTimer = setInterval(() => {
    if (opts.reply.raw.writableEnded) return
    if (keepalivePending) return
    keepalivePending = true
    void writeSseComment(opts.reply.raw, `keepalive ${Date.now()}`)
      .catch(() => {
        /* best-effort */
      })
      .finally(() => {
        keepalivePending = false
      })
  }, keepaliveMs)

  let inactivityTimer: ReturnType<typeof setInterval> | null = null
  if (inactivityMs && opts.onInactivity) {
    const tickMs = Math.max(100, Math.min(10_000, Math.floor(inactivityMs / 6)))
    inactivityTimer = setInterval(() => {
      if (opts.reply.raw.writableEnded) return
      const pending = opts.pendingDecision?.() ?? null
      if (pending) {
        // Deliberately blocked on a human decision — never a stall.
        if (decisionTimeoutMs > 0 && Date.now() - pending.since > decisionTimeoutMs) {
          if (!decisionTimedOut) {
            decisionTimedOut = true
            inactivityTripped = true
            try { opts.onDecisionTimeout?.(pending) } catch { /* best-effort */ }
          }
        }
        return
      }
      if (Date.now() - lastEventAt > inactivityMs) {
        if (!inactivityTripped) {
          inactivityTripped = true
          try { opts.onInactivity!() } catch { /* best-effort */ }
        }
      }
    }, tickMs)
  }

  return {
    recordEvent() { lastEventAt = Date.now() },
    isInactivityTripped() { return inactivityTripped },
    isDecisionTimedOut() { return decisionTimedOut },
    dispose() {
      try { clearInterval(keepaliveTimer) } catch { /* noop */ }
      if (inactivityTimer) {
        try { clearInterval(inactivityTimer) } catch { /* noop */ }
      }
    },
  }
}

/** Inactivity window resolved from env, with the documented default. */
export function resolveAgentInactivityMs(): number {
  const parsed = Number.parseInt(
    process.env.SEPILOTD_AGENT_INACTIVITY_MS ?? String(DEFAULT_INACTIVITY_MS),
    10,
  )
  return Number.isFinite(parsed) && parsed > 0
    ? Math.max(1000, parsed)
    : DEFAULT_INACTIVITY_MS
}

/**
 * Bound on how long a run may stay blocked waiting for a human approval or
 * answer, from `SEPILOTD_PENDING_DECISION_TIMEOUT_MS`. `0` means "wait
 * indefinitely". Kept separate from `SEPILOTD_AGENT_INACTIVITY_MS` because the
 * two describe different run states and deserve different bounds.
 */
export function resolvePendingDecisionTimeoutMs(): number {
  const raw = process.env.SEPILOTD_PENDING_DECISION_TIMEOUT_MS
  if (raw === undefined || raw.trim() === '') return DEFAULT_PENDING_DECISION_TIMEOUT_MS
  const parsed = Number.parseInt(raw, 10)
  if (!Number.isFinite(parsed) || parsed < 0) return DEFAULT_PENDING_DECISION_TIMEOUT_MS
  if (parsed === 0) return 0
  return Math.max(1000, parsed)
}

export interface PendingDecisionTimeoutFrame {
  code: 'APPROVAL_TIMEOUT' | 'QUESTION_TIMEOUT'
  reason: 'awaiting_approval' | 'awaiting_answer'
  message: string
  provider?: string
  model?: string
  lastEventType?: string
  pendingDecision: PendingDecision
  /** Structured stop cause so surfaces can offer "approve pending"/"resume". */
  stopReason: RunStopReason
}

export interface AgentInactivityFrame {
  code: 'AGENT_INACTIVITY'
  reason: 'no_model_output' | 'stalled'
  message: string
  provider?: string
  model?: string
  lastEventType?: string
  /** Structured stop cause so surfaces can offer "retry"/"resume". */
  stopReason: RunStopReason
}

export type AgentTimeoutFrame = AgentInactivityFrame | PendingDecisionTimeoutFrame

/**
 * Build the error payload for a run aborted because a *pending human decision*
 * never arrived. This is intentionally not an `AGENT_INACTIVITY` frame: nothing
 * stalled, so the remedy is to answer the outstanding decision (or run on a
 * surface that can answer it), not to retry or split the request.
 */
export function describePendingDecisionTimeout(input: {
  decisionTimeoutMs: number
  pending: PendingDecision
  activityLabel?: string
  provider?: string
  model?: string
  lastEventType?: string
}): PendingDecisionTimeoutFrame {
  const label = input.activityLabel ?? 'Agent'
  const waited = input.decisionTimeoutMs > 0
    ? ` for ${input.decisionTimeoutMs}ms`
    : ''
  if (input.pending.kind === 'approval') {
    const tool = input.pending.label ? ` for tool '${input.pending.label}'` : ''
    return {
      code: 'APPROVAL_TIMEOUT',
      reason: 'awaiting_approval',
      message:
        `${label} is blocked on a pending approval (id ${input.pending.id})${tool} and no decision arrived${waited}. `
        + `Nothing stalled — approve or deny it (\`sepilot approve ${input.pending.id}\` / \`sepilot deny ${input.pending.id}\`), `
        + 'or re-run on a surface that can answer approval prompts. '
        + 'Raise SEPILOTD_PENDING_DECISION_TIMEOUT_MS (0 = wait indefinitely) if you need a longer window.',
      provider: input.provider,
      model: input.model,
      lastEventType: input.lastEventType,
      pendingDecision: input.pending,
      stopReason: stopReasonApprovalTimeout({
        requestId: input.pending.id,
        tool: input.pending.label,
        layer: 'approval',
      }),
    }
  }
  const prompt = input.pending.label ? `: ${input.pending.label}` : ''
  return {
    code: 'QUESTION_TIMEOUT',
    reason: 'awaiting_answer',
    message:
      `${label} is blocked on a pending question (id ${input.pending.id})${prompt} and no answer arrived${waited}. `
      + `Nothing stalled — answer it (\`sepilot answer ${input.pending.id} <reply>\`), or re-run on a surface that can answer questions. `
      + 'Raise SEPILOTD_PENDING_DECISION_TIMEOUT_MS (0 = wait indefinitely) if you need a longer window.',
    provider: input.provider,
    model: input.model,
    lastEventType: input.lastEventType,
    pendingDecision: input.pending,
    stopReason: stopReasonApprovalTimeout({
      requestId: input.pending.id,
      layer: 'question',
    }),
  }
}

export interface AgentInactivityFrameInput {
  inactivityMs: number
  provider?: string
  model?: string
  /** True if the run emitted any model output (token/tool/message/error) before going silent. */
  sawModelOutput?: boolean
  /** Type of the last agent event seen before the run went silent. */
  lastEventType?: string
  /** Human label for the activity that stalled — e.g. "Agent", "Resume". */
  activityLabel?: string
}

/**
 * Build the structured `AGENT_INACTIVITY` error payload sent on the SSE
 * `error` frame when the inactivity watchdog fires. `reason` is the
 * machine-readable cause; `message` is operationally framed English that
 * surfaces may localise/soften. The important distinction: a run that
 * produced *no* model output almost always means the provider/model call
 * itself never returned — point at provider health, not at a stuck tool or
 * an over-long request.
 */
export function describeAgentInactivity(input: AgentInactivityFrameInput): AgentInactivityFrame {
  const label = input.activityLabel ?? 'Agent'
  const modelLabel = input.provider && input.model
    ? `${input.provider} / ${input.model}`
    : input.model ?? input.provider
  if (input.sawModelOutput === false) {
    const where = modelLabel ? ` from the provider/model (${modelLabel})` : ''
    return {
      code: 'AGENT_INACTIVITY',
      reason: 'no_model_output',
      message:
        `${label} produced no output for ${input.inactivityMs}ms — no token or tool call arrived${where}. `
        + 'The provider/model likely did not respond; check that the provider endpoint is reachable and the model is available, or switch to a faster model.',
      provider: input.provider,
      model: input.model,
      lastEventType: input.lastEventType,
      stopReason: stopReasonInactivity({
        budgetMs: input.inactivityMs,
        layer: 'no_model_output',
      }),
    }
  }
  const after = input.lastEventType ? ` after the last step (${input.lastEventType})` : ''
  return {
    code: 'AGENT_INACTIVITY',
    reason: 'stalled',
    message:
      `${label} produced no output for ${input.inactivityMs}ms${after} — a tool or the model stream stalled. `
      + 'Aborted to release the run slot; retry, or break the request into smaller steps.',
    provider: input.provider,
    model: input.model,
    lastEventType: input.lastEventType,
    stopReason: stopReasonInactivity({ budgetMs: input.inactivityMs, layer: 'stalled' }),
  }
}

/**
 * Watches an agent event stream and, if the inactivity watchdog later fires,
 * lets the route describe *why* it stalled: a run that never produced a token
 * or tool call (`reason: 'no_model_output'` — the provider/model call itself
 * hung) versus one that made progress and then went quiet (`reason: 'stalled'`).
 * Call `note()` for every event consumed, then `describe()` from the catch
 * block that handles `isInactivityTripped()`.
 */
export interface AgentInactivityProbe {
  note(event: AgentEvent): void
  /** The outstanding human decision blocking the run, if any. */
  pendingDecision(): PendingDecision | null
  describe(opts: {
    inactivityMs: number
    decisionTimeoutMs?: number
    provider?: string
    model?: string
    activityLabel?: string
  }): ReturnType<typeof describeAgentInactivity> | PendingDecisionTimeoutFrame
}

/**
 * Provider-stream heartbeat events keep the user informed, but they do not
 * prove that the provider produced a token or completed a step. Counting them
 * as activity would let a dead model request refresh the watchdog forever.
 */
export function isSubstantiveAgentActivity(event: AgentEvent): boolean {
  return !(
    event.type === 'thinking'
    && event.content === MODEL_STREAM_WAITING_THINKING
  )
}

export function createAgentInactivityProbe(): AgentInactivityProbe {
  let sawModelOutput = false
  let lastEventType: string | undefined
  // Pending-decision state machine: an approval/question request blocks the
  // run, and *any* later event proves the run moved on again (an approval
  // response, the tool result, the next token). No event can arrive while the
  // run is genuinely blocked, so "next event clears" is exact, not heuristic.
  let pending: PendingDecision | null = null
  return {
    note(event) {
      lastEventType = event.type
      if (event.type === 'approval_request') {
        pending = {
          kind: 'approval',
          id: event.requestId,
          label: event.toolCall?.name,
          since: Date.now(),
        }
      } else if (event.type === 'question_request') {
        pending = {
          kind: 'question',
          id: event.questionId,
          label: event.prompt,
          since: Date.now(),
        }
      } else {
        pending = null
      }
      if (
        event.type === 'text_delta'
        || event.type === 'message'
        || event.type === 'action_progress'
        || event.type === 'tool_call'
        || event.type === 'tool_result'
        || event.type === 'approval_request'
        || event.type === 'question_request'
        || event.type === 'error'
      ) {
        sawModelOutput = true
      }
    },
    pendingDecision() { return pending },
    describe(opts) {
      if (pending) {
        return describePendingDecisionTimeout({
          decisionTimeoutMs: opts.decisionTimeoutMs ?? resolvePendingDecisionTimeoutMs(),
          pending,
          activityLabel: opts.activityLabel,
          provider: opts.provider,
          model: opts.model,
          lastEventType,
        })
      }
      return describeAgentInactivity({
        inactivityMs: opts.inactivityMs,
        provider: opts.provider,
        model: opts.model,
        activityLabel: opts.activityLabel,
        sawModelOutput,
        lastEventType,
      })
    },
  }
}

export function buildSseResponseHeaders(
  request: FastifyRequest,
  headers: Record<string, string>,
): Record<string, string> {
  return buildDaemonCorsHeaders(request, headers)
}

/**
 * Track a long-lived SSE response in the daemon's `connectionRegistry` so
 * `/system/clients` and the desktop tray "N connected" indicator reflect
 * agent streams (chat-stream, cowork, session-resume, notifications-watch)
 * the same way they reflect WebSocket clients.
 *
 * Self-unregistering: the helper attaches its own listeners to `aborted`,
 * `close`, and `finish` so call sites don't need to remember to clean up.
 * Returns a manual `unregister` for tests / unusual teardown flows; safe to
 * ignore in normal route code.
 *
 * No-op (returns a stub handle) when the registry isn't wired — useful for
 * hand-built test Fastify instances.
 */
export function trackSseConnection(
  app: FastifyInstance,
  request: FastifyRequest,
  reply: FastifyReply,
  options: { label: string },
): { unregister: () => void } {
  const registry = app.connectionRegistry
  if (!registry) return { unregister: () => {} }

  const client = resolveClientLabel(request.authContext, resolveRequestSurface(request))
  const id = registry.add({ kind: 'sse', label: options.label, client })

  let removed = false
  const unregister = () => {
    if (removed) return
    removed = true
    registry.remove(id)
    request.raw.off('aborted', unregister)
    reply.raw.off('close', unregister)
    reply.raw.off('finish', unregister)
  }

  request.raw.on('aborted', unregister)
  reply.raw.on('close', unregister)
  reply.raw.on('finish', unregister)

  return { unregister }
}

export function registerSseDisconnectHandler(
  request: FastifyRequest,
  reply: FastifyReply,
  onDisconnect: () => void,
): () => void {
  let released = false

  const release = () => {
    if (released) {
      return
    }
    released = true
    request.raw.off('aborted', handleAbort)
    reply.raw.off('close', handleClose)
    reply.raw.off('finish', release)
  }

  const handleAbort = () => {
    release()
    onDisconnect()
  }

  const handleClose = () => {
    const responseFinished = reply.raw.writableEnded || reply.raw.writableFinished
    release()
    if (!responseFinished) {
      onDisconnect()
    }
  }

  request.raw.on('aborted', handleAbort)
  reply.raw.on('close', handleClose)
  reply.raw.on('finish', release)

  return release
}
