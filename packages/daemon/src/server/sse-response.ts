import type { RunStopReason } from '@sepilotd/core'
import { stopReasonApprovalTimeout, stopReasonInactivity } from '../agent/stop-reason.js'
import type { AgentEvent } from '@sepilotd/core'
import type { FastifyReply, FastifyRequest } from 'fastify'
import type { FastifyInstance } from 'fastify'
import { buildDaemonCorsHeaders } from './cors.js'
import { resolveClientLabel, resolveRequestSurface } from './request-surface.js'
import { writeSseComment } from './sse-write.js'
import { resolveApprovalTimeoutMs } from './runtime/approvals.js'
import './fastify-types.js'

const DEFAULT_KEEPALIVE_MS = 15000
const DEFAULT_INACTIVITY_MS = 5 * 60_000
export const MODEL_STREAM_WAITING_THINKING = 'Still waiting for the model stream...'

export interface SseLifecycleOptions {
  reply: FastifyReply
  keepaliveMs?: number
  inactivityMs?: number
  onInactivity?: () => void
  pendingDecision?: () => PendingDecision | null
  decisionTimeoutMs?: number
  onDecisionTimeout?: (pending: PendingDecision) => void
}

export interface PendingDecision {
  kind: 'approval' | 'question'
  id: string
  label?: string
  since: number
}

export interface SseLifecycle {
  recordEvent(): void
  isInactivityTripped(): boolean
  isDecisionTimedOut(): boolean
  dispose(): void
}

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
        if (decisionTimeoutMs > 0 && Date.now() - pending.since > decisionTimeoutMs) {
          if (!decisionTimedOut) {
            decisionTimedOut = true
            inactivityTripped = true
            try { opts.onDecisionTimeout?.(pending) } catch {}
          }
        }
        return
      }
      if (Date.now() - lastEventAt > inactivityMs) {
        if (!inactivityTripped) {
          inactivityTripped = true
          try { opts.onInactivity!() } catch {}
        }
      }
    }, tickMs)
  }

  return {
    recordEvent() { lastEventAt = Date.now() },
    isInactivityTripped() { return inactivityTripped },
    isDecisionTimedOut() { return decisionTimedOut },
    dispose() {
      try { clearInterval(keepaliveTimer) } catch {}
      if (inactivityTimer) {
        try { clearInterval(inactivityTimer) } catch {}
      }
    },
  }
}

export function resolveAgentInactivityMs(): number {
  const parsed = Number.parseInt(
    process.env.SEPILOTD_AGENT_INACTIVITY_MS ?? String(DEFAULT_INACTIVITY_MS),
    10,
  )
  return Number.isFinite(parsed) && parsed > 0
    ? Math.max(1000, parsed)
    : DEFAULT_INACTIVITY_MS
}

export function resolvePendingDecisionTimeoutMs(): number {
  return resolveApprovalTimeoutMs()
}

export interface PendingDecisionTimeoutFrame {
  code: 'APPROVAL_TIMEOUT' | 'QUESTION_TIMEOUT'
  reason: 'awaiting_approval' | 'awaiting_answer'
  message: string
  provider?: string
  model?: string
  lastEventType?: string
  pendingDecision: PendingDecision
  stopReason: RunStopReason
}

export interface AgentInactivityFrame {
  code: 'AGENT_INACTIVITY'
  reason: 'no_model_output' | 'stalled'
  message: string
  provider?: string
  model?: string
  lastEventType?: string
  stopReason: RunStopReason
}

export type AgentTimeoutFrame = AgentInactivityFrame | PendingDecisionTimeoutFrame

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
        `${label} is paused on a pending approval (id ${input.pending.id})${tool}: no decision arrived${waited}. `
        + `Nothing stalled and nothing was denied — the run is parked with its checkpoint. Approve or deny it (\`sepilot approve ${input.pending.id}\` / \`sepilot deny ${input.pending.id}\`) `
        + 'and the run continues from where it stopped. '
        + 'Raise SEPILOTD_APPROVAL_TIMEOUT_MS (legacy SEPILOTD_PENDING_DECISION_TIMEOUT_MS; 0 = wait indefinitely) if you need a longer window.',
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
      + 'Raise SEPILOTD_APPROVAL_TIMEOUT_MS (legacy SEPILOTD_PENDING_DECISION_TIMEOUT_MS; 0 = wait indefinitely) if you need a longer window.',
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
  sawModelOutput?: boolean
  lastEventType?: string
  activityLabel?: string
}

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

export interface AgentInactivityProbe {
  note(event: AgentEvent): void
  pendingDecision(): PendingDecision | null
  describe(opts: {
    inactivityMs: number
    decisionTimeoutMs?: number
    provider?: string
    model?: string
    activityLabel?: string
  }): ReturnType<typeof describeAgentInactivity> | PendingDecisionTimeoutFrame
}

export function isSubstantiveAgentActivity(event: AgentEvent): boolean {
  if (event.type === 'subagent_progress') return isSubstantiveAgentActivity(event.inner)
  return !(
    event.type === 'thinking'
    && event.content === MODEL_STREAM_WAITING_THINKING
  )
}

export function createAgentInactivityProbe(): AgentInactivityProbe {
  let sawModelOutput = false
  let lastEventType: string | undefined
  let pending: PendingDecision | null = null
  const children = new Map<string, PendingDecision>()
  const currentPending = () => pending ?? children.values().next().value ?? null
  return {
    note(event) {
      lastEventType = event.type
      if (event.type === 'subagent_progress') {
        const inner = event.inner
        const prior = children.get(event.subagentId)
        if (inner.type === 'approval_request') {
          if (prior?.id !== inner.requestId) children.set(event.subagentId, {
            kind: 'approval', id: inner.requestId, label: inner.toolCall.name, since: Date.now(),
          })
          sawModelOutput = true
        } else if ((inner.type === 'approval_response' && inner.requestId === prior?.id)
          || inner.type === 'tool_result' || inner.type === 'tool_call') {
          children.delete(event.subagentId)
          sawModelOutput = true
        }
        return
      }
      if (event.type === 'tool_result' || event.type === 'done' || event.type === 'error') children.clear()
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
    pendingDecision() { return currentPending() },
    describe(opts) {
      const decision = currentPending()
      if (decision) {
        return describePendingDecisionTimeout({
          decisionTimeoutMs: opts.decisionTimeoutMs ?? resolvePendingDecisionTimeoutMs(),
          pending: decision,
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
