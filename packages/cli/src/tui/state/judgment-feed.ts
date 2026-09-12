// Formats the daemon's judgment/decision events into one-line transcript
// entries so the operator can see *what the agent decided and why* while a
// run is otherwise silent (long tool runs, approval waits). Pure so the
// mapping can be unit-tested without rendering the TUI.

const FEED_LINE_MAX_CHARS = 160

interface ReasoningStepPayload {
  type: 'reasoning_step'
  label: string
  detail?: string
}

interface ActionProgressPayload {
  type: 'action_progress'
  summary: string
  nextStep: string
  toolNames?: string[]
}

interface ModeRouteDecisionPayload {
  type: 'mode_route_decision'
  chosen: string
  persona?: string
  reason?: string
  confidence?: number
  fallback?: boolean
}

interface QualityGateVerdictPayload {
  type: 'quality_gate_verdict'
  phase: string
  decision: string
  blockingReason?: string
}

interface BacktrackPayload {
  type: 'backtrack'
  phase: string
  reason?: string
  attempt?: number
}

interface SteeringConsumedPayload {
  type: 'steering_consumed'
  message: string
  kind: 'instruction' | 'question'
}

interface RecoveryPayload {
  type: 'recovery'
  message: string
  recoverable: boolean
}

export type JudgmentFeedPayload =
  | ReasoningStepPayload
  | ActionProgressPayload
  | ModeRouteDecisionPayload
  | QualityGateVerdictPayload
  | BacktrackPayload
  | SteeringConsumedPayload
  | RecoveryPayload

function clip(text: string): string {
  const singleLine = text.replace(/\s+/g, ' ').trim()
  if (singleLine.length <= FEED_LINE_MAX_CHARS) return singleLine
  return `${singleLine.slice(0, FEED_LINE_MAX_CHARS - 1)}…`
}

/**
 * Returns the transcript line for a judgment event, or null when the
 * payload is not a judgment event (callers can pass any stream payload).
 */
export function formatJudgmentFeedLine(payload: { type: string } & Record<string, unknown>): string | null {
  switch (payload.type) {
    case 'reasoning_step': {
      // Reasoning is transient progress, already shown in the live status
      // region. Persisting every step in scrollback makes the final answer
      // difficult to find and exposes implementation chatter as user output.
      return null
    }
    case 'action_progress': {
      const event = payload as unknown as ActionProgressPayload
      return clip(`↳ ${event.summary} → ${event.nextStep}`)
    }
    case 'mode_route_decision': {
      const event = payload as unknown as ModeRouteDecisionPayload
      const persona = event.persona ? ` (${event.persona})` : ''
      const fallback = event.fallback ? ' [fallback]' : ''
      const reason = event.fallback && event.reason ? ` — ${event.reason}` : ''
      return clip(`Mode: ${event.chosen}${persona}${fallback}${reason}`)
    }
    case 'quality_gate_verdict': {
      const event = payload as unknown as QualityGateVerdictPayload
      const reason = event.blockingReason ? ` — ${event.blockingReason}` : ''
      return clip(`Quality check (${event.phase}): ${event.decision}${reason}`)
    }
    case 'backtrack': {
      const event = payload as unknown as BacktrackPayload
      const attempt = typeof event.attempt === 'number' ? ` (attempt ${event.attempt})` : ''
      const reason = event.reason ? `: ${event.reason}` : ''
      return clip(`Retrying ${event.phase}${attempt}${reason}`)
    }
    case 'steering_consumed': {
      const event = payload as unknown as SteeringConsumedPayload
      const kindLabel = event.kind === 'question' ? 'question' : 'instruction'
      return clip(`Follow-up ${kindLabel} applied: ${event.message}`)
    }
    case 'recovery': {
      const event = payload as unknown as RecoveryPayload
      return clip(`Recovery: ${event.message}`)
    }
    default:
      return null
  }
}
