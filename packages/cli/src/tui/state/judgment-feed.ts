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

export function formatJudgmentFeedLine(payload: { type: string } & Record<string, unknown>): string | null {
  switch (payload.type) {
    case 'reasoning_step': {
      return null
    }
    case 'action_progress': {
      const event = payload as unknown as ActionProgressPayload
      return [`Now · ${event.summary.trim()}`, event.nextStep?.trim() ? `Next · ${event.nextStep.trim()}` : null].filter(Boolean).join('\n')
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
      if (event.recoverable !== false) return null
      return clip(`Recovery: ${event.message}`)
    }
    default:
      return null
  }
}
