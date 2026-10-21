export type RememberedApprovalScope = 'session' | 'always'
export type BroadApprovalScope = 'run' | 'session-all'
export type ApprovalScope = 'once' | RememberedApprovalScope | BroadApprovalScope
export type AutoApprovalScope = RememberedApprovalScope | BroadApprovalScope

export type ApprovalEvaluation = 'approved' | 'denied' | 'prompt'
export type ApprovalDecisionStatus = 'approved' | 'feedback' | 'denied'

export interface ApprovalDecision {
  decision: ApprovalDecisionStatus
  approved: boolean
  note?: string
  autoApproval?: { rule: ApprovalRule; scope: AutoApprovalScope }
  timedOut?: boolean
  stop?: boolean
}

export interface ApprovalRule {
  tool: string
  pattern: string
}

export interface RememberedDecision extends ApprovalRule {
  scope: RememberedApprovalScope
  approved: boolean
  sessionId?: string
  createdAt: string
  hitCount?: number
  lastHitAt?: string
}

export interface RememberedDecisionInput extends ApprovalRule {
  scope: RememberedApprovalScope
  approved: boolean
  sessionId?: string
}

export interface RememberedDecisionMatch extends ApprovalRule {
  scope: RememberedApprovalScope
  sessionId?: string
}

export interface ApprovalEvaluationResult {
  verdict: ApprovalEvaluation
  rule?: RememberedDecision
}

export const STALE_RULE_THRESHOLD_DAYS = 7

export function isStaleRememberedDecision(
  decision: RememberedDecision,
  now: number = Date.now(),
  thresholdDays: number = STALE_RULE_THRESHOLD_DAYS,
): boolean {
  if (decision.lastHitAt) return false
  if ((decision.hitCount ?? 0) > 0) return false
  if (!decision.createdAt) return false
  const created = new Date(decision.createdAt).getTime()
  if (!Number.isFinite(created)) return false
  const ageDays = (now - created) / (1000 * 60 * 60 * 24)
  return ageDays >= thresholdDays
}

export interface IApprovalDecisionStore {
  evaluate(params: {
    sessionId: string
    tool: string
    input: Record<string, unknown>
  }): ApprovalEvaluationResult

  remember(params: {
    sessionId: string
    tool: string
    input: Record<string, unknown>
    approved: boolean
    scope: RememberedApprovalScope
  }): ApprovalRule

  describeRule(tool: string, input: Record<string, unknown>): ApprovalRule

  list(): RememberedDecision[]

  upsert(params: RememberedDecisionInput): RememberedDecision

  update(
    match: RememberedDecisionMatch,
    params: RememberedDecisionInput,
  ): RememberedDecision | null

  remove(match: RememberedDecisionMatch): boolean

  clear(params?: {
    sessionId?: string
    scope?: RememberedApprovalScope
    stale?: boolean
    tool?: string
    pattern?: string
    approved?: boolean
  }): void
}
