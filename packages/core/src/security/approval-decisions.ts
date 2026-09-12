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
  /**
   * Populated by ApprovalRegistry when a remembered session/always rule
   * short-circuits the operator prompt. Carries the matched rule + scope
   * so the agent loop can emit a positive `auto_approval` AgentEvent (vs
   * silently proceeding to tool_call). Surfaces use it to render a
   * "auto-approved by <scope> rule '<pattern>'" line so the operator can
   * tell why no prompt appeared.
   */
  autoApproval?: { rule: ApprovalRule; scope: AutoApprovalScope }
  /**
   * Set when the prompt expired instead of the operator answering it. Carried
   * separately from `decision: 'denied'` because an unanswered prompt is not a
   * refusal — surfaces and the agent loop must be able to offer a retry rather
   * than reporting that the user said no.
   */
  timedOut?: boolean
  /**
   * Structured "deny and stop". A plain denial hands the model one
   * side-effect-free turn to propose a different approach or finish with what
   * is already done; `stop: true` ends the run immediately instead. Carried as
   * a flag rather than a note so surfaces do not have to encode intent in
   * free text.
   */
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
  /**
   * Bumps every time evaluate() short-circuits a tool call against this
   * rule. Operators read it from `decisions list` to tell *which* rules
   * are actually carrying their weight — a rule with hitCount=0 weeks
   * after registration is a candidate for cleanup, while a high-count
   * rule signals "this kind of tool runs unsupervised; double-check
   * the policy is still appropriate". Without it, the decisions list
   * is just a static set with no signal about real usage.
   */
  hitCount?: number
  /**
   * ISO timestamp of the most recent evaluate() match. Distinguishes
   * "never used" (undefined) from "registered today, used today" vs
   * "stale rule from last quarter" so cleanup tooling can surface
   * abandoned-but-still-active rules.
   */
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

/**
 * Result of evaluating a tool call against the decision store. The verdict
 * tells the caller whether to short-circuit; `rule` carries the matched
 * remembered decision when the verdict is `approved` or `denied`. Callers
 * forward `rule` to operators so they can see which scope/pattern triggered
 * the auto-decision.
 */
export interface ApprovalEvaluationResult {
  verdict: ApprovalEvaluation
  rule?: RememberedDecision
}

/**
 * Days after which a rule that has never matched is treated as stale.
 * Single source of truth shared by:
 *   - cli `decisions list` (stale? marker, --stale filter)
 *   - daemon clear({stale}) predicate
 *   - mock fixture in e2e
 * Picked deliberately: short enough that an operator returning from a
 * week off can still act on the signal, long enough that one-off
 * sessions (24h) don't generate noise.
 */
export const STALE_RULE_THRESHOLD_DAYS = 7

/**
 * A rule is "stale" when nothing has matched it AND it has been around
 * longer than the threshold. Centralised here so cli + daemon + mock
 * share one implementation; otherwise the cli could flag a rule as
 * stale that the daemon's clear({stale}) wouldn't actually remove
 * (silent ux divergence).
 */
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
    /** When true, only remove entries that match isStaleRememberedDecision. */
    stale?: boolean
    /**
     * When set, only remove entries with this exact tool name. Pairs
     * with cli `decisions list --tool` so the operator can list-then-
     * prune the same set of rules.
     */
    tool?: string
    /** Exact derived rule pattern. Used by UI/API single-rule deletion. */
    pattern?: string
    /** Exact remembered outcome filter. */
    approved?: boolean
  }): void
}
