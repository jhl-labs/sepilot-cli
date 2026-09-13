import { randomUUID } from 'node:crypto'
import type {
  ApprovalDecision,
  ApprovalDecisionStatus,
  ApprovalRule,
  ApprovalScope,
  AutoApprovalScope,
  IApprovalDecisionStore,
  ISessionStore,
  RememberedApprovalScope,
  ToolCall,
} from '@sepilotd/core'
import { describeRuleFor, matchesRule } from './approval-rules.js'

const RUN_AUTO_APPROVAL_TTL_MS = 60 * 60_000
const APPROVAL_TIMEOUT_FLOOR_MS = 60_000
export const DEFAULT_APPROVAL_TIMEOUT_MS = 10 * 60_000

/**
 * Thrown to the waiting run when an approval prompt reaches its parking
 * delay. It is deliberately an `AbortError` so every engine treats it like
 * the abort-without-cancel path: the run unwinds, releases its lease, and
 * leaves the pending approval and its checkpoint standing for a later answer.
 */
export class ApprovalParkedError extends Error {
  readonly name = 'AbortError'
  readonly code = 'APPROVAL_PARKED'
  constructor(
    readonly requestId: string,
    readonly tool: string,
    readonly sessionId: string,
  ) {
    super(`Approval for '${tool}' is still pending (request ${requestId}); the run was parked until a decision arrives`)
  }
}

export function isApprovalParkedError(error: unknown): error is ApprovalParkedError {
  return error instanceof ApprovalParkedError
    || (error instanceof Error && (error as { code?: unknown }).code === 'APPROVAL_PARKED')
}

export interface PendingApproval {
  requestId: string
  sessionId: string
  runId?: string
  toolCallId: string
  tool: string
  input: Record<string, unknown>
  requestedAt: string
  /** When the prompt parks its run (nobody answered in time). Informational after `state: 'parked'`. */
  expiresAt: string
  /**
   * `live`: a run is blocked on this prompt right now. `parked`: the parking
   * delay elapsed, the run was unwound, and answering resumes it from its
   * checkpoint. `stale`: reconstructed from session history after a restart.
   */
  state: 'live' | 'stale' | 'parked'
  parkedAt?: string
  resumeAvailable?: boolean
  suggestedRule?: ApprovalRule
}

interface PendingApprovalEntry extends PendingApproval {
  resolve: (decision: ApprovalDecision) => void
  reject: (error: Error) => void
  onParked?: (approval: PendingApproval) => void
  timeout: ReturnType<typeof setTimeout> | null
}

export interface ApprovalRegistryHooks {
  onPending?: (approval: PendingApproval) => void
  /** The parking delay elapsed: the request is still open, the run is paused. */
  onTimeout?: (approval: PendingApproval) => void
}

interface BroadApprovalDecision {
  sessionId: string
  decision: ApprovalDecision
  scope: AutoApprovalScope
  rule: ApprovalRule
  createdAt: number
  expiresAt?: number
}

export interface RespondOptions {
  scope?: ApprovalScope
  approvedBy?: string
  /**
   * Optional user-provided rule. When the WebToolCard's "Edit pattern…"
   * field is used, the user types a broader/narrower glob than what
   * describeRuleFor would derive. Persist it verbatim so the user's
   * intent survives — otherwise upsert via decisions.remember collapses
   * the pattern back to the derived shape and silently loses the edit.
   */
  rule?: ApprovalRule
}

export class ApprovalRegistry {
  private readonly pending = new Map<string, PendingApprovalEntry>()
  private readonly runAutoApprovals = new Map<string, BroadApprovalDecision>()
  private readonly sessionAutoApprovals = new Map<string, BroadApprovalDecision>()

  constructor(
    private readonly sessions: ISessionStore,
    private readonly decisions?: IApprovalDecisionStore,
    private readonly hooks: ApprovalRegistryHooks = {},
  ) {}

  waitForApproval(params: {
    sessionId: string
    toolCall: ToolCall
    requestId?: string
    runId?: string
    timeoutMs?: number
    /** Skip remembered run/session/always decisions and open a fresh prompt. */
    forcePrompt?: boolean
    /**
     * Abort the wait — not the request. The pending entry and its saved
     * checkpoint stay, so whoever comes back can still answer; only this run
     * stops occupying the session while nobody is there to reply.
     */
    signal?: AbortSignal
    /**
     * Called when the parking delay elapses. A caller that owns the run's
     * stop path (stream routes) uses it to abort the run through `signal`;
     * without it the wait rejects with `ApprovalParkedError` so headless runs
     * still unwind instead of blocking forever.
     */
    onParked?: (approval: PendingApproval) => void
  }): ApprovalDecision | Promise<ApprovalDecision> {
    const {
      sessionId,
      toolCall,
      requestId = toolCall.id,
      runId,
      timeoutMs = resolveApprovalTimeoutMs(),
      forcePrompt = false,
      signal,
      onParked,
    } = params

    const autoDecision = forcePrompt ? null : this.tryAutoApproval({ sessionId, toolCall, runId })
    if (autoDecision) return autoDecision

    const requestedAt = new Date().toISOString()
    const expiresAt = new Date(Date.now() + timeoutMs).toISOString()
    const suggestedRule = this.decisions?.describeRule(
      toolCall.name,
      toolCall.arguments,
    )

    return new Promise<ApprovalDecision>((resolve, reject) => {
      // The run is going away — stop waiting, but leave the request standing.
      // Rejecting unwinds the run before it reaches the tool-result path that
      // deletes the approval checkpoint, which is what makes a later
      // `/approvals/resume` possible at all.
      if (signal?.aborted) {
        reject(signal.reason instanceof Error ? signal.reason : new Error('Approval wait aborted'))
        return
      }
      const onAbort = () => {
        reject(
          signal?.reason instanceof Error ? signal.reason : new Error('Approval wait aborted'),
        )
      }
      signal?.addEventListener('abort', onAbort, { once: true })

      // Parking, not denial. The request stays answerable and the checkpoint
      // written before this wait stays on disk; only the run stops occupying
      // the session. `timeoutMs <= 0` means wait indefinitely.
      const timeout = timeoutMs > 0
        ? setTimeout(() => this.park(requestId), timeoutMs)
        : null

      const entry: PendingApprovalEntry = {
        requestId,
        sessionId,
        runId,
        toolCallId: toolCall.id,
        tool: toolCall.name,
        input: toolCall.arguments,
        requestedAt,
        expiresAt,
        state: 'live',
        suggestedRule,
        resolve: (decision) => {
          signal?.removeEventListener('abort', onAbort)
          resolve(decision)
        },
        reject: (error) => {
          signal?.removeEventListener('abort', onAbort)
          reject(error)
        },
        onParked,
        timeout,
      }
      this.pending.set(requestId, entry)
      this.hooks.onPending?.(toPendingApproval(entry))
    })
  }

  private park(requestId: string): void {
    const entry = this.pending.get(requestId)
    if (!entry || entry.state === 'parked') return
    entry.timeout = null
    entry.state = 'parked'
    entry.parkedAt = new Date().toISOString()
    entry.resumeAvailable = true
    const approval = toPendingApproval(entry)
    this.hooks.onTimeout?.(approval)
    if (entry.onParked) {
      entry.onParked(approval)
      return
    }
    entry.reject(new ApprovalParkedError(entry.requestId, entry.tool, entry.sessionId))
  }

  /** The parked approval for a request, if the registry still holds one. */
  getParked(requestId: string): PendingApproval | undefined {
    const entry = this.pending.get(requestId)
    return entry && entry.state === 'parked' ? toPendingApproval(entry) : undefined
  }

  /**
   * Drop a parked entry without journaling a decision — for callers that
   * resume the run from its checkpoint and record the decision themselves.
   */
  releaseParked(requestId: string): PendingApproval | undefined {
    const entry = this.pending.get(requestId)
    if (!entry || entry.state !== 'parked') return undefined
    this.pending.delete(requestId)
    return toPendingApproval(entry)
  }

  /**
   * Pre-authorise every tool call a future session will make.
   *
   * Exists for scheduled jobs. A job fires when nobody is at the screen, so any
   * tool its run needs that requires approval simply times out — the job stays
   * enabled and never accomplishes anything. Autonomy alone cannot fix that:
   * the policy check is `mode === 'supervised' || (autonomy === 'supervised' && …)`,
   * so a tool whose *policy* mode is supervised (fs.write, terminal.run) still
   * prompts no matter how the run's autonomy is set. An auto-approval short
   * circuits ahead of that check, which is why this is the mechanism that works.
   *
   * Scope is exactly one job: the scheduler runs each job under the stable
   * session id `scheduler-<jobId>`, so keying on that grants nothing outside
   * that job's own runs. Contrast `always`, which the approval dialog already
   * offers — that is a global tool+pattern rule and would open the tool in
   * every conversation.
   *
   * Caller must have explicit user consent; nothing here asks.
   *
   * LIMITATION — does not survive a daemon restart. `sessionAutoApprovals` is
   * in-memory, while scheduled jobs outlive restarts, so an unattended job
   * reverts to stalling on its first approval prompt after one. The durable
   * store cannot hold this as-is: `matchesRule` refuses to match across tools,
   * so a blanket "any tool for this job" has no single-row representation, and
   * the tools a job will need are unknown when it is created.
   *
   * The fix is to make it a property of the job (a persisted `unattended`
   * column) and have the scheduler executor re-establish this grant before each
   * run of a job that carries it. That keeps the durable flag where it belongs
   * — on the job, visible and toggleable — and leaves this method as the
   * per-run mechanism it already is.
   */
  grantSessionAutoApproval(sessionId: string, approvedBy = 'user'): void {
    this.sessionAutoApprovals.set(sessionId, {
      sessionId,
      decision: { decision: 'approved', approved: true, note: `pre-authorised by ${approvedBy}` },
      scope: 'session-all',
      rule: broadApprovalRule('session-all'),
      createdAt: Date.now(),
    })
  }

  /** Drop a pre-authorisation — used when the job it belonged to is removed. */
  revokeSessionAutoApproval(sessionId: string): void {
    this.sessionAutoApprovals.delete(sessionId)
  }

  tryAutoApproval(params: {
    sessionId: string
    toolCall: ToolCall
    runId?: string
  }): ApprovalDecision | null {
    const broadDecision = this.evaluateBroadApproval(
      params.sessionId,
      params.runId,
      params.toolCall,
    )
    if (broadDecision) return broadDecision
    if (!this.decisions) return null

    const evaluation = this.decisions.evaluate({
      sessionId: params.sessionId,
      tool: params.toolCall.name,
      input: params.toolCall.arguments,
    })
    if (evaluation.verdict !== 'prompt' && evaluation.rule) {
      return {
        decision: evaluation.verdict,
        approved: evaluation.verdict === 'approved',
        autoApproval: {
          rule: {
            tool: evaluation.rule.tool,
            pattern: evaluation.rule.pattern,
          },
          scope: evaluation.rule.scope,
        },
      }
    }
    return null
  }

  listForSession(sessionId: string): PendingApproval[] {
    return [...this.pending.values()]
      .filter((entry) => entry.sessionId === sessionId)
      .map((entry) => toPendingApproval(entry))
      .sort((left, right) => left.requestedAt.localeCompare(right.requestedAt))
  }

  listAll(): PendingApproval[] {
    return [...this.pending.values()]
      .map((entry) => toPendingApproval(entry))
      .sort((left, right) => left.requestedAt.localeCompare(right.requestedAt))
  }

  cancelForSession(
    sessionId: string,
    note = 'Session was deleted before approval was answered',
    options: { preserveParked?: boolean } = {},
  ): number {
    let cancelled = 0
    for (const entry of [...this.pending.values()]) {
      if (entry.sessionId !== sessionId || (options.preserveParked && entry.state === 'parked')) {
        continue
      }
      if (entry.timeout) clearTimeout(entry.timeout)
      this.pending.delete(entry.requestId)
      entry.resolve({
        decision: 'denied',
        approved: false,
        note,
      })
      cancelled += 1
    }
    this.sessionAutoApprovals.delete(sessionId)
    for (const [runId, decision] of [...this.runAutoApprovals.entries()]) {
      if (decision.sessionId === sessionId) {
        this.runAutoApprovals.delete(runId)
      }
    }
    return cancelled
  }

  async respond(
    requestId: string,
    approved: boolean | ApprovalDecision,
    options: RespondOptions | string = {},
  ): Promise<ApprovalRespondResult> {
    const normalized: RespondOptions = typeof options === 'string'
      ? { approvedBy: options }
      : options
    const scope: ApprovalScope = normalized.scope ?? 'once'
    const approvedBy = normalized.approvedBy ?? 'user'

    const entry = this.pending.get(requestId)
    if (!entry) {
      return { resolved: false }
    }
    const decision = normalizeApprovalDecision(approved)
    const parked = entry.state === 'parked'

    if (entry.timeout) clearTimeout(entry.timeout)

    try {
      await this.sessions.appendEvent(entry.sessionId, {
        type: 'approval_response',
        id: randomUUID(),
        timestamp: new Date().toISOString(),
        requestId,
        decision: decision.decision,
        approved: decision.approved,
        note: decision.note,
        approvedBy,
        scope,
      })
    } catch (appendError) {
      // The auto-denial timer is already cleared, so leaving the entry
      // pending would park the paused tool execution forever. Fail safe the
      // same way the durability rule below demands: no durable consent
      // record, no side effect — release the execution as a denial and let
      // the caller see the journaling failure.
      this.pending.delete(requestId)
      entry.resolve({
        decision: 'denied',
        approved: false,
        note: 'Approval could not be recorded (consent journaling failed); the action was denied.',
      })
      throw appendError
    }

    // Only enable future reuse after the consent event is durable. Generated
    // rules are exact-input hashes; a human-edited wildcard remains visible
    // verbatim so surfaces can explain precisely what may short-circuit.
    let rule: ApprovalRule | undefined
    if (decision.decision !== 'feedback') {
      rule = this.rememberBroadApproval(entry, decision, scope)
    }
    if (
      !rule
      && this.decisions
      && isRememberedApprovalScope(scope)
      && decision.decision !== 'feedback'
    ) {
      if (normalized.rule) {
        const persisted = this.decisions.upsert({
          tool: normalized.rule.tool,
          pattern: normalized.rule.pattern,
          scope,
          approved: decision.approved,
          sessionId: scope === 'session' ? entry.sessionId : undefined,
        })
        rule = { tool: persisted.tool, pattern: persisted.pattern }
      } else {
        rule = this.decisions.remember({
          sessionId: entry.sessionId,
          tool: entry.tool,
          input: entry.input,
          approved: decision.approved,
          scope,
        })
      }
    }

    // The durable approval evidence must exist before the paused execution is
    // released. Otherwise a session-store failure can run a side effect with
    // no corresponding consent record.
    this.pending.delete(requestId)
    entry.resolve(decision)

    return parked
      ? { resolved: true, rule, parked: toPendingApproval(entry) }
      : { resolved: true, rule }
  }

  private evaluateBroadApproval(
    sessionId: string,
    runId: string | undefined,
    toolCall: ToolCall,
  ): ApprovalDecision | null {
    this.pruneExpiredRunApprovals()
    if (runId) {
      const runDecision = this.runAutoApprovals.get(runId)
      if (
        runDecision
        && runDecision.sessionId === sessionId
        && matchesRule(runDecision.rule, toolCall.name, toolCall.arguments)
      ) {
        return toAutoApprovalDecision(runDecision)
      }
    }
    const sessionDecision = this.sessionAutoApprovals.get(sessionId)
    return sessionDecision ? toAutoApprovalDecision(sessionDecision) : null
  }

  private rememberBroadApproval(
    entry: PendingApprovalEntry,
    decision: ApprovalDecision,
    scope: ApprovalScope,
  ): ApprovalRule | undefined {
    if (scope === 'run') {
      if (!entry.runId) return undefined
      const rule = entry.suggestedRule ?? describeRuleFor(entry.tool, entry.input)
      this.runAutoApprovals.set(entry.runId, {
        sessionId: entry.sessionId,
        decision,
        scope,
        rule,
        createdAt: Date.now(),
        expiresAt: Date.now() + RUN_AUTO_APPROVAL_TTL_MS,
      })
      return rule
    }

    if (scope === 'session-all') {
      const rule = broadApprovalRule('session-all')
      this.sessionAutoApprovals.set(entry.sessionId, {
        sessionId: entry.sessionId,
        decision,
        scope,
        rule,
        createdAt: Date.now(),
      })
      return rule
    }

    return undefined
  }

  private pruneExpiredRunApprovals(now = Date.now()): void {
    for (const [runId, decision] of [...this.runAutoApprovals.entries()]) {
      if (decision.expiresAt !== undefined && decision.expiresAt <= now) {
        this.runAutoApprovals.delete(runId)
      }
    }
  }
}

/**
 * The single pending-decision bound. After this delay an unanswered approval
 * parks its run (checkpoint kept, lease released, answer resumes it).
 *
 * `SEPILOTD_APPROVAL_TIMEOUT_MS` is the knob (default 10 min, floor 60 s).
 * `SEPILOTD_PENDING_DECISION_TIMEOUT_MS` is the legacy stream-backstop name;
 * when set it overrides, and `0` there keeps its old meaning of "wait
 * indefinitely" (returns 0: no parking timer).
 */
export function resolveApprovalTimeoutMs(): number {
  const legacy = process.env.SEPILOTD_PENDING_DECISION_TIMEOUT_MS
  if (legacy !== undefined && legacy.trim() !== '') {
    const parsed = Number.parseInt(legacy, 10)
    if (Number.isFinite(parsed) && parsed >= 0) {
      return parsed === 0 ? 0 : Math.max(1000, parsed)
    }
  }
  const raw = Number(process.env.SEPILOTD_APPROVAL_TIMEOUT_MS ?? '')
  if (Number.isFinite(raw) && raw > 0) {
    return Math.max(APPROVAL_TIMEOUT_FLOOR_MS, raw)
  }
  return DEFAULT_APPROVAL_TIMEOUT_MS
}

function toPendingApproval(entry: PendingApprovalEntry): PendingApproval {
  const {
    resolve: _resolve,
    reject: _reject,
    onParked: _onParked,
    timeout: _timeout,
    ...approval
  } = entry
  return { ...approval }
}

export interface ApprovalRespondResult {
  resolved: boolean
  /**
   * Set when the response enables a future auto-decision. For
   * session/always and run this is the exact-by-default tool rule;
   * session-all returns its explicit broad synthetic rule. Undefined for
   * once-scope, feedback, or unknown requestIds.
   */
  rule?: ApprovalRule
  /**
   * Set when the answered request had already parked its run. The caller
   * owns resuming that run from its approval checkpoint; the decision is
   * journaled here, so the resume must not record it again.
   */
  parked?: PendingApproval
}

function normalizeApprovalDecision(
  input: boolean | ApprovalDecision,
): ApprovalDecision {
  if (typeof input === 'boolean') {
    return {
      decision: input ? 'approved' : 'denied',
      approved: input,
    }
  }

  const decision: ApprovalDecisionStatus =
    input.decision
    ?? (input.approved ? 'approved' : 'denied')

  return {
    decision,
    approved: decision === 'approved',
    note: input.note,
    ...(input.timedOut !== undefined ? { timedOut: input.timedOut } : {}),
    // "deny & stop" is only meaningful on a denial; never let it ride along
    // with an approval or feedback decision.
    ...(decision === 'denied' && input.stop !== undefined ? { stop: input.stop } : {}),
  }
}

function isRememberedApprovalScope(scope: ApprovalScope): scope is RememberedApprovalScope {
  return scope === 'session' || scope === 'always'
}

function broadApprovalRule(scope: AutoApprovalScope): ApprovalRule {
  switch (scope) {
    case 'session-all':
      return { tool: '*', pattern: 'all tools in this session' }
    case 'run':
      // Run-scoped approvals are invocation-pattern scoped and are created in
      // rememberBroadApproval. This fallback is unreachable but remains
      // fail-closed for exhaustive callers.
      return { tool: '__invalid__', pattern: '__invalid__' }
    default:
      return { tool: '*', pattern: 'remembered broad approval' }
  }
}

function toAutoApprovalDecision(decision: BroadApprovalDecision): ApprovalDecision {
  return {
    decision: decision.decision.decision,
    approved: decision.decision.approved,
    note: decision.decision.note,
    autoApproval: {
      rule: decision.rule,
      scope: decision.scope,
    },
  }
}
