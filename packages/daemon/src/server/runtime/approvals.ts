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
  expiresAt: string
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
    forcePrompt?: boolean
    signal?: AbortSignal
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

  getParked(requestId: string): PendingApproval | undefined {
    const entry = this.pending.get(requestId)
    return entry && entry.state === 'parked' ? toPendingApproval(entry) : undefined
  }

  releaseParked(requestId: string): PendingApproval | undefined {
    const entry = this.pending.get(requestId)
    if (!entry || entry.state !== 'parked') return undefined
    this.pending.delete(requestId)
    return toPendingApproval(entry)
  }

  grantSessionAutoApproval(sessionId: string, approvedBy = 'user'): void {
    this.sessionAutoApprovals.set(sessionId, {
      sessionId,
      decision: { decision: 'approved', approved: true, note: `pre-authorised by ${approvedBy}` },
      scope: 'session-all',
      rule: broadApprovalRule('session-all'),
      createdAt: Date.now(),
    })
  }

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
      this.pending.delete(requestId)
      entry.resolve({
        decision: 'denied',
        approved: false,
        note: 'Approval could not be recorded (consent journaling failed); the action was denied.',
      })
      throw appendError
    }

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
  rule?: ApprovalRule
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
