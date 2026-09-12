import { mkdir, readFile, rename, writeFile } from 'node:fs/promises'
import { dirname } from 'node:path'
import type {
  ApprovalEvaluationResult,
  ApprovalRule,
  IApprovalDecisionStore,
  RememberedDecision,
  RememberedDecisionInput,
  RememberedDecisionMatch,
} from '@sepilotd/core'
import { isStaleRememberedDecision } from '@sepilotd/core'
import { createLogger } from '../../logger.js'
import { isNodeFsError } from '../../utils/fs-error.js'
import { describeRuleFor, matchesRule } from './approval-rules.js'

const log = createLogger('approval-decisions')

interface DecisionStoreOptions {
  persistentPath?: string
}

function now(): string {
  return new Date().toISOString()
}

function normalizeText(value: string, field: string): string {
  const trimmed = value.trim()
  if (!trimmed) {
    throw new Error(`${field} is required`)
  }
  return trimmed
}

function normalizeDecisionInput(params: RememberedDecisionInput): RememberedDecisionInput {
  const tool = normalizeText(params.tool, 'tool')
  const pattern = normalizeText(params.pattern, 'pattern')
  if (params.scope === 'session') {
    return {
      ...params,
      tool,
      pattern,
      sessionId: normalizeText(params.sessionId ?? '', 'sessionId'),
    }
  }
  return {
    ...params,
    tool,
    pattern,
    sessionId: undefined,
  }
}

function normalizeDecisionMatch(match: RememberedDecisionMatch): RememberedDecisionMatch {
  const tool = normalizeText(match.tool, 'tool')
  const pattern = normalizeText(match.pattern, 'pattern')
  if (match.scope === 'session') {
    return {
      ...match,
      tool,
      pattern,
      sessionId: normalizeText(match.sessionId ?? '', 'sessionId'),
    }
  }
  return {
    ...match,
    tool,
    pattern,
    sessionId: undefined,
  }
}

function matchesExactDecision(
  decision: RememberedDecision,
  match: RememberedDecisionMatch,
): boolean {
  if (decision.tool !== match.tool) return false
  if (decision.pattern !== match.pattern) return false
  if (decision.scope !== match.scope) return false
  if (match.scope === 'session') return decision.sessionId === match.sessionId
  return true
}

function createDecision(
  params: RememberedDecisionInput,
  existing?: RememberedDecision,
): RememberedDecision {
  const decision: RememberedDecision = {
    tool: params.tool,
    pattern: params.pattern,
    scope: params.scope,
    approved: params.approved,
    sessionId: params.scope === 'session' ? params.sessionId : undefined,
    createdAt: existing?.createdAt ?? now(),
    hitCount: existing?.hitCount ?? 0,
  }
  if (existing?.lastHitAt) {
    decision.lastHitAt = existing.lastHitAt
  }
  return decision
}

export class ApprovalDecisionStore implements IApprovalDecisionStore {
  private sessionDecisions = new Map<string, RememberedDecision[]>()
  private persistentDecisions: RememberedDecision[] = []
  private persistentLoaded = false
  private persistentPath?: string
  private pendingPersist: Promise<void> = Promise.resolve()

  constructor(options: DecisionStoreOptions = {}) {
    this.persistentPath = options.persistentPath
  }

  async flush(): Promise<void> {
    await this.pendingPersist
  }

  async initialize(): Promise<void> {
    if (!this.persistentPath || this.persistentLoaded) {
      this.persistentLoaded = true
      return
    }
    let raw: string
    try {
      raw = await readFile(this.persistentPath, 'utf8')
    } catch (err) {
      if (isNodeFsError(err, 'ENOENT')) {
        // First run — no remembered "Always allow" decisions yet.
        this.persistentLoaded = true
        return
      }
      log.error('failed to read remembered approval decisions', {
        path: this.persistentPath,
        error: err instanceof Error ? err.message : String(err),
      })
      throw err
    }

    try {
      const parsed = JSON.parse(raw) as { decisions?: RememberedDecision[] }
      if (Array.isArray(parsed.decisions)) {
        this.persistentDecisions = parsed.decisions.filter(
          (entry) => entry.scope === 'always',
        )
      }
    } catch (err) {
      // Corrupted decision file — rotate aside so the operator can
      // see the rule set was wiped out and reissue if they want
      // those "Always allow" rules back. Silently restarting empty
      // would mean every previously-remembered tool re-prompts and
      // the operator has no idea why.
      const aside = `${this.persistentPath}.broken-${Date.now()}`
      log.error('approval decisions file unparseable; rotating aside', {
        path: this.persistentPath,
        rotated: aside,
        error: err instanceof Error ? err.message : String(err),
      })
      await rename(this.persistentPath, aside)
      this.persistentDecisions = []
    }
    this.persistentLoaded = true
  }

  describeRule(tool: string, input: Record<string, unknown>): ApprovalRule {
    return describeRuleFor(tool, input)
  }

  evaluate(params: {
    sessionId: string
    tool: string
    input: Record<string, unknown>
  }): ApprovalEvaluationResult {
    const { sessionId, tool, input } = params
    const sessionMatches = this.sessionDecisions.get(sessionId) ?? []
    for (const entry of sessionMatches) {
      if (matchesRule(entry, tool, input)) {
        // Bump usage counters so `decisions list` can show *which*
        // rules are actually short-circuiting prompts — a hitCount=0
        // rule registered weeks ago is a cleanup candidate, while a
        // hot rule signals heavy unsupervised tool execution that
        // operators should re-examine. Session-scope rules live
        // in-memory only, so no persistence call here.
        entry.hitCount = (entry.hitCount ?? 0) + 1
        entry.lastHitAt = now()
        return {
          verdict: entry.approved ? 'approved' : 'denied',
          rule: entry,
        }
      }
    }
    for (const entry of this.persistentDecisions) {
      if (matchesRule(entry, tool, input)) {
        // Always-scope rules outlive the daemon process, so the
        // hit counter must persist alongside the rule itself —
        // otherwise restarts would reset the operator's view of
        // which rules are in active use.
        entry.hitCount = (entry.hitCount ?? 0) + 1
        entry.lastHitAt = now()
        this.schedulePersist()
        return {
          verdict: entry.approved ? 'approved' : 'denied',
          rule: entry,
        }
      }
    }
    return { verdict: 'prompt' }
  }

  remember(params: {
    sessionId: string
    tool: string
    input: Record<string, unknown>
    approved: boolean
    scope: 'session' | 'always'
  }): ApprovalRule {
    const rule = this.describeRule(params.tool, params.input)
    const decision = createDecision({
      tool: rule.tool,
      pattern: rule.pattern,
      scope: params.scope,
      approved: params.approved,
      sessionId: params.scope === 'session' ? params.sessionId : undefined,
    })
    this.replaceDecision(decision)
    return rule
  }

  private schedulePersist(): void {
    this.pendingPersist = this.pendingPersist
      .then(() => this.persist())
      .catch(() => undefined)
  }

  list(): RememberedDecision[] {
    const sessionEntries = [...this.sessionDecisions.values()].flat()
    return [...this.persistentDecisions, ...sessionEntries]
  }

  upsert(params: RememberedDecisionInput): RememberedDecision {
    const normalized = normalizeDecisionInput(params)
    const existing = this.findExact(normalized)
    const decision = createDecision(normalized, existing)
    this.replaceDecision(decision)
    return decision
  }

  update(
    match: RememberedDecisionMatch,
    params: RememberedDecisionInput,
  ): RememberedDecision | null {
    const normalizedMatch = normalizeDecisionMatch(match)
    const existing = this.findExact(normalizedMatch)
    if (!existing) return null

    this.remove(normalizedMatch)
    const normalizedParams = normalizeDecisionInput(params)
    const decision = createDecision(normalizedParams, existing)
    this.replaceDecision(decision)
    return decision
  }

  remove(match: RememberedDecisionMatch): boolean {
    const normalized = normalizeDecisionMatch(match)
    if (normalized.scope === 'session') {
      const sessionId = normalized.sessionId!
      const bucket = this.sessionDecisions.get(sessionId) ?? []
      const remaining = bucket.filter((entry) => !matchesExactDecision(entry, normalized))
      if (remaining.length === bucket.length) return false
      if (remaining.length === 0) this.sessionDecisions.delete(sessionId)
      else this.sessionDecisions.set(sessionId, remaining)
      return true
    }

    const remaining = this.persistentDecisions.filter(
      (entry) => !matchesExactDecision(entry, normalized),
    )
    if (remaining.length === this.persistentDecisions.length) return false
    this.persistentDecisions = remaining
    this.schedulePersist()
    return true
  }

  clear(params: {
    sessionId?: string
    scope?: 'session' | 'always'
    stale?: boolean
    tool?: string
    pattern?: string
    approved?: boolean
  } = {}): void {
    const now = Date.now()
    const hasFilter = Boolean(
      params.stale
      || params.tool
      || params.pattern
      || params.approved !== undefined,
    )
    // `keep` returns true when the entry should *survive* the clear.
    // Filters compose: an entry is removed only if it matches every
    // active filter (stale AND tool). Without the all-filters-must-
    // match contract, clear --tool fs.write --stale would also wipe
    // hot fs.write rules, which contradicts "prune the cleanup
    // candidates I just listed".
    const keep = (entry: RememberedDecision): boolean => {
      if (params.stale && !isStaleRememberedDecision(entry, now)) return true
      if (params.tool && entry.tool !== params.tool) return true
      if (params.pattern && entry.pattern !== params.pattern) return true
      if (params.approved !== undefined && entry.approved !== params.approved) return true
      return false
    }

    if (!params.scope || params.scope === 'session') {
      if (params.sessionId) {
        if (hasFilter) {
          const bucket = this.sessionDecisions.get(params.sessionId) ?? []
          const remaining = bucket.filter(keep)
          if (remaining.length === 0) this.sessionDecisions.delete(params.sessionId)
          else this.sessionDecisions.set(params.sessionId, remaining)
        } else {
          this.sessionDecisions.delete(params.sessionId)
        }
      } else if (hasFilter) {
        for (const [id, bucket] of this.sessionDecisions) {
          const remaining = bucket.filter(keep)
          if (remaining.length === 0) this.sessionDecisions.delete(id)
          else this.sessionDecisions.set(id, remaining)
        }
      } else {
        this.sessionDecisions.clear()
      }
    }
    if (!params.scope || params.scope === 'always') {
      if (hasFilter) {
        this.persistentDecisions = this.persistentDecisions.filter(keep)
      } else {
        this.persistentDecisions = []
      }
      this.schedulePersist()
    }
  }

  private findExact(match: RememberedDecisionMatch): RememberedDecision | undefined {
    const normalized = normalizeDecisionMatch(match)
    if (normalized.scope === 'session') {
      const bucket = this.sessionDecisions.get(normalized.sessionId!) ?? []
      return bucket.find((entry) => matchesExactDecision(entry, normalized))
    }
    return this.persistentDecisions.find((entry) => matchesExactDecision(entry, normalized))
  }

  private replaceDecision(decision: RememberedDecision): void {
    if (decision.scope === 'session') {
      const sessionId = decision.sessionId!
      const bucket = this.sessionDecisions.get(sessionId) ?? []
      const withoutDuplicate = bucket.filter(
        (entry) => !matchesExactDecision(entry, decision),
      )
      withoutDuplicate.push(decision)
      this.sessionDecisions.set(sessionId, withoutDuplicate)
      return
    }

    this.persistentDecisions = this.persistentDecisions.filter(
      (entry) => !matchesExactDecision(entry, decision),
    )
    this.persistentDecisions.push(decision)
    this.schedulePersist()
  }

  private async persist(): Promise<void> {
    if (!this.persistentPath) return
    try {
      await mkdir(dirname(this.persistentPath), { recursive: true })
      await writeFile(
        this.persistentPath,
        `${JSON.stringify({ decisions: this.persistentDecisions }, null, 2)}\n`,
        { mode: 0o600 },
      )
    } catch (err) {
      // Persistence failures are non-fatal: the in-memory layer
      // still serves this session's prompts. But "Always allow" is
      // an explicit user contract — log so the operator can see
      // their rule won't survive a daemon restart, and they can
      // address the underlying disk/permission problem.
      log.error('failed to persist remembered approval decisions', {
        path: this.persistentPath,
        decisionCount: this.persistentDecisions.length,
        error: err instanceof Error ? err.message : String(err),
      })
    }
  }
}
