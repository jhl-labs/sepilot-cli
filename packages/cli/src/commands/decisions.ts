import chalk from 'chalk'
import type { DaemonRememberedApproval } from '@sepilotd/api-client'
import {
  STALE_RULE_THRESHOLD_DAYS,
  isStaleRememberedDecision,
} from '@sepilotd/api-client'
import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'
import { friendlyErrorMessage, printApiError } from '../utils/error-message.js'
import { detectCliLocale } from '../utils/locale.js'

const DECISIONS_COPY = {
  en: {
    unknownScopeList: (raw: string) => `Unknown --scope: ${raw}. Use session|always (or omit).`,
    unknownScopeClear: (raw: string) => `Unknown --scope: ${raw}. Use session|always (or omit to clear all).`,
    justNow: 'just now',
    minutesAgo: (n: number) => `${n}m ago`,
    hoursAgo: (n: number) => `${n}h ago`,
    daysAgo: (n: number) => `${n}d ago`,
    monthsAgo: (n: number) => `${n}mo ago`,
    noMatchedFilter: (filter: string) => `No remembered approval decisions matched filter (${filter}).`,
    noDecisions: 'No remembered approval decisions.',
    headlineStale: 'Stale remembered approval decisions',
    headline: 'Remembered approval decisions',
    headlineSuffix: (total: number, filter: string, showing: number) =>
      `(${total} total${filter}, showing ${showing}):`,
    approved: 'approved',
    denied: 'denied',
    sessionLabelPrefix: (id: string) => ` · session ${id}`,
    hitsPrefix: (n: number) => ` · hits: ${n}`,
    lastPrefix: (label: string) => ` · last: ${label}`,
    stale: ' (stale?)',
    staleHint: (count: number, days: number) =>
      `  ${count} rule${count === 1 ? '' : 's'} flagged stale (no hits, ≥${days}d old). List with \`sepilot decisions list --stale\`, prune with \`sepilot decisions clear --scope <session|always>\`.`,
    moreHint: (count: number) =>
      `\n  …${count} more. Use --limit <n> or --limit 0 to show all.`,
    failedList: (msg: string) => `Failed to list decisions: ${msg}`,
    usageShow: 'Usage: sepilot decisions show <tool>',
    noDecisionsForTool: (tool: string) => `No remembered approval decisions for tool '${tool}'.`,
    headlineForTool: (tool: string, count: number) =>
      `Remembered approval decisions for '${tool}' (${count}):`,
    patternLabel: 'pattern:   ',
    sessionIdLabel: 'sessionId: ',
    createdAtLabel: 'createdAt: ',
    hitCountLabel: 'hitCount:  ',
    lastHitAtLabel: 'lastHitAt: ',
    neverMatched: 'never matched',
    failedShow: (msg: string) => `Failed to show decisions: ${msg}`,
    clearedDecisions: (filter: string) => `Cleared remembered decisions${filter}.`,
    failedClear: (msg: string) => `Failed to clear decisions: ${msg}`,
    all: 'all',
  },
  ko: {
    unknownScopeList: (raw: string) => `알 수 없는 --scope: ${raw}. session|always를 사용하세요 (또는 생략).`,
    unknownScopeClear: (raw: string) => `알 수 없는 --scope: ${raw}. session|always를 사용하세요 (또는 전부 지우려면 생략).`,
    justNow: '방금',
    minutesAgo: (n: number) => `${n}분 전`,
    hoursAgo: (n: number) => `${n}시간 전`,
    daysAgo: (n: number) => `${n}일 전`,
    monthsAgo: (n: number) => `${n}개월 전`,
    noMatchedFilter: (filter: string) => `필터(${filter})에 일치하는 기억된 승인 결정이 없습니다.`,
    noDecisions: '기억된 승인 결정이 없습니다.',
    headlineStale: '오래된 기억된 승인 결정',
    headline: '기억된 승인 결정',
    headlineSuffix: (total: number, filter: string, showing: number) =>
      `(전체 ${total}개${filter}, ${showing}개 표시):`,
    approved: '승인됨',
    denied: '거부됨',
    sessionLabelPrefix: (id: string) => ` · 세션 ${id}`,
    hitsPrefix: (n: number) => ` · 히트: ${n}`,
    lastPrefix: (label: string) => ` · 마지막: ${label}`,
    stale: ' (오래됨?)',
    staleHint: (count: number, days: number) =>
      `  ${count}개 규칙이 오래된 것으로 표시됨 (히트 없음, ≥${days}일 경과). \`sepilot decisions list --stale\`로 목록 확인, \`sepilot decisions clear --scope <session|always>\`로 정리.`,
    moreHint: (count: number) =>
      `\n  …${count}개 더. 전부 표시하려면 --limit <n> 또는 --limit 0을 사용하세요.`,
    failedList: (msg: string) => `결정 목록 조회 실패: ${msg}`,
    usageShow: '사용법: sepilot decisions show <도구>',
    noDecisionsForTool: (tool: string) => `도구 '${tool}'에 대한 기억된 승인 결정이 없습니다.`,
    headlineForTool: (tool: string, count: number) =>
      `'${tool}'에 대한 기억된 승인 결정 (${count}개):`,
    patternLabel: 'pattern:   ',
    sessionIdLabel: 'sessionId: ',
    createdAtLabel: 'createdAt: ',
    hitCountLabel: 'hitCount:  ',
    lastHitAtLabel: 'lastHitAt: ',
    neverMatched: '일치한 적 없음',
    failedShow: (msg: string) => `결정 표시 실패: ${msg}`,
    clearedDecisions: (filter: string) => `기억된 결정 지움${filter}.`,
    failedClear: (msg: string) => `결정 지우기 실패: ${msg}`,
    all: '전체',
  },
} as const

type DecisionsCopy = (typeof DECISIONS_COPY)[keyof typeof DECISIONS_COPY]

function decisionsCopy(): DecisionsCopy {
  return DECISIONS_COPY[detectCliLocale()] ?? DECISIONS_COPY.en
}

export interface DecisionsListOptions {
  url?: string
  /**
   * When set, narrow output to rules that look stale: zero hits AND
   * createdAt older than STALE_RULE_THRESHOLD_DAYS. Lets automation
   * pipe `decisions list --stale --json | jq '.decisions[].pattern'`
   * into a cleanup script without re-implementing the staleness rule
   * client-side.
   */
  stale?: boolean
  /**
   * Cap the rendered rows. Default 20 mirrors `sessions list`; '0'
   * (or any non-positive integer) means "show all" — useful when
   * piping to grep/jq. Filter happens client-side after sort, so the
   * row order (most-recently-matched first) is preserved.
   */
  limit?: string
  /**
   * Narrow by rule scope. session|always (omit for all). Pairs with
   * `decisions clear --scope` so the operator can preview-then-prune
   * with the same flag spelled the same way.
   */
  scope?: string
  /**
   * Narrow by exact tool name (e.g. `terminal.run`, `fs.write`).
   * Useful for "show me every rule that lets the agent touch <tool>"
   * audits — without this, an operator on a daemon with rules for a
   * dozen tools would have to grep stdout. Exact match (not glob)
   * because tool names are well-known constants and partial matches
   * would silently include unrelated tools.
   */
  tool?: string
}

export interface DecisionsClearOptions {
  url?: string
  scope?: string
  session?: string
  /**
   * When set, ask the daemon to remove only rules that match
   * isStaleRememberedDecision. Pairs naturally with `decisions list
   * --stale` — see what would be pruned, then prune it. Combinable
   * with --scope so an operator can narrow ("clear stale always
   * rules only"); --session is intentionally ignored under --stale
   * because session-scoped staleness rarely makes sense (sessions
   * are short-lived already).
   */
  stale?: boolean
  /**
   * Exact tool name. Pairs with `decisions list --tool` for
   * preview-then-prune by tool. Combinable with --scope/--stale —
   * filters compose, so `clear --tool fs.write --stale` removes
   * only stale fs.write rules.
   */
  tool?: string
}

const VALID_CLEAR_SCOPES = new Set(['session', 'always'])
const VALID_LIST_SCOPES = new Set<'session' | 'always'>(['session', 'always'])

function parseListScope(raw: string | undefined): 'session' | 'always' | undefined {
  if (!raw) return undefined
  const trimmed = raw.trim().toLowerCase() as 'session' | 'always'
  if (VALID_LIST_SCOPES.has(trimmed)) return trimmed
  console.error(chalk.red(decisionsCopy().unknownScopeList(raw)))
  process.exit(1)
}

export function formatPastTime(iso: string | undefined, now: number): string | undefined {
  if (!iso) return undefined
  const past = new Date(iso).getTime()
  if (!Number.isFinite(past)) return undefined
  const deltaMs = now - past
  if (deltaMs < 0) return undefined
  const copy = decisionsCopy()
  const totalSeconds = Math.floor(deltaMs / 1000)
  if (totalSeconds < 60) return copy.justNow
  const minutes = Math.floor(totalSeconds / 60)
  if (minutes < 60) return copy.minutesAgo(minutes)
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return copy.hoursAgo(hours)
  const days = Math.floor(hours / 24)
  if (days < 30) return copy.daysAgo(days)
  const months = Math.floor(days / 30)
  return copy.monthsAgo(months)
}

/**
 * Thin wrapper over the core `isStaleRememberedDecision` helper so
 * callers in chat-stream-printer / chat shell keep their existing
 * import path. The actual predicate lives in core (single source of
 * truth shared with the daemon's clear({stale}) — without that, the
 * cli could flag a rule the daemon refuses to delete).
 */
export function isStaleRule(
  decision: DaemonRememberedApproval,
  now: number,
): boolean {
  return isStaleRememberedDecision(decision, now)
}

/**
 * Sort order: most-recently-matched first, then never-matched (those
 * fall to the bottom, ordered by createdAt asc so the *oldest* unused
 * rule is at the very bottom — the natural cleanup target). Without
 * this, decisions list shows insertion order, so a rule registered
 * yesterday and used 100 times sits below a forgotten rule from
 * months ago.
 */
export function compareDecisions(
  a: DaemonRememberedApproval,
  b: DaemonRememberedApproval,
): number {
  const aLast = a.lastHitAt ? new Date(a.lastHitAt).getTime() : null
  const bLast = b.lastHitAt ? new Date(b.lastHitAt).getTime() : null
  if (aLast !== null && bLast !== null) return bLast - aLast
  if (aLast !== null) return -1
  if (bLast !== null) return 1
  // Both never matched — oldest first (oldest is the most likely
  // cleanup candidate).
  const aCreated = a.createdAt ? new Date(a.createdAt).getTime() : 0
  const bCreated = b.createdAt ? new Date(b.createdAt).getTime() : 0
  return aCreated - bCreated
}

function parseClearScope(raw: string | undefined): 'session' | 'always' | undefined {
  if (!raw) return undefined
  const trimmed = raw.trim().toLowerCase()
  if (VALID_CLEAR_SCOPES.has(trimmed)) return trimmed as 'session' | 'always'
  console.error(chalk.red(decisionsCopy().unknownScopeClear(raw)))
  process.exit(1)
}

export async function decisionsListCommand(
  options: DecisionsListOptions = {},
): Promise<void> {
  const copy = decisionsCopy()
  const scopeFilter = parseListScope(options.scope)
  const toolFilter = options.tool?.trim() || undefined
  // --limit '0' (or any non-positive integer) means "show all" so an
  // operator piping to grep/jq doesn't have to remember a giant
  // number. NaN / missing falls back to the default 20 — same
  // contract as `sessions list --limit`.
  const limitArg = options.limit ? Number.parseInt(options.limit, 10) : 20
  const limit = Number.isFinite(limitArg) && limitArg >= 0 ? limitArg : 20
  const client = new DaemonClient(options.url)
  try {
    const result = await client.listRememberedApprovals()
    const allDecisions = result.decisions ?? []
    // Filters compose: tool/scope first (cheap), stale second (date
    // math). Order doesn't change semantics but keeps the predicate
    // cost ordering rational.
    const toolFiltered = toolFilter
      ? allDecisions.filter((d) => d.tool === toolFilter)
      : allDecisions
    const scopeFiltered = scopeFilter
      ? toolFiltered.filter((d) => d.scope === scopeFilter)
      : toolFiltered
    const filteredDecisions = options.stale
      ? scopeFiltered.filter((d) => isStaleRule(d, Date.now()))
      : scopeFiltered
    const filtered: typeof result = (options.stale || scopeFilter || toolFilter)
      ? { ...result, decisions: filteredDecisions }
      : result
    output(filtered, (data) => {
      const items = data.decisions ?? []
      if (items.length === 0) {
        // Status-aware empty copy so operators can tell "filter
        // matched nothing" from "list view broke".
        const filterParts = [
          options.stale ? 'stale' : null,
          scopeFilter ? `scope=${scopeFilter}` : null,
          toolFilter ? `tool=${toolFilter}` : null,
        ].filter(Boolean).join(', ')
        if (filterParts) {
          return chalk.gray(copy.noMatchedFilter(filterParts))
        }
        return chalk.gray(copy.noDecisions)
      }
      const now = Date.now()
      // Sorting + stale-flagging happen client-side so the daemon
      // contract stays simple (returns a flat list); the cli is
      // already the place that owns presentation.
      const sortedItems = [...items].sort(compareDecisions)
      const visibleItems = limit === 0 ? sortedItems : sortedItems.slice(0, limit)
      const staleCount = sortedItems.filter((d) => isStaleRule(d, now)).length
      const lines: string[] = []
      const filterLabelParts = [
        options.stale ? 'stale-only' : null,
        scopeFilter ? `scope=${scopeFilter}` : null,
        toolFilter ? `tool=${toolFilter}` : null,
      ].filter(Boolean).join(', ')
      const headlineFilter = filterLabelParts ? `, ${filterLabelParts}` : ''
      const headlinePrefix = options.stale
        ? copy.headlineStale
        : copy.headline
      // Headline mirrors `sessions list` shape: total + filter +
      // showing. When everything fits, "showing N" matches "N total"
      // — kept verbose so a future paginated list reads consistently.
      lines.push(chalk.cyan(
        `${headlinePrefix} ${copy.headlineSuffix(sortedItems.length, headlineFilter, visibleItems.length)}`,
      ))
      for (const decision of visibleItems) {
        // Showing tool + scope + approved/denied is the minimum an
        // operator needs to recognise a leftover rule. createdAt
        // helps spot stale decisions a session ran into months ago.
        // hits surfaces the runtime usage signal — a hits=0 rule
        // weeks after registration is a cleanup candidate, while a
        // hot rule warrants a re-check of whether short-circuiting
        // is still appropriate. lastHit shows recency without
        // forcing the operator to compute "how old is this iso?".
        const verb = decision.approved ? chalk.green(copy.approved) : chalk.red(copy.denied)
        const sessionLabel = decision.sessionId ? copy.sessionLabelPrefix(decision.sessionId) : ''
        const created = decision.createdAt ? chalk.gray(` · ${decision.createdAt}`) : ''
        const hits = typeof decision.hitCount === 'number'
          ? chalk.gray(copy.hitsPrefix(decision.hitCount))
          : ''
        const lastLabel = formatPastTime(decision.lastHitAt, now)
        const last = lastLabel ? chalk.gray(copy.lastPrefix(lastLabel)) : ''
        const stale = isStaleRule(decision, now)
          ? chalk.yellow(copy.stale)
          : ''
        lines.push(
          `  [${decision.scope}] ${decision.tool} → ${verb}${sessionLabel}${created}${hits}${last}${stale}`,
        )
      }
      if (staleCount > 0 && !options.stale) {
        // Trailing hint only fires when there's actually something
        // to clean up AND the operator hasn't already filtered to
        // stale-only — otherwise it's redundant noise. Mention the
        // cli verb so the operator can act in one step.
        lines.push('')
        lines.push(
          chalk.gray(copy.staleHint(staleCount, STALE_RULE_THRESHOLD_DAYS)),
        )
      }
      if (visibleItems.length < sortedItems.length) {
        // Mirror the sessions-list "…N more" hint so operators
        // hitting the default 20-row cap can immediately see the
        // escape hatch (`--limit 0`).
        lines.push(
          chalk.gray(copy.moreHint(sortedItems.length - visibleItems.length)),
        )
      }
      return lines.join('\n')
    })
  } catch (err) {
    if (printApiError(err)) {
      process.exit(1)
    }
    console.error(chalk.red(copy.failedList(friendlyErrorMessage(err))))
    process.exit(1)
  }
}

export interface DecisionsShowOptions {
  url?: string
}

/**
 * Verbose detail view for every rule that matches a tool name.
 * `decisions list --tool <name>` already narrows the list, but stays
 * one-row-per-rule for at-a-glance scanning. `show` is the
 * complement: 1 block per rule with full ISO timestamps + relative
 * ages + the staleness verdict — the view an operator opens when
 * deciding whether to keep / extend / prune a specific rule.
 */
export async function decisionsShowCommand(
  tool: string,
  options: DecisionsShowOptions = {},
): Promise<void> {
  const copy = decisionsCopy()
  const trimmedTool = tool?.trim()
  if (!trimmedTool) {
    console.error(chalk.red(copy.usageShow))
    process.exit(1)
  }
  const client = new DaemonClient(options.url)
  try {
    const result = await client.listRememberedApprovals()
    const allDecisions = result.decisions ?? []
    const matches = allDecisions.filter((d) => d.tool === trimmedTool)
    const filtered: typeof result = { ...result, decisions: matches }
    output(filtered, (data) => {
      const items = data.decisions ?? []
      if (items.length === 0) {
        return chalk.gray(copy.noDecisionsForTool(trimmedTool))
      }
      const now = Date.now()
      const sortedItems = [...items].sort(compareDecisions)
      const lines: string[] = []
      lines.push(chalk.cyan(copy.headlineForTool(trimmedTool, items.length)))
      for (const decision of sortedItems) {
        // 1 block per rule. Field-per-line layout makes long
        // sessionId / pattern values readable; relative ages sit
        // alongside the iso timestamp so an operator can grok
        // "5d ago (2026-04-28T...)" without computing the delta.
        const verb = decision.approved ? chalk.green(copy.approved) : chalk.red(copy.denied)
        const stale = isStaleRule(decision, now)
        lines.push('')
        lines.push(`  [${decision.scope}] ${decision.tool} → ${verb}${stale ? chalk.yellow(copy.stale) : ''}`)
        lines.push(chalk.gray(`    ${copy.patternLabel} ${decision.pattern}`))
        if (decision.sessionId) {
          lines.push(chalk.gray(`    ${copy.sessionIdLabel} ${decision.sessionId}`))
        }
        if (decision.createdAt) {
          const createdAge = formatPastTime(decision.createdAt, now)
          lines.push(chalk.gray(
            `    ${copy.createdAtLabel} ${decision.createdAt}${createdAge ? ` (${createdAge})` : ''}`,
          ))
        }
        if (typeof decision.hitCount === 'number') {
          lines.push(chalk.gray(`    ${copy.hitCountLabel} ${decision.hitCount}`))
        }
        if (decision.lastHitAt) {
          const lastAge = formatPastTime(decision.lastHitAt, now)
          lines.push(chalk.gray(
            `    ${copy.lastHitAtLabel} ${decision.lastHitAt}${lastAge ? ` (${lastAge})` : ''}`,
          ))
        } else {
          // Surfacing "lastHitAt: never matched" up front saves the
          // operator from interpreting the absence of a field as
          // "the daemon forgot to record it".
          lines.push(chalk.gray(`    ${copy.lastHitAtLabel} ${copy.neverMatched}`))
        }
      }
      return lines.join('\n')
    })
  } catch (err) {
    if (printApiError(err)) {
      process.exit(1)
    }
    console.error(chalk.red(copy.failedShow(friendlyErrorMessage(err))))
    process.exit(1)
  }
}

export async function decisionsClearCommand(
  options: DecisionsClearOptions = {},
): Promise<void> {
  const copy = decisionsCopy()
  const scope = parseClearScope(options.scope)
  const tool = options.tool?.trim() || undefined
  const client = new DaemonClient(options.url)
  try {
    await client.clearRememberedApprovals({
      scope,
      sessionId: options.session,
      stale: options.stale,
      tool,
    })
    const filterLabel = [
      options.stale ? 'stale-only' : null,
      scope ? `scope=${scope}` : null,
      tool ? `tool=${tool}` : null,
      options.session ? `session=${options.session}` : null,
    ].filter(Boolean).join(', ')
    output(
      { ok: true, cleared: true, filter: filterLabel || copy.all },
      () => chalk.green(copy.clearedDecisions(filterLabel ? ` (${filterLabel})` : '')),
    )
  } catch (err) {
    if (printApiError(err)) {
      process.exit(1)
    }
    console.error(chalk.red(copy.failedClear(friendlyErrorMessage(err))))
    process.exit(1)
  }
}
