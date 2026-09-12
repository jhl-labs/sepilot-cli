import chalk from 'chalk'
import type { DaemonSessionRunbook, SessionEvent } from '@sepilotd/api-client'
import {
  buildSessionEvidenceSurfaceModel,
  formatAcceptanceCriteriaCount,
  formatToolCall,
  toolExecutionPostureLabel,
} from '@sepilotd/api-client'
import { DaemonClient } from '../client/http.js'
import { output, outputError, writeOutputText } from '../output/formatter.js'
import { friendlyErrorMessage, printApiError, printStreamError } from '../utils/error-message.js'
import { createInteractiveCliChatStreamPrinter } from '../utils/chat-stream-printer.js'
import { formatChatStreamFailure } from '../utils/chat-stream-error.js'
import { isSubstantiveAskStreamEvent } from './ask.js'
import { resolveCliStreamIdleMs } from '../utils/stream-idle.js'
import {
  openChatStreamWithConnectTimeout,
  resolveCliStreamConnectMs,
} from '../utils/stream-connect.js'
import {
  forwardDaemonStreamWithResumeRecovery,
  isTerminalCliDaemonChatEvent,
} from '../utils/stream-resume.js'

function relativeTimeLabel(deltaMs: number): string {
  if (deltaMs <= 0) return 'expired'
  const totalSeconds = Math.floor(deltaMs / 1000)
  if (totalSeconds < 60) return `${totalSeconds}s`
  const minutes = Math.floor(totalSeconds / 60)
  if (minutes < 60) return `${minutes}m`
  const hours = Math.floor(minutes / 60)
  const remMin = minutes % 60
  return remMin === 0 ? `${hours}h` : `${hours}h${remMin}m`
}

// Compact "updated X ago" label — covers anything from seconds to
// weeks. Unlike relativeTimeLabel above it doesn't bail on long
// gaps; sessions list commonly shows runs from days/weeks ago.
function elapsedSinceLabel(iso: string, now: number): string {
  const updated = new Date(iso).getTime()
  if (Number.isNaN(updated)) return ''
  const delta = Math.max(0, now - updated)
  const minutes = Math.floor(delta / 60_000)
  if (minutes < 1) return 'just now'
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  if (days < 7) return `${days}d ago`
  const weeks = Math.floor(days / 7)
  return `${weeks}w ago`
}

function isNotFoundError(err: unknown): boolean {
  const msg = friendlyErrorMessage(err)
  return /\b404\b/.test(msg) || /not_found|not found/i.test(msg)
}

function isDaemonUnreachable(err: unknown): boolean {
  const cause = (err as { cause?: { code?: string } }).cause
  if (
    cause?.code === 'ECONNREFUSED' ||
    cause?.code === 'ECONNRESET' ||
    cause?.code === 'ENOTFOUND'
  ) {
    return true
  }
  if (err instanceof TypeError && /fetch failed/i.test(err.message)) {
    return true
  }
  return false
}

const DEFAULT_EVENT_TAIL = 25

export function resolveEventTail(raw: string | undefined): number {
  if (raw === undefined || raw.trim() === '') return DEFAULT_EVENT_TAIL
  const parsed = Number.parseInt(raw, 10)
  if (!Number.isFinite(parsed) || parsed < 1) return DEFAULT_EVENT_TAIL
  return Math.min(parsed, 5000)
}

function snippet(text: string, max = 80): string {
  // Collapse code-fence newlines so a leaked ```a2ui block doesn't
  // bleed across the rest of the events list.
  const flat = text.replace(/\s+/g, ' ').trim()
  return flat.length > max ? `${flat.slice(0, max - 1)}…` : flat
}

function formatSessionEvent(event: SessionEvent): string {
  switch (event.type) {
    case 'user_message':
    case 'assistant_message':
      return snippet(event.content)
    case 'memory_context':
      return snippet(event.items.map((item) => item.citationLabel).join(', '))
    case 'tool_call':
      return event.tool
    case 'approval_request': {
      // Surface status + reason note on the audit trail. Without these
      // an operator browsing `sessions show` after the fact can't tell
      // approved from denied, and can't recall *why* they denied — both
      // critical for "what did I just consent to?" reviews.
      const status = (event as { status?: string }).status
      const note = (event as { note?: string }).note
      const statusLabel = status && status !== 'pending' ? ` [${status}]` : ''
      const noteLabel = note ? ` — ${snippet(note, 100)}` : ''
      return `${event.tool}${statusLabel}${noteLabel}`
    }
    case 'tool_result': {
      const recovery =
        event.recovery === 'journal'
          ? '[saved execution] '
          : event.recovery === 'probe'
            ? '[verified recovery] '
            : ''
      const posture = event.executionPosture
        ? `[${toolExecutionPostureLabel(event.executionPosture)}] `
        : ''
      return `${recovery}${posture}${snippet(event.output)}`
    }
    case 'context_compact':
      return snippet(event.summary)
    case 'run_contract':
      return `${event.contract.acceptanceCriteria.length} ACs — ${snippet(event.contract.summary)}`
    case 'delegation_state':
      return `[${event.claimHealth}] ${snippet(event.detail)}`
    case 'session_start':
      return `${event.metadata.provider}/${event.metadata.model}`
    case 'session_end':
      return `${event.totalTokens.input + event.totalTokens.output} tokens`
    case 'approval_response':
      return event.approved ? `approved by ${event.approvedBy}` : `denied by ${event.approvedBy}`
    case 'node_trace':
      return `${event.node} ${Math.round(event.durationMs / 1000)}s${event.nextEdge ? ` → ${event.nextEdge}` : ''}`
    case 'llm_request': {
      const digest = (event as { requestDigest?: { model?: string; messageCount?: number; systemPromptChars?: number; toolSchemaChars?: number; auxiliary?: boolean } }).requestDigest
      const source = String(event.turnId ?? '').split(':').at(-1) ?? ''
      const size = digest
        ? ` msgs=${digest.messageCount ?? '?'} sys=${Math.round((digest.systemPromptChars ?? 0) / 1000)}k tools=${Math.round((digest.toolSchemaChars ?? 0) / 1000)}k`
        : ''
      return `${digest?.auxiliary ? 'aux ' : ''}${source} iter=${event.iteration}${size}`
    }
    case 'recovery':
      return `${event.kind} → ${event.action}: ${snippet(event.message, 100)}`
    case 'phase_change':
      return event.enteredPhase
        ? `→ ${event.enteredPhase}${event.closedPhase ? ` (closed ${event.closedPhase.phase}: ${event.closedPhase.usage.inputTokens} in / ${event.closedPhase.usage.outputTokens} out)` : ''}`
        : 'phase closed'
    case 'mode_route_decision':
      return `${event.chosen}${event.fallback ? ' (fallback)' : ''}${event.reason ? ` — ${snippet(event.reason, 80)}` : ''}`
    case 'router_decision': {
      const decision = (event as { decision?: { mode?: string; confidence?: string; fallback?: boolean; skipped?: boolean; reason?: string } }).decision
      if (!decision) return ''
      if (decision.skipped) return `${decision.mode ?? '?'} (router skipped — ${snippet(decision.reason ?? '', 60)})`
      return `${decision.mode ?? '?'} ${decision.confidence ?? ''}${decision.fallback ? ' fallback' : ''}${decision.reason ? ` — ${snippet(decision.reason, 60)}` : ''}`
    }
    case 'provider_attempt':
      return `${event.provider}/${event.model} attempt ${event.attempt} ${event.status}${event.errorCode ? ` (${event.errorCode})` : ''}`
    case 'edit_checkpoint_opened':
      return snippet(String((event as { checkpoint?: { label?: string } }).checkpoint?.label ?? ''))
    case 'memory_summary':
      return `${String((event as { semanticMemoriesExtracted?: number }).semanticMemoriesExtracted ?? 0)} memories extracted`
    case 'state_board':
      return 'board snapshot'
    case 'auto_approval':
      // Mirrors the live `[auto-approved]` cli banner so audit
      // browsers see the same scope/pattern as a live operator did —
      // bridges the gap between the live ux signal and post-hoc
      // `sessions show` review.
      return `${event.tool} auto-${event.decision} via ${event.scope} rule '${event.rule.pattern}'`
    default:
      return ''
  }
}

function formatProviderFallbackSummary(
  fallbacks: number,
  attempts: number,
  finalProvider?: string,
  finalModel?: string,
): string {
  const parts = [`${fallbacks} fallback${fallbacks === 1 ? '' : 's'}`]
  if (attempts > 0) {
    parts.push(`${attempts} attempt${attempts === 1 ? '' : 's'}`)
  }
  if (finalProvider && finalModel) {
    parts.push(`final ${finalProvider}/${finalModel}`)
  }
  return parts.join(' · ')
}

function formatEvidenceStatus(status: string): string {
  switch (status) {
    case 'supported':
      return chalk.green('supported')
    case 'failed':
      return chalk.red('failed')
    case 'blocked':
      return chalk.yellow('blocked')
    case 'unverified':
      return chalk.yellow('unverified')
    case 'attention_needed':
      return chalk.yellow('attention needed')
    case 'ready':
      return chalk.green('ready')
    default:
      return status.replace(/_/g, ' ')
  }
}

function formatRunbookStatus(status: DaemonSessionRunbook['status']): string {
  switch (status) {
    case 'ready':
      return chalk.green('ready')
    case 'recoverable':
      return chalk.yellow('recoverable')
    case 'blocked':
      return chalk.red('blocked')
    case 'needs_attention':
      return chalk.yellow('needs attention')
  }
}

function formatRunbookSeverity(severity: string): string {
  switch (severity) {
    case 'critical':
      return chalk.red('critical')
    case 'warning':
      return chalk.yellow('warning')
    default:
      return chalk.gray(severity)
  }
}

function formatRunbookMechanicalValidationStatus(status: string): string {
  switch (status) {
    case 'ready':
      return chalk.yellow('ready')
    case 'already_validated':
      return chalk.green('already validated')
    case 'not_needed':
      return chalk.gray('not needed')
    case 'missing_project_context':
      return chalk.yellow('missing project context')
    case 'unsupported_project':
      return chalk.yellow('unsupported project')
    default:
      return chalk.gray(status)
  }
}

function formatSessionRunbook(runbook: DaemonSessionRunbook): string {
  const lines = [
    `Session runbook: ${runbook.session.id}`,
    `Title: ${runbook.session.title}`,
    `Status: ${formatRunbookStatus(runbook.status)}`,
    `Headline: ${runbook.headline}`,
    `Evidence: ${runbook.summary.validationRuns} validation(s), ${runbook.summary.validationFailures} failure(s), ${runbook.summary.filesChanged} file change(s)`,
  ]

  if (runbook.signals.length > 0) {
    lines.push('\nSignals:')
    for (const signal of runbook.signals.slice(0, 8)) {
      lines.push(`  [${formatRunbookSeverity(signal.severity)}] ${signal.title}`)
      lines.push(chalk.gray(`    ${signal.detail}`))
    }
    if (runbook.signals.length > 8) {
      lines.push(chalk.gray(`  …${runbook.signals.length - 8} more signal(s)`))
    }
  }

  lines.push('\nMechanical validation:')
  lines.push(
    `  ${formatRunbookMechanicalValidationStatus(runbook.mechanicalValidation.status)}` +
      `${runbook.mechanicalValidation.toolchain ? ` · ${runbook.mechanicalValidation.toolchain}` : ''}`,
  )
  lines.push(chalk.gray(`  ${runbook.mechanicalValidation.reason}`))
  for (const command of runbook.mechanicalValidation.commands.slice(0, 6)) {
    lines.push(`  [${command.kind}] ${command.label}`)
    lines.push(`    ${chalk.cyan(command.command)}`)
    lines.push(chalk.gray(`    ${command.reason}`))
  }
  if (runbook.mechanicalValidation.commands.length > 6) {
    lines.push(
      chalk.gray(
        `  …${runbook.mechanicalValidation.commands.length - 6} more validation command(s)`,
      ),
    )
  }

  if (runbook.actions.length > 0) {
    lines.push('\nActions:')
    for (const action of runbook.actions) {
      const review = action.requiresReview ? ' · review required' : ''
      lines.push(`  [${action.priority}${review}] ${action.title}`)
      lines.push(`    ${chalk.cyan(action.command)}`)
      lines.push(chalk.gray(`    ${action.description}`))
    }
  }

  if (runbook.evidence.acceptanceCriteria.length > 0) {
    lines.push('\nAcceptance:')
    for (const criterion of runbook.evidence.acceptanceCriteria.slice(0, 6)) {
      lines.push(`  - [${formatEvidenceStatus(criterion.status)}] ${criterion.text}`)
      if (criterion.reason) {
        lines.push(chalk.gray(`    ${criterion.reason}`))
      }
    }
    if (runbook.evidence.acceptanceCriteria.length > 6) {
      lines.push(
        chalk.gray(`  …${runbook.evidence.acceptanceCriteria.length - 6} more criterion/criteria`),
      )
    }
  }

  lines.push('\nSupport bundle:')
  lines.push(
    `  ${runbook.supportBundle.recommended ? chalk.yellow('recommended') : chalk.gray('optional')} · ${runbook.supportBundle.reason}`,
  )
  lines.push(`  ${chalk.cyan(runbook.supportBundle.command)}`)

  return lines.join('\n')
}

function truncateId(id: string, width: number): string {
  if (id.length <= width) return id.padEnd(width)
  return `${id.slice(0, width - 1)}…`
}

function truncateTitle(title: string | undefined, width: number): string {
  const value = title?.trim() || '(untitled)'
  if (value.length <= width) return value.padEnd(width)
  return `${value.slice(0, width - 1)}…`
}

// Branch/import sessions are tagged on the daemon side. Surface a
// small lineage glyph in cli output so the operator can tell a
// derived run from a fresh one without --json. Matches the web
// sidebar badge vocabulary.
function lineageGlyph(tags: string[] = []): string {
  if (tags.some((tag) => tag.startsWith('branch:'))) return chalk.cyan('⌥')
  if (tags.includes('imported:share')) return chalk.cyan('⇣')
  return ' '
}

const VALID_SESSION_STATUSES = new Set<'active' | 'completed' | 'abandoned'>([
  'active',
  'completed',
  'abandoned',
])

function parseSessionStatus(
  raw: string | undefined,
): 'active' | 'completed' | 'abandoned' | undefined {
  if (!raw) return undefined
  const trimmed = raw.trim().toLowerCase() as 'active' | 'completed' | 'abandoned'
  if (VALID_SESSION_STATUSES.has(trimmed)) return trimmed
  console.error(chalk.red(`Unknown --status: ${raw}. Use active|completed|abandoned (or omit).`))
  process.exit(1)
}

export async function sessionsCommand(options: {
  url?: string
  query?: string
  limit?: string
  status?: string
}) {
  const client = new DaemonClient(options.url)
  const status = parseSessionStatus(options.status)
  // Always request approval counters from the cli — the column is
  // the headline operator signal for "which sessions actually
  // exercised the consent flow vs. ran on auto?". Web/desktop list
  // views still default to metrics=false to keep their list cheap.
  const limitArg = options.limit ? Number.parseInt(options.limit, 10) : 20
  const limit = Number.isFinite(limitArg) && limitArg >= 0 ? limitArg : 20
  const first = await client.sessions(options.query, {
    metrics: true, status, page: 1, perPage: Math.min(limit || 100, 100),
  })
  let page = first
  const items = [...first.items]
  const seen = new Set(items.map((item) => item.id))
  while (page.hasNextPage && (limit === 0 || items.length < limit)) {
    const next = await client.sessions(options.query, {
      metrics: true, status, page: page.page + 1, perPage: page.perPage,
    })
    const fresh = next.items.filter((item) => !seen.has(item.id))
    if (next.page <= page.page || (next.hasNextPage && fresh.length === 0)) {
      throw new Error('Session pagination did not advance; retry the list request.')
    }
    for (const item of fresh) {
      seen.add(item.id)
      items.push(item)
    }
    page = next
  }
  const selected = limit === 0 ? items : items.slice(0, limit)
  const data = {
    ...first,
    items: selected,
    perPage: selected.length || first.perPage,
    hasNextPage: page.hasNextPage || items.length > selected.length,
  }
  output(data, (d) => {
    if (!d.items?.length) {
      // Friendlier empty-list copy under --status — "no sessions"
      // alone leaves operators wondering whether the daemon's empty
      // or the filter excluded them.
      return status ? `No ${status} sessions.` : 'No sessions.'
    }
    const items = limit === 0 ? d.items : d.items.slice(0, limit)
    const filterLabel = status ? `, status=${status}` : ''
    const lines = [`Sessions (${d.totalCount} total${filterLabel}, showing ${items.length}):\n`]
    // Column header — same widths as the data rows below, in chalk.dim
    // so it sits behind the data but still anchors the eye.
    const showCost = items.some((s) => s.totalCost > 0)
    const costHeader = showCost ? `  ${'COST'.padStart(8)}` : ''
    const now = Date.now()
    lines.push(
      chalk.dim(
        `  ${'ID'.padEnd(18)}  ${'TITLE'.padEnd(30)}  ${'STATUS'.padEnd(10)}  ${'MSGS'.padStart(8)}${costHeader}  ${'UPDATED'.padEnd(10)}  PROVIDER/MODEL  (⌥=branch ⇣=import)`,
      ),
    )
    for (const s of items) {
      // Only render the approvals column when at least one signal
      // exists — a quiet session shouldn't drag a "appr: 0/0/0" tag
      // along that adds noise to the row. Auto-only sessions still
      // light up because autoApprovalsApproved bumps the threshold.
      const counters = s.approvalCounters
      const hasApprovalActivity =
        counters && (counters.approvalsRequested > 0 || counters.autoApprovalsApproved > 0)
      const approvalsLabel = hasApprovalActivity
        ? `  appr: ${counters.approvalsRequested}/${counters.approvalsApproved}/${counters.approvalsDenied}` +
          (counters.autoApprovalsApproved > 0 ? ` · auto: ${counters.autoApprovalsApproved}` : '')
        : ''
      // Tint the status column so a 200-row list lets the operator
      // skim active vs. archived without parsing the word every line.
      // Mirrors the web sidebar treatment: green=active, dim=completed,
      // red=abandoned.
      const paddedStatus = s.status.padEnd(10)
      const statusLabel =
        s.status === 'active'
          ? chalk.green(paddedStatus)
          : s.status === 'abandoned'
            ? chalk.red(paddedStatus)
            : chalk.dim(paddedStatus)
      // Render cost only when at least one row in the page actually
      // accrued spend; an all-$0 page would otherwise carry a column
      // of zeros that adds visual noise.
      const costLabel = showCost
        ? '  ' +
          (s.totalCost > 0
            ? (s.totalCost < 0.01
                ? `$${s.totalCost.toFixed(4)}`
                : s.totalCost < 1
                  ? `$${s.totalCost.toFixed(3)}`
                  : `$${s.totalCost.toFixed(2)}`
              ).padStart(8)
            : ''.padStart(8))
        : ''
      const updatedLabel = chalk.dim(elapsedSinceLabel(s.updatedAt, now).padEnd(10))
      const lineage = lineageGlyph(s.tags)
      lines.push(
        `${lineage} ${truncateId(s.id, 18)}  ${truncateTitle(s.title, 30)}  ${statusLabel}  ${String(s.messageCount).padStart(3)} msgs${costLabel}  ${updatedLabel}  ${s.provider}/${s.model}${approvalsLabel}`,
      )
    }
    if (d.totalCount > items.length) {
      lines.push(
        `\n  …${d.totalCount - items.length} more. Use --limit <n> or --limit 0 to show all.`,
      )
    }
    return lines.join('\n')
  })
}

export async function sessionDetailCommand(id: string, options: { url?: string; tail?: string }) {
  const client = new DaemonClient(options.url)
  const tailLimit = resolveEventTail(options.tail)
  let data
  try {
    data = await client.session(id)
  } catch (err) {
    const unreachable = isDaemonUnreachable(err)
    const notFound = !unreachable && isNotFoundError(err)
    const errorPayload = unreachable
      ? { ok: false, id, error: 'daemon-unreachable' }
      : notFound
        ? { ok: false, id, error: 'session-not-found' }
        : { ok: false, id, error: friendlyErrorMessage(err) }
    outputError(errorPayload, () =>
      unreachable
        ? chalk.red('Cannot connect to sepilotd.\nIs the daemon running? Start with: sepilot start')
        : notFound
          ? chalk.red(`Session not found: ${id}`)
          : chalk.red(`Failed to fetch session ${id}: ${friendlyErrorMessage(err)}`),
    )
    process.exit(1)
  }
  output(data, (d) => {
    // After a long-running session, an operator's first question is
    // "what has this cost me so far?" — surface accumulated tokens and
    // cost up front so the headline doesn't drop them under the events
    // log where nobody scrolls. The mock daemon records 22 tokens / turn,
    // so this becomes the most useful number on screen after 20+ rounds.
    const tokens = d.totalTokens
      ? `${(d.totalTokens.input ?? 0).toLocaleString()} in / ${(d.totalTokens.output ?? 0).toLocaleString()} out`
      : null
    const cost = typeof d.totalCost === 'number' ? `$${d.totalCost.toFixed(4)}` : null
    // Same status colour vocabulary as `sessions list` so the
    // operator can land on the same row twice and read the same
    // signal without re-learning it.
    const statusLabel =
      d.status === 'active'
        ? chalk.green(d.status)
        : d.status === 'abandoned'
          ? chalk.red(d.status)
          : chalk.dim(d.status)
    const showNow = Date.now()
    const updatedRelative = elapsedSinceLabel(d.updatedAt, showNow)
    const createdRelative = elapsedSinceLabel(d.createdAt, showNow)
    const branchTag = (d.tags ?? []).find((tag) => tag.startsWith('branch:'))
    const branchSourceId = branchTag ? branchTag.slice('branch:'.length).trim() : null
    const isImportedShare = (d.tags ?? []).includes('imported:share')
    const lines = [
      `Session: ${d.id}`,
      `Title: ${d.title}`,
      `Status: ${statusLabel}`,
      `Provider: ${d.provider}/${d.model}`,
      `Messages: ${d.messageCount}`,
      `Updated: ${chalk.dim(updatedRelative)} (${d.updatedAt})`,
      `Created: ${chalk.dim(createdRelative)} (${d.createdAt})`,
    ]
    if (branchSourceId) {
      lines.push(`Lineage: ${chalk.cyan('⌥ branch')} from ${chalk.dim(branchSourceId)}`)
    } else if (isImportedShare) {
      lines.push(`Lineage: ${chalk.cyan('⇣ imported')} from a public share`)
    }
    if (tokens || cost) {
      const tokenLine = tokens ?? ''
      const costLine = cost ? ` (cost ${cost})` : ''
      lines.push(`Tokens: ${tokenLine}${costLine}`)
    }
    // Approvals at-a-glance: surface traceMetrics counts up front so an
    // operator auditing "what consent decisions did this session
    // accumulate?" doesn't have to re-render with --json | jq. Auto-
    // approval counts are split onto their own sub-line so the
    // explicit-vs-rule-driven distinction is obvious — without it,
    // a 1-prompt session that ran 20 tools via a remembered rule
    // looks identical to a single-tool session, masking the actual
    // automation level.
    const trace = d.traceMetrics
    if (trace) {
      const providerAttemptEvents = trace.providerAttemptEvents ?? 0
      const providerAttemptCount = trace.providerAttemptsStarted ?? 0
      const providerFallbackCount = trace.providerFallbacks ?? 0
      if (providerAttemptEvents > 0 || providerAttemptCount > 0 || providerFallbackCount > 0) {
        lines.push(
          `Provider fallback: ${formatProviderFallbackSummary(
            providerFallbackCount,
            providerAttemptCount,
            trace.finalProvider,
            trace.finalModel,
          )}`,
        )
      }
      const requested = trace.approvalRequests ?? 0
      const approved = trace.approvalApproved ?? 0
      const denied = trace.approvalDenied ?? 0
      const feedback = trace.approvalFeedback ?? 0
      const autoApproved = trace.autoApprovalsApproved ?? 0
      const autoDenied = trace.autoApprovalsDenied ?? 0
      if (requested > 0 || autoApproved > 0 || autoDenied > 0) {
        const parts = [`${requested} requested`, `${approved} approved`, `${denied} denied`]
        if (feedback > 0) parts.push(`${feedback} feedback`)
        lines.push(`\nApprovals: ${parts.join(' · ')}`)
        if (autoApproved > 0 || autoDenied > 0) {
          // The auto: line only fires when something actually
          // short-circuited; keeping it conditional means a session
          // with no remembered rules doesn't get a noisy "auto: 0/0".
          const autoParts = [`${autoApproved} approved`]
          if (autoDenied > 0) autoParts.push(`${autoDenied} denied`)
          lines.push(chalk.gray(`  auto (remembered rule): ${autoParts.join(' · ')}`))
        }
      }
    }
    if (d.runContract) {
      const criteria = d.runContract.acceptanceCriteria ?? []
      lines.push(`\nRun contract: ${formatAcceptanceCriteriaCount(criteria.length)}`)
      lines.push(`  ${d.runContract.summary}`)
      for (const criterion of criteria.slice(0, 5)) {
        lines.push(`  - ${criterion.text}`)
      }
      if (criteria.length > 5) {
        lines.push(chalk.gray(`  …${criteria.length - 5} more`))
      }
    }
    if (d.contractLedger) {
      const ledger = d.contractLedger
      const summary = ledger.summary
      lines.push(
        `\nContract ledger: ${formatEvidenceStatus(ledger.status)} · ` +
          `${summary.confirmedSections}/${summary.sections} confirmed · ` +
          `${summary.safeDefaults} default(s) · ${summary.blockers} blocker(s)`,
      )
      const attentionSections = ledger.sections
        .filter((section) => ['missing', 'weak', 'blocked'].includes(section.status))
        .slice(0, 3)
      for (const section of attentionSections) {
        const marker = section.status === 'blocked' ? '!' : '-'
        lines.push(chalk.yellow(`  ${marker} ${section.name}: ${section.summary}`))
      }
      for (const blocker of ledger.blockers.slice(0, 3)) {
        const color = blocker.severity === 'blocker' ? chalk.red : chalk.yellow
        lines.push(color(`  ! ${blocker.code}: ${blocker.message}`))
      }
    }
    if (d.contextEngine) {
      const context = d.contextEngine
      const sources = context.sources
      lines.push(
        `Context version: ${context.revision} (${context.status}, ${context.eventCount} events)`,
      )
      lines.push(
        chalk.gray(
          `  memory ${sources.memoryContext.items} item(s) · ` +
            `compactions ${sources.compaction.events} · ` +
            `summaries ${sources.memorySummary.events} · ` +
            `decisions ${sources.workingMemory.decisions}`,
        ),
      )
    }
    const evidenceSurface = buildSessionEvidenceSurfaceModel({
      evidenceManifest: d.evidenceManifest,
      evaluationGate: d.evaluationGate,
    })
    if (d.evidenceManifest) {
      const evidence = d.evidenceManifest
      const summary = evidence.summary
      lines.push(
        `\nEvidence manifest: ${formatEvidenceStatus(evidence.status)} · ${summary.artifacts} artifact(s) · ${summary.validationRuns} validation(s) · ${summary.filesChanged} file change(s)`,
      )
      if (summary.acceptanceCriteria > 0) {
        const criteria = evidence.acceptanceCriteria.slice(0, 5)
        for (const criterion of criteria) {
          lines.push(`  - [${formatEvidenceStatus(criterion.status)}] ${criterion.text}`)
        }
        if (evidence.acceptanceCriteria.length > criteria.length) {
          lines.push(chalk.gray(`  …${evidence.acceptanceCriteria.length - criteria.length} more`))
        }
      }
      for (const risk of evidence.risks.filter((item) => item.severity === 'warning').slice(0, 3)) {
        lines.push(chalk.yellow(`  ! ${risk.message}`))
      }
      if ((evidenceSurface?.readbacks.length ?? 0) > 0) {
        const visibleReadbacks = evidenceSurface!.readbacks.slice(-5)
        for (const readback of visibleReadbacks) {
          const detail = [
            readback.path ?? readback.label,
            readback.tool,
            readback.hash,
          ].filter(Boolean).join(' · ')
          lines.push(`  - [read-back ${formatEvidenceStatus(readback.status)}] ${detail}`)
        }
        if (evidenceSurface!.readbacks.length > visibleReadbacks.length) {
          lines.push(
            chalk.gray(
              `  …${evidenceSurface!.readbacks.length - visibleReadbacks.length} earlier read-back(s)`,
            ),
          )
        }
      }
    }
    if (d.evaluationGate) {
      const gate = d.evaluationGate
      const verdict = evidenceSurface?.approved === true
        ? chalk.green('approved')
        : evidenceSurface?.approved === false
          ? chalk.yellow('not approved')
          : evidenceSurface?.state === 'attention_needed'
            ? chalk.yellow('attention needed')
            : chalk.cyan('in progress')
      lines.push(`\nEvaluation gate: ${formatEvidenceStatus(gate.status)} · ${verdict}`)
      if (evidenceSurface?.verdictReason) {
        lines.push(chalk.gray(`  ${evidenceSurface.verdictReason}`))
      }
      lines.push(
        `  mechanical: ${formatEvidenceStatus(gate.stages.mechanical.status)} · ${gate.stages.mechanical.summary}`,
      )
      lines.push(
        `  semantic: ${formatEvidenceStatus(gate.stages.semantic.status)} · ${gate.stages.semantic.summary}`,
      )
      lines.push(
        `  consensus: ${formatEvidenceStatus(gate.stages.consensus.status)} · ${gate.stages.consensus.summary}`,
      )
      lines.push(
        `  artifact bundle: ${formatEvidenceStatus(gate.artifactBundle.status)} · ` +
          `${gate.artifactBundle.summary.hashedFiles}/${gate.artifactBundle.summary.files} hashed · ` +
          `${gate.artifactBundle.summary.skippedFiles} skipped`,
      )
      if (gate.acceptanceVerification) {
        const verification = gate.acceptanceVerification
        lines.push(
          `  acceptance verifier: ${formatEvidenceStatus(verification.status)} · ` +
            `${verification.summary.verifiedAssertions}/${verification.summary.assertions} assertion(s) verified · ` +
            `${verification.summary.failedAssertions} failed · ${verification.summary.unverifiedAssertions} unverified`,
        )
        for (const report of verification.reports
          .filter((item) => item.status === 'failed' || item.status === 'unverified')
          .slice(0, 3)) {
          lines.push(
            chalk.yellow(
              `    ! [${formatEvidenceStatus(report.status)}] ${report.acceptanceCriterionText}`,
            ),
          )
        }
      }
      if (gate.consensusTriggers.primaryTrigger) {
        lines.push(
          `  consensus trigger: ${gate.consensusTriggers.primaryTrigger.code} · ${gate.consensusTriggers.primaryTrigger.message}`,
        )
      }
      for (const risk of gate.risks.filter((item) => item.severity === 'warning').slice(0, 3)) {
        lines.push(chalk.yellow(`  ! ${risk.message}`))
      }
    }
    // Working memory: condensed view of what the session has actually
    // *decided* and what's still open. Lifts keyDecisions (incl.
    // auto_approval entries from the durable journal) out of --json so
    // an operator scanning `sessions show` after the fact sees a
    // human-readable answer to "what did this run end up agreeing to?"
    // — without it, the only audit path was to scroll the events tail
    // and infer from approval_request/auto_approval rows directly.
    const wm = d.workingMemory
    if (wm) {
      const wmLines: string[] = []
      if (wm.activeTodo) {
        wmLines.push(`  active todo: ${wm.activeTodo}`)
      }
      const recentDecisions = (wm.keyDecisions ?? []).slice(-5)
      if (recentDecisions.length > 0) {
        wmLines.push(chalk.gray(`  decisions (last ${recentDecisions.length}):`))
        const now = Date.now()
        for (const decision of recentDecisions) {
          const age = decision.timestamp
            ? relativeTimeLabel(now - new Date(decision.timestamp).getTime())
            : null
          // relativeTimeLabel returns 'expired' for past timestamps;
          // re-frame as "<X> ago" so an audit reader doesn't see
          // "expired" applied to a recorded decision.
          const ageLabel = age && age !== 'expired' ? ` (${age} ago)` : ''
          wmLines.push(`    · ${decision.summary}${chalk.gray(ageLabel)}`)
        }
      }
      const openQuestions = wm.openQuestions ?? []
      if (openQuestions.length > 0) {
        wmLines.push(chalk.gray(`  open questions (${openQuestions.length}):`))
        for (const question of openQuestions.slice(0, 3)) {
          wmLines.push(`    ? ${question}`)
        }
        if (openQuestions.length > 3) {
          wmLines.push(chalk.gray(`    …${openQuestions.length - 3} more`))
        }
      }
      if (wmLines.length > 0) {
        lines.push('\nWorking memory:')
        lines.push(...wmLines)
      }
    }
    if (d.pendingApprovals?.length) {
      const now = Date.now()
      // Headline count first so an operator scrolling past `sepilot
      // session show <id>` notices the run is paused on a decision; the
      // per-row detail (args preview, age, expiry, decide hint) follows
      // underneath, mirroring what the chat shell's `/approvals` command
      // shows so an operator switching surfaces sees the same content.
      lines.push(
        `\nPending approvals: ${chalk.yellow(String(d.pendingApprovals.length))} ` +
          `(use \`/approve <id>\` or \`/deny <id>\` from the chat shell)`,
      )
      for (const approval of d.pendingApprovals) {
        const askedAgo = approval.requestedAt
          ? relativeTimeLabel(now - new Date(approval.requestedAt).getTime())
          : null
        // `relativeTimeLabel(<=0)` returns 'expired'; combining that
        // with ' until timeout' produced the awkward 'expired until
        // timeout'. Render the past-the-deadline case as a bare
        // 'expired' label so an operator can scan for it visually.
        const expiresIn = approval.expiresAt
          ? relativeTimeLabel(new Date(approval.expiresAt).getTime() - now)
          : null
        const expiryLabel = !expiresIn
          ? null
          : expiresIn === 'expired'
            ? 'expired'
            : `${expiresIn} until timeout`
        const stateTags = `${approval.state}${approval.resumeAvailable ? ', resumable' : ''}`
        const preview = formatToolCall({ name: approval.tool, arguments: approval.input }, 200)
        lines.push('')
        lines.push(`  [${stateTags}] ${preview}`)
        const meta = [approval.requestId, askedAgo ? `asked ${askedAgo} ago` : null, expiryLabel]
          .filter(Boolean)
          .join(' · ')
        if (meta) {
          lines.push(`    ${chalk.gray(meta)}`)
        }
      }
    }
    if (d.events) {
      // Show only the tail so a 100-turn session doesn't flood the
      // terminal, but tell the operator how many earlier events are
      // hidden — silent truncation made `sessions show` look like the
      // full log when it wasn't, which is the regression this branch
      // fixes. `--json` carries the full events array for tooling.
      lines.push(`\nEvents (${d.events.length}):`)
      if (d.events.length > tailLimit) {
        lines.push(
          chalk.gray(
            `  …${d.events.length - tailLimit} earlier events hidden. Use \`--tail <n>\` or \`--json\` for the full log.`,
          ),
        )
      }
      for (const e of d.events.slice(-tailLimit)) {
        lines.push(`  [${e.type}] ${formatSessionEvent(e)}`)
      }
    }
    return lines.join('\n')
  })
}

export async function sessionRunbookCommand(id: string, options: { url?: string }) {
  const client = new DaemonClient(options.url)
  try {
    const data = await client.sessionRunbook(id)
    output(data, formatSessionRunbook)
  } catch (err) {
    const unreachable = isDaemonUnreachable(err)
    const notFound = !unreachable && isNotFoundError(err)
    const errorPayload = unreachable
      ? { ok: false, id, error: 'daemon-unreachable' }
      : notFound
        ? { ok: false, id, error: 'session-not-found' }
        : { ok: false, id, error: friendlyErrorMessage(err) }
    outputError(errorPayload, () =>
      unreachable
        ? chalk.red('Cannot connect to sepilotd.\nIs the daemon running? Start with: sepilot start')
        : notFound
          ? chalk.red(`Session not found: ${id}`)
          : chalk.red(`Failed to build runbook for session ${id}: ${friendlyErrorMessage(err)}`),
    )
    process.exit(1)
  }
}

export async function sessionResumeCommand(
  id: string,
  options: { url?: string; force?: boolean },
) {
  const client = new DaemonClient(options.url)
  const printer = createInteractiveCliChatStreamPrinter({
    questionHintMode: 'cli',
    showArtifacts: true,
    showDiagnostics: false,
  })
  const streamIdleMs = resolveCliStreamIdleMs()
  const streamConnectMs = resolveCliStreamConnectMs()
  const aborter = new AbortController()
  let lastTick = Date.now()
  const watchdog = setInterval(
    () => {
      if (Date.now() - lastTick > streamIdleMs) {
        aborter.abort(new Error('stream-idle-timeout'))
      }
    },
    Math.min(5000, streamIdleMs / 4),
  )

  const tick = (event: unknown) => {
    if (isSubstantiveAskStreamEvent(event)) lastTick = Date.now()
    printer.handleEvent(event as Parameters<typeof printer.handleEvent>[0])
  }

  try {
    const response = await openChatStreamWithConnectTimeout(
      client.resumeSessionStream(id, { force: options.force }, { signal: aborter.signal }),
      {
        timeoutMs: streamConnectMs,
        abort: (error) => aborter.abort(error),
      },
    )
    if (!response.ok || !response.body) {
      throw new Error(await formatChatStreamFailure(response))
    }
    await forwardDaemonStreamWithResumeRecovery(response, {
      aborter,
      getSessionId: () => id,
      onEvent: tick,
      openResumeStream: (sessionId) => client.resumeSessionStream(
        sessionId,
        { force: options.force },
        { signal: aborter.signal },
      ),
      isTerminalEvent: isTerminalCliDaemonChatEvent,
      streamIdleMs,
    })
    if (printer.hadError()) process.exit(1)
  } catch (err) {
    if (printStreamError(err, { sessionId: id })) {
      process.exit(1)
    }
    if (printApiError(err, { hint: `Inspect with: sepilot sessions show ${id}` })) {
      process.exit(1)
    }
    console.error(chalk.red(`Failed to resume session ${id}: ${friendlyErrorMessage(err)}`))
    process.exit(1)
  } finally {
    clearInterval(watchdog)
  }
}

export async function sessionDeleteCommand(id: string, options: { url?: string }) {
  const client = new DaemonClient(options.url)
  try {
    await client.deleteSession(id)
  } catch (err) {
    const unreachable = isDaemonUnreachable(err)
    const notFound = !unreachable && isNotFoundError(err)
    const errorPayload = unreachable
      ? { ok: false, id, error: 'daemon-unreachable' }
      : notFound
        ? { ok: false, id, error: 'session-not-found' }
        : { ok: false, id, error: friendlyErrorMessage(err) }
    outputError(errorPayload, () =>
      unreachable
        ? chalk.red('Cannot connect to sepilotd.\nIs the daemon running? Start with: sepilot start')
        : notFound
          ? chalk.red(`Session not found: ${id}`)
          : chalk.red(`Failed to delete session ${id}: ${friendlyErrorMessage(err)}`),
    )
    process.exit(1)
  }
  output({ ok: true, id, deleted: true }, () => `Session deleted: ${chalk.bold(id)}`)
}

export async function sessionRenameCommand(id: string, title: string, options: { url?: string }) {
  const trimmed = (title ?? '').trim()
  if (!trimmed) {
    outputError({ ok: false, id, error: 'invalid-title' }, () =>
      chalk.red('A non-empty title is required.'),
    )
    process.exit(1)
  }
  const client = new DaemonClient(options.url)
  try {
    const updated = await client.updateSession(id, { title: trimmed })
    output(
      { ok: true, id, title: updated.title },
      () => `Session renamed: ${chalk.bold(id)} → ${chalk.cyan(updated.title)}`,
    )
  } catch (err) {
    const unreachable = isDaemonUnreachable(err)
    const notFound = !unreachable && isNotFoundError(err)
    const errorPayload = unreachable
      ? { ok: false, id, error: 'daemon-unreachable' }
      : notFound
        ? { ok: false, id, error: 'session-not-found' }
        : { ok: false, id, error: friendlyErrorMessage(err) }
    outputError(errorPayload, () =>
      unreachable
        ? chalk.red('Cannot connect to sepilotd.\nIs the daemon running? Start with: sepilot start')
        : notFound
          ? chalk.red(`Session not found: ${id}`)
          : chalk.red(`Failed to rename session ${id}: ${friendlyErrorMessage(err)}`),
    )
    process.exit(1)
  }
}

const VALID_STATUS_VALUES = ['active', 'completed', 'abandoned'] as const
type SessionStatusValue = (typeof VALID_STATUS_VALUES)[number]

function isStatusValue(value: string): value is SessionStatusValue {
  return (VALID_STATUS_VALUES as readonly string[]).includes(value)
}

export async function sessionStatusCommand(id: string, status: string, options: { url?: string }) {
  const normalized = (status ?? '').trim().toLowerCase()
  if (!isStatusValue(normalized)) {
    outputError({ ok: false, id, error: 'invalid-status' }, () =>
      chalk.red(`Status must be one of: ${VALID_STATUS_VALUES.join(', ')}`),
    )
    process.exit(1)
  }
  const client = new DaemonClient(options.url)
  try {
    const updated = await client.updateSession(id, { status: normalized })
    output(
      { ok: true, id, status: updated.status },
      () => `Session status: ${chalk.bold(id)} → ${chalk.cyan(updated.status)}`,
    )
  } catch (err) {
    const unreachable = isDaemonUnreachable(err)
    const notFound = !unreachable && isNotFoundError(err)
    const errorPayload = unreachable
      ? { ok: false, id, error: 'daemon-unreachable' }
      : notFound
        ? { ok: false, id, error: 'session-not-found' }
        : { ok: false, id, error: friendlyErrorMessage(err) }
    outputError(errorPayload, () =>
      unreachable
        ? chalk.red('Cannot connect to sepilotd.\nIs the daemon running? Start with: sepilot start')
        : notFound
          ? chalk.red(`Session not found: ${id}`)
          : chalk.red(`Failed to update session ${id}: ${friendlyErrorMessage(err)}`),
    )
    process.exit(1)
  }
}

export async function sessionBranchCommand(
  id: string,
  options: { url?: string; fromEventIndex?: string },
) {
  const client = new DaemonClient(options.url)
  const rawFromEventIndex = options.fromEventIndex?.trim()
  const parsedFromEventIndex = rawFromEventIndex ? Number.parseInt(rawFromEventIndex, 10) : null

  if (
    rawFromEventIndex &&
    (parsedFromEventIndex === null ||
      !Number.isInteger(parsedFromEventIndex) ||
      parsedFromEventIndex < 0)
  ) {
    console.error('`--from-event-index` must be a non-negative integer.')
    process.exit(1)
  }

  const fromEventIndex = parsedFromEventIndex ?? undefined

  const data = await client.branchSession(
    id,
    fromEventIndex === undefined ? undefined : { fromEventIndex },
  )

  output(data, (d) =>
    [
      `Branched session ${d.sourceId} -> ${d.branchId}`,
      `Copied events: ${d.copiedEvents}`,
      fromEventIndex === undefined
        ? 'Fork point: session tail'
        : `Fork point: event ${fromEventIndex}`,
    ].join('\n'),
  )
}

export async function sessionExportCommand(
  id: string,
  options: { url?: string; format?: string; output?: string; sanitize?: boolean },
) {
  const client = new DaemonClient(options.url)
  const format = options.format ?? 'markdown'
  const qs = new URLSearchParams({ format })
  if (options.sanitize) qs.set('sanitize', '1')
  const res = await client.fetch(
    `/api/v1/sessions/${encodeURIComponent(id)}/export?${qs.toString()}`,
  )
  if (!res.ok) {
    if (res.status === 404) {
      console.error(chalk.red(`Session not found: ${id}`))
    } else {
      console.error(chalk.red(`Failed to export session ${id}: HTTP ${res.status}`))
    }
    process.exit(1)
  }

  let content: string
  if (format === 'json') {
    const raw = (await res.json()) as unknown
    // Unwrap the standard daemon envelope { data: ... } so the exported
    // JSON file is the session payload itself, not an envelope wrapper.
    const payload =
      raw && typeof raw === 'object' && 'data' in (raw as Record<string, unknown>)
        ? (raw as { data: unknown }).data
        : raw
    content = JSON.stringify(payload, null, 2)
  } else {
    content = await res.text()
  }

  if (options.output) {
    const { writeFile } = await import('node:fs/promises')
    try {
      await writeFile(options.output, content, 'utf-8')
    } catch (err) {
      const code = (err as { code?: string }).code
      if (code === 'EISDIR') {
        console.error(chalk.red(`--output path is a directory: ${options.output}`))
      } else if (code === 'EACCES') {
        console.error(chalk.red(`--output not writable: ${options.output} (permission denied)`))
      } else if (code === 'ENOENT') {
        console.error(chalk.red(`--output parent directory missing: ${options.output}`))
      } else {
        console.error(chalk.red(`Failed to write ${options.output}: ${friendlyErrorMessage(err)}`))
      }
      process.exit(1)
    }
    if (options.output === '/dev/stdout' || options.output === '/dev/fd/1') {
      // Keep redirected JSON/Markdown parseable: a success footer on stdout
      // becomes an invalid trailing token for consumers such as jq.
      console.error(chalk.green('Output written to stdout'))
    } else {
      console.log(chalk.green(`Output written to ${options.output}`))
    }
  } else {
    writeOutputText(content)
  }
}

export async function sessionCompactCommand(id: string, options: { url?: string }) {
  const client = new DaemonClient(options.url)
  const data = await client.compactSession(id)
  output(data, (d) => {
    const counts = [
      d.removedMessageCount ? `compacted ${d.removedMessageCount} earlier messages` : null,
      d.preservedMessageCount ? `preserved ${d.preservedMessageCount} recent messages` : null,
    ]
      .filter(Boolean)
      .join(', ')
    return [
      `Compacted: ${d.originalTokens} -> ${d.compactedTokens} tokens (saved ${d.savedTokens})`,
      counts || null,
    ]
      .filter(Boolean)
      .join('\n')
  })
}

export const __testables = {
  elapsedSinceLabel,
  formatSessionEvent,
  formatProviderFallbackSummary,
  formatEvidenceStatus,
  formatRunbookMechanicalValidationStatus,
  formatRunbookSeverity,
  formatRunbookStatus,
  isDaemonUnreachable,
  isNotFoundError,
  lineageGlyph,
  relativeTimeLabel,
  formatSessionRunbook,
  snippet,
  truncateId,
  truncateTitle,
}
