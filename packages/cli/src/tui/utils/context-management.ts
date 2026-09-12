import type {
  DaemonResumableRun,
  DaemonSessionDetail,
  DaemonSessionMeta,
  SessionEvent,
} from '@sepilotd/api-client'
import {
  resumableRunActionLabel,
  resumableRunCopy,
} from '@sepilotd/api-client'
import {
  describeRewindTurns,
  listRewindTargets,
  type RewindTarget,
} from './session-history.js'

const DOT = '·'
const BAR_WIDTH = 12

type ContextManagementFocus = 'map' | 'resume' | 'rewind'

interface ContextUsageSnapshot {
  input: number
  output: number
  cost: number
}

export interface BuildContextManagementSummaryOptions {
  session: DaemonSessionDetail
  usage: ContextUsageSnapshot
  /** Input tokens in the latest provider request, or the daemon's preflight estimate. */
  contextInputTokens?: number | null
  /** Whether `contextInputTokens` is the daemon's preflight estimate. */
  contextEstimated?: boolean
  contextWindow?: number | null
  projectName?: string | null
  mode: string
  autonomy: string
  thinkingLevel: string
  maxTokens: number | null
  artifactCount: number
  pendingApprovalTool?: string | null
  focus?: ContextManagementFocus
}

interface SessionEventCounts {
  userTurns: number
  assistantMessages: number
  toolCalls: number
  approvals: number
  compactions: number
}

function shortId(id: string | null | undefined): string {
  return id ? id.slice(0, 8) : 'none'
}

function cleanTitle(value: string | null | undefined): string {
  const title = value?.trim()
  return title || '(untitled)'
}

function formatNumber(value: number): string {
  return Number.isFinite(value) ? Math.max(0, value).toLocaleString() : '0'
}

function formatCost(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return '$0'
  if (value < 0.01) return `$${value.toFixed(4)}`
  if (value < 1) return `$${value.toFixed(3)}`
  return `$${value.toFixed(2)}`
}

function formatRelativeTime(value: string | null | undefined): string {
  if (!value) return 'unknown'
  const timestamp = Date.parse(value)
  if (!Number.isFinite(timestamp)) return value

  const diffMinutes = Math.max(0, Math.floor((Date.now() - timestamp) / 60_000))
  if (diffMinutes < 1) return 'just now'
  if (diffMinutes < 60) return `${diffMinutes}m ago`
  const diffHours = Math.floor(diffMinutes / 60)
  if (diffHours < 24) return `${diffHours}h ago`
  const diffDays = Math.floor(diffHours / 24)
  if (diffDays < 30) return `${diffDays}d ago`
  return new Date(timestamp).toISOString().slice(0, 10)
}

function renderUsageBar(percent: number): string {
  const normalized = Math.max(0, Math.min(100, percent))
  const filled = Math.round((normalized / 100) * BAR_WIDTH)
  return `${'█'.repeat(filled)}${'░'.repeat(BAR_WIDTH - filled)}`
}

function contextPercent(
  contextTokens: number | null,
  contextWindow?: number | null,
): number | null {
  return contextTokens != null && contextWindow && contextWindow > 0
    ? Math.min(999, Math.round((contextTokens / contextWindow) * 100))
    : null
}

function formatContextPressure(
  usage: ContextUsageSnapshot,
  contextWindow: number | null | undefined,
  contextTokens: number | null,
  estimated: boolean,
): string {
  const percent = contextPercent(contextTokens, contextWindow)
  if (contextTokens == null) {
    return `unknown ctx ${DOT} billed ${formatNumber(usage.input)} in / ${formatNumber(usage.output)} out${contextWindow && contextWindow > 0 ? ` ${DOT} ${formatNumber(contextWindow)} window` : ` ${DOT} window unknown`}`
  }
  const tokenLabel = `${estimated ? '~' : ''}${formatNumber(contextTokens)}`
  if (percent == null || !contextWindow) {
    return `${tokenLabel} ctx ${DOT} billed ${formatNumber(usage.input)} in / ${formatNumber(usage.output)} out ${DOT} window unknown`
  }

  return `${tokenLabel} / ${formatNumber(contextWindow)} ctx ${DOT} ${renderUsageBar(percent)} ${percent}% ${DOT} billed ${formatNumber(usage.input)} in / ${formatNumber(usage.output)} out`
}

function countSessionEvents(events: SessionEvent[]): SessionEventCounts {
  return events.reduce<SessionEventCounts>(
    (counts, event) => {
      switch (event.type) {
        case 'user_message':
          counts.userTurns += 1
          break
        case 'assistant_message':
          counts.assistantMessages += 1
          break
        case 'tool_call':
          counts.toolCalls += 1
          break
        case 'approval_request':
          counts.approvals += 1
          break
        case 'context_compact':
          counts.compactions += 1
          break
      }
      return counts
    },
    {
      userTurns: 0,
      assistantMessages: 0,
      toolCalls: 0,
      approvals: 0,
      compactions: 0,
    },
  )
}

function formatResumableRunTarget(run: DaemonResumableRun): string {
  const currentTools = run.currentTools ?? (run.currentTool ? [run.currentTool] : [])
  const currentToolCount =
    run.currentToolCount
    ?? (currentTools.length > 0 ? currentTools.length : (run.currentTool ? 1 : 0))

  if (currentToolCount <= 1) {
    return run.currentTool ?? currentTools[0] ?? 'checkpoint'
  }

  const summarizedTools = currentTools.slice(0, 3).join(', ')
  return summarizedTools
    ? `${currentToolCount} tool calls (${summarizedTools}${currentTools.length > 3 ? ', ...' : ''})`
    : `${currentToolCount} tool calls`
}

function formatResumeStatus(run?: DaemonResumableRun): string {
  if (!run) {
    return `no interrupted checkpoint ${DOT} /resume opens recent history when no session is active`
  }

  const safety = run.forceRequired ? 'force required' : 'ready'
  return `${resumableRunActionLabel(run)} ${DOT} ${run.mode}/${run.stage} ${DOT} ${formatResumableRunTarget(run)} ${DOT} ${safety}`
}

function formatRewindTargetInline(target: RewindTarget): string {
  return `/rewind ${target.turns} before "${target.userPreview}"`
}

function formatRewindTargetDetail(target: RewindTarget): string {
  return `  /rewind ${target.turns}  keeps ${formatNumber(target.copiedEvents)} events, drops ${formatNumber(target.omittedEvents)}  before "${target.userPreview}"`
}

function buildRewindLines(
  session: DaemonSessionDetail,
  focus: ContextManagementFocus,
): string[] {
  const targets = listRewindTargets(session.events, focus === 'rewind' ? 5 : 3)
  if (targets.length === 0) {
    return ['Rewind   no user turn boundary yet']
  }

  if (focus === 'rewind') {
    return [
      'Rewind targets',
      ...targets.map(formatRewindTargetDetail),
    ]
  }

  return [
    `Rewind   ${targets.slice(0, 2).map(formatRewindTargetInline).join(`  ${DOT}  `)}`,
  ]
}

export function buildContextManagementSummary({
  session,
  usage,
  contextInputTokens,
  contextEstimated = false,
  contextWindow,
  projectName,
  mode,
  autonomy,
  thinkingLevel,
  maxTokens,
  artifactCount,
  pendingApprovalTool,
  focus = 'map',
}: BuildContextManagementSummaryOptions): string {
  const counts = countSessionEvents(session.events)
  const contextTokens = contextInputTokens != null && contextInputTokens >= 0
    ? contextInputTokens
    : null
  const percent = contextPercent(contextTokens, contextWindow)
  const title = focus === 'resume'
    ? 'Resume Readiness'
    : focus === 'rewind'
      ? 'Rewind Map'
      : 'Context Map'
  const nextResumeCommand = session.resumableRun?.forceRequired
    ? '/resume --force'
    : '/resume'
  const compactHint = percent != null && percent >= 80
    ? '/compact now'
    : '/compact when context gets heavy'
  const resumeNote = focus === 'resume' && session.resumableRun
    ? `Note     ${resumableRunCopy(session.resumableRun)}`
    : null

  return [
    title,
    `Session  ${shortId(session.id)} ${DOT} ${cleanTitle(session.title)} ${DOT} updated ${formatRelativeTime(session.updatedAt)}`,
    `Stack    ${session.provider}/${session.model} ${DOT} mode ${mode} ${DOT} ${autonomy} ${DOT} thinking ${thinkingLevel}`,
    `Project  ${projectName ?? 'none'} ${DOT} max output ${maxTokens === null ? 'provider default' : formatNumber(maxTokens)} ${DOT} cost ${formatCost(usage.cost)}`,
    `Tokens   ${formatContextPressure(usage, contextWindow, contextTokens, contextEstimated)}`,
    `History  ${counts.userTurns} turns ${DOT} ${counts.assistantMessages} replies ${DOT} ${counts.toolCalls} tools ${DOT} ${counts.compactions} compactions ${DOT} ${artifactCount} artifacts`,
    `Resume   ${formatResumeStatus(session.resumableRun)}`,
    resumeNote,
    pendingApprovalTool
      ? `Approval pending  ${pendingApprovalTool} ${DOT} resolve it before switching, rewinding, or resuming`
      : null,
    ...buildRewindLines(session, focus),
    `Next     ${nextResumeCommand} ${DOT} /rewind 1 ${DOT} ${compactHint} ${DOT} /session`,
  ].filter((line): line is string => Boolean(line)).join('\n')
}

export function buildSessionLoadedSummary(session: DaemonSessionDetail): string {
  return [
    'Session Loaded',
    `Session  ${shortId(session.id)} ${DOT} ${cleanTitle(session.title)}`,
    `Stack    ${session.provider}/${session.model} ${DOT} ${session.status} ${DOT} ${session.messageCount} messages ${DOT} updated ${formatRelativeTime(session.updatedAt)}`,
    'Next     type to continue, /context for the map, /rewind 1 to fork safely',
  ].join('\n')
}

export function buildRecentResumeSummary(
  session: DaemonSessionMeta,
  totalSessions: number,
): string {
  return [
    'Resume History',
    `Opening  ${shortId(session.id)} ${DOT} ${cleanTitle(session.title)}`,
    `Stack    ${session.provider}/${session.model} ${DOT} ${session.status} ${DOT} ${session.messageCount} messages ${DOT} updated ${formatRelativeTime(session.updatedAt)}`,
    `Scope    picked newest of ${formatNumber(totalSessions)} session${totalSessions === 1 ? '' : 's'}`,
  ].join('\n')
}

export function buildResumePreflightSummary(
  session: DaemonSessionDetail,
  force: boolean,
): string {
  const run = session.resumableRun
  return [
    'Resume Checkpoint',
    `Session  ${shortId(session.id)} ${DOT} ${cleanTitle(session.title)}`,
    `Action   ${run ? resumableRunActionLabel(run) : 'Resume Run'}${force ? ' (forced)' : ''}`,
    run
      ? `Safety   ${run.mode}/${run.stage} ${DOT} ${formatResumableRunTarget(run)} ${DOT} checkpoint ${formatRelativeTime(run.checkpointedAt)}`
      : `Safety   no saved checkpoint was reported; daemon will attempt the latest recoverable state`,
    `Note     ${resumableRunCopy(run)}`,
    run?.forceRequired && !force
      ? 'Hint     replay confirmation may be required; use /resume --force if you intentionally want to replay'
      : null,
  ].filter((line): line is string => Boolean(line)).join('\n')
}

export function buildRewindPlanSummary(
  session: DaemonSessionDetail,
  target: RewindTarget,
): string {
  return [
    'Rewind Branch',
    `Source   ${shortId(session.id)} ${DOT} ${cleanTitle(session.title)}`,
    `Fork     before "${target.userPreview}"`,
    `History  keeps ${formatNumber(target.copiedEvents)} events ${DOT} drops ${formatNumber(target.omittedEvents)} events ${DOT} source stays intact`,
  ].join('\n')
}

export function buildRewindSuccessSummary(
  sourceSessionId: string,
  branchSessionId: string,
  turns: number,
  copiedEvents: number,
  target: RewindTarget,
): string {
  return [
    'Rewind Complete',
    `Branch   ${shortId(sourceSessionId)} -> ${shortId(branchSessionId)}`,
    `Fork     ${describeRewindTurns(turns)} back ${DOT} kept ${formatNumber(copiedEvents)} events ${DOT} before "${target.userPreview}"`,
    'Next     replace the forked turn with a new instruction, or /session branch again from here',
  ].join('\n')
}

export function buildBranchSuccessSummary(
  sourceSessionId: string,
  branchSessionId: string,
  copiedEvents: number,
): string {
  return [
    'Branch Created',
    `Branch   ${shortId(sourceSessionId)} -> ${shortId(branchSessionId)}`,
    `History  copied ${formatNumber(copiedEvents)} events at the tail`,
    'Next     continue in the branch; the source session is unchanged',
  ].join('\n')
}
