import chalk from 'chalk'
import {
  isStaleRememberedDecision,
  type DaemonRememberedApproval,
  type DaemonObservabilityEvent,
  type DaemonObservabilityPrivacySettings,
  type DaemonObservabilitySnapshot,
  type DaemonSessionMeta,
} from '@sepilotd/api-client'
import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'
import { friendlyErrorMessage, printApiError } from '../utils/error-message.js'

export interface StatsOptions {
  url?: string
}

export interface StatsSnapshot {
  sessions: {
    total: number
    active: number
    completed: number
    abandoned: number
    pendingApprovals: number
  }
  approvals: {
    requested: number
    approved: number
    denied: number
    autoApproved: number
  }
  decisions: {
    total: number
    bySession: number
    byAlways: number
    stale: number
    /**
     * Top-N rules sorted by hitCount desc — operator's "where is the
     * automation budget actually going?" answer. Rules with zero
     * hits are excluded so the slot doesn't surface a wall of "0"s
     * on a daemon with mostly-unused fixtures.
     */
    topHits: Array<{
      tool: string
      pattern: string
      scope: string
      hitCount: number
    }>
  }
  observability?: DaemonObservabilitySnapshot | null
  privacy?: DaemonObservabilityPrivacySettings | null
  recentEvents?: DaemonObservabilityEvent[]
  crashEvents?: DaemonObservabilityEvent[]
}

export function buildStats(
  sessions: { items?: DaemonSessionMeta[] } | undefined,
  decisions: DaemonRememberedApproval[],
): StatsSnapshot {
  const items = sessions?.items ?? []
  const sessionCounters = items.reduce(
    (acc, s) => {
      acc.total += 1
      if (s.status === 'active') acc.active += 1
      else if (s.status === 'completed') acc.completed += 1
      else if (s.status === 'abandoned') acc.abandoned += 1
      const c = s.approvalCounters
      if (c) {
        acc.requested += c.approvalsRequested ?? 0
        acc.approved += c.approvalsApproved ?? 0
        acc.denied += c.approvalsDenied ?? 0
        acc.autoApproved += c.autoApprovalsApproved ?? 0
      }
      return acc
    },
    {
      total: 0,
      active: 0,
      completed: 0,
      abandoned: 0,
      requested: 0,
      approved: 0,
      denied: 0,
      autoApproved: 0,
    },
  )

  const now = Date.now()
  const decisionCounters = decisions.reduce(
    (acc, d) => {
      acc.total += 1
      if (d.scope === 'session') acc.bySession += 1
      else if (d.scope === 'always') acc.byAlways += 1
      if (isStaleRememberedDecision(d, now)) acc.stale += 1
      return acc
    },
    { total: 0, bySession: 0, byAlways: 0, stale: 0 },
  )
  const topHits = [...decisions]
    .filter((d) => (d.hitCount ?? 0) > 0)
    .sort((a, b) => (b.hitCount ?? 0) - (a.hitCount ?? 0))
    .slice(0, 3)
    .map((d) => ({
      tool: d.tool,
      pattern: d.pattern,
      scope: d.scope,
      hitCount: d.hitCount ?? 0,
    }))

  // pendingApprovals across all sessions isn't surfaced by sessions
  // list (it's a per-session-detail field), so we don't pre-compute
  // it here — keep the headline accurate to what the list response
  // already gives us. A future revision can add a daemon-side
  // pending count if operators ask for it.
  return {
    sessions: {
      total: sessionCounters.total,
      active: sessionCounters.active,
      completed: sessionCounters.completed,
      abandoned: sessionCounters.abandoned,
      pendingApprovals: 0,
    },
    approvals: {
      requested: sessionCounters.requested,
      approved: sessionCounters.approved,
      denied: sessionCounters.denied,
      autoApproved: sessionCounters.autoApproved,
    },
    decisions: {
      total: decisionCounters.total,
      bySession: decisionCounters.bySession,
      byAlways: decisionCounters.byAlways,
      stale: decisionCounters.stale,
      topHits,
    },
  }
}

/**
 * Render the dashboard layout for a given snapshot. Exported so the
 * chat shell `/stats` slash can reuse the exact same layout — keeps
 * cli verb and chat slash visually identical so an operator
 * pivoting between surfaces doesn't have to re-orient.
 */
export function formatStatsSnapshot(data: StatsSnapshot): string {
  const lines: string[] = []
  // Each section starts with a single-line headline followed by
  // bullet rows; an operator scrolling fast sees the headline,
  // a careful read picks up the bullets. Empty bullets get a
  // muted "—" so the section doesn't collapse and look broken.
  lines.push(chalk.cyan('Daemon snapshot'))
  lines.push('')
  lines.push(chalk.bold('Sessions'))
  lines.push(
    `  total: ${data.sessions.total}  ·  active: ${data.sessions.active}  ·  completed: ${data.sessions.completed}  ·  abandoned: ${data.sessions.abandoned}`,
  )
  lines.push('')
  lines.push(chalk.bold('Approvals'))
  if (data.approvals.requested === 0 && data.approvals.autoApproved === 0) {
    lines.push(chalk.gray('  no approval activity recorded'))
  } else {
    lines.push(
      `  requested: ${data.approvals.requested}  ·  approved: ${data.approvals.approved}  ·  denied: ${data.approvals.denied}`,
    )
    if (data.approvals.autoApproved > 0) {
      lines.push(chalk.gray(
        `  auto (remembered rule): ${data.approvals.autoApproved} approved`,
      ))
    }
  }
  lines.push('')
  lines.push(chalk.bold('Observability (7d)'))
  const obs = data.observability
  if (!obs) {
    lines.push(chalk.gray('  no observability snapshot available'))
  } else {
    lines.push(
      `  events: ${obs.reliability.totalEvents}  ·  errors: ${obs.reliability.errorEvents}  ·  crashes: ${obs.reliability.crashReports}`,
    )
    lines.push(
      `  satisfaction: ${formatPercent(obs.quality.explicitSatisfaction)}  ·  implicit acceptance: ${formatPercent(obs.quality.implicitAcceptance)}`,
    )
    lines.push(
      `  tasks: ${obs.productivity.tasksCompleted}/${obs.productivity.tasksStarted} completed  ·  completion: ${formatPercent(obs.productivity.completionRate)}  ·  throughput: ${formatNumber(obs.productivity.assistedTaskThroughputPerDay)}/day`,
    )
    lines.push(
      `  channel resolution: ${formatPercent(obs.productivity.channelResolutionRate)}  ·  tool success: ${formatPercent(obs.productivity.toolSuccessRate)}`,
    )
    if (obs.comparison) {
      lines.push(chalk.gray(
        `  vs previous: completion ${formatDeltaPercent(obs.comparison.completionRateDelta)}  ·  throughput ${formatDeltaNumber(obs.comparison.throughputPerDayDelta)}/day  ·  satisfaction ${formatDeltaPercent(obs.comparison.satisfactionDelta)}  ·  errors ${formatDeltaCount(obs.comparison.errorEventsDelta)}`,
      ))
    }
    if (obs.trend && obs.trend.length > 0) {
      lines.push(chalk.gray(
        `  completion trend: ${formatSparkline(obs.trend.map((point) => point.completionRate))}`,
      ))
      lines.push(chalk.gray(
        `  throughput trend: ${formatSparkline(obs.trend.map((point) => point.tasksCompleted))}`,
      ))
    }
    if (obs.alerts && obs.alerts.length > 0) {
      lines.push(chalk.yellow('  alerts:'))
      for (const alert of obs.alerts.slice(0, 3)) {
        const marker = alert.level === 'critical'
          ? chalk.red(alert.level)
          : alert.level === 'warning'
            ? chalk.yellow(alert.level)
            : chalk.gray(alert.level)
        lines.push(chalk.yellow(`    · [${marker}] ${alert.title}: ${alert.detail}`))
      }
    }
    if (obs.segments && obs.segments.length > 0) {
      lines.push(chalk.gray('  top segments:'))
      for (const segment of obs.segments.slice(0, 3)) {
        lines.push(chalk.gray(
          `    · ${segment.label}: ${segment.tasksCompleted}/${segment.tasksStarted} tasks  ·  errors ${segment.errorEvents}  ·  satisfaction ${formatPercent(segment.explicitSatisfaction)}`,
        ))
      }
    }
    if (obs.hotspots?.errorEvents.length) {
      lines.push(chalk.gray('  error hotspots:'))
      for (const hotspot of obs.hotspots.errorEvents.slice(0, 3)) {
        lines.push(chalk.gray(
          `    · ${hotspot.label}: ${hotspot.count}${hotspot.source ? ` (${hotspot.source})` : ''}`,
        ))
      }
    }
    if (data.privacy) {
      lines.push(
        chalk.gray(
          `  retention: ${data.privacy.retentionDays}d  ·  collection: ${data.privacy.localCollectionEnabled ? 'on' : 'paused'}  ·  feedback: ${data.privacy.feedbackCollectionEnabled ? 'on' : 'paused'}`,
        ),
      )
    }
    if (data.crashEvents && data.crashEvents.length > 0) {
      lines.push(chalk.yellow('  recent crashes:'))
      for (const event of data.crashEvents.slice(0, 3)) {
        lines.push(chalk.yellow(
          `    · ${formatEventTime(event.timestamp)} ${event.source}:${event.eventType}${event.sessionId ? ` (${event.sessionId})` : ''}`,
        ))
      }
    }
    if (data.recentEvents && data.recentEvents.length > 0) {
      lines.push(chalk.gray('  recent events:'))
      for (const event of data.recentEvents.slice(0, 5)) {
        const marker = event.severity === 'fatal' || event.severity === 'error'
          ? chalk.red(event.severity)
          : chalk.gray(event.severity)
        lines.push(
          chalk.gray(
            `    · ${formatEventTime(event.timestamp)} [${marker}] ${event.source}:${event.eventType}${event.surface ? `@${event.surface}` : ''}`,
          ),
        )
      }
    }
  }
  lines.push('')
  lines.push(chalk.bold('Decisions'))
  if (data.decisions.total === 0) {
    lines.push(chalk.gray('  no remembered approval rules'))
  } else {
    const stalePart = data.decisions.stale > 0
      ? `  ·  ${chalk.yellow(`stale: ${data.decisions.stale}`)}`
      : ''
    lines.push(
      `  total: ${data.decisions.total}  ·  session: ${data.decisions.bySession}  ·  always: ${data.decisions.byAlways}${stalePart}`,
    )
    if (data.decisions.topHits.length > 0) {
      lines.push(chalk.gray(`  top by hits:`))
      for (const top of data.decisions.topHits) {
        lines.push(
          chalk.gray(
            `    · [${top.scope}] ${top.tool} → ${top.pattern}  (${top.hitCount} hits)`,
          ),
        )
      }
    }
  }
  return lines.join('\n')
}

function formatPercent(value: number | null | undefined): string {
  return value === null || typeof value === 'undefined' || !Number.isFinite(value)
    ? 'n/a'
    : `${Math.round(value * 100)}%`
}

function formatNumber(value: number | null | undefined): string {
  return typeof value === 'number' && Number.isFinite(value)
    ? value.toFixed(2)
    : 'n/a'
}

function formatDeltaPercent(value: number | null | undefined): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'n/a'
  const rounded = Math.round(value * 100)
  return `${rounded >= 0 ? '+' : ''}${rounded}pp`
}

function formatDeltaNumber(value: number | null | undefined): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'n/a'
  return `${value >= 0 ? '+' : ''}${value.toFixed(2)}`
}

function formatDeltaCount(value: number | null | undefined): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'n/a'
  return `${value >= 0 ? '+' : ''}${value}`
}

function formatSparkline(values: Array<number | null | undefined>): string {
  const numeric = values.map((value) =>
    typeof value === 'number' && Number.isFinite(value) ? value : 0
  )
  if (numeric.length === 0) return 'n/a'
  const max = Math.max(...numeric)
  if (max <= 0) return numeric.map(() => '▁').join('')
  const chars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
  return numeric
    .map((value) => chars[Math.min(chars.length - 1, Math.round(
      (value / max) * (chars.length - 1),
    ))])
    .join('')
}

function formatEventTime(value: string): string {
  const parsed = Date.parse(value)
  if (!Number.isFinite(parsed)) return value
  return new Date(parsed).toISOString().slice(11, 19)
}

export async function statsCommand(options: StatsOptions = {}): Promise<void> {
  const client = new DaemonClient(options.url)
  try {
    // Two parallel calls: sessions list (with metrics so we can sum
    // approval counters) + decisions list. The compute is cheap and
    // running them concurrently avoids serial latency on the
    // dashboard view.
    const [sessions, decisionsResp] = await Promise.all([
      client.sessions(undefined, { metrics: true }),
      client.listRememberedApprovals(),
    ])
    const snapshot = buildStats(sessions, decisionsResp.decisions ?? [])
    const [observability, privacy, recentEvents, crashEvents] = await Promise.all([
      client.observabilitySnapshot('7d').catch(() => null),
      client.observabilityPrivacy().catch(() => null),
      client.observabilityEvents({ limit: 5 }).catch(() => []),
      client.observabilityCrashes(3).catch(() => []),
    ])
    snapshot.observability = observability
    snapshot.privacy = privacy
    snapshot.recentEvents = recentEvents
    snapshot.crashEvents = crashEvents
    output(snapshot, formatStatsSnapshot)
  } catch (err) {
    if (printApiError(err)) {
      process.exit(1)
    }
    console.error(chalk.red(`Failed to load daemon snapshot: ${friendlyErrorMessage(err)}`))
    process.exit(1)
  }
}
