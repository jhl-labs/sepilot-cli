import type { ActivityItem } from '@sepilotd/api-client'
import type { ToolCallState } from '../types.js'

interface BuildToolActivitySummaryOptions {
  activities?: ActivityItem[]
  toolCalls?: ToolCallState[]
  runActive?: boolean
}

type StatusCounts = Record<'running' | 'pending' | 'success' | 'error', number>

function createEmptyCounts(): StatusCounts {
  return {
    running: 0,
    pending: 0,
    success: 0,
    error: 0,
  }
}

function summarizePhase(detail: string): string | null {
  const normalized = detail.replace(/\s+/g, ' ').trim()
  if (!normalized) {
    return null
  }

  const phase = normalized
    .split(/·|[.!?](?:\s|$)|\n/)[0]
    ?.trim()
  return phase && phase.length > 0 ? phase : null
}

function formatCount(count: number, singular: string, plural = singular): string {
  return `${count} ${count === 1 ? singular : plural}`
}

function truncateSegment(value: string, maxLength = 48): string {
  if (value.length <= maxLength) {
    return value
  }

  return `${value.slice(0, Math.max(0, maxLength - 1)).trimEnd()}…`
}

function summarizeLatestActivity(
  activities: ActivityItem[],
): string | null {
  const latest = [...activities]
    .reverse()
    .find((activity) => (
      activity.status === 'running'
      || activity.status === 'pending'
      || activity.label === 'Still working'
      || activity.label === 'Progress update'
      || activity.label === 'Reasoning'
    ))

  if (!latest) {
    return null
  }

  if (latest.status === 'pending') {
    return truncateSegment(`${latest.label} pending`)
  }

  if (latest.label === 'Progress update' || latest.label === 'Still working' || latest.label === 'Reasoning') {
    return truncateSegment(summarizePhase(latest.detail) ?? latest.label.toLowerCase())
  }

  return truncateSegment(latest.label)
}

export function buildToolActivitySummary({
  activities = [],
  toolCalls = [],
  runActive = false,
}: BuildToolActivitySummaryOptions): string {
  const counts = createEmptyCounts()
  // Count only tool/approval executions. Narration items (thinking,
  // reasoning steps, state notes) are pushed with a permanent `running`
  // status and never resolved, so counting them inflates the tally into a
  // bogus "17 running" while a single tool executes.
  const sourceStatuses = activities.length > 0
    ? activities
        .filter((activity) => activity.kind === 'tool' || activity.kind === 'approval')
        .map((activity) => activity.status)
    : toolCalls.map((tool) => tool.status)

  for (const status of sourceStatuses) {
    if (status === 'running' || status === 'pending' || status === 'success' || status === 'error') {
      counts[status] += 1
    }
  }

  const latestActivitySummary = summarizeLatestActivity(activities)

  const segments: string[] = []

  if (runActive) {
    segments.push(latestActivitySummary ? `run active · ${latestActivitySummary}` : 'run active')
  }

  // A "running" tally is only meaningful while the run is live — activity
  // items keep a stale `running` status after the run ends (the stream
  // controller pushes fresh result items instead of reconciling the original),
  // so once the run is idle we report what actually happened instead of a
  // bogus "14 running" count that lingers on screen.
  if (runActive && counts.running > 0) {
    segments.push(formatCount(counts.running, 'running'))
  }
  if (counts.pending > 0) {
    segments.push(formatCount(counts.pending, 'waiting'))
  }
  if (counts.error > 0) {
    segments.push(formatCount(counts.error, 'error', 'errors'))
  }
  if (!runActive && counts.running === 0 && counts.pending === 0 && counts.success > 0) {
    segments.push(`${counts.success} completed`)
  }

  if (segments.length === 0 && latestActivitySummary) {
    segments.push(`latest update · ${latestActivitySummary}`)
  }

  if (segments.length === 0) {
    return 'idle'
  }

  return segments.join(' · ')
}
