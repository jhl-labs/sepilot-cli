import type { ActivityItem } from '@sepilotd/api-client'
import { formatElapsed } from './streaming.js'

const DEFAULT_MAX_ACTIVITY_ITEMS = 24

export interface RunProgressActivityInput {
  id: string
  startedAt: number
  now: number
  streamStatus: string | null
  currentMessage: string
  pendingApprovalToolName?: string | null
  pendingQuestionPrompt?: string | null
}

function normalizeStatus(status: string | null): string | null {
  const normalized = status
    ?.replace(/[.:…]+$/g, '')
    .replace(/\s+/g, ' ')
    .trim()

  return normalized && normalized.length > 0 ? normalized : null
}

function describeVisiblePhase({
  streamStatus,
  currentMessage,
  pendingApprovalToolName,
  pendingQuestionPrompt,
}: Pick<
  RunProgressActivityInput,
  'streamStatus' | 'currentMessage' | 'pendingApprovalToolName' | 'pendingQuestionPrompt'
>): string {
  if (pendingApprovalToolName) {
    return `waiting for approval on ${pendingApprovalToolName}`
  }
  if (pendingQuestionPrompt) {
    return `waiting for answer: ${pendingQuestionPrompt}`
  }

  const status = normalizeStatus(streamStatus)
  if (status?.startsWith('Thinking')) {
    return 'thinking'
  }
  if (status?.startsWith('Running ')) {
    return status.replace(/^Running\s+/, 'running ')
  }
  if (currentMessage.trim().length > 0 || status?.startsWith('Streaming')) {
    return 'streaming response'
  }
  if (status) {
    return status.toLowerCase()
  }

  return 'waiting for daemon event'
}

export function buildRunProgressActivity({
  id,
  startedAt,
  now,
  streamStatus,
  currentMessage,
  pendingApprovalToolName,
  pendingQuestionPrompt,
}: RunProgressActivityInput): ActivityItem {
  const elapsedLabel = formatElapsed(now - startedAt)
  const phase = describeVisiblePhase({
    streamStatus,
    currentMessage,
    pendingApprovalToolName,
    pendingQuestionPrompt,
  })
  const outputHint = currentMessage.trim().length > 0
    ? `${currentMessage.trim().length} chars of visible assistant output so far`
    : 'no visible assistant output yet'

  return {
    id,
    kind: 'state',
    label: 'Still working',
    detail: `${phase} · ${outputHint} · waiting for the next stream event`,
    status: pendingApprovalToolName || pendingQuestionPrompt ? 'pending' : 'running',
    meta: elapsedLabel,
  }
}

export function upsertRunProgressActivity(
  activities: ActivityItem[],
  activity: ActivityItem,
  maxActivityItems = DEFAULT_MAX_ACTIVITY_ITEMS,
): ActivityItem[] {
  const withoutPrevious = activities.filter((item) => item.id !== activity.id)
  return [...withoutPrevious, activity].slice(-Math.max(1, maxActivityItems))
}
