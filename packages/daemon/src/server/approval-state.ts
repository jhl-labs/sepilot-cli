import type { SessionEvent } from '@sepilotd/core'
import type { PendingApproval } from './runtime/approvals.js'

type ApprovalResponseEvent = Extract<SessionEvent, { type: 'approval_response' }>

export function buildPendingApprovals(
  sessionId: string,
  events: SessionEvent[],
  liveApprovals: PendingApproval[] = [],
): PendingApproval[] {
  const resolvedRequestIds = new Set<string>()
  const completedToolCallIds = new Set<string>()

  for (const event of events) {
    if (event.type === 'approval_response') {
      resolvedRequestIds.add(event.requestId)
    }

    if (event.type === 'tool_result') {
      completedToolCallIds.add(event.toolCallId)
    }
  }

  const pending = new Map<string, PendingApproval>()

  for (const approval of liveApprovals) {
    if (
      resolvedRequestIds.has(approval.requestId)
      || completedToolCallIds.has(approval.toolCallId)
    ) {
      continue
    }
    pending.set(approval.requestId, approval)
  }

  for (const event of events) {
    if (event.type !== 'approval_request') {
      continue
    }

    const requestId = event.id
    if (
      pending.has(requestId)
      || resolvedRequestIds.has(requestId)
      || completedToolCallIds.has(event.toolCallId)
    ) {
      continue
    }

    pending.set(requestId, {
      requestId,
      sessionId,
      toolCallId: event.toolCallId,
      tool: event.tool,
      input: event.input,
      requestedAt: event.timestamp,
      expiresAt: event.timestamp,
      state: 'stale',
    })
  }

  return [...pending.values()].sort((left, right) =>
    left.requestedAt.localeCompare(right.requestedAt),
  )
}

export function findPendingApproval(
  sessionId: string,
  events: SessionEvent[],
  requestId: string,
): PendingApproval | null {
  return (
    buildPendingApprovals(sessionId, events).find(
      (approval) => approval.requestId === requestId,
    ) ?? null
  )
}

/**
 * Return the latest recorded response only when the request itself exists in
 * this session. This lets approval responses be safely retried without turning
 * an already-handled notification action into a misleading 404.
 */
export function findApprovalResolution(
  events: SessionEvent[],
  requestId: string,
): ApprovalResponseEvent | null {
  const wasRequested = events.some(
    (event) => event.type === 'approval_request' && event.id === requestId,
  )
  if (!wasRequested) return null

  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index]
    if (event?.type === 'approval_response' && event.requestId === requestId) {
      return event
    }
  }
  return null
}
