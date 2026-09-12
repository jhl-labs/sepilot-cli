import type { ApprovalDecisionStatus } from '@sepilotd/core'
import type {
  ActivityItem,
  ActivityStatus,
  ApprovalState,
  Message,
} from './chat-surface-types.js'

function normalizeApprovalDecision(
  approved: boolean | ApprovalDecisionStatus,
): ApprovalDecisionStatus {
  return typeof approved === 'boolean'
    ? (approved ? 'approved' : 'denied')
    : approved
}

export function approvalResolutionLabel(
  approved: boolean | ApprovalDecisionStatus,
): string {
  switch (normalizeApprovalDecision(approved)) {
    case 'approved':
      return 'Approval granted'
    case 'feedback':
      return 'Approval needs changes'
    case 'denied':
      return 'Approval denied'
  }
}

export function approvalResolutionStatus(
  approved: boolean | ApprovalDecisionStatus,
): ActivityStatus {
  return normalizeApprovalDecision(approved) === 'approved'
    ? 'success'
    : 'error'
}

export function approvalResolutionToolMeta(
  approved: boolean | ApprovalDecisionStatus,
  note?: string,
): string {
  switch (normalizeApprovalDecision(approved)) {
    case 'approved':
      return 'Approval granted, waiting for result'
    case 'feedback':
      return note ? `Needs changes: ${note}` : 'Needs changes before execution'
    case 'denied':
      return note ? `Approval denied: ${note}` : 'Approval denied'
  }
}

export function approvalResolutionStatusText(
  approved: boolean | ApprovalDecisionStatus,
): string {
  switch (normalizeApprovalDecision(approved)) {
    case 'approved':
      return 'Tool approved…'
    case 'feedback':
      return 'Tool needs changes'
    case 'denied':
      return 'Tool denied'
  }
}

export function approvalResolutionActivityDetail(surface: string): string {
  return `User responded from ${surface} surface`
}

export function approvalRecoveryStartDetail(
  toolName: string,
  approved: boolean | ApprovalDecisionStatus,
): string {
  switch (normalizeApprovalDecision(approved)) {
    case 'approved':
      return `Approval restored for ${toolName}`
    case 'feedback':
      return `Feedback recorded for ${toolName}`
    case 'denied':
      return `Denial recorded for ${toolName}`
  }
}

export function createApprovalResolutionActivity(options: {
  id: string
  approved: boolean | ApprovalDecisionStatus
  detail: string
}): ActivityItem {
  return {
    id: options.id,
    kind: 'approval',
    label: approvalResolutionLabel(options.approved),
    detail: options.detail,
    status: approvalResolutionStatus(options.approved),
  }
}

export function applyApprovalResolution(
  messages: Message[],
  toolCallId: string,
  approved: boolean | ApprovalDecisionStatus,
  note?: string,
): Message[] {
  const decision = normalizeApprovalDecision(approved)
  return messages.map((message) => (
    message.role === 'tool' && message.id === toolCallId
      ? {
          ...message,
          toolNeedsApproval: false,
          approvalRequestId: undefined,
          approvalState: undefined,
          resumeAvailable: undefined,
          toolStatus: decision === 'approved' ? 'running' : 'error',
          toolMeta: approvalResolutionToolMeta(decision, note),
        }
      : message
  ))
}

export function approvalNoteCopy(
  approvalState?: ApprovalState,
  resumeAvailable?: boolean,
): string {
  if (approvalState === 'stale') {
    return resumeAvailable
      ? 'The daemon still has a checkpoint for this paused run. Responding will resume the exact run from the approval boundary.'
      : 'The original run is no longer attached. Responding will record the decision, refresh the session state, and start a fresh run from the latest context.'
  }
  return 'Review the tool input before continuing the run.'
}
