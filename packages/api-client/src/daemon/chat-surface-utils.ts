import { toolExecutionPostureLabel as formatToolExecutionPostureLabel } from '@sepilotd/core'
import type { ToolCall, ToolExecutionPosture } from '@sepilotd/core'
import type { AgentRunContract } from '@sepilotd/core'
import type { DaemonMemoryContextItem, DaemonUsage } from './types.js'
import type {
  ArtifactEventHandlerParams,
  Message,
  ToolRecovery,
  ToolStatus,
} from './chat-surface-types.js'
import {
  isAssistantMessage,
  isContextMessage,
  isSystemMessage,
  isToolMessage,
  isTodoMessage,
  isUserMessage,
} from './chat-surface-message-guards.js'

// Surface helpers split out by domain. Re-exported here so existing import
// sites keep working while the file shrinks toward formatting/reconciliation
// concerns only.
export {
  applyApprovalResolution,
  approvalNoteCopy,
  approvalRecoveryStartDetail,
  approvalResolutionActivityDetail,
  approvalResolutionLabel,
  approvalResolutionStatus,
  approvalResolutionStatusText,
  approvalResolutionToolMeta,
  createApprovalResolutionActivity,
} from './chat-surface-approval.js'
export {
  delegationHealthDetail,
  delegationHealthLabel,
  delegationHealthStatus,
} from './chat-surface-delegation.js'
export { toolExecutionPostureLabel } from '@sepilotd/core'

export function formatToolInput(input?: Record<string, unknown>): string {
  return JSON.stringify(input ?? {}, null, 2)
}

export function formatToolCall(toolCall: Partial<ToolCall>, limit = 160): string {
  return `${toolCall.name ?? 'tool'}(${JSON.stringify(toolCall.arguments ?? {}).slice(0, limit)})`
}

export function summarizeText(content: string, limit = 180): string {
  const normalized = content.replace(/\s+/g, ' ').trim()
  if (!normalized) return 'No details'
  return normalized.length > limit ? `${normalized.slice(0, limit - 1)}…` : normalized
}

export function formatAcceptanceCriteriaCount(
  count: number,
  style: 'long' | 'short' = 'long',
): string {
  if (style === 'short') {
    return `${count} ${count === 1 ? 'AC' : 'ACs'}`
  }
  return `${count} acceptance ${count === 1 ? 'criterion' : 'criteria'}`
}

function criterionText(criterion: AgentRunContract['acceptanceCriteria'][number]): string {
  return typeof criterion === 'string' ? criterion : criterion.text
}

function findRunContractFocusCriterion(contract: AgentRunContract): string | undefined {
  return contract.acceptanceCriteria
    .map(criterionText)
    .find((text) =>
      /current review\/audit follow-up|browser screenshot|rendered ui|visual qa|design plan|스크린샷|시각\s*(?:검수|리뷰)|디자인\s*(?:계획|브리프)/i.test(text),
    )
}

export function formatRunContractDetail(contract: AgentRunContract, limit = 180): string {
  const focus = findRunContractFocusCriterion(contract)
  const parts = [contract.summary, focus ? `Focus: ${focus}` : '']
    .filter((part) => part.trim().length > 0)
  return summarizeText(parts.join(' · '), limit)
}

export function formatMemoryContextMessage(items: DaemonMemoryContextItem[]): string {
  if (items.length === 0) {
    return ''
  }

  return [
    'Relevant context',
    ...items.map((item, index) => `${index + 1}. ${item.citationLabel}\n${item.snippet}`),
  ].join('\n\n')
}

export function summarizeMemoryContextItems(items: DaemonMemoryContextItem[]): {
  detail: string
  meta?: string
} {
  if (items.length === 0) {
    return { detail: 'No saved context was used' }
  }

  const documentCount = items.filter((item) => item.kind === 'document').length
  const memoryCount = items.length - documentCount
  const parts: string[] = []
  if (memoryCount > 0) {
    parts.push(`${memoryCount} memor${memoryCount === 1 ? 'y' : 'ies'}`)
  }
  if (documentCount > 0) {
    parts.push(`${documentCount} document${documentCount === 1 ? '' : 's'}`)
  }

  const meta = items
    .slice(0, 2)
    .map((item) => item.citationLabel)
    .join(' · ')

  return {
    detail: `Using ${parts.join(' and ')}`,
    meta: meta || undefined,
  }
}

export function mergeArtifacts<T extends { id: string }>(previous: T[], incoming: T[]): T[] {
  const nextById = new Map(previous.map((artifact) => [artifact.id, artifact]))
  for (const artifact of incoming) {
    nextById.set(artifact.id, artifact)
  }
  return Array.from(nextById.values())
}

export function artifactCaptureDetail(count: number): string {
  return `${count} artifact${count > 1 ? 's' : ''} captured for this session`
}

export function toolResultRecoveryLabel(recovery?: ToolRecovery): string | undefined {
  switch (recovery) {
    case 'journal':
      return 'saved execution'
    case 'probe':
      return 'verified recovery'
    default:
      return undefined
  }
}

function appendExecutionPostureLabel(label: string, posture?: ToolExecutionPosture): string {
  const postureLabel = formatToolExecutionPostureLabel(posture)
  return postureLabel ? `${label} · ${postureLabel}` : label
}

function isToolTerminalStatus(status?: ToolStatus): status is 'success' | 'error' {
  return status === 'success' || status === 'error'
}

function shouldPreserveLiveMessage(stored: Message, live?: Message): boolean {
  if (!live || live.role !== stored.role) {
    return false
  }

  if (isToolMessage(live) && isToolMessage(stored)) {
    if (Boolean(live.toolResult) && !stored.toolResult) {
      return true
    }

    if (isToolTerminalStatus(live.toolStatus) && !isToolTerminalStatus(stored.toolStatus)) {
      return true
    }

    if (!live.toolNeedsApproval && stored.toolNeedsApproval) {
      return true
    }
  }

  if (isAssistantMessage(live) && isAssistantMessage(stored)) {
    if (live.content.trim().length > stored.content.trim().length) {
      return true
    }

    if (live.usage && !stored.usage) {
      return true
    }
  }

  if (isUserMessage(live) && isUserMessage(stored)) {
    return live.content.trim().length > stored.content.trim().length
  }

  if (isContextMessage(live) && isContextMessage(stored)) {
    return (
      live.contextItems.length > stored.contextItems.length ||
      live.content.trim().length > stored.content.trim().length
    )
  }

  if (isTodoMessage(live) && isTodoMessage(stored)) {
    return (
      live.todoItems.length > stored.todoItems.length ||
      live.content.trim().length > stored.content.trim().length
    )
  }

  if (isSystemMessage(live) && isSystemMessage(stored)) {
    return live.content.trim().length > stored.content.trim().length
  }

  return false
}

function shouldAppendLiveMessage(message: Message): boolean {
  if (isUserMessage(message)) {
    return Boolean(message.content.trim())
  }

  if (isAssistantMessage(message)) {
    return Boolean(message.content.trim() || message.usage)
  }

  if (isContextMessage(message)) {
    return Boolean(message.contextItems.length || message.content.trim())
  }

  if (isTodoMessage(message)) {
    return Boolean(message.todoItems.length || message.content.trim())
  }

  if (isSystemMessage(message)) {
    return Boolean(message.content.trim())
  }

  // Tool messages: keep them once they've materialized any payload.
  return Boolean(message.content.trim() || message.toolResult)
}

export function reconcileSurfaceMessages(
  storedMessages: Message[],
  liveMessages: Message[],
): Message[] {
  const liveById = new Map(liveMessages.map((message) => [message.id, message]))
  const storedIds = new Set(storedMessages.map((message) => message.id))
  let changed = false

  const reconciled = storedMessages.map((message) => {
    const liveMessage = liveById.get(message.id)
    if (shouldPreserveLiveMessage(message, liveMessage)) {
      changed = true
      return liveMessage!
    }
    return message
  })

  for (const liveMessage of liveMessages) {
    if (storedIds.has(liveMessage.id) || !shouldAppendLiveMessage(liveMessage)) {
      continue
    }
    reconciled.push(liveMessage)
    changed = true
  }

  return changed ? reconciled : storedMessages
}

export function createArtifactEventHandler<T extends { id: string }>(
  params: ArtifactEventHandlerParams<T>,
): (artifacts: T[]) => void {
  return (artifacts: T[]) => {
    if (artifacts.length === 0) return

    params.setArtifacts((prev) => mergeArtifacts(prev, artifacts))
    params.pushActivity({
      id: params.createId(),
      kind: 'result',
      label: 'Artifacts saved',
      detail: artifactCaptureDetail(artifacts.length),
      status: 'success',
    })
  }
}

export function normalizeUsage(
  usage?: Partial<DaemonUsage> | Message['usage'],
): Message['usage'] | undefined {
  if (!usage) return undefined
  return {
    inputTokens: usage.inputTokens ?? 0,
    outputTokens: usage.outputTokens ?? 0,
    estimatedCost: 'costUsd' in usage ? usage.costUsd : usage.estimatedCost,
  }
}

export function mapStoredToolStatus(
  status: 'pending' | 'approved' | 'denied' | 'executing',
): ToolStatus {
  switch (status) {
    case 'pending':
      return 'pending'
    case 'denied':
      return 'error'
    default:
      return 'running'
  }
}

export function toolMetaFromStoredStatus(
  status: 'pending' | 'approved' | 'denied' | 'executing',
): string {
  switch (status) {
    case 'pending':
      return 'Awaiting approval'
    case 'approved':
      return 'Approved for execution'
    case 'denied':
      return 'Denied by policy'
    case 'executing':
      return 'Running now'
  }
}

export function toolStatusLabel(message: Message): string {
  if (message.toolNeedsApproval && message.toolStatus === 'pending') {
    return 'Needs approval'
  }

  switch (message.toolStatus) {
    case 'success':
      return 'Completed'
    case 'error':
      return 'Failed'
    case 'pending':
      return 'Pending'
    default:
      return 'Running'
  }
}

export function toolMetaFromResultStatus(
  status: 'success' | 'error' | 'timeout' | 'cancelled',
  recovery?: ToolRecovery,
  output?: string,
  executionPosture?: ToolExecutionPosture,
): string {
  if (status === 'success' && recovery === 'journal') {
    return appendExecutionPostureLabel('Recovered from saved execution', executionPosture)
  }

  if (status === 'success' && recovery === 'probe') {
    return appendExecutionPostureLabel('Recovered from verified side effect', executionPosture)
  }

  let label: string
  switch (status) {
    case 'success':
      label = 'Execution finished'
      break
    case 'timeout':
      label = appendReason('Execution timed out', output)
      break
    case 'cancelled':
      label = 'Execution cancelled'
      break
    default:
      label = failedAttemptGuardMeta(output)
        ?? appendReason('Execution failed', output)
      break
  }
  return appendExecutionPostureLabel(label, executionPosture)
}

function failedAttemptGuardMeta(output: string | undefined): string | null {
  if (!output?.startsWith('[Failed-attempt guard]')) return null
  const recordedFailure = output
    .split(/\r?\n/)
    .find((line) => line.startsWith('Recorded failure:'))
    ?.slice('Recorded failure:'.length)
    .trim()
  return recordedFailure
    ? `Retry blocked · original failure: ${truncateReason(recordedFailure)}`
    : 'Retry blocked after an earlier identical failure'
}

function appendReason(label: string, output: string | undefined): string {
  const reason = extractFirstLine(output)
  return reason ? `${label}: ${reason}` : label
}

const REASON_MAX_LEN = 160

function extractFirstLine(text: string | undefined): string | null {
  if (!text) return null
  const firstLine = text.split(/\r?\n/, 1)[0]?.trim() ?? ''
  if (!firstLine) return null
  return truncateReason(firstLine)
}

function truncateReason(reason: string): string {
  if (reason.length <= REASON_MAX_LEN) return reason

  const headLength = Math.ceil((REASON_MAX_LEN - 1) / 2)
  const tailLength = Math.floor((REASON_MAX_LEN - 1) / 2)
  return `${reason.slice(0, headLength).trimEnd()}…${reason.slice(-tailLength).trimStart()}`
}
