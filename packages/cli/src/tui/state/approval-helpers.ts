// Pure helpers around the approval-related slices of useChat. Pulled
// out so the conversion (payload -> cli ApprovalRequest) and the
// stale-recovery prompt copy can be unit-tested without rendering
// useChat or driving a real daemon stream.

import type { ApprovalRequest } from '../types.js'

export interface ApprovalRequestPayload {
  requestId: string
  phase?: string
  toolCall?: {
    id?: string
    name?: string
    arguments?: Record<string, unknown>
  }
  suggestedRule?: ApprovalRequest['suggestedRule']
  previewDiff?: string
  context?: string
}

/**
 * Translate a daemon `approval_request` stream payload into the cli
 * ApprovalRequest shape. Falls back when the payload omits parts that
 * older daemon builds didn't emit:
 *   - missing toolCall.id  -> fresh uuid (so the cli has a stable key)
 *   - missing toolCall.name -> 'unknown' (so the picker still renders)
 *   - missing arguments     -> {}
 */
export function approvalRequestFromPayload(opts: {
  payload: ApprovalRequestPayload
  sessionId: string | null
  phase?: string | null
  repeatCount?: number
  generateToolCallId?: () => string
}): ApprovalRequest {
  const generateId = opts.generateToolCallId ?? (() => crypto.randomUUID())
  const payloadPhase =
    typeof opts.payload.phase === 'string' && opts.payload.phase.trim().length > 0
      ? opts.payload.phase.trim()
      : null
  const phase =
    payloadPhase ??
    (typeof opts.phase === 'string' && opts.phase.trim().length > 0 ? opts.phase.trim() : null)
  return {
    requestId: opts.payload.requestId,
    sessionId: opts.sessionId ?? undefined,
    toolCallId: opts.payload.toolCall?.id ?? generateId(),
    toolName: opts.payload.toolCall?.name ?? 'unknown',
    input: (opts.payload.toolCall?.arguments ?? {}) as Record<string, unknown>,
    phase: phase ?? undefined,
    repeatCount: opts.repeatCount && opts.repeatCount > 1 ? opts.repeatCount : undefined,
    state: 'live',
    suggestedRule: opts.payload.suggestedRule,
    previewDiff:
      typeof opts.payload.previewDiff === 'string' && opts.payload.previewDiff.trim().length > 0
        ? opts.payload.previewDiff
        : undefined,
    context:
      typeof opts.payload.context === 'string' && opts.payload.context.trim().length > 0
        ? opts.payload.context.trim()
        : undefined,
  }
}

export function buildApprovalRepeatKey(input: {
  toolName: string
  input: Record<string, unknown>
}): string {
  return `${input.toolName}:${stableStringify(input.input)}`
}

function stableStringify(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map((item) => stableStringify(item)).join(',')}]`
  }
  if (value && typeof value === 'object') {
    return `{${Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, item]) => `${JSON.stringify(key)}:${stableStringify(item)}`)
      .join(',')}}`
  }
  return JSON.stringify(value)
}

/**
 * Copy that goes into the SYSTEM_MESSAGE after a stale-run approval
 * resolves: the approve / deny verbs lead to different recovery paths
 * (a new chat run vs. continuing without the tool) and the operator
 * needs to see why the next event in the transcript is one or the
 * other.
 */
export function buildStaleApprovalNotice(opts: { approved: boolean }): string {
  return opts.approved
    ? 'Recorded approval for a stale run. Starting a fresh recovery run from the latest session state.'
    : 'Recorded denial for a stale run. Continuing from the latest session state without that tool.'
}

export function buildMissingApprovalNotice(): string {
  return 'Approval request no longer exists on the daemon (the run may have died). Modal dismissed.'
}

export function buildAlreadyHandledApprovalNotice(): string {
  return 'That approval was already handled (or its run ended) — nothing to do.'
}
