import type { Message, ToolResultMetadata } from '@sepilotd/core'
import { CURRENT_AGENT_TURN_USER_METADATA_KEY } from './turn-context.js'

export type ApprovalFailureStatus = 'denied'

export const APPROVAL_FAILURE_STATUS_METADATA_KEY = 'approvalFailureStatus'
/** Operator chose "deny & stop": the denial ends the run immediately. */
export const APPROVAL_STOP_REQUESTED_METADATA_KEY = 'approvalStopRequested'
/** Free-text note the operator attached to a denial. */
export const APPROVAL_DENIAL_NOTE_METADATA_KEY = 'approvalDenialNote'
/**
 * Provenance of a tool result that was refused before execution. Policy
 * blocks (deny rules, autonomy, workspace boundary) and approval outcomes
 * (denied, timed out, cancelled, missing handler) are friction the model must
 * route around, not evidence that the action itself fails. Loop-control
 * counters (stuck-repeat failure bucket, failed-attempt guard) skip entries
 * carrying this key; only the exact-repeat threshold still sees them.
 */
export const TOOL_RESULT_BLOCK_SOURCE_METADATA_KEY = 'toolResultBlockSource'
export type ToolResultBlockSource = 'policy' | 'approval'

/** Number of policy/approval blocks in one run after which the model is told to stop probing. */
export const POLICY_FRICTION_WARNING_THRESHOLD = 3

export function blockSourceFromMetadata(
  metadata: ToolResultMetadata | Record<string, unknown> | undefined,
): ToolResultBlockSource | null {
  const value = metadata?.[TOOL_RESULT_BLOCK_SOURCE_METADATA_KEY]
  return value === 'policy' || value === 'approval' ? value : null
}

export function approvalFailureStatusFromDecision(
  decision: unknown,
): ApprovalFailureStatus | null {
  if (decision === 'denied') return 'denied'
  return null
}

export interface TrustedApprovalDenial {
  toolCallId?: string
  toolName?: string
  /** True when the operator asked for the run to stop with this denial. */
  stop?: boolean
  note?: string
}

function messageText(message: Message): string {
  if (typeof message.content === 'string') return message.content
  return message.content
    .filter((part): part is Extract<(typeof message.content)[number], { type: 'text' }> =>
      part.type === 'text')
    .map((part) => part.text)
    .join('\n')
}

export function isTrustedApprovalDenialResult(result: {
  output: string
  status: 'success' | 'error'
  metadata?: ToolResultMetadata
}): boolean {
  return result.status === 'error'
    && result.metadata?.[APPROVAL_FAILURE_STATUS_METADATA_KEY] === 'denied'
    && /^\s*\[approval:denied\]/iu.test(result.output)
}

/** Denial details carried on a trusted denial tool result's metadata. */
export function trustedApprovalDenialDetails(
  metadata: ToolResultMetadata | undefined,
): Pick<TrustedApprovalDenial, 'stop' | 'note'> {
  const note = metadata?.[APPROVAL_DENIAL_NOTE_METADATA_KEY]
  return {
    ...(metadata?.[APPROVAL_STOP_REQUESTED_METADATA_KEY] === true ? { stop: true } : {}),
    ...(typeof note === 'string' && note.length > 0 ? { note } : {}),
  }
}

/**
 * Mutable per-run approval state shared by the react loop's tool-batch sites.
 * `denial` is set once a plain denial grants the side-effect-free turn;
 * `toolTurnsRemaining` bounds how many read-only tool turns that grace may
 * spend before the model must answer.
 */
export interface ApprovalGraceState {
  denial: TrustedApprovalDenial | null
  toolTurnsRemaining: number
  frictionWarningIssued: boolean
}

export function createApprovalGraceState(): ApprovalGraceState {
  return { denial: null, toolTurnsRemaining: 1, frictionWarningIssued: false }
}

export type ApprovalOutcomeAfterTools = 'terminate' | 'grace_started' | null

/**
 * Inspect the tool results just appended and update the grace state:
 * - `terminate`: a `deny & stop` denial, or a second denial while a grace turn
 *   is already open — the run must end with `approval_denied`.
 * - `grace_started`: first plain denial — the grace system message was pushed
 *   and the caller should continue with a read-only catalog.
 * Also injects the policy-friction warning once the block count crosses the
 * threshold. Pure bookkeeping: no text parsing, only trusted metadata.
 */
export function observeApprovalOutcomeAfterTools(
  messages: Message[],
  state: ApprovalGraceState,
): ApprovalOutcomeAfterTools {
  let outcome: ApprovalOutcomeAfterTools = null
  const denial = latestTrustedApprovalDenial(messages)
  if (denial) {
    if (denial.stop) {
      state.denial = denial
      return 'terminate'
    }
    if (!state.denial) {
      state.denial = denial
      state.toolTurnsRemaining = 1
      messages.push(buildApprovalDenialGraceMessage(denial))
      outcome = 'grace_started'
    } else if (denial.toolCallId !== state.denial.toolCallId) {
      state.denial = denial
      return 'terminate'
    }
  }
  const friction = countBlockedToolResultsInCurrentTurn(messages)
  if (friction > POLICY_FRICTION_WARNING_THRESHOLD && !state.frictionWarningIssued) {
    state.frictionWarningIssued = true
    messages.push(buildPolicyFrictionMessage(friction))
  }
  return outcome
}

/**
 * Return the latest operator denial in this user turn. Both the trusted
 * metadata and the bracketed status tag are required so an ordinary tool
 * cannot stop a run by spoofing denial-shaped output.
 */
function currentTurnMessages(messages: readonly Message[]): readonly Message[] {
  let turnStart = -1
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]
    if (
      message.role === 'user'
      && message.metadata?.[CURRENT_AGENT_TURN_USER_METADATA_KEY] === true
    ) {
      turnStart = index
      break
    }
  }
  if (turnStart < 0) {
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      if (messages[index]?.role === 'user') {
        turnStart = index
        break
      }
    }
  }

  return turnStart >= 0 ? messages.slice(turnStart) : messages
}

/**
 * Count tool results in the current user turn that were refused by policy or
 * by a human before execution. This is the run's "friction" — distinct from
 * tool failures, which the stuck/failed-attempt guards own.
 */
export function countBlockedToolResultsInCurrentTurn(messages: readonly Message[]): number {
  let count = 0
  for (const message of currentTurnMessages(messages)) {
    if (message.role === 'tool' && blockSourceFromMetadata(message.metadata)) count += 1
  }
  return count
}

export function buildPolicyFrictionMessage(count: number): Message {
  return {
    role: 'system',
    content: [
      `[Policy friction] ${count} actions were blocked by policy or approval in this run.`,
      'Stop probing alternatives that need the same permission; a rephrased or',
      'differently-shaped call will be blocked the same way.',
      'Either finish with what is possible using the tools that are allowed, or',
      'reply with `INCOMPLETE:` naming the exact permission needed and the exact',
      'command or action that requires it.',
    ].join(' '),
    metadata: { reminderKind: 'policy_friction' },
  }
}

/**
 * System message for the single side-effect-free turn a denial grants. The
 * tool catalog offered alongside it is filtered to read-only tools, so the
 * text and the catalog enforce the same boundary.
 */
export function buildApprovalDenialGraceMessage(denial: TrustedApprovalDenial): Message {
  const tool = denial.toolName ?? 'the requested tool'
  return {
    role: 'system',
    content: [
      `[Approval denied] The user declined to run \`${tool}\`.`,
      ...(denial.note ? [`User note: ${denial.note}`] : []),
      'Do not retry it and do not work around it with another side-effecting tool;',
      'only read-only inspection tools are available for this turn.',
      'Choose one: propose a materially different approach and ask the user',
      'whether to proceed, finish with what is already done, or answer with',
      '`INCOMPLETE:` naming the declined action as the blocker.',
    ].join(' '),
    metadata: { reminderKind: 'approval_denial_grace' },
  }
}

export function latestTrustedApprovalDenial(
  messages: readonly Message[],
): TrustedApprovalDenial | null {
  const currentTurn = currentTurnMessages(messages)
  const toolNamesById = new Map<string, string>()
  for (const message of currentTurn) {
    for (const call of message.toolCalls ?? []) {
      toolNamesById.set(call.id, call.name)
    }
  }

  for (let index = currentTurn.length - 1; index >= 0; index -= 1) {
    const message = currentTurn[index]!
    if (
      message.role !== 'tool'
      || message.metadata?.[APPROVAL_FAILURE_STATUS_METADATA_KEY] !== 'denied'
      || !/^\s*\[approval:denied\]/iu.test(messageText(message))
    ) {
      continue
    }
    const note = message.metadata?.[APPROVAL_DENIAL_NOTE_METADATA_KEY]
    return {
      ...(message.toolCallId ? { toolCallId: message.toolCallId } : {}),
      ...(message.name
        ? { toolName: message.name }
        : message.toolCallId && toolNamesById.has(message.toolCallId)
          ? { toolName: toolNamesById.get(message.toolCallId)! }
          : {}),
      ...(message.metadata?.[APPROVAL_STOP_REQUESTED_METADATA_KEY] === true ? { stop: true } : {}),
      ...(typeof note === 'string' && note.length > 0 ? { note } : {}),
    }
  }
  return null
}

export function buildApprovalDeniedTurnOutput(toolName: string | undefined, input: string): string {
  const tool = toolName ?? 'tool'
  if (/[가-힣]/u.test(input)) {
    return [
      `INCOMPLETE: ${tool} 실행 승인이 거절되어 이번 작업을 중단했습니다.`,
      '거절 이후 다른 쓰기나 명령 실행은 시도하지 않았습니다.',
      '계속하려면 새 메시지로 원하는 변경 범위를 알려주세요.',
    ].join(' ')
  }
  return [
    `INCOMPLETE: Approval to run ${tool} was denied, so this turn stopped.`,
    'No other writes or commands were attempted after the denial.',
    'Send a new message with the approach you want if you would like to continue.',
  ].join(' ')
}
