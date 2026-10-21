import type { Message, ToolResultMetadata } from '@sepilotd/core'
import { CURRENT_AGENT_TURN_USER_METADATA_KEY } from './turn-context.js'

export type ApprovalFailureStatus = 'denied'

export const APPROVAL_FAILURE_STATUS_METADATA_KEY = 'approvalFailureStatus'
export const APPROVAL_STOP_REQUESTED_METADATA_KEY = 'approvalStopRequested'
export const APPROVAL_DENIAL_NOTE_METADATA_KEY = 'approvalDenialNote'
export const TOOL_RESULT_BLOCK_SOURCE_METADATA_KEY = 'toolResultBlockSource'
export type ToolResultBlockSource = 'policy' | 'approval'

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

export function trustedApprovalDenialDetails(
  metadata: ToolResultMetadata | undefined,
): Pick<TrustedApprovalDenial, 'stop' | 'note'> {
  const note = metadata?.[APPROVAL_DENIAL_NOTE_METADATA_KEY]
  return {
    ...(metadata?.[APPROVAL_STOP_REQUESTED_METADATA_KEY] === true ? { stop: true } : {}),
    ...(typeof note === 'string' && note.length > 0 ? { note } : {}),
  }
}

export interface ApprovalGraceState {
  denial: TrustedApprovalDenial | null
  toolTurnsRemaining: number
  frictionWarningIssued: boolean
}

export function createApprovalGraceState(): ApprovalGraceState {
  return { denial: null, toolTurnsRemaining: 1, frictionWarningIssued: false }
}

export type ApprovalOutcomeAfterTools = 'terminate' | 'grace_started' | null

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
