import type { Message } from '@sepilotd/core'
import { TOOL_RESULT_STATUS_METADATA_KEY } from './memory-write-completion.js'
import { CURRENT_AGENT_TURN_USER_METADATA_KEY } from './turn-context.js'

export const POLICY_FAILURE_REASON_METADATA_KEY = 'policyFailureReason'
export const READ_ONLY_INVOCATION_REPAIR_METADATA_KEY = 'readOnlyInvocationRepair'
export const READ_ONLY_INVOCATION_REPAIR_TOOL_CALL_ID_METADATA_KEY =
  'readOnlyInvocationRepairToolCallId'
export const TOOL_RESULT_EXECUTION_OBSERVED_METADATA_KEY = 'toolResultExecutionObserved'
export const TOOL_RESULT_SECURITY_EFFECT_METADATA_KEY = 'toolResultSecurityEffect'

const READ_ONLY_TERMINAL_POLICY_REASON = "ReadOnly mode blocks tool 'terminal.run'"
const SHELL_WRAPPER_EXECUTABLES = new Set([
  'bash',
  'cmd',
  'dash',
  'fish',
  'ksh',
  'powershell',
  'pwsh',
  'sh',
  'zsh',
])

export interface TrustedPolicyFailure {
  reason: string
  toolCallId?: string
  toolName?: string
}

function messageText(message: Message): string {
  if (typeof message.content === 'string') return message.content
  return message.content
    .filter((part): part is Extract<(typeof message.content)[number], { type: 'text' }> =>
      part.type === 'text')
    .map((part) => part.text)
    .join('\n')
}

export function currentTurnMessages(messages: readonly Message[]): readonly Message[] {
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
 * Successful evidence must belong to the active user turn. An older session
 * result cannot justify answering a new request after one of its required
 * reads was blocked.
 */
export function hasSuccessfulToolEvidenceInCurrentTurn(
  messages: readonly Message[],
): boolean {
  return currentTurnMessages(messages).some((message) =>
    message.role === 'tool'
    && message.metadata?.[TOOL_RESULT_STATUS_METADATA_KEY] === 'success')
}

/**
 * A requested live action is evidenced by a tool result from this user turn,
 * even when the command itself reports failure. For example, a test process
 * exiting 1 is still fresh execution evidence that the agent must summarize.
 * Policy denial is handled as a separate terminal state before completion
 * guards run, so it cannot be misrepresented as a successful execution.
 */
export function hasToolResultEvidenceInCurrentTurn(
  messages: readonly Message[],
): boolean {
  return currentTurnMessages(messages).some((message) =>
    message.role === 'tool'
    && message.metadata?.[TOOL_RESULT_EXECUTION_OBSERVED_METADATA_KEY] === true)
}

/**
 * Return the latest policy rejection in this user turn. The metadata is added
 * by the daemon after policy evaluation and is never copied from tool output,
 * so a plugin cannot spoof a policy blocker by printing matching text.
 */
export function latestTrustedPolicyFailure(
  messages: readonly Message[],
  ignoredToolCallIds: ReadonlySet<string> = new Set(),
): TrustedPolicyFailure | null {
  const currentTurn = currentTurnMessages(messages)
  const toolNamesById = new Map<string, string>()
  for (const message of currentTurn) {
    for (const call of message.toolCalls ?? []) {
      toolNamesById.set(call.id, call.name)
    }
  }

  for (let index = currentTurn.length - 1; index >= 0; index -= 1) {
    const message = currentTurn[index]!
    const reason = message.metadata?.[POLICY_FAILURE_REASON_METADATA_KEY]
    const text = messageText(message)
    if (
      message.role !== 'tool'
      || (message.toolCallId ? ignoredToolCallIds.has(message.toolCallId) : false)
      || typeof reason !== 'string'
      || reason.trim().length === 0
      || !/^\s*Tool\s+[A-Za-z0-9_.-]+\s+blocked:/iu.test(text)
      || !text.includes(reason)
    ) {
      continue
    }
    return {
      reason,
      ...(message.toolCallId ? { toolCallId: message.toolCallId } : {}),
      ...(message.name
        ? { toolName: message.name }
        : message.toolCallId && toolNamesById.has(message.toolCallId)
          ? { toolName: toolNamesById.get(message.toolCallId)! }
          : {}),
    }
  }
  return null
}

function executableName(value: unknown): string {
  if (typeof value !== 'string') return ''
  return value.trim().split(/[\\/]/).at(-1)?.toLowerCase().replace(/\.exe$/u, '') ?? ''
}

function hasShellCommandFlag(executable: string, args: unknown): boolean {
  if (!Array.isArray(args)) return false
  const normalizedArgs = args.filter((arg): arg is string => typeof arg === 'string')
  if (executable === 'cmd') {
    return normalizedArgs.some((arg) => /^\/(?:c|k)$/iu.test(arg))
  }
  if (executable === 'powershell' || executable === 'pwsh') {
    return normalizedArgs.some((arg) => /^-(?:c|command|encodedcommand)$/iu.test(arg))
  }
  return normalizedArgs.some((arg) => /^-[A-Za-z]*c[A-Za-z]*$/u.test(arg))
}

function toolCallForFailure(
  messages: readonly Message[],
  failure: TrustedPolicyFailure,
) {
  if (!failure.toolCallId) return null
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const call = messages[index]?.toolCalls?.find((candidate) =>
      candidate.id === failure.toolCallId)
    if (call) return call
  }
  return null
}

/**
 * A shell wrapper rejected by ReadOnly is an invocation-shape failure, not
 * evidence that the requested observation itself is forbidden. The runtime
 * never unwraps or executes the script: it may only give the model one chance
 * to reconstruct the operation as a normal policy-checked tool call.
 */
export function isRepairableReadOnlyInvocationFailure(
  messages: readonly Message[],
  failure: TrustedPolicyFailure,
): boolean {
  if (
    failure.toolName !== 'terminal.run'
    || failure.reason !== READ_ONLY_TERMINAL_POLICY_REASON
  ) return false

  const call = toolCallForFailure(messages, failure)
  if (!call || call.name !== 'terminal.run') return false
  const executable = executableName(call.arguments.executable)
  return SHELL_WRAPPER_EXECUTABLES.has(executable)
    && hasShellCommandFlag(executable, call.arguments.args)
}

export function countReadOnlyInvocationRepairMessages(messages: readonly Message[]): number {
  return messages.filter((message) =>
    message.role === 'system'
    && message.metadata?.[READ_ONLY_INVOCATION_REPAIR_METADATA_KEY] === true)
    .length
}

export function readOnlyInvocationRepairToolCallIds(
  messages: readonly Message[],
): string[] {
  return messages.flatMap((message) => {
    const value = message.metadata?.[READ_ONLY_INVOCATION_REPAIR_TOOL_CALL_ID_METADATA_KEY]
    return message.role === 'system' && typeof value === 'string' && value
      ? [value]
      : []
  })
}

export function buildReadOnlyInvocationRepairMessage(
  failure: TrustedPolicyFailure,
): Message {
  return {
    role: 'system',
    content: [
      'The trusted runtime policy blocked a terminal.run shell wrapper in READ-ONLY autonomy.',
      `Exact policy reason: ${failure.reason}`,
      'This is one bounded opportunity to repair only the invocation shape; it does not grant permission to bypass policy.',
      'Reconstruct the requested observation from the user request, not by parsing or replaying the rejected shell script.',
      'If the same observation can use a registered read-only tool, or a direct policy-allowlisted executable plus argv, emit exactly one structured tool call now.',
      'For a bounded watch/follow where silence validly means no changes, use the tool timeoutMs field and timeoutOutcome observation_complete; use success_if_output only when captured stdout is required for a usable result.',
      'Once that observation window completes, interpret the result instead of adding a delay-only call or repeating the same observation.',
      'Do not use shell timeout commands, separators, pipelines, redirects, backgrounding, or trailing echo commands.',
      'Do not use another shell wrapper, process/delegation tool, or a state-changing substitute.',
      'If no policy-compliant equivalent exists, return INCOMPLETE with the exact policy reason and do not call a tool.',
    ].join('\n'),
    metadata: {
      [READ_ONLY_INVOCATION_REPAIR_METADATA_KEY]: true,
      ...(failure.toolCallId
        ? { [READ_ONLY_INVOCATION_REPAIR_TOOL_CALL_ID_METADATA_KEY]: failure.toolCallId }
        : {}),
    },
  }
}

export function buildReadOnlyPolicyBlockSynthesisMessage(
  failure: TrustedPolicyFailure,
): Message {
  return {
    role: 'system',
    content: [
      'The trusted runtime policy blocked a tool call in READ-ONLY autonomy.',
      `Blocked tool: ${failure.toolName ?? 'unknown'}.`,
      `Exact policy reason: ${failure.reason}`,
      'Give one concise final response now and do not request any more tools.',
      'If successful read-only evidence already satisfies the original request, start with ANSWER:, summarize that evidence, and clearly mention the blocked optional follow-up as a ReadOnly policy block.',
      'Otherwise start with INCOMPLETE:, include the exact policy reason verbatim, and clearly say that the required blocked action was not executed.',
      'Attribute evidence only to tools with successful result messages. Never claim that a blocked, failed, skipped, or unrequested tool ran.',
      'Do not call this PLAN mode and do not suggest exiting PLAN mode.',
    ].join('\n'),
  }
}

export function buildReadOnlyPolicyBlockedTurnOutput(
  failure: TrustedPolicyFailure,
  input: string,
): string {
  const tool = failure.toolName ?? 'tool'
  if (/[가-힣]/u.test(input)) {
    return [
      `INCOMPLETE: 읽기 전용 정책으로 ${tool} 실행이 차단되었습니다: ${failure.reason}`,
      '차단된 작업은 실행되지 않았으며, 현재의 읽기 전용 제한을 변경하거나 우회하지 않았습니다.',
    ].join(' ')
  }
  return [
    `INCOMPLETE: Read-only policy blocked ${tool}: ${failure.reason}`,
    'The blocked action was not executed, and the current read-only restriction was not changed or bypassed.',
  ].join(' ')
}
