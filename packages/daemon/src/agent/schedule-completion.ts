import type { Message } from '@sepilotd/core'
import { CURRENT_AGENT_TURN_USER_METADATA_KEY } from './turn-context.js'
import { TOOL_RESULT_STATUS_METADATA_KEY } from './memory-write-completion.js'
import {
  isExplicitScheduleCreateRequest,
  SCHEDULE_CREATE_TOOL_NAME,
} from './schedule-intent.js'

export { isExplicitScheduleCreateRequest, SCHEDULE_CREATE_TOOL_NAME }
export const CHANNEL_REMINDER_TOOL_NAME = 'memory.remind_at'
export const SCHEDULE_EVIDENCE_TOOL_NAMES = [
  SCHEDULE_CREATE_TOOL_NAME,
  CHANNEL_REMINDER_TOOL_NAME,
] as const

const SCHEDULE_COMPLETION_RECOVERY_REMINDER_KIND = 'schedule_completion'

export type ScheduleCompletionOutcome = 'success' | 'failed' | 'missing'

type ToolCallHistoryEntry = {
  tool: string
  status: 'success' | 'error'
}

function isScheduleEvidenceTool(name: string | undefined): boolean {
  return SCHEDULE_EVIDENCE_TOOL_NAMES.some((candidate) => candidate === name)
}

export function scheduleCompletionOutcomeFromHistory(
  history: readonly ToolCallHistoryEntry[] | undefined,
): ScheduleCompletionOutcome {
  const attempts = (history ?? []).filter((entry) => isScheduleEvidenceTool(entry.tool))
  if (attempts.length === 0) return 'missing'
  return attempts.some((entry) => entry.status === 'success') ? 'success' : 'failed'
}

export function scheduleCompletionOutcomeFromMessages(
  messages: readonly Message[],
): ScheduleCompletionOutcome {
  const currentTurnMessages = messagesAfterLatestUser(messages)
  const toolNamesById = new Map<string, string>()
  for (const message of currentTurnMessages) {
    for (const toolCall of message.toolCalls ?? []) {
      if (isScheduleEvidenceTool(toolCall.name)) toolNamesById.set(toolCall.id, toolCall.name)
    }
  }

  let attempted = false
  for (const message of currentTurnMessages) {
    if (message.role !== 'tool') continue
    const toolName = message.name
      ?? (message.toolCallId ? toolNamesById.get(message.toolCallId) : undefined)
    if (!isScheduleEvidenceTool(toolName)) continue
    attempted = true
    if (message.metadata?.[TOOL_RESULT_STATUS_METADATA_KEY] === 'success') return 'success'
  }
  return attempted ? 'failed' : 'missing'
}

export function availableScheduleEvidenceTools(
  availableToolNames: Iterable<string>,
): string[] {
  const available = new Set(availableToolNames)
  return SCHEDULE_EVIDENCE_TOOL_NAMES.filter((name) => available.has(name))
}

export function buildScheduleCompletionRecoveryMessage(
  availableToolNames: readonly string[],
): Message {
  const hasScheduler = availableToolNames.includes(SCHEDULE_CREATE_TOOL_NAME)
  const hasChannelReminder = availableToolNames.includes(CHANNEL_REMINDER_TOOL_NAME)
  return {
    role: 'system',
    metadata: { reminderKind: SCHEDULE_COMPLETION_RECOVERY_REMINDER_KIND },
    content: [
      '[Schedule completion guard]',
      'The user explicitly requested a future reminder or scheduled action, but this run has no successful scheduling-tool result. The previous confirmation draft was withheld from the user.',
      hasScheduler
        ? 'Call schedule_create exactly once now with the requested time and a self-contained instruction. This is the required path for Desktop/mobile personal chat.'
        : '',
      !hasScheduler && hasChannelReminder
        ? 'Call memory.remind_at exactly once now with the requested time and literal reminder text. Use it only because this channel-scoped turn can route the reminder back to the user.'
        : '',
      'Do not claim that the reminder was registered unless the tool returns success. On failure, report INCOMPLETE with the concrete error instead of promising future delivery.',
    ].filter(Boolean).join('\n'),
  }
}

export function countScheduleCompletionRecoveryPrompts(messages: readonly Message[]): number {
  return messagesAfterLatestUser(messages).filter((message) => (
    message.role === 'system'
    && message.metadata?.reminderKind === SCHEDULE_COMPLETION_RECOVERY_REMINDER_KIND
  )).length
}

export function buildScheduleCompletionFailureOutput(
  outcome: Exclude<ScheduleCompletionOutcome, 'success'>,
  userInput: string,
): string {
  if (/[가-힣]/u.test(userInput)) {
    return outcome === 'failed'
      ? 'INCOMPLETE: 일정 또는 알림 도구 호출이 실패하여 요청한 알림이 등록됐다고 확인할 수 없습니다.'
      : 'INCOMPLETE: 일정 또는 알림 도구의 성공 결과가 없어 요청한 알림이 등록됐다고 확인할 수 없습니다.'
  }
  return outcome === 'failed'
    ? 'INCOMPLETE: The scheduling tool failed, so the requested reminder was not confirmed registered.'
    : 'INCOMPLETE: No successful scheduling-tool result exists, so the requested reminder was not confirmed registered.'
}

function messagesAfterLatestUser(messages: readonly Message[]): readonly Message[] {
  const latestUserIndex = currentTurnUserMessageIndex(messages)
  return latestUserIndex >= 0 ? messages.slice(latestUserIndex + 1) : messages
}

function currentTurnUserMessageIndex(messages: readonly Message[]): number {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]
    if (
      message?.role === 'user'
      && message.metadata?.[CURRENT_AGENT_TURN_USER_METADATA_KEY] === true
    ) return index
  }
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === 'user') return index
  }
  return -1
}
