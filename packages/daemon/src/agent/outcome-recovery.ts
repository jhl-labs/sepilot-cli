import type { Message, TodoItem } from '@sepilotd/core'
import { parseTodoItems } from '../tools/todo.js'
import {
  APPROVAL_FAILURE_STATUS_METADATA_KEY,
  type ApprovalFailureStatus,
} from './approval-failure.js'
import {
  actionCompletionFailureReason,
  evaluateActionCompletion,
  type ActionCompletionEvaluation,
} from './action-completion.js'
import { POLICY_FAILURE_REASON_METADATA_KEY } from './policy-failure.js'

export type RunOutcomeStatus = 'complete' | 'needs_recovery'

export interface RunOutcomeEvaluation {
  status: RunOutcomeStatus
  reasons: string[]
  userRequest: string
  answerSnippet: string
  hasToolResultSinceLastUser: boolean
  availableToolNames: string[]
  actionCompletion: ActionCompletionEvaluation
}

export const MAX_OUTCOME_RECOVERY_REPAIRS = 6

const PLACEHOLDER_PATTERNS = [
  /\?\s*(?:원|%|주|개|달러|USD|KRW|시세|가격|평가|수익)/i,
  /(?:현재가|시세|가격|평가금액|수익금|수익률)\s*[:|]\s*\?\s*(?:원|%)?/i,
  /\b(?:TBD|N\/A)\b/i,
  /(?:현재가|시세|가격|평가금액|수익금|수익률)\s*[:|]\s*(?:unknown|unavailable|not available)\b/i,
  /\|\s*(?:unknown|unavailable|not available)\s*\|/i,
]

const TOOL_FAILURE_PATTERNS = [
  /^\[approval:(?:denied|needs-changes)\]/i,
  /^\[error:/i,
  /\bUnknown tool:/i,
  /^Tool execution requires approval \(no approval handler\)/i,
  /\bTool\s+[A-Za-z0-9_.-]+\s+blocked:/i,
  /\btool_use_failed\b/i,
]

const APPROVAL_FAILURE_STATUS_PATTERN = /^\[approval:(denied)\]/i
const REPORTED_APPROVAL_FAILURE_STATUS_PATTERN =
  /^\s*(?:(?:ANSWER|INCOMPLETE|FINAL)\s*:\s*)?\[approval:(denied)\]/i
const PERMANENT_ERROR_CODE_PATTERN = /^\[error:\s*([A-Z0-9_]+_PERMANENT)\]/i
const REPORTED_PERMANENT_ERROR_CODE_PATTERN =
  /(?:^|\r?\n)\s*INCOMPLETE\s*:[^\r\n]{0,512}\[error:\s*([A-Z0-9_]+_PERMANENT)\]/i

function extractMessageText(message: Message): string {
  if (typeof message.content === 'string') {
    return message.content
  }

  return message.content
    .filter((part): part is { type: 'text'; text: string } => part.type === 'text')
    .map((part) => part.text)
    .join('\n')
}

function lastUserMessageText(messages: Message[]): string {
  return extractMessageText(
    [...messages].reverse().find((message) => message.role === 'user') ?? {
      role: 'user',
      content: '',
    },
  )
}

function messagesSinceLastUser(messages: Message[]): Message[] {
  const lastUserIndex = messages
    .map((message, index) => ({ message, index }))
    .filter(({ message }) => message.role === 'user')
    .at(-1)?.index

  if (lastUserIndex == null) {
    return []
  }
  return messages.slice(lastUserIndex + 1)
}

/**
 * Latest todo list since the last user turn, taken from structured
 * `todowrite` tool-call arguments (first-class todo state) — never scraped
 * out of the tool's rendered output text.
 */
function latestTodoItems(messages: Message[]): TodoItem[] | null {
  for (const message of [...messages].reverse()) {
    if (message.role !== 'assistant' || !message.toolCalls?.length) continue
    for (const call of [...message.toolCalls].reverse()) {
      if (call.name !== 'todowrite') continue
      const items = parseTodoItems((call.arguments as { items?: unknown } | undefined)?.items)
      if (items) return items
    }
  }
  return null
}

function incompleteTodoSummary(items: TodoItem[] | null): string | null {
  if (!items) return null
  const incomplete = items.filter((item) => item.status !== 'completed' && item.status !== 'cancelled')
  if (incomplete.length === 0) return null
  const summary = normalizeSnippet(
    incomplete.slice(0, 5).map((item) => `[${item.status}] ${item.content}`).join('; '),
    220,
  )
  return summary || 'the todo list still has pending, in-progress, or blocked items'
}

function normalizeSnippet(text: string, limit = 280): string {
  const normalized = text.trim().replace(/\s+/g, ' ')
  if (!normalized) {
    return ''
  }
  return normalized.length > limit ? `${normalized.slice(0, limit - 1)}...` : normalized
}

function latestToolResultIsFailure(messages: Message[]): boolean {
  const latestToolMessage = [...messages].reverse().find((message) => message.role === 'tool')
  if (!latestToolMessage) {
    return false
  }
  const text = extractMessageText(latestToolMessage)
  return TOOL_FAILURE_PATTERNS.some((pattern) => pattern.test(text))
}

function approvalFailureStatus(text: string): ApprovalFailureStatus | null {
  const status = APPROVAL_FAILURE_STATUS_PATTERN.exec(text.trim())?.[1]?.toLowerCase()
  return status === 'denied' ? status : null
}

function recordedApprovalFailureStatus(message: Message): ApprovalFailureStatus | null {
  const status = message.metadata?.[APPROVAL_FAILURE_STATUS_METADATA_KEY]
  return status === 'denied' ? status : null
}

function reportedApprovalFailureStatus(text: string): ApprovalFailureStatus | null {
  const status = REPORTED_APPROVAL_FAILURE_STATUS_PATTERN.exec(text)?.[1]?.toLowerCase()
  return status === 'denied' ? status : null
}

function clearlyReportsLatestApprovalFailure(messages: Message[], answer: string): boolean {
  const latestToolMessage = [...messages].reverse().find((message) => message.role === 'tool')
  if (!latestToolMessage) return false

  const latestStatus = approvalFailureStatus(extractMessageText(latestToolMessage))
  return latestStatus !== null
    && recordedApprovalFailureStatus(latestToolMessage) === latestStatus
    && reportedApprovalFailureStatus(answer) === latestStatus
}

export function clearlyReportsLatestPermanentFailure(messages: Message[], answer: string): boolean {
  const latestToolMessage = [...messages].reverse().find((message) => message.role === 'tool')
  if (!latestToolMessage) return false
  const toolCode = PERMANENT_ERROR_CODE_PATTERN
    .exec(extractMessageText(latestToolMessage).trim())?.[1]?.toUpperCase()
  const reportedCode = REPORTED_PERMANENT_ERROR_CODE_PATTERN
    .exec(answer)?.[1]?.toUpperCase()
  return Boolean(toolCode && reportedCode && toolCode === reportedCode)
}

export function clearlyReportsLatestPolicyFailure(messages: Message[], answer: string): boolean {
  const latestToolMessage = [...messages].reverse().find((message) => message.role === 'tool')
  if (!latestToolMessage || !/^\s*(?:ANSWER|INCOMPLETE)\s*:/iu.test(answer)) return false

  const reason = latestToolMessage.metadata?.[POLICY_FAILURE_REASON_METADATA_KEY]
  const toolText = extractMessageText(latestToolMessage)
  const trustedPolicyFailure = typeof reason === 'string'
    && reason.trim().length > 0
    && /^\s*Tool\s+[A-Za-z0-9_.-]+\s+blocked:/iu.test(toolText)
    && toolText.includes(reason)
  if (!trustedPolicyFailure) return false

  if (/^\s*INCOMPLETE\s*:/iu.test(answer) && answer.includes(reason)) return true

  const hasSuccessfulEvidence = messages.some((message) =>
    message.role === 'tool' && message.metadata?.toolResultStatus === 'success')
  const reportsReadOnlyBlock = /(?:ReadOnly|read[- ]?only|읽기\s*전용)/iu.test(answer)
    && /(?:block(?:ed)?|차단|거부)/iu.test(answer)
  return hasSuccessfulEvidence && reportsReadOnlyBlock
}

export function evaluateRunOutcome(options: {
  messages: Message[]
  content: string
  availableToolNames?: string[]
  commandEvidenceToolNames?: string[]
  inferActionsFromWording?: boolean
  /**
   * Structured todo loop state (AgentState.todoList). When provided it wins
   * over deriving the list from todowrite tool calls in the history.
   */
  todoList?: TodoItem[]
}): RunOutcomeEvaluation {
  const { messages, content } = options
  const availableToolNames = options.availableToolNames ?? []
  const userRequest = lastUserMessageText(messages)
  const answer = content.trim()
  const answerSnippet = normalizeSnippet(answer)
  const recentMessages = messagesSinceLastUser(messages)
  const hasToolResultSinceLastUser = recentMessages.some((message) => message.role === 'tool')
  const incompleteTodo = incompleteTodoSummary(options.todoList ?? latestTodoItems(recentMessages))
  const actionCompletion = evaluateActionCompletion({
    messages,
    content: answer,
    userInput: userRequest,
    availableToolNames,
    commandEvidenceToolNames: options.commandEvidenceToolNames,
    inferActionsFromWording: options.inferActionsFromWording,
  })
  const reasons: string[] = []

  if (PLACEHOLDER_PATTERNS.some((pattern) => pattern.test(answer))) {
    reasons.push('the answer still contains unresolved placeholders such as ? values')
  }

  if (incompleteTodo) {
    reasons.push(`the latest todo list still has incomplete items: ${incompleteTodo}`)
  }

  const reportsApprovalBlocker = clearlyReportsLatestApprovalFailure(recentMessages, answer)
  const reportsPermanentBlocker = clearlyReportsLatestPermanentFailure(recentMessages, answer)
  const reportsPolicyBlocker = clearlyReportsLatestPolicyFailure(recentMessages, answer)
  const actionReason = actionCompletionFailureReason(actionCompletion)
  if (
    actionReason
    && !reportsApprovalBlocker
    && !reportsPermanentBlocker
    && !reportsPolicyBlocker
  ) {
    reasons.push(actionReason)
  }

  if (
    latestToolResultIsFailure(recentMessages)
    && !reportsApprovalBlocker
    && !reportsPermanentBlocker
    && !reportsPolicyBlocker
  ) {
    reasons.push('the latest tool result failed and the run stopped before a recovery tool call or clear blocker')
  }

  return {
    status: reasons.length > 0 ? 'needs_recovery' : 'complete',
    reasons,
    userRequest: normalizeSnippet(userRequest, 220),
    answerSnippet,
    hasToolResultSinceLastUser,
    availableToolNames,
    actionCompletion,
  }
}

export function shouldRepairRunOutcome(options: {
  evaluation: RunOutcomeEvaluation
  repairedCount: number
  isLastIteration?: boolean
  maxRepairs?: number
}): boolean {
  const {
    evaluation,
    repairedCount,
    isLastIteration = false,
    maxRepairs = MAX_OUTCOME_RECOVERY_REPAIRS,
  } = options

  return evaluation.status === 'needs_recovery' && repairedCount < maxRepairs && !isLastIteration
}

export function buildRunOutcomeRecoveryMessage(evaluation: RunOutcomeEvaluation): Message {
  const reasonLines = evaluation.reasons.map((reason) => `- ${reason}`)
  const toolHint =
    evaluation.availableToolNames.length > 0
      ? `Available tools include: ${evaluation.availableToolNames.slice(0, 20).join(', ')}.`
      : 'No tools are available in this run.'

  return {
    role: 'system',
    content: [
      '[Outcome supervisor] Your previous assistant reply did not fully satisfy the user request.',
      'Reasons:',
      ...reasonLines,
      `User request: ${evaluation.userRequest || '(unavailable)'}`,
      evaluation.answerSnippet ? `Previous answer excerpt: ${evaluation.answerSnippet}` : '',
      toolHint,
      'Retry now with a different approach. Do not send another answer with placeholders, stale values, or unresolved table cells.',
      evaluation.actionCompletion.hasUnexecutedToolText
        ? 'The prior tool_call:/terminal.run: prose was not executed. Emit a real structured tool call using an exact available tool name; never simulate a tool call in assistant text.'
        : '',
      'If a tool failed because the name was invalid, retry with an exact available tool name from the list.',
      'If a prior tool failed, change the tool, query, URL, or arguments instead of repeating the exact same call.',
      'If the prior reply was only a plan or progress report, use the final-answer protocol: call the next tool now, or reply with INCOMPLETE: and the concrete blocker.',
      'Only ask the user for data when it is private, unavailable to tools, or genuinely impossible to verify; in that case state the exact blocker.',
    ]
      .filter(Boolean)
      .join('\n'),
  }
}
