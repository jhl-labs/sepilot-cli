import type { Message, ISessionStore, PlannerWorkingMemory, TodoItem } from '@sepilotd/core'
import { getCompactContinuationMessage } from './session-compaction.js'
import { summarizeToolOutputForAgentContext } from './tool-output.js'
import { coalesceAdjacentAssistantToolCallMessages } from './tool-protocol-context.js'
import { replayConversationTextControls } from './turn-context.js'
import type { AgentSeedContract } from './graph/types.js'
import {
  buildStateBoard,
  formatStateBoard,
  isStateBoardEnabled,
  type AgentStateBoardSource,
} from './graph/state-board.js'

export async function loadSessionContext(
  sessionStore: ISessionStore,
  sessionId: string,
  maxMessages?: number,
): Promise<Message[]> {
  const events = await sessionStore.getEvents(sessionId)
  const messages: Message[] = []
  let rehydratedContract: AgentSeedContract | undefined
  let rehydratedTodos: TodoItem[] | undefined
  let rehydratedPlannerMemory: PlannerWorkingMemory | undefined
  const toolNameByCallId = new Map<string, string>()
  const messageCountAfterEvent = new Map<string, number>()

  for (const event of events) {
    if (event.type === 'context_compact' && event.throughEventId && messageCountAfterEvent.has(event.throughEventId)) {
      const tail = messages.slice(messageCountAfterEvent.get(event.throughEventId)!)
      messages.length = 0
      messages.push({
        role: 'system',
        content: getCompactContinuationMessage(event.summary, {
          recentMessagesPreserved: (event.preservedMessages?.length ?? 0) > 0,
          suppressFollowUpQuestions: true,
        }),
      })
      if (event.preservedMessages?.length) messages.push(...event.preservedMessages)
      messages.push(...tail)
      messageCountAfterEvent.set(event.id, messages.length)
      continue
    }
    switch (event.type) {
      case 'user_message':
        messages.push({ role: 'user', content: replayConversationTextControls(event.content) })
        break
      case 'run_contract':
        rehydratedContract = event.contract
        break
      case 'todo_list':
        rehydratedTodos = event.items
        break
      case 'planner_working_memory_updated':
        rehydratedPlannerMemory = event.workingMemory
        break
      case 'assistant_message':
        messages.push({ role: 'assistant', content: replayConversationTextControls(event.content) })
        break
      case 'tool_call':
        toolNameByCallId.set(event.id, event.tool)
        messages.push({
          role: 'assistant',
          content: '',
          toolCalls: [{ id: event.id, name: event.tool, arguments: event.input }],
        })
        break
      case 'tool_result':
        {
          const toolName = toolNameByCallId.get(event.toolCallId)
          messages.push({
            role: 'tool',
            content: summarizeToolOutputForAgentContext(
              toolName ?? '',
              event.output,
              event.toolCallId,
            ),
            toolCallId: event.toolCallId,
            ...(toolName ? { name: toolName } : {}),
            metadata: {
              ...(event.metadata ?? {}),
              status: event.status,
            },
          })
        }
        break
      case 'context_compact':
        messages.length = 0
        messages.push({
          role: 'system',
          content: getCompactContinuationMessage(event.summary, {
            recentMessagesPreserved: (event.preservedMessages?.length ?? 0) > 0,
            suppressFollowUpQuestions: true,
          }),
        })
        if (event.preservedMessages?.length) {
          messages.push(...event.preservedMessages)
        }
        break
    }
    messageCountAfterEvent.set(event.id, messages.length)
  }

  const limit = maxMessages ?? DEFAULT_SESSION_CONTEXT_MAX_MESSAGES
  const normalizedMessages = coalesceAdjacentAssistantToolCallMessages(messages)
  const trimmed = stripStalePlannerBlocks(trimMessagesToLimit(normalizedMessages, limit))
  const boardMessage = buildRehydratedBoardMessage(
    rehydratedContract,
    rehydratedTodos,
    rehydratedPlannerMemory,
  )
  return boardMessage ? [boardMessage, ...trimmed] : trimmed
}

export function buildRehydratedBoardMessage(
  contract: AgentSeedContract | undefined,
  todos: TodoItem[] | undefined,
  plannerWorkingMemory: PlannerWorkingMemory | undefined,
): Message | null {
  if (!isStateBoardEnabled()) return null
  if (!contract && !(todos && todos.length > 0) && !plannerWorkingMemory) return null
  const source: AgentStateBoardSource = {
    seedContract: contract,
    todoList: todos,
    plannerWorkingMemory,
  }
  const board = buildStateBoard(source)
  board.evidenceSection = null
  const formatted = formatStateBoard(board)
  return formatted ? { role: 'system', content: formatted } : null
}

export const DEFAULT_SESSION_CONTEXT_MAX_MESSAGES = 50

export function resolveSessionContextMaxMessages(): number {
  const raw = Number(process.env.SEPILOTD_SESSION_CONTEXT_MAX_MESSAGES)
  if (Number.isFinite(raw) && raw >= 4) {
    return Math.floor(raw)
  }
  return DEFAULT_SESSION_CONTEXT_MAX_MESSAGES
}

export function trimMessagesToLimit(
  messages: Message[],
  maxMessages: number | undefined,
): Message[] {
  if (maxMessages === undefined || messages.length <= maxMessages) {
    return messages
  }
  let keepFrom = messages.length - maxMessages
  while (keepFrom > 0 && messages[keepFrom]?.role === 'tool') {
    keepFrom -= 1
  }
  return messages.slice(keepFrom)
}

const PLANNER_BLOCK_REGEX = /\n?```(?:planner|json:planner)\s*\n[\s\S]*?\n?```\s*$/i

export function stripStalePlannerBlocks(messages: Message[]): Message[] {
  let lastAssistantIdx = -1
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i]
    if (message.role !== 'assistant' || typeof message.content !== 'string') continue
    if (message.toolCalls && message.toolCalls.length > 0) continue
    if (message.content.trim().length === 0) continue
    if (lastAssistantIdx < 0) lastAssistantIdx = i
    if (PLANNER_BLOCK_REGEX.test(message.content)) {
      lastAssistantIdx = i
      break
    }
  }
  if (lastAssistantIdx < 0) return messages
  let changed = false
  const next = messages.map((message, i) => {
    if (i === lastAssistantIdx) return message
    if (message.role !== 'assistant' || typeof message.content !== 'string') return message
    if (!PLANNER_BLOCK_REGEX.test(message.content)) return message
    changed = true
    return { ...message, content: message.content.replace(PLANNER_BLOCK_REGEX, '').trimEnd() }
  })
  return changed ? next : messages
}

export function estimateTokens(messages: Message[]): number {
  let chars = 0
  for (const msg of messages) {
    if (typeof msg.content === 'string') chars += msg.content.length
    else if (Array.isArray(msg.content)) {
      for (const part of msg.content) {
        if (part.type === 'text') chars += part.text.length
      }
    }
  }
  return Math.ceil(chars / 4)
}
