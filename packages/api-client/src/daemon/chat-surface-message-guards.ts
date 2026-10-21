import type {
  ApprovalState,
  Message,
  ToolStatus,
} from './chat-surface-types.js'
import type { TodoItem } from '@sepilotd/core'
import type { DaemonMemoryContextItem } from './types.js'

export interface ToolMessageView extends Message {
  role: 'tool'
  toolName: string
  toolStatus?: ToolStatus
  toolResult?: string
  toolInput?: Record<string, unknown>
  toolMeta?: string
  toolNeedsApproval?: boolean
  approvalRequestId?: string
  approvalState?: ApprovalState
  resumeAvailable?: boolean
}

export interface ContextMessageView extends Message {
  role: 'context'
  contextItems: DaemonMemoryContextItem[]
}

export interface TodoMessageView extends Message {
  role: 'todo'
  todoItems: TodoItem[]
}

export interface AssistantMessageView extends Message {
  role: 'assistant'
}

export interface UserMessageView extends Message {
  role: 'user'
}

export interface SystemMessageView extends Message {
  role: 'system'
}

export function isToolMessage(message: Message): message is ToolMessageView {
  return message.role === 'tool' && typeof message.toolName === 'string'
}

export function isContextMessage(
  message: Message,
): message is ContextMessageView {
  return (
    message.role === 'context'
    && Array.isArray(message.contextItems)
  )
}

export function isTodoMessage(message: Message): message is TodoMessageView {
  return message.role === 'todo' && Array.isArray(message.todoItems)
}

export function isAssistantMessage(
  message: Message,
): message is AssistantMessageView {
  return message.role === 'assistant'
}

export function isUserMessage(message: Message): message is UserMessageView {
  return message.role === 'user'
}

export function isSystemMessage(
  message: Message,
): message is SystemMessageView {
  return message.role === 'system'
}
