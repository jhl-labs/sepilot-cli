// Type guards for the surface Message union.
//
// Message is currently a flat type with optional fields per role:
//
//   interface Message {
//     role: 'user' | 'assistant' | 'tool' | 'context' | 'todo' | 'system'
//     content: string
//     toolName?: string         // only meaningful when role === 'tool'
//     contextItems?: ...        // only meaningful when role === 'context'
//     todoItems?: ...           // only meaningful when role === 'todo'
//     ...
//   }
//
// The compiler does not stop a caller from doing
// `(message as Message).toolName` when role is `'user'`. Migrating to a
// proper tagged union is a 99-call-site change across cli/web/desktop and
// will land separately. In the meantime these guards give callers a safe
// way to narrow Message and have the compiler check tool-only / context-
// only / todo-only field access — without touching the Message definition.

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
  // toolName is the only field the producers always set when role === 'tool'.
  // We check both the role and that the producer set the name to defend
  // against half-built messages still in flight on the surface side.
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
