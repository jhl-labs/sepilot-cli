import { randomUUID } from 'node:crypto'
import type { SessionEvent } from '@sepilotd/core'

function collectCompletedToolCallIds(events: SessionEvent[]): Set<string> {
  return new Set(
    events
      .filter(
        (event): event is Extract<SessionEvent, { type: 'tool_result' }> =>
          event.type === 'tool_result',
      )
      .map((event) => event.toolCallId),
  )
}

function collectResolvedApprovalRequestIds(events: SessionEvent[]): Set<string> {
  return new Set(
    events
      .filter(
        (event): event is Extract<SessionEvent, { type: 'approval_response' }> =>
          event.type === 'approval_response',
      )
      .map((event) => event.requestId),
  )
}

function isTransferableEvent(
  event: SessionEvent,
  completedToolCallIds: Set<string>,
  resolvedApprovalRequestIds: Set<string>,
): boolean {
  switch (event.type) {
    case 'user_message':
    case 'assistant_message':
    case 'memory_context':
    case 'context_compact':
    case 'provider_attempt':
    case 'run_contract':
    case 'todo_list':
    case 'delegation_state':
    case 'delegation_result':
    case 'approval_response':
    case 'auto_approval':
    case 'cowork_plan':
    case 'cowork_task_start':
    case 'cowork_task_complete':
    case 'cowork_task_failed':
    case 'cowork_synthesizing':
    case 'cowork_discuss_request':
    case 'cowork_discuss_response':
      return true
    case 'approval_request':
      return resolvedApprovalRequestIds.has(event.id)
    case 'tool_call':
      return completedToolCallIds.has(event.id)
    case 'tool_result':
      return true
    default:
      return false
  }
}

export function buildImportedSessionEvents(
  events: SessionEvent[],
  importedAtMs = Date.now(),
): SessionEvent[] {
  const completedToolCallIds = collectCompletedToolCallIds(events)
  const resolvedApprovalRequestIds = collectResolvedApprovalRequestIds(events)
  const transferableEvents = events.filter((event) =>
    isTransferableEvent(event, completedToolCallIds, resolvedApprovalRequestIds),
  )

  const toolCallIdMap = new Map<string, string>()
  const resolveToolCallId = (sourceToolCallId: string): string => {
    const existing = toolCallIdMap.get(sourceToolCallId)
    if (existing) {
      return existing
    }

    const nextId = randomUUID()
    toolCallIdMap.set(sourceToolCallId, nextId)
    return nextId
  }

  return transferableEvents.map((event, index) => {
    const timestamp = new Date(importedAtMs + index).toISOString()

    switch (event.type) {
      case 'user_message':
        // Attachment ids are capabilities in the source daemon's authenticated
        // file registry. They are not transferable with a JSON-only share and
        // must not be allowed to alias an unrelated local upload after import.
        return {
          type: 'user_message',
          id: randomUUID(),
          timestamp,
          content: event.content,
        }
      case 'tool_call': {
        const id = resolveToolCallId(event.id)
        return {
          ...event,
          id,
          timestamp,
        }
      }
      case 'tool_result':
        return {
          ...event,
          id: randomUUID(),
          timestamp,
          toolCallId: resolveToolCallId(event.toolCallId),
        }
      default:
        return {
          ...event,
          id: randomUUID(),
          timestamp,
        }
    }
  })
}
