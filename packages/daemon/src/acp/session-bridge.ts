import {
  streamDaemonChatEvents,
  type DaemonChatStreamPayload,
  type DaemonClient,
} from '@sepilotd/api-client'
import type { AcpBridge } from './handlers.js'

export interface AcpSessionBridgeDeps {
  client: Pick<DaemonClient, 'chatStream'>
}

type AcpSessionUpdate = Record<string, unknown>
type AcpUpdateListener = (update: AcpSessionUpdate) => void

function textContent(text: string): { type: 'text'; text: string } {
  return { type: 'text', text }
}

function toolCallTitle(payload: Extract<DaemonChatStreamPayload, { type: 'tool_call' }>): string {
  return payload.toolCall?.name || 'tool'
}

function mapDaemonStreamPayload(payload: DaemonChatStreamPayload): AcpSessionUpdate | null {
  if (!('type' in payload)) return null

  switch (payload.type) {
    case 'text_delta':
      return {
        sessionUpdate: 'agent_message_chunk',
        content: textContent(payload.text ?? ''),
      }
    case 'message':
      return {
        sessionUpdate: 'agent_message_chunk',
        content: textContent(payload.content ?? ''),
      }
    case 'thinking':
      return {
        sessionUpdate: 'agent_thought_chunk',
        content: textContent(payload.content ?? ''),
      }
    case 'tool_call':
      return {
        sessionUpdate: 'tool_call',
        toolCallId: payload.toolCall?.id ?? 'tool',
        title: toolCallTitle(payload),
        status: 'pending',
        content: [],
        locations: [],
        rawInput: payload.toolCall?.arguments ?? {},
      }
    case 'tool_result':
      return {
        sessionUpdate: 'tool_call_update',
        toolCallId: payload.toolCallId,
        status: payload.status === 'success' ? 'completed' : 'failed',
        content: [{
          type: 'content',
          content: textContent(payload.output ?? ''),
        }],
        rawOutput: { output: payload.output ?? '', status: payload.status },
      }
    case 'error':
      throw new Error(payload.error?.message ?? 'daemon chat stream failed')
    default:
      return null
  }
}

/**
 * Bridge between ACP handlers and a running daemon. Prefer the daemon chat SSE
 * endpoint so ACP clients receive standard `session/update` notifications
 * while the turn is still running.
 */
export function createAcpSessionBridge(deps: AcpSessionBridgeDeps): AcpBridge {
  const controllers = new Map<string, AbortController>()
  const listeners = new Map<string, Set<AcpUpdateListener>>()

  function emit(threadId: string, update: AcpSessionUpdate): void {
    const active = listeners.get(threadId)
    if (!active) return
    for (const listener of [...active]) {
      listener(update)
    }
  }

  return {
    subscribe(threadId, cb) {
      const typed = cb as AcpUpdateListener
      let active = listeners.get(threadId)
      if (!active) {
        active = new Set()
        listeners.set(threadId, active)
      }
      active.add(typed)
      return () => {
        const current = listeners.get(threadId)
        if (!current) return
        current.delete(typed)
        if (current.size === 0) {
          listeners.delete(threadId)
        }
      }
    },

    async send(threadId, content) {
      const existing = controllers.get(threadId)
      existing?.abort()
      const controller = new AbortController()
      controllers.set(threadId, controller)
      try {
        const response = await deps.client.chatStream(
          content,
          threadId,
          undefined,
          { signal: controller.signal },
        )
        if (!response.ok) {
          throw new Error(`${response.status}: ${await response.text()}`)
        }
        for await (const payload of streamDaemonChatEvents(response)) {
          const update = mapDaemonStreamPayload(payload)
          if (update) {
            emit(threadId, update)
          }
        }
      } finally {
        if (controllers.get(threadId) === controller) {
          controllers.delete(threadId)
        }
      }
    },

    async cancel(threadId) {
      const controller = controllers.get(threadId)
      if (controller) {
        controller.abort()
        controllers.delete(threadId)
      }
    },
  }
}
