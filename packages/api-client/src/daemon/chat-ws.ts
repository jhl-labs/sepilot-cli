import type { DaemonChatStreamPayload, DaemonWsAgentEvent, DaemonWsEvent } from './types.js'

function normalizeAgentWsEvent(event: DaemonWsAgentEvent): DaemonChatStreamPayload {
  const { type, ...payload } = event
  return {
    ...payload,
    type: type.slice('agent.'.length),
  } as DaemonChatStreamPayload
}

export function normalizeWsEvent(event: DaemonWsEvent): DaemonChatStreamPayload | null {
  switch (event.type) {
    case 'chat.session':
      return { sessionId: event.sessionId }
    case 'chat.artifacts':
      return { artifacts: event.artifacts, sessionId: event.sessionId }
    case 'warning':
      return event
    case 'error':
      return { type: 'error', error: event.error }
    case 'pong':
      return null
    default:
      return normalizeAgentWsEvent(event)
  }
}
