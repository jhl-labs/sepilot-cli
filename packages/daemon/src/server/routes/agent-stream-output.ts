import type { AgentEvent } from '@sepilotd/core'
import type { RuntimeServices } from '../runtime.js'
import { createAgentOutputTracker } from '../../agent/event-output.js'
import { persistAgentSessionEvent } from '../session-events.js'

export interface AgentStreamOutputTracker {
  consume: (event: AgentEvent) => void
  finalContent: () => string
  providerFailureContent: (error: { code?: string; message?: string }) => string | null
  syntheticMessageEvent: () => Extract<AgentEvent, { type: 'message' }> | null
  persistSyntheticMessage: (
    runtime: RuntimeServices,
    sessionId: string,
    send?: (event: string, data: unknown) => void,
  ) => Promise<string>
}

export function createAgentStreamOutputTracker(): AgentStreamOutputTracker {
  const tracker = createAgentOutputTracker()

  return {
    consume: tracker.consume,
    finalContent: tracker.finalContent,
    providerFailureContent: tracker.providerFailureContent,
    syntheticMessageEvent: tracker.syntheticMessageEvent,
    async persistSyntheticMessage(runtime, sessionId, send) {
      const syntheticMessage = tracker.syntheticMessageEvent()
      if (!syntheticMessage) {
        return tracker.finalContent()
      }

      send?.(syntheticMessage.type, syntheticMessage)
      await persistAgentSessionEvent(runtime.sessions, sessionId, syntheticMessage)
      return syntheticMessage.content
    },
  }
}
