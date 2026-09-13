// Shared shape for the "live stream slot" useChat keeps while an agent
// run is in flight. Pulled out of useChat.ts so abort-outcome and other
// stream-related helpers can refer to the same type without importing
// the giant hook file.

import type { Dispatch, MutableRefObject } from 'react'
import {
  getDaemonChatRunAbortMeta,
  type createStreamEventController,
  type AgentState as SurfaceAgentState,
  type Message as SurfaceMessage,
  type RunStopReason,
  type StreamEventControllerBindings,
} from '@sepilotd/api-client'
import type { Message } from '../types.js'
import { cliMessageToSurfaceMessage } from '../utils/surface.js'
import type { ChatAction } from './chat-reducer.js'

export interface LiveStreamSlot {
  assistantId: string
  agentState: SurfaceAgentState
  messages: SurfaceMessage[]
  controller: ReturnType<typeof createStreamEventController>
  /** Structured stop cause once the daemon's terminal frame arrived. */
  stopReason?: RunStopReason | null
}

/**
 * Translate a daemon-chat abort error into the surface-side cleanup it
 * implies: mark the live stream as done, fill in fallback assistant
 * content if the model never produced any, sync the surface messages,
 * and emit the user-facing abort detail message. Returns true when the
 * error was actually a recognised abort (so the caller can swallow it
 * instead of bubbling it to error reporting). Returns false for any
 * other error so the caller routes it through the normal error path.
 *
 * Free function: the live stream ref and dispatcher are passed in so
 * the helper can be unit-tested without rendering useChat.
 */
export function handleAbortOutcome(opts: {
  error: unknown
  liveStreamRef: MutableRefObject<LiveStreamSlot | null>
  dispatch: Dispatch<ChatAction>
}): boolean {
  const abortMeta = getDaemonChatRunAbortMeta(opts.error)
  if (!abortMeta) return false

  const liveStream = opts.liveStreamRef.current
  if (liveStream) {
    liveStream.agentState = 'done'
    if (!liveStream.controller.getAssistantContent()) {
      liveStream.controller.updateAssistant(abortMeta.assistantFallbackText)
    }
    opts.dispatch({
      type: 'SYNC_SURFACE_MESSAGES',
      messages: liveStream.messages,
      assistantId: liveStream.assistantId,
      agentState: 'done',
    })
    opts.liveStreamRef.current = null
  }

  opts.dispatch({ type: 'SET_STREAM_STATUS', status: null })
  opts.dispatch({ type: 'SYSTEM_MESSAGE', content: abortMeta.detail })
  return true
}

/**
 * Build the initial surface-message list a fresh live stream starts
 * with: every existing cli message converted to a surface message,
 * optionally a freshly-emitted user turn, and a placeholder assistant
 * message at the tail. Pure: no refs touched, no dispatch fired.
 */
export function buildInitialLiveStreamMessages(opts: {
  storedMessages: Message[]
  assistantId: string
  userMessage?: { id: string; content: string }
}): SurfaceMessage[] {
  return [
    ...opts.storedMessages
      .map((message) => cliMessageToSurfaceMessage(message))
      .filter((message): message is SurfaceMessage => message !== null),
    ...(opts.userMessage
      ? [{
          id: opts.userMessage.id,
          role: 'user' as const,
          content: opts.userMessage.content,
        }]
      : []),
    {
      id: opts.assistantId,
      role: 'assistant' as const,
      content: '',
    },
  ]
}

/**
 * Build the four-setter bindings object the stream controller writes
 * back through. setMessages and setAgentState only fire when the live
 * stream slot still points at the same assistantId — preventing a
 * late-arriving event from a superseded run from clobbering the
 * current one. setActivities / setStatus / setError go straight to
 * the dispatcher.
 */
export function createLiveStreamBindings(opts: {
  assistantId: string
  liveStreamRef: MutableRefObject<LiveStreamSlot | null>
  dispatch: Dispatch<ChatAction>
  sanitizeAssistantContent?: (content: string, stopReason: RunStopReason | null) => string
}): StreamEventControllerBindings {
  const isCurrent = (): LiveStreamSlot | null => {
    const current = opts.liveStreamRef.current
    return current && current.assistantId === opts.assistantId ? current : null
  }
  const sanitizeMessages = (messages: SurfaceMessage[]): SurfaceMessage[] => {
    const sanitizeAssistantContent = opts.sanitizeAssistantContent
    if (!sanitizeAssistantContent) return messages
    return messages.map((message) => {
      if (message.id !== opts.assistantId || message.role !== 'assistant') {
        return message
      }
      const content = sanitizeAssistantContent(
        message.content,
        opts.liveStreamRef.current?.stopReason ?? null,
      )
      return content === message.content ? message : { ...message, content }
    })
  }
  return {
    setMessages: (next) => {
      const current = isCurrent()
      if (!current) return
      const resolved = typeof next === 'function'
        ? next(current.messages)
        : next
      current.messages = sanitizeMessages(resolved)
      opts.dispatch({
        type: 'SYNC_SURFACE_MESSAGES',
        messages: current.messages,
        assistantId: current.assistantId,
        agentState: current.agentState,
      })
    },
    setActivities: (next) => {
      opts.dispatch({ type: 'SET_ACTIVITIES', activities: next })
    },
    setAgentState: (next) => {
      const current = isCurrent()
      if (!current) return
      current.agentState = next
    },
    setStatus: (next) => {
      opts.dispatch({ type: 'SET_STREAM_STATUS', status: next })
    },
    setThinking: (content) => {
      if (!isCurrent()) return
      opts.dispatch({ type: 'SET_THINKING', content })
    },
    setError: (next) => {
      if (!next) return
      opts.dispatch({ type: 'ERROR', message: next })
    },
    setStopReason: (reason) => {
      const current = isCurrent()
      if (!current) return
      current.stopReason = reason
    },
    setContextUsage: (context) => {
      if (!isCurrent()) return
      opts.dispatch({ type: 'SET_CONTEXT_USAGE', context })
    },
    setProviderWait: (wait) => {
      if (!isCurrent()) return
      opts.dispatch({ type: 'SET_PROVIDER_WAIT', wait })
    },
  }
}
