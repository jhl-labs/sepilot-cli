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
import type { TurnOutcome } from '../utils/turn-outcome.js'

export interface LiveStreamSlot {
  assistantId: string
  agentState: SurfaceAgentState
  messages: SurfaceMessage[]
  controller: ReturnType<typeof createStreamEventController>
  stopReason?: RunStopReason | null
  outcome?: TurnOutcome
}

export function handleAbortOutcome(opts: {
  error: unknown
  liveStreamRef: MutableRefObject<LiveStreamSlot | null>
  dispatch: Dispatch<ChatAction>
}): boolean {
  const abortMeta = getDaemonChatRunAbortMeta(opts.error)
  if (!abortMeta) return false

  const liveStream = opts.liveStreamRef.current
  opts.dispatch({ type: 'SET_TURN_OUTCOME', outcome: { kind: abortMeta.kind } })
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
      const current = isCurrent()
      if (!current) return
      opts.dispatch({ type: 'ERROR', message: next, outcome: current.outcome })
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
