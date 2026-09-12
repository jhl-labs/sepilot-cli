import {
  resolveDaemonChatRunAbortReason,
} from './chat-run-control.js'
import {
  reconnectingJsonSseStream,
  streamDaemonChatEvents,
} from './stream.js'
import type {
  DaemonArtifact,
  DaemonChatResult,
  DaemonChatStreamPayload,
} from './types.js'
import type {
  DaemonChatEventHandler,
  DaemonStreamEventHandler,
  SseChatParams,
} from './chat-transport-types.js'

export function createIncompleteDaemonChatStreamError(): Error {
  return new Error('Daemon chat stream ended before completion.')
}

export function isTerminalDaemonChatPayload(
  payload: DaemonChatStreamPayload,
): payload is Extract<DaemonChatStreamPayload, { type: 'done' | 'error' }> {
  return 'type' in payload
    && (payload.type === 'done' || payload.type === 'error')
}

export async function forwardDaemonStream<T>(
  response: Response,
  onEvent: DaemonStreamEventHandler<T>,
  options?: {
    isTerminalEvent?: (event: T) => boolean
  },
): Promise<void> {
  let sawTerminalEvent = false

  for await (const event of streamDaemonChatEvents<T>(response)) {
    if (options?.isTerminalEvent?.(event)) {
      sawTerminalEvent = true
    }
    await onEvent(event)
  }

  if (options?.isTerminalEvent && !sawTerminalEvent) {
    throw createIncompleteDaemonChatStreamError()
  }
}

export async function emitDaemonChatResult(
  result: DaemonChatResult,
  onEvent: DaemonChatEventHandler,
  options?: { artifacts?: DaemonArtifact[] },
): Promise<void> {
  await onEvent({ sessionId: result.sessionId })

  for (const toolCall of result.toolCalls ?? []) {
    await onEvent({ type: 'tool_call', toolCall })
  }

  if ((options?.artifacts?.length ?? 0) > 0) {
    await onEvent({ artifacts: options?.artifacts ?? [] })
  }

  await onEvent({ type: 'message', content: result.content })
  await onEvent({ type: 'done', usage: result.usage })
}

export async function streamChatViaSse(
  params: SseChatParams,
): Promise<void> {
  try {
    let sawTerminalEvent = false
    for await (const event of reconnectingJsonSseStream<DaemonChatStreamPayload>(
      ({ signal, lastEventId }) =>
        params.client.chatStream(
          params.message,
          params.sessionId,
          {
            ...params.options,
            ...(lastEventId ? { lastEventId } : {}),
          },
          { signal },
        ),
      {
        signal: params.signal,
        lastEventId: params.options?.lastEventId,
        reconnectWithoutLastEventId: false,
      },
    )) {
      if (isTerminalDaemonChatPayload(event.data)) {
        sawTerminalEvent = true
        await params.onEvent(event.data)
        break
      }
      await params.onEvent(event.data)
    }
    if (!sawTerminalEvent) {
      throw createIncompleteDaemonChatStreamError()
    }
  } catch (error) {
    throw resolveDaemonChatRunAbortReason(params.signal, error) ?? error
  }
}

export async function streamChatWithFallback(params: {
  runWebSocketChat: () => Promise<void>
  fallbackStream: () => Promise<void>
  shouldFallback?: (error: unknown) => boolean
}): Promise<void> {
  try {
    await params.runWebSocketChat()
  } catch (error) {
    if (params.shouldFallback && !params.shouldFallback(error)) {
      throw error
    }
    await params.fallbackStream()
  }
}
