import {
  resolveDaemonChatRunAbortReason,
} from './chat-run-control.js'
import {
  sanitizeChatOptions,
  type ChatStreamOptions,
} from './http.js'
import { normalizeWsEvent } from './chat-ws.js'
import {
  createIncompleteDaemonChatStreamError,
  isTerminalDaemonChatPayload,
} from './chat-transport-stream.js'
import type { DaemonWsEvent } from './types.js'
import type { DaemonWsClient } from './ws.js'
import type {
  BrowserWebSocketChatParams,
  BrowserWebSocketFactory,
  BrowserWebSocketLike,
  DaemonChatEventHandler,
} from './chat-transport-types.js'

export function getDaemonWsUrl(baseUrl: string): string {
  return `${baseUrl.replace(/^http/, 'ws').replace(/\/$/, '')}/api/v1/ws`
}

export async function runBrowserWebSocketChat(
  params: BrowserWebSocketChatParams,
): Promise<void> {
  const {
    baseUrl,
    message,
    sessionId,
    options,
    socketRef,
    socketFactory,
    signal,
    onEvent,
  } = params

  const createSocket: BrowserWebSocketFactory =
    socketFactory
    ?? ((url: string) => new WebSocket(url) as unknown as BrowserWebSocketLike)

  return new Promise((resolve, reject) => {
    const socket = createSocket(getDaemonWsUrl(baseUrl))
    let settled = false

    socketRef?.current?.close()
    if (socketRef) {
      socketRef.current = socket
    }

    const cleanup = () => {
      socket.removeEventListener('open', handleOpen)
      socket.removeEventListener('message', handleMessage)
      socket.removeEventListener('error', handleError)
      socket.removeEventListener('close', handleClose)
      signal?.removeEventListener('abort', handleAbort)
      if (socketRef?.current === socket) {
        socketRef.current = null
      }
    }

    const finish = (callback: () => void) => {
      if (settled) return
      settled = true
      cleanup()
      try {
        socket.close()
      } catch {
        // Ignore close failures during teardown.
      }
      callback()
    }

    const handleOpen = () => {
      const normalizedOptions = sanitizeChatOptions(options)
      socket.send(JSON.stringify({
        type: 'chat.send',
        message,
        sessionId,
        ...normalizedOptions,
      }))
    }

    const handleMessage = (messageEvent: { data?: unknown }) => {
      void (async () => {
        try {
          const raw = JSON.parse(String(messageEvent.data)) as DaemonWsEvent
          const normalized = normalizeWsEvent(raw)
          if (normalized) {
            await onEvent(normalized)
          }

          if (
            raw.type === 'agent.done'
            || raw.type === 'agent.error'
            || raw.type === 'error'
          ) {
            finish(resolve)
          }
        } catch {
          // Ignore malformed payloads from the server.
        }
      })()
    }

    const handleError = () => {
      finish(() => reject(new Error('WebSocket connection failed')))
    }

    const handleClose = () => {
      const abortReason = resolveDaemonChatRunAbortReason(signal)
      finish(() => reject(abortReason ?? new Error('WebSocket connection closed')))
    }

    const handleAbort = () => {
      const abortReason = resolveDaemonChatRunAbortReason(signal)
      finish(() => reject(abortReason ?? new Error('Run cancelled by user.')))
    }

    socket.addEventListener('open', handleOpen)
    socket.addEventListener('message', handleMessage)
    socket.addEventListener('error', handleError)
    socket.addEventListener('close', handleClose)

    if (signal?.aborted) {
      handleAbort()
      return
    }

    signal?.addEventListener('abort', handleAbort, { once: true })
  })
}

export async function runDaemonWsClientChat(params: {
  wsClient: DaemonWsClient
  message: string
  sessionId?: string
  options?: ChatStreamOptions
  onEvent: DaemonChatEventHandler
}): Promise<void> {
  const { wsClient, message, sessionId, options, onEvent } = params
  let sawTerminalEvent = false

  await wsClient.connect()
  for await (const event of wsClient.chat(message, sessionId, options)) {
    const normalized = normalizeWsEvent(event)
    if (normalized) {
      if (isTerminalDaemonChatPayload(normalized)) {
        sawTerminalEvent = true
      }
      await onEvent(normalized)
    }
  }

  if (!sawTerminalEvent) {
    throw createIncompleteDaemonChatStreamError()
  }
}

export function isRetryableLocalWsError(error: unknown): boolean {
  // Treat any failure that looks like "WS isn't usable here" as retryable so
  // the caller falls back to plain HTTP+SSE. Covers:
  //   - timeout / connect / ECONNREFUSED / socket   (transport-level fails)
  //   - "Unexpected server response: 404|426"        (no WS handler at the
  //     path — happens against test mocks and during rolling daemon upgrades
  //     where /ws hasn't been wired yet)
  return error instanceof Error
    && /timeout|connect|ECONNREFUSED|socket|unexpected server response/i.test(
      error.message,
    )
}

export function shouldFallbackFromInteractiveWsError(
  error: unknown,
  options: { sawEvent: boolean; aborted?: boolean },
): boolean {
  if (options.aborted || options.sawEvent) return false
  const message = error instanceof Error ? error.message : String(error)
  if (/stream-idle-timeout|Daemon chat stream ended before completion/i.test(message)) {
    return false
  }
  return isRetryableLocalWsError(error)
}
