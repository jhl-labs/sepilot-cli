import { forwardDaemonStream } from './chat-transport-stream.js'
import type {
  DaemonSessionsWatchPayload,
  DaemonSessionWatchPayload,
} from './types.js'
import type {
  SessionsWatchParams,
  SessionWatchParams,
} from './chat-transport-types.js'

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError'
}

export function watchDaemonSession(
  params: SessionWatchParams,
): {
  close: () => void
  done: Promise<void>
} {
  const controller = new AbortController()
  const done = (async () => {
    try {
      const response = await params.client.watchSessionStream(params.sessionId, {
        signal: controller.signal,
      })
      await forwardDaemonStream<DaemonSessionWatchPayload>(
        response,
        params.onEvent,
      )
    } catch (error) {
      if (!isAbortError(error)) {
        throw error
      }
    }
  })()

  return {
    close: () => controller.abort(),
    done,
  }
}

export function watchDaemonSessions(
  params: SessionsWatchParams,
): {
  close: () => void
  done: Promise<void>
} {
  const controller = new AbortController()
  const done = (async () => {
    try {
      const response = await params.client.watchSessionsStream(
        params.query,
        {
          ...params.options,
          signal: controller.signal,
        },
      )
      await forwardDaemonStream<DaemonSessionsWatchPayload>(
        response,
        params.onEvent,
      )
    } catch (error) {
      if (!isAbortError(error)) {
        throw error
      }
    }
  })()

  return {
    close: () => controller.abort(),
    done,
  }
}
