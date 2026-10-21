import type { LiveView, LiveViewStill } from '@sepilotd/core'
import type { DaemonClient } from './http.js'
import { parseJsonSseStream } from './stream.js'

export interface LiveViewFrameEvent {
  event: 'ready' | 'frame' | 'change' | 'heartbeat' | (string & {})
  data: unknown
  id?: string
}

const VIEWS_PATH = '/api/v1/views'
const BROWSER_PATH = '/api/v1/browser'

export interface BrowserConnectionSummary {
  id: string
  sessionId: string | null
  mode: 'human' | 'agent'
}

export function createLiveViewApi(client: DaemonClient) {
  return {
    list: (sessionId?: string) =>
      client.request<LiveView[]>(
        `${VIEWS_PATH}${sessionId ? `?sessionId=${encodeURIComponent(sessionId)}` : ''}`,
      ),
    still: (id: string) =>
      client.request<LiveViewStill>(`${VIEWS_PATH}/${encodeURIComponent(id)}/still`),
    connections: () => client.request<BrowserConnectionSummary[]>(`${BROWSER_PATH}/connections`),
    control: (connectionId: string, sessionId: string, mode: 'human' | 'agent') =>
      client.request<unknown>(`${BROWSER_PATH}/control`, {
        method: 'POST',
        body: { id: connectionId, sessionId, mode },
      }),
    async *frames(
      id: string,
      options: { signal?: AbortSignal } = {},
    ): AsyncIterable<LiveViewFrameEvent> {
      const response = await client.fetch(`${VIEWS_PATH}/${encodeURIComponent(id)}/frames`, {
        method: 'GET',
        signal: options.signal,
        headers: { accept: 'text/event-stream' },
      })
      if (!response.ok || !response.body) {
        throw new Error(`View frame stream failed (${response.status})`)
      }
      for await (const frame of parseJsonSseStream<unknown>(response.body)) {
        yield { event: frame.event, data: frame.data, id: frame.id }
      }
    },
  }
}

export type LiveViewApi = ReturnType<typeof createLiveViewApi>
