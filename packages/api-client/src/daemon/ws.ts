import type { ApprovalDecisionStatus } from '@sepilotd/core'
import WebSocket from 'ws'
import { buildMemoryScopeHeaders, type MemoryScope, type Resolvable } from '../http.js'
import { sanitizeChatOptions, type ChatStreamOptions } from './http.js'
import type { DaemonWsEvent } from './types.js'

export type DaemonWsChatOptions = ChatStreamOptions

export interface DaemonWsClientOptions {
  baseUrl?: string
  token?: string | null
  /** UI/client surface label sent as X-Sepilotd-Surface for routing and diagnostics. */
  surface?: string | null
  /**
   * Memory scope sent as X-Memory-Scope-* handshake headers. Resolved lazily
   * when connecting so callers can discover daemon-owned scope at runtime.
   */
  memoryScope?: Resolvable<MemoryScope | null | undefined>
  pairedDevice?: {
    deviceId: string
    sign(payload: string): string | Promise<string>
  } | null
}

function parseOptions(options?: string | DaemonWsClientOptions): Required<DaemonWsClientOptions> {
  if (typeof options === 'string') {
    return {
      baseUrl: options,
      token: null,
      surface: null,
      memoryScope: null,
      pairedDevice: null,
    }
  }
  return {
    baseUrl: options?.baseUrl ?? 'http://127.0.0.1:17600',
    token: options?.token ?? null,
    surface: options?.surface ?? null,
    memoryScope: options?.memoryScope ?? null,
    pairedDevice: options?.pairedDevice ?? null,
  }
}

function resolve<T>(value: Resolvable<T>): Promise<T> {
  if (typeof value === 'function') {
    return Promise.resolve((value as () => T | Promise<T>)())
  }
  return Promise.resolve(value)
}

function buildPairedDevicePayload(deviceId: string, timestamp: string): string {
  return `sepilotd-ws-connect:${deviceId}:${timestamp}`
}

export class DaemonWsClient {
  private ws: WebSocket | null = null
  private connectPromise: Promise<void> | null = null
  private connectionGeneration = 0
  // A single WebSocket multiplexes all messages into one queue, so two
  // concurrent chats on the same connection would cross-talk (one chat's
  // agent.done would end the other). Enforce a single in-flight chat per
  // connection to keep event streams correlated.
  private chatInFlight = false
  readonly baseUrl: string
  private readonly token: string | null
  private readonly surface: string | null
  private readonly memoryScope: Required<DaemonWsClientOptions>['memoryScope']
  private readonly pairedDevice: Required<DaemonWsClientOptions>['pairedDevice']

  constructor(options?: string | DaemonWsClientOptions) {
    const resolved = parseOptions(options)
    this.baseUrl = resolved.baseUrl.replace(/^http/, 'ws')
    this.token = resolved.token
    this.surface = resolved.surface
    this.memoryScope = resolved.memoryScope
    this.pairedDevice = resolved.pairedDevice
  }

  connect(): Promise<void> {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return Promise.resolve()
    }
    if (this.connectPromise) {
      return this.connectPromise
    }

    const generation = ++this.connectionGeneration
    const pending = this.openConnection(generation)
    this.connectPromise = pending
    const clearPending = () => {
      if (this.connectPromise === pending) {
        this.connectPromise = null
      }
    }
    void pending.then(clearPending, clearPending)
    return pending
  }

  private async openConnection(generation: number): Promise<void> {
    const headers: Record<string, string> = {}
    if (this.token) {
      headers.Authorization = `Bearer ${this.token}`
    }
    const surface = this.surface?.trim()
    if (surface) {
      headers['X-Sepilotd-Surface'] = surface
    }
    Object.assign(headers, buildMemoryScopeHeaders(await resolve(this.memoryScope)))
    if (this.pairedDevice) {
      const timestamp = new Date().toISOString()
      const payload = buildPairedDevicePayload(this.pairedDevice.deviceId, timestamp)
      headers['X-Sepilot-Device-Id'] = this.pairedDevice.deviceId
      headers['X-Sepilot-Device-Timestamp'] = timestamp
      headers['X-Sepilot-Device-Signature'] = await this.pairedDevice.sign(payload)
    }

    if (generation !== this.connectionGeneration) {
      throw new Error('WebSocket connection cancelled')
    }

    return new Promise((resolve, reject) => {
      const socket = new WebSocket(`${this.baseUrl}/api/v1/ws`, {
        headers: Object.keys(headers).length > 0 ? headers : undefined,
      })
      this.ws = socket

      let settled = false
      const settle = (callback: () => void) => {
        if (settled) return
        settled = true
        clearTimeout(timeout)
        socket.removeListener('open', onOpen)
        socket.removeListener('error', onConnectError)
        socket.removeListener('close', onConnectClose)
        callback()
      }
      const clearSocket = () => {
        if (this.ws === socket) {
          this.ws = null
        }
      }
      const terminate = () => {
        clearSocket()
        try {
          if (socket.readyState === WebSocket.CONNECTING) {
            // ws reports termination during the opening handshake as an
            // `error`; retain a listener after connect handlers are removed.
            socket.once('error', () => {})
          }
          socket.terminate()
        } catch {
          // The socket may already have closed while the attempt was settling.
        }
      }

      const onOpen = () => {
        if (generation !== this.connectionGeneration) {
          settle(() => {
            terminate()
            reject(new Error('WebSocket connection cancelled'))
          })
          return
        }
        settle(() => {
          // Keep lifecycle listeners after the handshake so a close between
          // chats invalidates the cached socket and the next connect retries.
          socket.once('close', clearSocket)
          socket.on('error', () => {})
          resolve()
        })
      }
      const onConnectError = (error: Error) => {
        settle(() => {
          terminate()
          reject(error)
        })
      }
      const onConnectClose = () => {
        settle(() => {
          clearSocket()
          reject(new Error('WebSocket closed before the connection completed'))
        })
      }

      const timeout = setTimeout(() => {
        settle(() => {
          terminate()
          reject(new Error('WebSocket connection timeout'))
        })
      }, 5000)

      socket.once('open', onOpen)
      socket.once('error', onConnectError)
      socket.once('close', onConnectClose)
    })
  }

  async *chat(
    message: string,
    sessionId?: string,
    options?: DaemonWsChatOptions,
  ): AsyncIterable<DaemonWsEvent> {
    if (this.chatInFlight) {
      throw new Error(
        'A chat is already in progress on this connection; open a separate connection for concurrent chats.',
      )
    }
    const ws = this.ws
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      throw new Error('Not connected')
    }
    this.chatInFlight = true
    try {
      this.send({
        type: 'chat.send',
        message,
        sessionId,
        ...sanitizeChatOptions(options),
      })
    } catch (error) {
      this.chatInFlight = false
      throw error
    }
    const socket: WebSocket = ws

    const queue: DaemonWsEvent[] = []
    let resolve: (() => void) | null = null
    let done = false
    let error: Error | null = null
    // Whether a terminal event (agent.done/agent.error/error) arrived before
    // the socket closed. A close without one is a premature truncation, not a
    // clean end, and must surface as an error instead of a silent stop.
    let terminalSeen = false

    const wake = () => {
      if (resolve) {
        resolve()
        resolve = null
      }
    }

    const onMessage = (raw: WebSocket.RawData) => {
      try {
        const msg = JSON.parse(raw.toString()) as DaemonWsEvent
        queue.push(msg)
        if (msg.type === 'agent.done' || msg.type === 'agent.error' || msg.type === 'error') {
          done = true
          terminalSeen = true
        }
      } catch {
        // Ignore malformed JSON from the server.
      }
      wake()
    }

    const onError = (err: Error) => {
      error = err
      done = true
      wake()
    }

    const onClose = () => {
      if (!terminalSeen && !error) {
        error = new Error('WebSocket closed before the chat completed (premature close)')
      }
      done = true
      wake()
    }

    socket.on('message', onMessage)
    socket.on('error', onError)
    socket.on('close', onClose)

    try {
      while (!done || queue.length > 0) {
        if (queue.length > 0) {
          yield queue.shift() as DaemonWsEvent
        } else if (!done) {
          await new Promise<void>((next) => {
            resolve = next
          })
        }
      }

      if (error) throw error
    } finally {
      socket.removeListener('message', onMessage)
      socket.removeListener('error', onError)
      socket.removeListener('close', onClose)
      this.chatInFlight = false
    }
  }

  send(message: Record<string, unknown>): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('Not connected')
    }

    this.ws.send(JSON.stringify(message))
  }

  respondApproval(
    requestId: string,
    approved: boolean | ApprovalDecisionStatus,
    options?: {
      note?: string
      scope?: 'once' | 'session' | 'always' | 'run' | 'session-all'
      rule?: { tool: string; pattern: string }
    },
  ): void {
    const note = options?.note
    const scope = options?.scope
    const rule = options?.rule
    this.send(
      typeof approved === 'boolean'
        ? { type: 'approval.respond', requestId, approved, note, scope, rule }
        : {
            type: 'approval.respond',
            requestId,
            decision: approved,
            approved: approved === 'approved',
            note,
            scope,
            rule,
          },
    )
  }

  ping(): void {
    this.send({ type: 'ping' })
  }

  close(): void {
    this.connectionGeneration += 1
    this.connectPromise = null
    const socket = this.ws
    this.ws = null
    if (!socket) return
    if (socket.readyState === WebSocket.CONNECTING) {
      socket.terminate()
      return
    }
    socket.close()
  }
}
