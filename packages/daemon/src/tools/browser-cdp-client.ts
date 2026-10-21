export type CdpEventListener = (params: Record<string, unknown>, sessionId?: string) => void

export interface CdpClient {
  send(method: string, params?: Record<string, unknown>, sessionId?: string, timeoutMs?: number): Promise<any>
  on(event: string, listener: CdpEventListener): () => void
  onClose(listener: () => void): () => void
  close(): Promise<void>
}

export type CdpConnectFn = (wsEndpoint: string) => Promise<CdpClient>

interface Pending {
  resolve: (value: unknown) => void
  reject: (error: Error) => void
  method: string
}

const CONNECT_TIMEOUT_MS = 60_000
const COMMAND_TIMEOUT_MS = 30_000

export class WebSocketCdpClient implements CdpClient {
  private nextId = 1
  private readonly pending = new Map<number, Pending>()
  private readonly listeners = new Map<string, Set<CdpEventListener>>()
  private readonly closeListeners = new Set<() => void>()
  private closed = false

  private constructor(private readonly socket: WebSocket) {
    socket.addEventListener('message', (event) => this.handleMessage(String((event as MessageEvent).data)))
    socket.addEventListener('close', () => this.handleClose())
    socket.addEventListener('error', () => this.handleClose())
  }

  static connect(wsEndpoint: string, timeoutMs = CONNECT_TIMEOUT_MS): Promise<WebSocketCdpClient> {
    return new Promise((resolve, reject) => {
      const socket = new WebSocket(wsEndpoint)
      const timer = setTimeout(() => {
        socket.close()
        reject(new Error(`Timed out connecting to Chrome after ${timeoutMs} ms; accept the remote debugging prompt in Chrome and try again`))
      }, timeoutMs)
      socket.addEventListener('open', () => {
        clearTimeout(timer)
        resolve(new WebSocketCdpClient(socket))
      })
      socket.addEventListener('error', () => {
        clearTimeout(timer)
        reject(new Error('Could not connect to Chrome remote debugging endpoint'))
      })
      socket.addEventListener('close', () => {
        clearTimeout(timer)
        reject(new Error('Chrome closed the remote debugging connection before it opened'))
      })
    })
  }

  private handleMessage(raw: string): void {
    let message: { id?: number; method?: string; params?: Record<string, unknown>; sessionId?: string; result?: unknown; error?: { message?: string } }
    try {
      message = JSON.parse(raw)
    } catch {
      return
    }
    if (typeof message.id === 'number') {
      const pending = this.pending.get(message.id)
      if (!pending) return
      this.pending.delete(message.id)
      if (message.error) pending.reject(new Error(`${pending.method}: ${message.error.message ?? 'CDP error'}`))
      else pending.resolve(message.result ?? {})
      return
    }
    if (message.method) {
      for (const listener of this.listeners.get(message.method) ?? []) {
        try {
          listener(message.params ?? {}, message.sessionId)
        } catch {
        }
      }
    }
  }

  private handleClose(): void {
    if (this.closed) return
    this.closed = true
    for (const pending of this.pending.values()) pending.reject(new Error(`${pending.method}: CDP connection closed`))
    this.pending.clear()
    for (const listener of this.closeListeners) listener()
    this.closeListeners.clear()
  }

  send(method: string, params: Record<string, unknown> = {}, sessionId?: string, timeoutMs = COMMAND_TIMEOUT_MS): Promise<any> {
    if (this.closed) return Promise.reject(new Error(`${method}: CDP connection closed`))
    const id = this.nextId++
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        if (this.pending.delete(id)) reject(new Error(`${method}: timed out after ${timeoutMs} ms`))
      }, timeoutMs)
      timer.unref?.()
      this.pending.set(id, {
        method,
        resolve: (value) => {
          clearTimeout(timer)
          resolve(value)
        },
        reject: (error) => {
          clearTimeout(timer)
          reject(error)
        },
      })
      this.socket.send(JSON.stringify(sessionId ? { id, method, params, sessionId } : { id, method, params }))
    })
  }

  on(event: string, listener: CdpEventListener): () => void {
    let set = this.listeners.get(event)
    if (!set) {
      set = new Set()
      this.listeners.set(event, set)
    }
    set.add(listener)
    return () => {
      set!.delete(listener)
    }
  }

  onClose(listener: () => void): () => void {
    if (this.closed) listener()
    else this.closeListeners.add(listener)
    return () => {
      this.closeListeners.delete(listener)
    }
  }

  async close(): Promise<void> {
    if (this.closed) return
    try {
      this.socket.close()
    } catch {
    }
    this.handleClose()
  }
}
