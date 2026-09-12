import { createServer, createConnection, type Server, type Socket } from 'node:net'
import { randomUUID } from 'node:crypto'

export interface P2PConfig {
  port: number
  deviceId: string
  deviceName: string
}

export interface P2PMessage {
  id: string
  from: string
  fromName: string
  type: 'task' | 'result' | 'status' | 'ping' | 'pong'
  payload: Record<string, unknown>
  timestamp: string
}

export class P2PServer {
  private server: Server | null = null
  private config: P2PConfig
  private connections = new Map<string, Socket>()
  private handlers: Array<(msg: P2PMessage, reply: (msg: P2PMessage) => void) => void> = []

  constructor(config: P2PConfig) {
    this.config = config
  }

  async start(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.server = createServer((socket) => {
        let buffer = ''

        socket.on('data', (data) => {
          buffer += data.toString()
          const lines = buffer.split('\n')
          buffer = lines.pop() ?? ''

          for (const line of lines) {
            if (!line.trim()) continue
            try {
              const msg = JSON.parse(line) as P2PMessage
              this.handleMessage(msg, socket)
            } catch { /* malformed JSON */ }
          }
        })

        socket.on('error', () => {})
        socket.on('close', () => {
          for (const [id, s] of this.connections) {
            if (s === socket) { this.connections.delete(id); break }
          }
        })
      })

      this.server.listen(this.config.port, '0.0.0.0', () => resolve())
      this.server.on('error', reject)
    })
  }

  async stop(): Promise<void> {
    for (const socket of this.connections.values()) {
      socket.destroy()
    }
    this.connections.clear()
    if (this.server) {
      await new Promise<void>((resolve) => this.server!.close(() => resolve()))
      this.server = null
    }
  }

  /** Connect to a remote device */
  async connect(host: string, port: number): Promise<string> {
    return new Promise((resolve, reject) => {
      const socket = createConnection({ host, port }, () => {
        // Send identification
        const id = randomUUID()
        this.send(socket, {
          id: randomUUID(), from: this.config.deviceId, fromName: this.config.deviceName,
          type: 'ping', payload: {}, timestamp: new Date().toISOString(),
        })
        this.connections.set(id, socket)
        resolve(id)
      })
      socket.on('error', reject)
    })
  }

  /** Send a message to a connected device */
  async sendTo(connectionId: string, type: P2PMessage['type'], payload: Record<string, unknown>): Promise<void> {
    const socket = this.connections.get(connectionId)
    if (!socket) throw new Error(`No connection: ${connectionId}`)
    this.send(socket, {
      id: randomUUID(), from: this.config.deviceId, fromName: this.config.deviceName,
      type, payload, timestamp: new Date().toISOString(),
    })
  }

  /** Register message handler */
  onMessage(handler: (msg: P2PMessage, reply: (msg: P2PMessage) => void) => void): void {
    this.handlers.push(handler)
  }

  getConnections(): string[] {
    return Array.from(this.connections.keys())
  }

  private handleMessage(msg: P2PMessage, socket: Socket): void {
    // Auto-respond to pings
    if (msg.type === 'ping') {
      this.send(socket, {
        id: randomUUID(), from: this.config.deviceId, fromName: this.config.deviceName,
        type: 'pong', payload: {}, timestamp: new Date().toISOString(),
      })
    }

    // Track connection
    if (!Array.from(this.connections.values()).includes(socket)) {
      this.connections.set(msg.from, socket)
    }

    const reply = (replyMsg: P2PMessage) => this.send(socket, replyMsg)
    for (const handler of this.handlers) {
      try { handler(msg, reply) } catch { /* handler error */ }
    }
  }

  private send(socket: Socket, msg: P2PMessage): void {
    try { socket.write(JSON.stringify(msg) + '\n') } catch { /* write error on closed socket */ }
  }
}
