import { createSocket, type Socket } from 'node:dgram'

export interface DiscoveredDevice {
  id: string
  name: string
  host: string
  port: number
  role: string
  lastSeen: Date
}

export interface MdnsConfig {
  enabled: boolean
  deviceId: string
  deviceName: string
  port: number
  role: string
  broadcastIntervalMs?: number
}

const MDNS_ADDR = '224.0.0.251'
const MDNS_PORT = 5353
const SERVICE_TYPE = '_sepilotd._tcp'

export class MdnsDiscovery {
  private config: MdnsConfig
  private socket: Socket | null = null
  private devices = new Map<string, DiscoveredDevice>()
  private broadcastTimer: ReturnType<typeof setInterval> | null = null
  private stopped = false
  private listeners: Array<(device: DiscoveredDevice) => void> = []

  constructor(config: MdnsConfig) {
    this.config = config
  }

  async start(): Promise<void> {
    if (!this.config.enabled) return
    this.stopped = false

    this.socket = createSocket({ type: 'udp4', reuseAddr: true })

    this.socket.on('message', (msg, rinfo) => {
      try {
        const data = JSON.parse(msg.toString())
        if (data.service !== SERVICE_TYPE) return
        if (data.id === this.config.deviceId) return // Ignore self

        const device: DiscoveredDevice = {
          id: data.id,
          name: data.name,
          host: rinfo.address,
          port: data.port,
          role: data.role,
          lastSeen: new Date(),
        }

        this.devices.set(device.id, device)
        for (const listener of this.listeners) {
          try { listener(device) } catch { /* handler error ignored */ }
        }
      } catch { /* malformed message */ }
    })

    await new Promise<void>((resolve) => {
      this.socket!.bind(MDNS_PORT, () => {
        try {
          this.socket!.addMembership(MDNS_ADDR)
          resolve()
        } catch {
          resolve() // Continue even if multicast fails (e.g., in Docker)
        }
      })
      this.socket!.on('error', () => resolve()) // Don't fail startup
    })

    // Broadcast presence
    if (this.stopped) return
    this.broadcast()
    this.broadcastTimer = setInterval(() => this.broadcast(), this.config.broadcastIntervalMs ?? 30000)
    this.broadcastTimer.unref?.()
  }

  async stop(): Promise<void> {
    this.stopped = true
    if (this.broadcastTimer) {
      clearInterval(this.broadcastTimer)
      this.broadcastTimer = null
    }
    if (this.socket) {
      try { this.socket.close() } catch { /* socket already closed */ }
      this.socket = null
    }
  }

  getDevices(): DiscoveredDevice[] {
    // Remove stale devices (not seen in 2 minutes)
    const cutoff = Date.now() - 120_000
    for (const [id, dev] of this.devices) {
      if (dev.lastSeen.getTime() < cutoff) this.devices.delete(id)
    }
    return Array.from(this.devices.values())
  }

  onDeviceFound(listener: (device: DiscoveredDevice) => void): void {
    this.listeners.push(listener)
  }

  private broadcast(): void {
    if (!this.socket) return
    const msg = JSON.stringify({
      service: SERVICE_TYPE,
      id: this.config.deviceId,
      name: this.config.deviceName,
      port: this.config.port,
      role: this.config.role,
    })
    try {
      this.socket.send(msg, 0, msg.length, MDNS_PORT, MDNS_ADDR)
    } catch { /* send error on closed socket */ }
  }
}
