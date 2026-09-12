import type { Timestamp, DeviceId } from '../types/common.js'
import type { Disposable } from '../disposable.js'

export interface DeviceMessage {
  id: string; from: DeviceId; to?: DeviceId
  type: 'task' | 'result' | 'status' | 'heartbeat' | 'ack' | 'custom'
  payload: Record<string, unknown>; timestamp: Timestamp; nonce: string
  expiresAt?: Timestamp; correlationId?: string; signature: string
}

export interface Device {
  id: DeviceId; name: string; role: 'server' | 'desktop' | 'edge'
  status: 'online' | 'offline' | 'busy'; lastSeen: Timestamp
  capabilities: string[]; platform: string; arch: string
}

export interface IMessageService {
  send(deviceId: DeviceId, message: DeviceMessage): Promise<void>
  broadcast(message: DeviceMessage): Promise<void>
  onMessage(callback: (message: DeviceMessage) => void): Disposable
  listDevices(): Promise<Device[]>
}
