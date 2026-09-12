import type { Timestamp } from '../types/common.js'

export interface AuditEvent {
  timestamp: Timestamp
  event: string
  device: string
  session?: string
  [key: string]: unknown
}

export interface IAuditLogger {
  log(event: AuditEvent): Promise<void>
  query(filter: { since?: Timestamp; event?: string; limit?: number }): Promise<AuditEvent[]>
}
