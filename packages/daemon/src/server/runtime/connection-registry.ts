import { randomUUID } from 'node:crypto'

export type ConnectionKind = 'ws' | 'sse'

export interface ConnectionInfo {
  id: string
  kind: ConnectionKind
  /** Free-form route label, e.g. 'chat-stream', '/api/v1/ws'. */
  label: string
  /** Best-effort surface tag — extension token label or 'master'. */
  client: string | null
  openedAt: number
}

export interface ConnectionSnapshot {
  count: number
  byKind: Record<ConnectionKind, number>
  /**
   * Milliseconds since the daemon last saw any inbound activity (HTTP request
   * completion, WS open/close, etc). The idle reaper uses this so a chain of
   * one-shot `sepilot ask` calls keeps the daemon alive even though no
   * long-lived client is attached.
   */
  idleMs: number
  clients: ConnectionInfo[]
}

export class ConnectionLimitExceededError extends Error {
  constructor(
    readonly kind: ConnectionKind,
    readonly limit: number,
  ) {
    super(`Connection limit exceeded for ${kind}: ${limit}`)
    this.name = 'ConnectionLimitExceededError'
  }
}

export interface ConnectionRegistryOptions {
  maxConnectionsByKind?: Partial<Record<ConnectionKind, number>>
}

/**
 * Central registry that the WS/SSE handlers and the idle-reaper share.
 *
 * Implemented as a plain in-memory store — daemons are single-process by PID
 * lock, so there's no cross-process state to reconcile. Calling `bumpActivity`
 * is cheap and safe to invoke from request hooks.
 */
export class ConnectionRegistry {
  private connections = new Map<string, ConnectionInfo>()
  private lastActivityAt = Date.now()

  constructor(private readonly options: ConnectionRegistryOptions = {}) {}

  add(input: { kind: ConnectionKind; label: string; client?: string | null }): string {
    const limit = this.options.maxConnectionsByKind?.[input.kind]
    if (typeof limit === 'number' && this.countByKind(input.kind) >= limit) {
      throw new ConnectionLimitExceededError(input.kind, limit)
    }

    const id = randomUUID()
    this.connections.set(id, {
      id,
      kind: input.kind,
      label: input.label,
      client: input.client ?? null,
      openedAt: Date.now(),
    })
    this.lastActivityAt = Date.now()
    return id
  }

  remove(id: string): void {
    if (this.connections.delete(id)) {
      this.lastActivityAt = Date.now()
    }
  }

  bumpActivity(): void {
    this.lastActivityAt = Date.now()
  }

  get count(): number { return this.connections.size }
  get idleMs(): number { return Date.now() - this.lastActivityAt }

  countByKind(kind: ConnectionKind): number {
    let count = 0
    for (const info of this.connections.values()) {
      if (info.kind === kind) count += 1
    }
    return count
  }

  snapshot(): ConnectionSnapshot {
    const byKind: Record<ConnectionKind, number> = { ws: 0, sse: 0 }
    const clients: ConnectionInfo[] = []
    for (const info of this.connections.values()) {
      byKind[info.kind] += 1
      clients.push(info)
    }
    return {
      count: this.connections.size,
      byKind,
      idleMs: this.idleMs,
      clients,
    }
  }
}
