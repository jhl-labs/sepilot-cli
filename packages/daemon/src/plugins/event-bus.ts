import { createLogger } from '../logger.js'

const log = createLogger('plugin-event')

export interface PluginEventMap {
  'session.created': { sessionId: string; provider?: string; model?: string }
  'session.deleted': { sessionId: string }
  'session.compacted': { sessionId: string }
  'tool.execute.before': {
    sessionId: string
    executionId: string
    tool: string
    input: Record<string, unknown>
  }
  'tool.execute.after': {
    sessionId: string
    executionId: string
    tool: string
    status: 'success' | 'error'
    durationMs?: number
  }
  'permission.asked': { sessionId: string; requestId: string; tool: string }
  'permission.replied': {
    sessionId: string
    requestId: string
    decision: 'approved' | 'denied' | 'feedback'
    note?: string
  }
  'file.edited': { sessionId: string; tool: string; path: string }
}

export type PluginEventName = keyof PluginEventMap

export type PluginEventHandler<E extends PluginEventName> = (
  payload: PluginEventMap[E],
) => void | Promise<void>

export type PluginEventUnsubscribe = () => void

export class PluginEventBus {
  private readonly handlers = new Map<PluginEventName, Set<(p: never) => void | Promise<void>>>()

  on<E extends PluginEventName>(
    event: E,
    handler: PluginEventHandler<E>,
  ): PluginEventUnsubscribe {
    let bucket = this.handlers.get(event)
    if (!bucket) {
      bucket = new Set()
      this.handlers.set(event, bucket)
    }
    bucket.add(handler as (p: never) => void | Promise<void>)
    return () => {
      this.handlers.get(event)?.delete(handler as (p: never) => void | Promise<void>)
    }
  }

  async emit<E extends PluginEventName>(
    event: E,
    payload: PluginEventMap[E],
  ): Promise<void> {
    const bucket = this.handlers.get(event)
    if (!bucket || bucket.size === 0) return
    const calls: Array<void | Promise<void>> = []
    for (const handler of bucket) {
      try {
        calls.push((handler as (p: PluginEventMap[E]) => void | Promise<void>)(payload))
      } catch (err) {
        // Plugin error must not break daemon flow — log and keep
        // dispatching to remaining handlers.
        log.error(`handler for ${event} threw`, {
          event,
          error: err instanceof Error ? err.message : String(err),
        })
      }
    }
    await Promise.allSettled(calls)
  }

  listenerCount(event: PluginEventName): number {
    return this.handlers.get(event)?.size ?? 0
  }

  dispose(): void {
    this.handlers.clear()
  }
}
