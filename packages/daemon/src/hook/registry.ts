import type { Disposable } from '@sepilotd/core'
import type { HookEvent, HookPayload, HookResult, IHookHandler, IHookRegistry } from '@sepilotd/core'

/**
 * Per-handler timeout for hook execution. A programmatic (in-process) hook
 * handler that hangs would otherwise stall the whole turn indefinitely
 * (command/outbound handlers are already bounded elsewhere). When a handler
 * exceeds this budget the registry logs and treats it as `continue` so the turn
 * proceeds. Operator-tunable; floored so it can't be set absurdly low.
 */
function resolveHookHandlerTimeoutMs(): number {
  const raw = Number(process.env.SEPILOTD_HOOK_HANDLER_TIMEOUT_MS)
  if (Number.isFinite(raw) && raw >= 100) return Math.floor(raw)
  return 10_000
}

export class HookRegistry implements IHookRegistry {
  private handlers = new Map<HookEvent, IHookHandler[]>()

  register(event: HookEvent, handler: IHookHandler): Disposable {
    if (!this.handlers.has(event)) {
      this.handlers.set(event, [])
    }
    const list = this.handlers.get(event)!
    list.push(handler)
    // Sort by priority (lower = first)
    list.sort((a, b) => a.priority - b.priority)

    return {
      dispose: () => {
        const idx = list.indexOf(handler)
        if (idx >= 0) list.splice(idx, 1)
      },
    }
  }

  async trigger(payload: HookPayload): Promise<HookResult> {
    const handlers = this.handlers.get(payload.event) ?? []
    let currentPayload = payload
    const timeoutMs = resolveHookHandlerTimeoutMs()

    for (const handler of handlers) {
      const result = await this.runHandlerWithTimeout(handler, currentPayload, timeoutMs)
      if (result.action === 'abort') return result
      if (result.action === 'skip') return result
      if (result.modifiedPayload) currentPayload = result.modifiedPayload
    }

    // Surface the chained payload so gating callers (e.g. pre:tool:execute)
    // can apply handler modifications such as rewritten tool arguments.
    return {
      action: 'continue',
      ...(currentPayload !== payload ? { modifiedPayload: currentPayload } : {}),
    }
  }

  private async runHandlerWithTimeout(
    handler: IHookHandler,
    payload: HookPayload,
    timeoutMs: number,
  ): Promise<HookResult> {
    let timer: ReturnType<typeof setTimeout> | undefined
    const timeout = new Promise<HookResult>((resolve) => {
      timer = setTimeout(() => {
        console.warn(
          `hook handler '${handler.id}' for event '${payload.event}' timed out after ${timeoutMs}ms; continuing`,
        )
        resolve({ action: 'continue' })
      }, timeoutMs)
      // Do not keep the process alive solely for this timer.
      if (typeof timer.unref === 'function') timer.unref()
    })
    try {
      return await Promise.race([handler.handle(payload), timeout])
    } finally {
      if (timer) clearTimeout(timer)
    }
  }

  listHandlers(event?: HookEvent): Array<{ event: HookEvent; id: string; priority: number }> {
    const result: Array<{ event: HookEvent; id: string; priority: number }> = []
    for (const [evt, handlers] of this.handlers) {
      if (event && evt !== event) continue
      for (const h of handlers) {
        result.push({ event: evt, id: h.id, priority: h.priority })
      }
    }
    return result
  }

  removeHandlers(
    matcher: (event: HookEvent, handler: IHookHandler) => boolean,
  ): number {
    let removed = 0

    for (const [event, handlers] of this.handlers) {
      const nextHandlers = handlers.filter((handler) => {
        if (!matcher(event, handler)) {
          return true
        }
        removed += 1
        return false
      })

      if (nextHandlers.length === 0) {
        this.handlers.delete(event)
        continue
      }

      this.handlers.set(event, nextHandlers)
    }

    return removed
  }
}
