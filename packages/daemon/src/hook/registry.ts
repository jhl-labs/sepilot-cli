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
  private backgroundExecutor?: (id: string, payload: HookPayload, execute: (signal: AbortSignal) => Promise<unknown>) => void

  setBackgroundExecutor(executor: NonNullable<HookRegistry['backgroundExecutor']>): void {
    this.backgroundExecutor = executor
  }

  startBackground(id: string, payload: HookPayload, execute: (signal: AbortSignal) => Promise<unknown>): void {
    if (!payload.event.startsWith('post:')) throw new Error('Background hooks cannot gate pre events')
    if (!this.backgroundExecutor) throw new Error('Background hook job service is unavailable')
    this.backgroundExecutor(id, payload, execute)
  }

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
        const current = this.handlers.get(event)
        const idx = current?.indexOf(handler) ?? -1
        if (idx >= 0) current!.splice(idx, 1)
      },
    }
  }

  async trigger(payload: HookPayload, signal?: AbortSignal): Promise<HookResult> {
    if (signal?.aborted) return { action: 'abort', reason: 'hook execution canceled' }
    const handlers = [...(this.handlers.get(payload.event) ?? [])]
    let currentPayload = payload
    const timeoutMs = resolveHookHandlerTimeoutMs()

    for (const handler of handlers) {
      const result = await this.runHandlerWithTimeout(handler, currentPayload, timeoutMs, signal)
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
    signal?: AbortSignal,
  ): Promise<HookResult> {
    let timer: ReturnType<typeof setTimeout> | undefined
    const controller = new AbortController()
    let abort: (() => void) | undefined
    const canceled = new Promise<HookResult>((resolve) => {
      abort = () => {
        resolve({ action: 'abort', reason: 'hook execution canceled' })
        controller.abort(signal?.reason)
      }
      signal?.addEventListener('abort', abort, { once: true })
      if (signal?.aborted) abort()
    })
    const timeout = new Promise<HookResult>((resolve) => {
      timer = setTimeout(() => {
        console.warn(
          `hook handler '${handler.id}' for event '${payload.event}' timed out after ${timeoutMs}ms; continuing`,
        )
        resolve({ action: 'continue' })
        controller.abort(new Error(`hook handler timed out after ${timeoutMs}ms`))
      }, timeoutMs)
      // Do not keep the process alive solely for this timer.
      if (typeof timer.unref === 'function') timer.unref()
    })
    try {
      if (signal?.aborted) return await canceled
      return await Promise.race([handler.handle(payload, controller.signal), timeout, canceled])
    } finally {
      if (timer) clearTimeout(timer)
      if (abort) signal?.removeEventListener('abort', abort)
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
