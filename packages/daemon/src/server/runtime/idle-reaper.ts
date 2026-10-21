import { createLogger } from '../../logger.js'
import type { ConnectionRegistry } from './connection-registry.js'
import type { DaemonShutdownController } from '../fastify-types.js'

const log = createLogger('idle-reaper')

export interface IdleReaperOptions {
  idleMs: number
  pollMs?: number
  isBusy?: () => boolean
}

export function startIdleReaper(
  registry: ConnectionRegistry,
  controller: DaemonShutdownController,
  options: IdleReaperOptions,
): { stop: () => void } {
  const idleMs = Math.max(0, options.idleMs)
  if (idleMs === 0) {
    return { stop: () => {} }
  }

  const pollMs =
    options.pollMs ?? Math.min(60_000, Math.max(10_000, Math.floor(idleMs / 6)))

  let fired = false
  const timer = setInterval(() => {
    if (fired) return
    if (registry.count > 0) return
    if (registry.idleMs < idleMs) return
    if (options.isBusy?.()) {
      log.debug('Idle window elapsed but background work is active — deferring shutdown')
      return
    }
    fired = true
    log.info('Idle reaper firing — no clients and no HTTP traffic for the configured window', {
      idleMs: registry.idleMs,
    })
    clearInterval(timer)
    void controller.shutdown('idle-reaper').catch((error) => {
      log.error('Idle reaper failed to shutdown daemon', {
        error: error instanceof Error ? error.message : String(error),
      })
    })
  }, pollMs)
  timer.unref?.()

  return { stop: () => clearInterval(timer) }
}

export function resolveIdleShutdownMs(): number {
  const raw = process.env.SEPILOTD_IDLE_SHUTDOWN_MS
  if (!raw) return 0
  const parsed = Number.parseInt(raw, 10)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 0
}
