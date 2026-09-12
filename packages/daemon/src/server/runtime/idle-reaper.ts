import { createLogger } from '../../logger.js'
import type { ConnectionRegistry } from './connection-registry.js'
import type { DaemonShutdownController } from '../fastify-types.js'

const log = createLogger('idle-reaper')

export interface IdleReaperOptions {
  /**
   * Idle window in ms. The reaper requires (a) zero active connections and
   * (b) no HTTP/WS activity for at least this long before triggering shutdown.
   * Set to 0 (the default) to disable entirely.
   */
  idleMs: number
  /** How often to recheck. Defaults to 1/6 of idleMs, clamped to [10s, 60s]. */
  pollMs?: number
  /**
   * Optional work predicate. Zero connections does NOT mean zero work: a
   * scheduled job, a swarm run, a delegation, or a non-streaming `POST /chat`
   * turn can be in-flight with no open WS/SSE connection. When this returns
   * true the reaper holds off, so background work is never killed mid-flight.
   */
  isBusy?: () => boolean
}

/**
 * Optional self-shutdown service: when both the connection count is zero AND
 * `lastActivityAt` is older than `idleMs`, hand off to the shutdown controller.
 *
 * Important: the activity bump is per-HTTP-request (see `app.ts`), so a chain
 * of one-shot `sepilot ask` invocations keeps the daemon alive even though
 * each call only opens a transient SSE stream. Long-lived WS / desktop tray
 * surfaces also count as connections so idle is genuinely idle.
 */
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
    // Genuine idle also requires no in-flight background work (scheduler,
    // swarm, active/queued runs). Connections alone under-count activity.
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

/**
 * Read the `SEPILOTD_IDLE_SHUTDOWN_MS` env var, defaulting to 0 (disabled).
 * Centralised here so the operator-knobs section in CLAUDE.md and the deploy
 * docs reference a single source of truth.
 */
export function resolveIdleShutdownMs(): number {
  const raw = process.env.SEPILOTD_IDLE_SHUTDOWN_MS
  if (!raw) return 0
  const parsed = Number.parseInt(raw, 10)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 0
}
