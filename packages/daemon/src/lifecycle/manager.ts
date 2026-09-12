import type { FastifyInstance } from 'fastify'
import type { LifecycleCapabilities } from '../server/runtime/capabilities.js'
import { createLogger } from '../logger.js'
import { closeAllDomainDbs } from '../storage/domain-db.js'

const log = createLogger('lifecycle')

// Hard upper bound on graceful shutdown. `app.close()` waits for in-flight
// requests and keep-alive sockets to drain; a wedged connection (e.g. a
// long-lived SSE reader or a stale dev-server keep-alive) can stall it
// indefinitely, leaving the process alive and holding the pid lock — which
// blocks the next start ("Another sepilotd instance is already running").
// After this deadline we force-exit so we never become that zombie.
const DEFAULT_SHUTDOWN_DEADLINE_MS = 15_000
const DEFAULT_SCHEDULER_DRAIN_MS = 10_000

export function resolveShutdownDeadlineMs(): number {
  const raw = Number(process.env.SEPILOTD_SHUTDOWN_DEADLINE_MS)
  return Number.isFinite(raw) && raw >= 1_000 ? Math.floor(raw) : DEFAULT_SHUTDOWN_DEADLINE_MS
}

function resolveSchedulerDrainMs(): number {
  const raw = Number(process.env.SEPILOTD_SCHEDULER_DRAIN_MS)
  const requested = Number.isFinite(raw) && raw >= 0
    ? Math.floor(raw)
    : DEFAULT_SCHEDULER_DRAIN_MS
  // Leave one second for channel/process/database teardown before the outer
  // force-exit deadline. A smaller outer deadline deliberately reduces the
  // scheduler window instead of letting the process die with a running row.
  return Math.min(requested, Math.max(0, resolveShutdownDeadlineMs() - 1_000))
}

/**
 * Run the daemon's full graceful-shutdown sequence: close the HTTP server,
 * stop channels/plugins/MCP/etc., flush telemetry, drain swarm runs.
 *
 * Exported separately from `LifecycleManager` so the HTTP shutdown endpoint
 * can trigger the exact same path that SIGTERM/SIGINT use, without needing to
 * fake-signal ourselves (which is unreliable on Windows).
 */
export async function performShutdown(
  app: FastifyInstance,
  runtime: LifecycleCapabilities,
  reason: string,
): Promise<void> {
  log.info(`${reason} received. Shutting down gracefully...`)

  try { runtime.serviceSupervisor.stopMonitor() } catch { /* ignore cleanup error */ }

  // Stop new scheduler claims immediately, but leave channels, providers, and
  // managed tools alive while existing executions get their bounded drain.
  // Starting this before app.close() also prevents new automatic work while
  // Fastify waits for in-flight HTTP requests.
  const handleSchedulerDrainError = (error: unknown): null => {
    log.warn('Scheduler graceful drain failed', {
      error: error instanceof Error ? error.message : String(error),
    })
    return null
  }
  let schedulerDrain: Promise<Awaited<ReturnType<typeof runtime.schedulerEngine.shutdown>> | null>
  try {
    // shutdown() stops admission synchronously before returning its drain
    // promise, so no timer tick can claim work between here and app.close().
    schedulerDrain = runtime.schedulerEngine
      .shutdown(resolveSchedulerDrainMs())
      .catch(handleSchedulerDrainError)
  } catch (error) {
    schedulerDrain = Promise.resolve(handleSchedulerDrainError(error))
  }

  try {
    await app.close()
    log.info('HTTP server closed.')
  } catch { /* ignore cleanup error */ }

  const schedulerResult = await schedulerDrain
  if (schedulerResult) {
    log.info('Scheduler stopped.', { ...schedulerResult })
  }

  for (const ch of runtime.channels) {
    try { await ch.stop() } catch { /* ignore cleanup error */ }
  }
  log.info('Channels stopped.')

  // Terminate agent-spawned background processes and language servers before
  // the rest of teardown so a restart never inherits orphaned children.
  // PLAN_019-T14 will extend this teardown with SIGHUP/WAL-checkpoint steps;
  // keep new cleanup calls additive here.
  try { await runtime.managedProcesses.stopAll() } catch { /* ignore cleanup error */ }
  try { runtime.lsp.stopAll() } catch { /* ignore cleanup error */ }

  try { await runtime.pluginLoader.shutdownAll() } catch { /* ignore cleanup error */ }
  try { await runtime.mcpManager.disconnectAll() } catch { /* ignore cleanup error */ }
  try { runtime.graphAgentLoader?.stop() } catch { /* ignore cleanup error */ }
  try { runtime.configWatcher?.stop() } catch { /* ignore cleanup error */ }
  try { await runtime.delegationWorker.stop() } catch { /* ignore cleanup error */ }
  try { await runtime.notificationRelayWorker.stop() } catch { /* ignore cleanup error */ }
  try { await runtime.mdns.stop() } catch { /* ignore cleanup error */ }
  try { await runtime.updater.stop() } catch { /* ignore cleanup error */ }
  try { await runtime.channelPipelineMonitor.flush() } catch { /* ignore cleanup error */ }
  try { runtime.semanticIndex.close() } catch { /* ignore cleanup error */ }
  try { runtime.usageTracker.close() } catch { /* ignore cleanup error */ }
  try { await runtime.swarmRunRegistry.shutdownAll() } catch { /* ignore cleanup error */ }
  try { await runtime.telemetry.shutdown() } catch { /* ignore cleanup error */ }
  // Domain repositories share a process-wide cache. Close it after all
  // consumers have drained so an embedded restart can safely select another
  // canonical data root without retaining handles into the previous profile.
  try { closeAllDomainDbs() } catch { /* ignore cleanup error */ }

  log.info('Shutdown complete.')
}

export class LifecycleManager {
  private app: FastifyInstance
  private runtime: LifecycleCapabilities
  private shuttingDown = false
  private shutdownPromise?: Promise<void>
  private exitAfterShutdown: boolean
  private afterShutdown?: () => Promise<void> | void
  private onReload?: () => Promise<void> | void

  constructor(
    app: FastifyInstance,
    runtime: LifecycleCapabilities,
    options: {
      exitAfterShutdown?: boolean
      afterShutdown?: () => Promise<void> | void
      /** Invoked on SIGHUP for a live config reload without full teardown. */
      onReload?: () => Promise<void> | void
    } = {},
  ) {
    this.app = app
    this.runtime = runtime
    this.exitAfterShutdown = options.exitAfterShutdown ?? true
    this.afterShutdown = options.afterShutdown
    this.onReload = options.onReload
  }

  /**
   * Idempotent shutdown trigger. Safe to invoke from a signal handler, the
   * `/system/shutdown` HTTP route, or the idle-reaper — multiple callers will
   * collapse onto the same in-flight teardown promise.
   *
   * `exitCode` is what the process reports once teardown finishes. It defaults
   * to 0 for deliberate shutdowns (signal, HTTP route, idle reaper). Crash
   * paths must pass a non-zero code: process supervisors run the daemon with
   * `Restart=on-failure`, so exiting 0 after an uncaughtException would look
   * like a clean stop and the daemon would never be restarted.
   */
  shutdown(reason: string, exitCode = 0): Promise<void> {
    if (this.shutdownPromise) return this.shutdownPromise
    this.shuttingDown = true
    this.shutdownPromise = this.runShutdown(reason, exitCode)
    return this.shutdownPromise
  }

  private async runShutdown(reason: string, exitCode: number): Promise<void> {
    const forceExitTimer = setTimeout(() => {
      log.warn(
        `Graceful shutdown exceeded ${resolveShutdownDeadlineMs()}ms — forcing exit.`,
      )
      if (this.exitAfterShutdown) process.exit(1)
    }, resolveShutdownDeadlineMs())
    forceExitTimer.unref()

    try {
      await performShutdown(this.app, this.runtime, reason)
      try {
        await this.afterShutdown?.()
      } catch {
        // Shutdown cleanup should not prevent process termination.
      }
    } finally {
      clearTimeout(forceExitTimer)
    }
    if (this.exitAfterShutdown) process.exit(exitCode)
  }

  /** True once a shutdown is in progress. Used by health to report draining. */
  isShuttingDown(): boolean {
    return this.shuttingDown
  }

  private onTerminationSignal(signal: string): void {
    // Double-signal escalation: if the operator sends a second SIGTERM/SIGINT
    // while a graceful shutdown is already running (e.g. teardown is wedged on
    // a stuck connection), honor the intent immediately instead of waiting out
    // the force-exit deadline.
    if (this.shuttingDown) {
      log.warn(`${signal} received during shutdown — escalating to immediate exit.`)
      if (this.exitAfterShutdown) process.exit(1)
      return
    }
    void this.shutdown(signal)
  }

  setupSignalHandlers(): void {
    process.on('SIGTERM', () => this.onTerminationSignal('SIGTERM'))
    process.on('SIGINT', () => this.onTerminationSignal('SIGINT'))
    // SIGHUP: conventional "reload configuration" signal. Prefer a live reload
    // when a handler is wired; otherwise fall back to graceful shutdown so the
    // signal is never silently ignored.
    process.on('SIGHUP', () => {
      if (this.shuttingDown) return
      if (this.onReload) {
        log.info('SIGHUP received — reloading configuration.')
        void Promise.resolve(this.onReload()).catch((error) => {
          log.warn('SIGHUP reload failed', {
            error: error instanceof Error ? error.message : String(error),
          })
        })
        return
      }
      this.onTerminationSignal('SIGHUP')
    })
  }

  /**
   * Route uncaughtException / unhandledRejection through the same graceful
   * shutdown path (with the force-exit deadline) instead of Node's default
   * exit(1), which skips app.close() draining, DB checkpoint/close, and swarm
   * jsonl flush. A crash should tear down as cleanly as a signal — but it must
   * still *report* as a crash, so both handlers exit non-zero. Node's own
   * default for these is exit(1); preserving that is what keeps
   * `Restart=on-failure` supervisors (scripts/sepilotd.service and the unit
   * `sepilotd daemon-service` generates) actually restarting the daemon.
   */
  setupCrashHandlers(): void {
    process.on('uncaughtException', (error) => {
      log.error('uncaughtException — shutting down gracefully', {
        error: error instanceof Error ? error.stack ?? error.message : String(error),
      })
      void this.shutdown('uncaughtException', 1)
    })
    process.on('unhandledRejection', (reason) => {
      log.error('unhandledRejection — shutting down gracefully', {
        error: reason instanceof Error ? reason.stack ?? reason.message : String(reason),
      })
      void this.shutdown('unhandledRejection', 1)
    })
  }
}
