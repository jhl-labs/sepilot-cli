import type { FastifyInstance } from 'fastify'
import type { LifecycleCapabilities } from '../server/runtime/capabilities.js'
import { createLogger } from '../logger.js'
import { stopEventRoutingFeature } from '../generated/feature-registration.js'
import { closeAllDomainDbs } from '../storage/domain-db.js'

const log = createLogger('lifecycle')

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
  return Math.min(requested, Math.max(0, resolveShutdownDeadlineMs() - 1_000))
}

export async function performShutdown(
  app: FastifyInstance,
  runtime: LifecycleCapabilities,
  reason: string,
): Promise<void> {
  log.info(`${reason} received. Shutting down gracefully...`)

  try { runtime.serviceSupervisor.stopMonitor() } catch {}

  const handleSchedulerDrainError = (error: unknown): null => {
    log.warn('Scheduler graceful drain failed', {
      error: error instanceof Error ? error.message : String(error),
    })
    return null
  }
  let schedulerDrain: Promise<Awaited<ReturnType<typeof runtime.schedulerEngine.shutdown>> | null>
  try {
    schedulerDrain = runtime.schedulerEngine
      .shutdown(resolveSchedulerDrainMs())
      .catch(handleSchedulerDrainError)
  } catch (error) {
    schedulerDrain = Promise.resolve(handleSchedulerDrainError(error))
  }

  try {
    await app.close()
    log.info('HTTP server closed.')
  } catch {}

  const schedulerResult = await schedulerDrain
  if (schedulerResult) {
    log.info('Scheduler stopped.', { ...schedulerResult })
  }

  try { await stopEventRoutingFeature(runtime) } catch {}

  for (const ch of runtime.channels) {
    try { await ch.stop() } catch {}
  }
  log.info('Channels stopped.')

  try { await runtime.managedProcesses.stopAll() } catch {}
  try { runtime.lsp.stopAll() } catch {}

  try { await runtime.pluginLoader.shutdownAll() } catch {}
  try { await runtime.mcpManager.disconnectAll() } catch {}
  try { runtime.graphAgentLoader?.stop() } catch {}
  try { runtime.configWatcher?.stop() } catch {}
  try { await runtime.delegationWorker.stop() } catch {}
  try { await runtime.notificationRelayWorker.stop() } catch {}
  try { await runtime.mdns.stop() } catch {}
  try { await runtime.updater.stop() } catch {}
  try { await runtime.channelPipelineMonitor.flush() } catch {}
  try { runtime.semanticIndex.close() } catch {}
  try { runtime.usageTracker.close() } catch {}
  try { await runtime.swarmRunRegistry.shutdownAll() } catch {}
  try { await runtime.telemetry.shutdown() } catch {}
  try { closeAllDomainDbs() } catch {}

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
      onReload?: () => Promise<void> | void
    } = {},
  ) {
    this.app = app
    this.runtime = runtime
    this.exitAfterShutdown = options.exitAfterShutdown ?? true
    this.afterShutdown = options.afterShutdown
    this.onReload = options.onReload
  }

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
      }
    } finally {
      clearTimeout(forceExitTimer)
    }
    if (this.exitAfterShutdown) process.exit(exitCode)
  }

  isShuttingDown(): boolean {
    return this.shuttingDown
  }

  private onTerminationSignal(signal: string): void {
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
