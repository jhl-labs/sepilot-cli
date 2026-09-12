import { createLogger } from '../../logger.js'
import { startChannelWebhookDispatcher } from '../../channels/webhook-dispatch.js'
import { createSchedulerTools } from '../../tools/scheduler.js'
import { buildSchedulerStack } from './services.js'
import { configureScheduler } from './scheduler.js'
import { startRetentionSweeper } from '../../retention/sweeper.js'
import { cleanupSessionRuntimeState } from '../routes/session-management.js'
import type { StartupCapabilities } from './capabilities.js'
import { validateScheduledSkillRefs } from '../../scheduler/skill-selection.js'

const log = createLogger('runtime.lifecycle')

/**
 * Run every side-effecting service on an assembled runtime graph:
 * channel adapters, mDNS discovery, auto-updater, scheduled jobs,
 * delegation worker, and graph agent loader. Kept separate from
 * graph construction so tests can opt out of side effects.
 */
export function startRuntime(runtime: StartupCapabilities): void {
  runtime.notificationRelayWorker.start()

  for (const channel of runtime.channels) {
    channel
      .start()
      .catch((error) =>
        log.error(`Failed to start channel ${channel.id}`, {
          error: String(error),
        }),
      )
  }
  startChannelWebhookDispatcher(runtime)

  runtime.mdns.start().catch(() => {
    // mDNS is optional.
  })

  runtime.updater.start().catch(() => {
    // Update checks are optional.
  })

  // Build the SQLite-backed scheduler stack now that all deps are available.
  const schedulerStack = buildSchedulerStack({
    channels: runtime.channels,
    channelPipelineCapabilities: runtime,
    providerRegistry: runtime.providerRegistry,
    toolRegistry: runtime.toolRegistry,
    skillRegistry: runtime.skillRegistry,
    policyEngine: runtime.policyEngine,
    sessions: runtime.sessions,
    getAutonomy: () => runtime.autonomy,
    auditLogger: runtime.auditLogger,
    dreaming: runtime.dreaming,
    semanticIndex: runtime.semanticIndex,
    config: runtime.config,
    grantUnattendedApproval: (schedulerSessionId) =>
      runtime.approvalRegistry?.grantSessionAutoApproval(schedulerSessionId, 'unattended scheduled job'),
    waitForApproval: (args) => runtime.approvalRegistry.waitForApproval(args),
  })

  // Register scheduler agent tools now that jobStore is live.
  for (const tool of createSchedulerTools({
    store: schedulerStack.jobStore,
    triggerSchedulerJob: schedulerStack.triggerSchedulerJob,
    defaultTimezone: schedulerStack.defaultTimezone,
    grantUnattendedApproval: (schedulerSessionId) =>
      runtime.approvalRegistry?.grantSessionAutoApproval(schedulerSessionId, 'schedule_create'),
    revokeUnattendedApproval: (schedulerSessionId) =>
      runtime.approvalRegistry?.revokeSessionAutoApproval(schedulerSessionId),
    validateSkillRefs: (refs) => validateScheduledSkillRefs(refs, {
      skillRegistry: runtime.skillRegistry,
      toolRegistry: runtime.toolRegistry,
      autonomy: runtime.autonomy,
    }),
  })) {
    runtime.toolRegistry.register(tool)
  }

  // Expose on runtime so routes and the pipeline can consume them.
  runtime.jobStore = schedulerStack.jobStore
  runtime.schedulerEngine = schedulerStack.engine
  runtime.parseWhen = schedulerStack.parseWhen
  runtime.schedulerDefaultTimezone = schedulerStack.defaultTimezone
  runtime.triggerSchedulerJob = schedulerStack.triggerSchedulerJob

  configureScheduler(schedulerStack)

  runtime.delegationWorker.start().catch((error) => {
    log.error('Failed to start delegation worker', {
      error: String(error),
    })
  })

  runtime.graphAgentLoader?.start()

  // Retention sweeper: age out sessions/memory/usage past the configured
  // window. No-op unless retention days are configured (or the env kill-switch
  // is set). Session deletion reuses the complete-deletion path so nothing is
  // orphaned.
  const retention = runtime.config.retention
  if (retention) {
    const intervalMs = Math.max(1, retention.sweepIntervalHours ?? 24) * 60 * 60 * 1000
    startRetentionSweeper(
      {
        listSessions: async () => {
          const page = await runtime.sessions.list({ page: 1, perPage: 100_000 })
          return page.items.map((session) => ({ id: session.id, updatedAt: session.updatedAt }))
        },
        deleteSession: async (id) => {
          await cleanupSessionRuntimeState(runtime, id)
          await runtime.sessions.delete(id)
        },
        deleteUsageOlderThan: (cutoff) => runtime.usageTracker.deleteOlderThan(cutoff),
        deleteMemoriesOlderThan: (cutoff) => runtime.semanticIndex.deleteOlderThan(cutoff),
        logger: log,
      },
      retention,
      intervalMs,
    )
  }
}
