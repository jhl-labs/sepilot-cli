import { createLogger } from '../../logger.js'
import { startChannelWebhookDispatcher } from '../../channels/webhook-dispatch.js'
import { createSchedulerTools } from '../../tools/scheduler.js'
import { buildSchedulerStack } from './services.js'
import { startEventRoutingFeature } from '../../generated/feature-registration.js'
import { configureScheduler } from './scheduler.js'
import { startRetentionSweeper } from '../../retention/sweeper.js'
import { cleanupSessionRuntimeState } from '../routes/session-management.js'
import type { StartupCapabilities } from './capabilities.js'
import { validateScheduledSkillRefs } from '../../scheduler/skill-selection.js'

const log = createLogger('runtime.lifecycle')

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

  const eventRoutingHooks = startEventRoutingFeature(runtime)

  runtime.mdns.start().catch(() => {
  })

  runtime.updater.start().catch(() => {
  })

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
    onRunFinished: eventRoutingHooks.onSchedulerRunFinished,
  })

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
