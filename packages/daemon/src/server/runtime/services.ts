import { inheritedPersonaMemory } from '../../memory/persona-scope.js'
import { createPersonaRepo } from '../../persona/repo.js'
import { persistedPersonaToPersona } from '../../agent/custom/persona-resolver.js'
import { randomUUID } from 'node:crypto'
import { join } from 'node:path'
import {
  DEFAULT_RETRY_POLICY,
  nextRetryAt,
  type AutonomyLevel,
  type IAuditLogger,
  type IDreamingMemoryStore,
  type IChannel,
  type ApprovalDecision,
  type ISessionStore,
  type ManualJobRunOptions,
  type ManualJobRunResult,
  type ISkillRegistry,
  type ToolCall,
  type TokenUsage,
} from '@sepilotd/core'
import type { ScheduledAgentSkillRef } from '@sepilotd/api-client'
import type { SepilotdConfig } from '../../config/schema.js'
import { AgentEngine } from '../../agent/engine.js'
import { createAgentOutputTracker } from '../../agent/event-output.js'
import { registerBuiltinGraphs, builtinGraphBuilders } from '../../agent/graph/presets/index.js'
import {
  UserAgentLoader,
  registerUserAgents,
  type BaseGraphBuilders,
} from '../../agent/user-agents/loader.js'
import { UserCommandStore } from '../../agent/user-commands/loader.js'
import { GraphAgentRegistry } from '../../agent/graph/registry.js'
import { YamlGraphAgentLoader } from '../../agent/graph/yaml-loader.js'
import { MdnsDiscovery } from '../../discovery/mdns.js'
import { GatewayClient, loadGatewayToken } from '../../gateway/client.js'
import { HookRegistry } from '../../hook/registry.js'
import { registerOutboundWebhooks } from '../../hooks/outbound-webhook.js'
import { registerCommandHooks } from '../../hooks/command-hook.js'
import { McpManager } from '../../mcp/manager.js'
import type { McpClientFeatureDeps } from '../../mcp/client.js'
import type { FileMemory } from '../../memory/file-memory.js'
import type { ScopedFileMemoryRegistry } from '../../memory/scoped-file-memory.js'
import { DreamingEngine } from '../../memory/dreaming.js'
import type { SemanticMemoryStore } from '../../memory/types.js'
import { TelemetryManager } from '../../observability/telemetry.js'
import { PluginLoader } from '../../plugins/loader.js'
import { LLMCache } from '../../providers/cache.js'
import { ProviderCircuitBreaker } from '../../providers/circuit-breaker.js'
import { ModelRouter } from '../../providers/model-router.js'
import type { ProviderRegistry } from '../../providers/registry.js'
import { createJobStore, type JobStore, type ScheduledJob } from '../../scheduler/job-store.js'
import { SchedulerEngine, type JobExecutionResult } from '../../scheduler/engine.js'
import { ScheduledDeliveryTracker } from '../../scheduler/delivery-control.js'
import { createSchedulerExecutor, type ChannelLookup } from '../../scheduler/executor.js'
import { createScriptMonitorRunner, scriptMonitorConfigFromMetadata } from '../../scheduler/script-monitor.js'
import { scheduledAgentExecutionResult } from '../../scheduler/agent-outcome.js'
import {
  createSchedulerDeliveryOutbox,
  createSchedulerDeliveryWorker,
  type SchedulerDeliveryWorker,
} from '../../scheduler/delivery-outbox.js'
import { createInternalJobRunner, ensureInternalJobs } from '../../scheduler/internal-jobs.js'
import { schedulerNotificationAudience } from '../../scheduler/notification-subscriptions.js'
import { nextRecurringRun, parseWhen, SchedulerParseError } from '../../scheduler/time-parser.js'
import { createNotificationsRepo } from '../../notifications/repo.js'
import { createObservabilityRepo } from '../../observability/events.js'
import { publishNotification } from '../../notifications/broker.js'
import { publishSchedulerRunNotification } from '../../notifications/publish.js'
import { ChannelAgentExecutor } from '../../channels/agent-executor.js'
import { ChannelSessionResolver } from '../../channels/session-resolver.js'
import type { ChannelPipelineCapabilities } from './capabilities.js'
import type { PolicyEngine } from '../../security/policy-engine.js'
import type { JsonlAuditLogger } from '../../security/audit-logger.js'
import type { ToolRegistry } from '../../tools/registry.js'
import { UpdateChecker } from '../../updater/checker.js'
import { DAEMON_VERSION } from '../../version.js'
import { RunLimiter } from './run-limiter.js'
import type { SecretVault } from '../../security/secret-vault.js'
import { createLogger } from '../../logger.js'
import type { ConfigWriter } from '../../mcp/config-writer.js'
import { resolveScheduledSkillRefsContext } from '../../scheduler/skill-selection.js'
import { persistAgentSessionEvent } from '../session-events.js'
import type { RunStopReason } from '@sepilotd/core'

const log = createLogger('runtime.services')

export async function buildGatewayClient(
  config: SepilotdConfig,
  dataDir: string,
): Promise<GatewayClient> {
  const gatewayToken = await loadGatewayToken(dataDir)
  return new GatewayClient(config.gateway.url, gatewayToken ?? undefined)
}

export function buildHookRegistry(
  config: SepilotdConfig,
  observer: {
    auditLogger?: IAuditLogger
    deviceName?: string
  } = {},
): HookRegistry {
  const hookRegistry = new HookRegistry()
  registerOutboundWebhooks(
    hookRegistry,
    config.hooks.outboundWebhooks,
    observer,
  )
  registerCommandHooks(hookRegistry, config.hooks.commandHooks ?? [])
  return hookRegistry
}

export function buildMcpManager(
  auditLogger?: IAuditLogger,
  vault?: SecretVault | null,
  deviceName?: string,
  featureDeps?: McpClientFeatureDeps,
  configWriter?: ConfigWriter | null,
): McpManager {
  const metricsAuditLogger = auditLogger && deviceName
    ? {
        log: async (entry: Record<string, unknown>) => {
          await auditLogger.log({
            timestamp: new Date().toISOString(),
            device: deviceName,
            event: typeof entry.event === 'string'
              ? entry.event
              : 'mcp.tool.call',
            ...entry,
          })
        },
      }
    : undefined
  return new McpManager(undefined, metricsAuditLogger, vault, undefined, featureDeps, configWriter)
}

export async function buildTelemetry(
  config: SepilotdConfig,
): Promise<TelemetryManager> {
  const telemetry = new TelemetryManager({
    enabled: config.observability.telemetry,
    otlpEndpoint: config.observability.otlpEndpoint,
    serviceName: 'sepilotd',
  })
  await telemetry.init()
  return telemetry
}

export function buildDiscovery(config: SepilotdConfig): MdnsDiscovery {
  return new MdnsDiscovery({
    enabled: true,
    deviceId: config.device.id,
    deviceName: config.device.name,
    port: config.daemon.port,
    role: config.device.role,
  })
}

export function buildUpdater(): UpdateChecker {
  return new UpdateChecker(DAEMON_VERSION, {
    channel: 'stable',
    autoCheck: true,
    checkIntervalMs: 24 * 60 * 60 * 1000,
  })
}

export interface SchedulerStackDeps {
  channels: IChannel[]
  channelPipelineCapabilities: ChannelPipelineCapabilities
  providerRegistry: Pick<ProviderRegistry, 'getDefault'>
  toolRegistry: ToolRegistry
  /** Registry used to rehydrate explicitly selected skills for scheduled runs. */
  skillRegistry?: ISkillRegistry
  policyEngine: PolicyEngine
  sessions: ISessionStore
  getAutonomy: () => AutonomyLevel
  auditLogger: JsonlAuditLogger
  dreaming: DreamingEngine
  semanticIndex: SemanticMemoryStore
  config: SepilotdConfig
  /**
   * Re-establish the standing approval an unattended job carries. The
   * `unattended` column survives restarts; the in-memory grant does not, so the
   * executor replays it before each fire.
   */
  grantUnattendedApproval?: (schedulerSessionId: string) => void
  /**
   * Resolve a tool approval for a headless scheduled run. Without it the
   * engine has no approval handler at all, so every tool that needs consent
   * fails outright — including on an unattended job, whose whole point is a
   * standing grant the registry already holds.
   */
  waitForApproval?: (args: {
    sessionId: string
    toolCall: ToolCall
    requestId: string
    runId?: string
    forcePrompt?: boolean
    signal?: AbortSignal
  }) => ApprovalDecision | Promise<ApprovalDecision>
}

export interface SchedulerStack {
  jobStore: JobStore
  engine: SchedulerEngine
  deliveryWorker: SchedulerDeliveryWorker
  parseWhen: typeof parseWhen
  /** IANA timezone applied to cron jobs that don't carry their own. */
  defaultTimezone: string
  triggerSchedulerJob: (id: string, options?: ManualJobRunOptions) => Promise<ManualJobRunResult>
}

function resolveDefaultTimezone(config: SepilotdConfig): string {
  const fromConfig = config.scheduler?.timezone
  if (fromConfig?.trim()) return fromConfig.trim()
  try {
    const tz = Intl.DateTimeFormat().resolvedOptions().timeZone
    if (tz) return tz
  } catch {
    /* fall through */
  }
  return 'UTC'
}

function hasSchedulerUsage(usage: TokenUsage): boolean {
  return usage.inputTokens + usage.outputTokens > 0
}

function normalizeManualSchedulerResult(
  result: string | void | JobExecutionResult | null,
): {
  output: string | void
  outcome: 'complete' | 'incomplete'
  failureReason?: string
} {
  if (typeof result === 'string' || result == null) {
    return { output: result ?? undefined, outcome: 'complete' }
  }
  return {
    output: typeof result.output === 'string' ? result.output : undefined,
    outcome: result.outcome === 'incomplete' ? 'incomplete' : 'complete',
    ...(result.outcome === 'incomplete' && result.failureReason?.trim()
      ? { failureReason: result.failureReason.trim() }
      : {}),
  }
}

export function buildSchedulerStack(deps: SchedulerStackDeps): SchedulerStack {
  const {
    channels,
    channelPipelineCapabilities,
    providerRegistry,
    toolRegistry,
    policyEngine,
    sessions,
    getAutonomy,
    auditLogger,
    dreaming,
    semanticIndex,
    config,
  } = deps

  const jobStore = createJobStore()
  const deliveryOutbox = createSchedulerDeliveryOutbox()

  const channelLookup: ChannelLookup = {
    get: (type: string) => channels.find((ch) => ch.type === type),
  }
  const deliveryWorker = createSchedulerDeliveryWorker(deliveryOutbox, channelLookup)

  const channelAgentExecutor = new ChannelAgentExecutor(channelPipelineCapabilities)
  const sessionResolver = new ChannelSessionResolver(channelPipelineCapabilities)

  const memoryDreaming = config.memory.dreaming ?? {
    enabled: false,
    schedule: '6h',
  }
  const memoryMaintenance = config.memory.maintenance ?? {
    enabled: false,
    schedule: '1d',
    maxAgeDays: 90,
    maxImportance: 0.1,
    dryRun: false,
  }
  const notifications = createNotificationsRepo()

  const appendSchedulerChatProgress = async (
    job: { id: string; name: string; instruction: string; parentSessionId: string | null; channelType: string | null; channelTarget: string | null },
    content: string,
  ): Promise<void> => {
    if (
      !job.parentSessionId
      || job.channelType
      || job.channelTarget
      || job.instruction.startsWith('__internal:')
    ) {
      return
    }

    try {
      const parentSession = await sessions.get(job.parentSessionId)
      if (!parentSession) {
        log.warn('scheduled progress has no parent session to append to', {
          parentSessionId: job.parentSessionId,
          schedulerJobId: job.id,
        })
        return
      }
      await sessions.appendEvent(job.parentSessionId, {
        type: 'assistant_message',
        id: randomUUID(),
        timestamp: new Date().toISOString(),
        content,
        origin: 'scheduler',
      })
    } catch (err) {
      log.warn('failed to append scheduled progress to parent session', {
        parentSessionId: job.parentSessionId,
        schedulerJobId: job.id,
        error: err instanceof Error ? err.message : String(err),
      })
    }
  }

  const internalJobRunner = createInternalJobRunner({
    dreaming,
    semanticIndex,
    notifications: {
      publish(input) {
        const item = notifications.upsert(input)
        publishNotification(item)
      },
    },
    memoryMaintenance: {
      maxAgeDays: memoryMaintenance.maxAgeDays,
      maxImportance: memoryMaintenance.maxImportance,
      dryRun: memoryMaintenance.dryRun,
    },
    retention: {
      prune() {
        // Observability prune honours the configured retentionDays; notification
        // prune uses its own age ceiling + newest-K cap.
        createObservabilityRepo().prune()
        notifications.prune()
      },
    },
  })

  /**
   * A scheduled run has no per-request model, so it must fall back to the same
   * model the user actually configured rather than an arbitrary catalog entry.
   */
  const resolveHeadlessModelId = (
    provider: { models: ReadonlyArray<{ id: string }> },
  ): string => {
    const configured = config.agent.defaultModel?.trim()
    if (configured && provider.models.some((model) => model.id === configured)) {
      return configured
    }
    return provider.models[0]?.id ?? 'default'
  }

  const runHeadlessAgent = async (
    instruction: string,
    opts: {
      sessionId: string
      agentSessionId?: string
      onAgentSessionReady?: () => void
      parentSessionId: string | null
      channelContext?: { channel: string; chatKey: string; triggerMessageId?: string }
      skillRefs?: ScheduledAgentSkillRef[]
      signal?: AbortSignal
      suppressDelivery?: boolean
    },
  ): Promise<string | void | JobExecutionResult> => {
    const provider = providerRegistry.getDefault()
    if (!provider) {
      throw new Error('SCHEDULER_NO_PROVIDER: no default provider configured')
    }
    const parentMemory = inheritedPersonaMemory(opts.parentSessionId ? await sessions.get(opts.parentSessionId) : null, createPersonaRepo().list())
    const selectedModel = resolveHeadlessModelId(provider)
    const auditStartedAt = Date.now()
    let auditSessionCreated = false
    const abandonAuditSession = async (): Promise<void> => {
      if (!opts.agentSessionId || !auditSessionCreated) return
      try {
        await sessions.updateMeta?.(opts.agentSessionId, { status: 'abandoned' })
      } catch (error) {
        log.warn('failed to mark scheduled agent audit session abandoned', {
          agentSessionId: opts.agentSessionId,
          error: error instanceof Error ? error.message : String(error),
        })
      }
    }
    if (opts.agentSessionId) {
      const existingAuditSession = await sessions.get(opts.agentSessionId)
      if (existingAuditSession) {
        throw new Error(`SCHEDULER_AGENT_SESSION_EXISTS: audit session ${opts.agentSessionId} already exists`)
      }
      const now = new Date().toISOString()
      try {
        await sessions.create({
          id: opts.agentSessionId,
          title: `Scheduled run ${opts.agentSessionId.slice(-8)}`,
          createdAt: now,
          updatedAt: now,
          provider: provider.id,
          model: selectedModel,
          device: config.device.name,
          status: 'active',
          tags: ['scheduler-run'],
          ...(parentMemory ? { memoryNamespace: parentMemory.memoryNamespace, personaIds: [parentMemory.persona.id] } : {}),
        })
        auditSessionCreated = true
        opts.onAgentSessionReady?.()
        await sessions.appendEvent(opts.agentSessionId, {
          type: 'user_message',
          id: randomUUID(),
          timestamp: now,
          content: instruction,
        })
      } catch (error) {
        await abandonAuditSession()
        throw error
      }
    }
    const skillContext = await resolveScheduledSkillRefsContext(
      opts.skillRefs ?? [],
      {
        skillRegistry: deps.skillRegistry,
        toolRegistry,
        autonomy: getAutonomy(),
      },
    ).catch(async (error) => {
      await abandonAuditSession()
      throw error
    })
    const waitForApproval = deps.waitForApproval
    const engine = new AgentEngine({
      provider,
      tools: toolRegistry,
      policy: policyEngine,
      autonomy: getAutonomy(),
      auditLogger,
      deviceName: config.device.name,
      usageTracker: deps.channelPipelineCapabilities.usageTracker,
      // A scheduled run collects before it reports, so it needs more turns than
      // an interactive chat answer. On the engine default of 10 a nightly
      // briefing over 13 tickers exhausted the budget and reported INCOMPLETE
      // with its report already written to disk.
      maxIterations: config.scheduler?.maxIterations ?? 40,
      strictFinalAnswerProtocol:
        process.env.SEPILOTD_STRICT_ANSWER_PROTOCOL === '1'
        || Boolean(skillContext.systemPrompt?.trim()),
      // Nobody is at the screen for a scheduled run, but the registry still
      // holds the standing grant an unattended job carries, and a job without
      // one should reach the surface's approval prompt rather than fail on the
      // spot with "no approval handler".
      ...(waitForApproval
        ? {
            approvalCallback: (toolCall, requestId, options) =>
              waitForApproval({
                sessionId: opts.sessionId,
                toolCall,
                requestId,
                forcePrompt: options?.forcePrompt,
                signal: options?.signal,
              }),
          }
        : {}),
    })
    const outputTracker = createAgentOutputTracker()
    const deliveryTracker = new ScheduledDeliveryTracker()
    const usage: TokenUsage = { inputTokens: 0, outputTokens: 0 }
    let runStopReason: RunStopReason | undefined
    const stopOnAbort = (): void => {
      void engine.stop().catch(() => undefined)
    }
    try {
      opts.signal?.throwIfAborted()
      opts.signal?.addEventListener('abort', stopOnAbort, { once: true })
      for await (const event of engine.run(instruction, {
        // Approval authority remains stable per job, but all execution-scoped
        // tool state and evidence use the unique persisted run session.
        sessionId: opts.agentSessionId ?? opts.sessionId,
        provider: provider.id,
        // `models[0]` is whatever order the provider returned — alphabetical for
        // Ollama — so a scheduled run could execute on a completely different
        // model from the one the user configured, including one that cannot call
        // tools. Prefer the configured default and fall back only if it is not
        // in the catalog.
        model: selectedModel,
        channelContext: opts.channelContext,
        ...(parentMemory ? { scopeTags: parentMemory.scopeTags } : {}),
        ...((skillContext.systemPrompt || parentMemory) ? { systemPrompt: [parentMemory ? persistedPersonaToPersona(parentMemory.persona).systemPromptAddition : '', skillContext.systemPrompt].filter(Boolean).join('\n\n') } : {}),
        ...(skillContext.refs.length > 0
          ? { toolAllowlist: toolRegistry.list().map((tool) => tool.name) }
          : {}),
        ...(skillContext.selectedSkillIds
          ? { selectedSkillIds: skillContext.selectedSkillIds }
          : {}),
        ...(skillContext.executionSkillIds
          ? { executionSkillIds: skillContext.executionSkillIds }
          : {}),
        ...(skillContext.skillToolNames ? { skillToolNames: skillContext.skillToolNames } : {}),
        ...(skillContext.skillExecutionPolicies
          ? { skillExecutionPolicies: skillContext.skillExecutionPolicies }
          : {}),
      })) {
        if (opts.agentSessionId && event.type !== 'done') {
          await persistAgentSessionEvent(sessions, opts.agentSessionId, event)
        }
        outputTracker.consume(event)
        deliveryTracker.consume(event)
        if (event.type === 'done') {
          usage.inputTokens = event.usage.inputTokens
          usage.outputTokens = event.usage.outputTokens
          runStopReason = event.stopReason
        }
      }
      opts.signal?.throwIfAborted()
    } catch (error) {
      await abandonAuditSession()
      throw error
    } finally {
      opts.signal?.removeEventListener('abort', stopOnAbort)
    }
    const finalContent = outputTracker.finalContent().trim()
    if (opts.agentSessionId) {
      try {
        if (finalContent) {
          await sessions.appendEvent(opts.agentSessionId, {
            type: 'assistant_message',
            id: randomUUID(),
            timestamp: new Date().toISOString(),
            content: finalContent,
            origin: 'scheduler',
          })
        }
        await sessions.appendEvent(opts.agentSessionId, {
          type: 'session_end',
          id: randomUUID(),
          timestamp: new Date().toISOString(),
          totalTokens: {
            input: usage.inputTokens,
            output: usage.outputTokens,
          },
          totalCost: 0,
          duration_ms: Date.now() - auditStartedAt,
          ...(runStopReason ? { stopReason: runStopReason } : {}),
        })
      } catch (error) {
        await abandonAuditSession()
        throw error
      }
    }
    // These two exits also end a run, so they carry the cause as well —
    // otherwise a job that produced no final answer records nothing to explain
    // why. Recording it does not by itself reclassify the run.
    const headlessExit = (): JobExecutionResult | undefined => {
      const carriesUsage = hasSchedulerUsage(usage)
      if (!carriesUsage && !runStopReason) return undefined
      return {
        ...(carriesUsage ? { usage } : {}),
        ...(runStopReason ? { stopReason: runStopReason } : {}),
      }
    }
    if (deliveryTracker.shouldSuppress()) {
      return headlessExit()
    }
    if (!finalContent) {
      return headlessExit()
    }

    if (opts.parentSessionId && !opts.suppressDelivery) {
      try {
        const parentSession = await sessions.get(opts.parentSessionId)
        if (parentSession) {
          await sessions.appendEvent(opts.parentSessionId, {
            type: 'assistant_message',
            id: randomUUID(),
            timestamp: new Date().toISOString(),
            content: finalContent,
            origin: 'scheduler',
          })
        } else {
          // The chat the job was created from was deleted while the job stayed
          // enabled. Restore it under the same id rather than dropping the
          // output: the job is still live, so the conversation it belongs to
          // should come back with it — that is where the user asked for this
          // and where they will look for it. Dropping the result instead left
          // a job that ran forever with nowhere to report.
          const now = new Date().toISOString()
          await sessions.create({
            id: opts.parentSessionId,
            title: 'Scheduled task',
            createdAt: now,
            updatedAt: now,
            provider: provider.id,
            model: selectedModel,
            device: config.device.name,
            status: 'active',
            tags: [],
          })
          await sessions.appendEvent(opts.parentSessionId, {
            type: 'assistant_message',
            id: randomUUID(),
            timestamp: now,
            content: finalContent,
          })
          log.info('restored the deleted chat a scheduled job belongs to', {
            parentSessionId: opts.parentSessionId,
            schedulerSessionId: opts.sessionId,
          })
        }
      } catch (err) {
        log.warn('failed to append scheduled headless result to parent session', {
          parentSessionId: opts.parentSessionId,
          schedulerSessionId: opts.sessionId,
          error: err instanceof Error ? err.message : String(err),
        })
      }
    }

    // A job with no channel binding runs through this headless path, so without
    // carrying the stop reason here the common case records no cause at all.
    return scheduledAgentExecutionResult({
      rawOutput: finalContent,
      usage,
      ...(runStopReason ? { stopReason: runStopReason } : {}),
    })
  }

  const executor = createSchedulerExecutor({
    jobStore,
    channelRegistry: channelLookup,
    channelAgentExecutor,
    sessionResolver,
    providerRegistry,
    runHeadlessAgent,
    runInternalJob: internalJobRunner,
    runScriptMonitor: createScriptMonitorRunner({
      updateMetadata(jobId, metadata) {
        jobStore.updateMetadata(jobId, metadata)
      },
      notify(input) {
        const job = jobStore.get(input.jobId)
        publishSchedulerRunNotification({
          ...input,
          audience: job ? schedulerNotificationAudience(job) : null,
        })
      },
    }),
    grantUnattendedApproval: deps.grantUnattendedApproval,
    onJobProgress: appendSchedulerChatProgress,
    deliveryOutbox,
    autonomy: getAutonomy(),
  })

  const defaultTimezone = resolveDefaultTimezone(config)
  const schedulerDeliveryPriority = (job: { metadata: Record<string, unknown> | null } | null | undefined): 'normal' | 'high' | 'critical' => {
    const priority = job?.metadata?.notificationPriority
    return priority === 'high' || priority === 'critical' ? priority : 'normal'
  }
  const activeManualJobIds = new Set<string>()
  const engine = new SchedulerEngine({
    activeManualJobIds: () => [...activeManualJobIds],
    store: jobStore,
    executor,
    defaultTimezone,
    maxConsecutiveFailures: config.scheduler?.maxConsecutiveFailures,
    enabled: () => config.scheduler?.enabled !== false,
    dailyTokenBudget: () => config.scheduler?.dailyTokenBudget,
    onEvent: (event) => {
      // Surface scheduler outcomes as in-app notifications. We notify on:
      //   - permanent failure (any kind of job)
      //   - missed one-shots (one-shot is implicit for `missed`)
      //   - successful completion of a one-shot job
      //   - successful completion of a parent-chat recurring job that produced output
      // Internal maintenance jobs (__internal:*) log on their own — skip those
      // across the board.
      try {
        if (event.type === 'failed' && !event.willRetry) {
          const job = jobStore.get(event.jobId)
          if (job?.instruction.startsWith('__internal:')) return
          publishSchedulerRunNotification({
            title: `Scheduled task failed: ${job?.name ?? event.jobId}`,
            body: event.reason.slice(0, 500),
            jobId: job?.id,
            runId: event.runId,
            audience: job ? schedulerNotificationAudience(job) : null,
            priority: 'high',
          })
        } else if (event.type === 'missed') {
          const job = jobStore.get(event.jobId)
          if (job?.instruction.startsWith('__internal:')) return
          const mins = Math.max(1, Math.round(event.missedByMs / 60_000))
          publishSchedulerRunNotification({
            title: `Scheduled task missed: ${job?.name ?? event.jobId}`,
            body: `The one-shot task did not run within ${mins} min of its scheduled time and was cancelled (the daemon may have been offline).`,
            jobId: job?.id,
            audience: job ? schedulerNotificationAudience(job) : null,
            priority: 'high',
          })
        } else if (event.type === 'auto-disabled') {
          const job = jobStore.get(event.jobId)
          if (!job || job.instruction.startsWith('__internal:')) return
          publishSchedulerRunNotification({
            title: `Scheduled task auto-disabled: ${job.name ?? event.jobId}`,
            body: `Disabled after ${event.consecutiveFailures} consecutive failures (threshold ${event.threshold}). Latest error: ${event.reason.slice(0, 400)}`,
            jobId: job.id,
            runId: event.runId,
            audience: schedulerNotificationAudience(job),
            priority: 'high',
          })
        } else if (event.type === 'budget-exhausted') {
          publishSchedulerRunNotification({
            title: 'Scheduler daily token budget exhausted',
            body: `Used ${event.usedTokens}/${event.budgetTokens} scheduler tokens for ${event.dateKey}. Automatic scheduled runs are paused until the next local day.`,
            audience: null,
            priority: 'high',
          })
        } else if (event.type === 'completed') {
          const job = jobStore.get(event.jobId)
          if (!job) return
          if (job.instruction.startsWith('__internal:')) return
          // Script monitors own transition/reminder delivery. A generic
          // completion notification on every probe would defeat deduplication.
          if (scriptMonitorConfigFromMetadata(job.metadata)) return
          const seconds = Math.max(1, Math.round(event.durationMs / 1000))
          const latestRun = jobStore.listRuns(job.id, 1)[0]
          const output = latestRun?.outputExcerpt?.trim()
          // A recurring run that produced nothing to read stays quiet — there is
          // no result to deliver and a heartbeat every few minutes is noise.
          // But anything the run did produce has to reach the user: a job
          // created from the scheduler UI has no parent chat to fall back on,
          // so gating on `parentSessionId` made every such job run invisibly
          // forever, which is indistinguishable from a scheduler that is broken.
          if (job.kind !== 'oneshot' && !output) return
          publishSchedulerRunNotification({
            title: `Scheduled task completed: ${job.name ?? event.jobId}`,
            body: output
              ? `Finished in ${seconds}s.\n\n${output.slice(0, 500)}`
              : `Finished in ${seconds}s.`,
            jobId: job.id,
            runId: event.runId,
            audience: schedulerNotificationAudience(job),
            priority: schedulerDeliveryPriority(job),
          })
        }
      } catch {
        // notifications are best-effort — never let one break the scheduler tick.
      }
    },
  })

  // Reconcile internal recurring jobs to configuration. LLM-backed memory
  // work is opt-in; only the bounded retention sweep remains on by default.
  // "6h" → "0 */6 * * *" via scheduleToCron.
  const dreamingSchedule = memoryDreaming.schedule ?? '6h'
  const maintenanceSchedule = memoryMaintenance.schedule ?? '1d'
  const userProfile = config.memory.userProfile ?? { enabled: false, schedule: '1d', section: 'User Profile' }
  const digest = config.proactivity?.digest
  ensureInternalJobs(jobStore, {
    dreaming: { schedule: dreamingSchedule, enabled: memoryDreaming.enabled ?? false },
    memoryMaintenance: { schedule: maintenanceSchedule, enabled: memoryMaintenance.enabled ?? false },
    userProfile: {
      schedule: userProfile.schedule ?? '1d',
      enabled: userProfile.enabled ?? false,
      section: userProfile.section ?? 'User Profile',
    },
    digest: digest
      ? {
          schedule: digest.schedule ?? '0 9 * * *',
          enabled: digest.enabled ?? false,
          channelType: digest.channelType,
          channelTarget: digest.channelTarget,
        }
      : undefined,
    retention: {
      schedule: 'daily',
      enabled: config.retention?.enabled ?? true,
    },
  })

  const timezoneForJob = (job: ScheduledJob): string | undefined => job.timezone ?? defaultTimezone

  const advanceManualRecurringJob = (
    job: ScheduledJob,
    startedAt: number,
    error: string | null,
  ): void => {
    if (!job.cron) {
      jobStore.updateStatus(job.id, 'failed', {
        lastRunAt: startedAt,
        lastError: 'recurring job has no cron expression',
      })
      return
    }

    try {
      const next = nextRecurringRun(job.cron, Date.now(), timezoneForJob(job))
      if (error) jobStore.rescheduleAfterFailure(job.id, next, startedAt, error)
      else jobStore.updateNextRun(job.id, next, startedAt)
    } catch (err) {
      const reason =
        err instanceof SchedulerParseError
          ? err.message
          : err instanceof Error
            ? err.message
            : String(err)
      jobStore.updateStatus(job.id, 'failed', { lastRunAt: startedAt, lastError: reason })
    }
  }

  const finalizeManualSchedulerJob = (
    job: ScheduledJob,
    startedAt: number,
    error: string | null,
  ): void => {
    // A late manual result must not undo cancellation or a definition edit.
    // Pause retains running status and its enabled=false flag survives finalization.
    if (jobStore.get(job.id)?.status !== 'running') return
    const scheduledFireWasDue = job.nextRunAt <= startedAt

    if (!scheduledFireWasDue) {
      if (error) jobStore.rescheduleAfterFailure(job.id, job.nextRunAt, startedAt, error)
      else jobStore.updateNextRun(job.id, job.nextRunAt, startedAt)
      return
    }

    if (!error) {
      if (job.kind === 'oneshot') {
        jobStore.updateStatus(job.id, 'completed', { lastRunAt: startedAt })
      } else {
        advanceManualRecurringJob(job, startedAt, null)
      }
      return
    }

    const attemptNo = job.attempt + 1
    const retryAt = nextRetryAt(Date.now(), attemptNo, {
      maxAttempts: job.maxAttempts,
      backoffMs: job.retryBackoffMs,
      maxBackoffMs: DEFAULT_RETRY_POLICY.maxBackoffMs,
    })
    if (retryAt != null) {
      jobStore.markRetry(job.id, retryAt, attemptNo, error)
      return
    }

    if (job.kind === 'oneshot') {
      jobStore.updateStatus(job.id, 'failed', { lastRunAt: startedAt, lastError: error })
    } else {
      advanceManualRecurringJob(job, startedAt, error)
    }
  }

  const executeManualSchedulerJob = async (
    job: ScheduledJob,
    runId: string,
    startedAt: number,
    options: ManualJobRunOptions,
  ): Promise<ManualJobRunResult> => {
    try {
      const { output, outcome, failureReason } = normalizeManualSchedulerResult(
        await executor(job, {
          runId,
          ...(options.suppressDelivery ? { suppressDelivery: true } : {}),
        }),
      )
      const durationMs = Date.now() - startedAt
      if (outcome === 'incomplete') {
        const reason = failureReason ?? 'Scheduled agent reported an incomplete result.'
        jobStore.recordRunFinish(runId, 'failed', {
          durationMs,
          error: reason,
          outputExcerpt: typeof output === 'string' && output.trim().length > 0
            ? output
            : null,
        })
        finalizeManualSchedulerJob(job, startedAt, reason)
        if (!options.suppressDelivery && !job.instruction.startsWith('__internal:')) {
          try {
            publishSchedulerRunNotification({
              title: `Scheduled task failed: ${job.name ?? job.id}`,
              body: reason.slice(0, 500),
              jobId: job.id,
              runId,
              audience: schedulerNotificationAudience(job),
            })
          } catch {
            // notification is best-effort
          }
        }
        return {
          started: true,
          jobId: job.id,
          runId,
          status: 'failed',
          ...(options.suppressDelivery ? { deliverySuppressed: true as const } : {}),
        }
      }
      jobStore.recordRunFinish(runId, 'success', {
        durationMs,
        outputExcerpt: typeof output === 'string' && output.trim().length > 0 ? output : null,
      })
      finalizeManualSchedulerJob(job, startedAt, null)
      if (!options.suppressDelivery && !job.instruction.startsWith('__internal:')) {
        try {
          const outputText = typeof output === 'string' ? output.trim() : ''
          const seconds = Math.max(1, Math.round(durationMs / 1000))
          publishSchedulerRunNotification({
            title: `Scheduled task completed: ${job.name ?? job.id}`,
            body: outputText
              ? `Manual run finished in ${seconds}s.\n\n${outputText.slice(0, 500)}`
              : `Manual run finished in ${seconds}s.`,
            jobId: job.id,
            runId,
            audience: schedulerNotificationAudience(job),
          })
        } catch {
          // notification is best-effort
        }
      }
      return {
        started: true,
        jobId: job.id,
        runId,
        status: 'success',
        ...(options.suppressDelivery ? { deliverySuppressed: true as const } : {}),
      }
    } catch (err) {
      const reason = err instanceof Error ? err.message : String(err)
      const durationMs = Date.now() - startedAt
      jobStore.recordRunFinish(runId, 'failed', { durationMs, error: reason })
      finalizeManualSchedulerJob(job, startedAt, reason)
      if (!options.suppressDelivery && !job.instruction.startsWith('__internal:')) {
        try {
          publishSchedulerRunNotification({
            title: `Scheduled task failed: ${job.name ?? job.id}`,
            body: reason.slice(0, 500),
            jobId: job.id,
            runId,
            audience: schedulerNotificationAudience(job),
          })
        } catch {
          // notification is best-effort
        }
      }
      return {
        started: true,
        jobId: job.id,
        runId,
        status: 'failed',
        ...(options.suppressDelivery ? { deliverySuppressed: true as const } : {}),
      }
    }
  }

  /**
   * Returns a rejected result when the job could not be locked for a run — it
   * is already running, or paused/disabled. Every started run returns its
   * canonical persisted identity. Callers may observe completion through run
   * history instead of holding one heartbeat-free HTTP request open.
   */
  const triggerSchedulerJob = async (
    id: string,
    options: ManualJobRunOptions = {},
  ): Promise<ManualJobRunResult> => {
    const job = jobStore.get(id)
    if (!job) throw new Error(`job not found: ${id}`)

    if (!jobStore.tryLockForRun(job.id)) {
      log.info('scheduler job could not be claimed; skipping manual trigger', {
        jobId: job.id,
        enabled: job.enabled,
        status: job.status,
      })
      return { started: false, jobId: job.id, runId: null, status: null }
    }

    const startedAt = Date.now()
    const runId = jobStore.recordRunStart(
      job.id,
      job.nextRunAt <= startedAt ? job.attempt + 1 : 0,
      startedAt,
    )
    activeManualJobIds.add(job.id)
    const completion = executeManualSchedulerJob(job, runId, startedAt, options)
      .finally(() => activeManualJobIds.delete(job.id))
    if (options.waitForCompletion === false) {
      void completion.catch((error) => {
        log.error('detached manual scheduler execution escaped terminal handling', {
          jobId: job.id,
          runId,
          error: error instanceof Error ? error.message : String(error),
        })
      })
      return {
        started: true,
        jobId: job.id,
        runId,
        status: 'running',
        ...(options.suppressDelivery ? { deliverySuppressed: true as const } : {}),
      }
    }
    return completion
  }

  return { jobStore, engine, deliveryWorker, parseWhen, defaultTimezone, triggerSchedulerJob }
}

export function buildLlmCache(): LLMCache {
  return new LLMCache()
}

export function buildProviderCircuitBreaker(): ProviderCircuitBreaker {
  return new ProviderCircuitBreaker({
    failureThreshold: 5,
    openDurationMs: 30_000,
  })
}

export function buildRunLimiter(): RunLimiter {
  return new RunLimiter({
    maxActive: 4,
    maxQueued: 4,
    queueTimeoutMs: 1_500,
  })
}

export function buildModelRouter(
  providerRegistry: ProviderRegistry,
): ModelRouter {
  return new ModelRouter(providerRegistry)
}

export async function buildPluginLoader(
  dataDir: string,
  config?: SepilotdConfig,
): Promise<PluginLoader> {
  return new PluginLoader(`${dataDir}/plugins`, {
    strict: config?.plugins.strict,
    loadTimeoutMs: config?.plugins.loadTimeoutMs,
    trustedSignatureKeys: config?.plugins.trustedSignatureKeys,
  })
}

export interface GraphRegistryServices {
  graphRegistry: GraphAgentRegistry
  graphAgentLoader?: YamlGraphAgentLoader
  userAgentLoader?: UserAgentLoader
  userCommandStore?: UserCommandStore
}

const baseBuilderAdapter: BaseGraphBuilders = {
  get: (id) => (builtinGraphBuilders as Record<string, (deps: any) => any>)[id],
  list: () => Object.keys(builtinGraphBuilders),
}

export async function buildGraphRegistry(
  dataDir?: string,
): Promise<GraphRegistryServices> {
  const graphRegistry = new GraphAgentRegistry()
  registerBuiltinGraphs(graphRegistry)
  if (!dataDir) {
    return { graphRegistry }
  }

  const graphAgentLoader = new YamlGraphAgentLoader(
    join(dataDir, 'agents'),
    graphRegistry,
  )
  await graphAgentLoader.init()

  const userAgentLoader = new UserAgentLoader({
    agentsDir: join(dataDir, 'user-agents'),
    baseBuilders: baseBuilderAdapter,
  })
  const records = await userAgentLoader.loadAll()
  registerUserAgents(graphRegistry, records, baseBuilderAdapter)

  const userCommandStore = new UserCommandStore({
    commandsDir: join(dataDir, 'commands'),
  })
  await userCommandStore.loadAll()

  return {
    graphRegistry,
    graphAgentLoader,
    userAgentLoader,
    userCommandStore,
  }
}

export function buildDreamingEngine(
  sessions: ISessionStore,
  memoryStore: IDreamingMemoryStore,
  providerRegistry: ProviderRegistry,
  fileMemory?: FileMemory,
  fileMemoryRegistry?: ScopedFileMemoryRegistry,
  modelSelection: Pick<SepilotdConfig['agent'], 'defaultModel' | 'auxModel'> = {},
): DreamingEngine {
  const dreaming = new DreamingEngine(sessions, memoryStore, fileMemory)
  if (fileMemoryRegistry) dreaming.setFileMemoryRegistry(fileMemoryRegistry)
  syncDreamingProvider(dreaming, providerRegistry, modelSelection)
  return dreaming
}

export function syncDreamingProvider(
  dreaming: DreamingEngine,
  providerRegistry: ProviderRegistry,
  modelSelection: Pick<SepilotdConfig['agent'], 'defaultModel' | 'auxModel'> = {},
): void {
  const defaultProvider = providerRegistry.getDefault()
  if (!defaultProvider) {
    dreaming.clearProvider()
    return
  }

  const configuredAuxModel = modelSelection.auxModel?.trim()
  const configuredDefaultModel = modelSelection.defaultModel?.trim()
  const configuredModel = [configuredAuxModel, configuredDefaultModel].find(
    (modelId): modelId is string => Boolean(
      modelId
      && defaultProvider.models.some((candidate) => candidate.id === modelId),
    ),
  )
  const model = configuredModel || defaultProvider.models[0]?.id || 'default'

  dreaming.setProvider(defaultProvider, model)
}

export async function enableJsonLoggingIfNeeded(
  config: SepilotdConfig,
): Promise<void> {
  if (!config.observability.telemetry) return
  const { setJsonFormat } = await import('../../logger.js')
  setJsonFormat(true)
}
