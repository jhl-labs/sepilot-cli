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
import { resolveRunIterationBudget } from '../../agent/iteration-budget.js'
import { AgentEngine } from '../../agent/engine.js'
import { stopReasonProviderError } from '../../agent/stop-reason.js'
import { buildProviderInstance, createProviderFactoryRegistry } from './providers.js'
import { createAgentOutputTracker } from '../../agent/event-output.js'
import { registerBuiltinGraphs, builtinGraphBuilders } from '../../agent/graph/presets/index.js'
import {
  UserAgentLoader,
  registerUserAgents,
  type BaseGraphBuilders,
} from '../../agent/user-agents/loader.js'
import { UserCommandStore } from '../../agent/user-commands/loader.js'
import { RoleBotRegistry } from '../../agent/bots/registry.js'
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
import type { SchedulerRunFinished } from '../../event-routing/producers/scheduler.js'
import { scheduledAgentExecutionResult } from '../../scheduler/agent-outcome.js'
import {
  createSchedulerDeliveryOutbox,
  createSchedulerDeliveryWorker,
  type SchedulerDeliveryWorker,
} from '../../scheduler/delivery-outbox.js'
import { createInternalJobRunner, ensureInternalJobs } from '../../scheduler/internal-jobs.js'
import { schedulerNotificationAudience } from '../../scheduler/notification-subscriptions.js'
import { nextScheduledRun, parseWhen, SchedulerParseError } from '../../scheduler/time-parser.js'
import { createNotificationsRepo } from '../../notifications/repo.js'
import { SessionInbox } from './session-inbox.js'
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
  providerRegistry: Pick<ProviderRegistry, 'getDefault'> & Partial<Pick<ProviderRegistry, 'get' | 'register'>>
  toolRegistry: ToolRegistry
  skillRegistry?: ISkillRegistry
  policyEngine: PolicyEngine
  sessions: ISessionStore
  getAutonomy: () => AutonomyLevel
  auditLogger: JsonlAuditLogger
  dreaming: DreamingEngine
  semanticIndex: SemanticMemoryStore
  config: SepilotdConfig
  grantUnattendedApproval?: (schedulerSessionId: string) => void
  waitForApproval?: (args: {
    sessionId: string
    toolCall: ToolCall
    requestId: string
    runId?: string
    forcePrompt?: boolean
    signal?: AbortSignal
  }) => ApprovalDecision | Promise<ApprovalDecision>
  onRunFinished?: (finished: SchedulerRunFinished) => void
}

export interface SchedulerStack {
  jobStore: JobStore
  engine: SchedulerEngine
  deliveryWorker: SchedulerDeliveryWorker
  parseWhen: typeof parseWhen
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
  stopReason?: RunStopReason
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
    ...(result.stopReason ? { stopReason: result.stopReason } : {}),
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
    onRunFinished,
  } = deps

  const notifyRunFinished = (
    job: ScheduledJob,
    run: { id: string; durationMs: number | null; error: string | null; outputExcerpt: string | null; stopReason?: string | null; agentSessionId?: string | null },
    status: SchedulerRunFinished['status'],
    trigger: SchedulerRunFinished['trigger'],
    willRetry?: boolean,
  ): void => {
    if (!onRunFinished) return
    if (job.instruction.startsWith('__internal:')) return
    if (scriptMonitorConfigFromMetadata(job.metadata)) return
    try {
      onRunFinished({
        job: { id: job.id, name: job.name, kind: job.kind, channelType: job.channelType },
        run,
        status,
        trigger,
        ...(willRetry !== undefined ? { willRetry } : {}),
      })
    } catch (error) {
      log.error('scheduler onRunFinished listener threw', { error: error instanceof Error ? error.message : String(error) })
    }
  }

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
        createObservabilityRepo().prune()
        notifications.prune()
      },
    },
  })

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
    const configuredProviderId = config.agent.defaultProvider?.trim()
    let provider = configuredProviderId
      ? providerRegistry.get?.(configuredProviderId)
      : providerRegistry.getDefault()
    if (!provider && configuredProviderId) {
      const configuredProvider = config.providers.find((entry) => entry.id === configuredProviderId)
      if (configuredProvider) {
        const built = buildProviderInstance(configuredProvider, createProviderFactoryRegistry())
        if (built.provider) {
          provider = providerRegistry.register?.(configuredProviderId, built.provider) ?? built.provider
        }
      }
    }
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
    const schedulerBudget = resolveRunIterationBudget({
      surface: 'scheduler',
      schedulerMaxIterations: config.scheduler?.maxIterations,
    })
    const engine = new AgentEngine({
      provider,
      tools: toolRegistry,
      policy: policyEngine,
      autonomy: getAutonomy(),
      auditLogger,
      deviceName: config.device.name,
      usageTracker: deps.channelPipelineCapabilities.usageTracker,
      maxIterations: schedulerBudget.maxIterations,
      maxContinuationCycles: schedulerBudget.maxContinuationCycles,
      strictFinalAnswerProtocol:
        process.env.SEPILOTD_STRICT_ANSWER_PROTOCOL === '1'
        || Boolean(skillContext.systemPrompt?.trim()),
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
    let runError: string | undefined
    let observedDone = false
    const stopOnAbort = (): void => {
      void engine.stop().catch(() => undefined)
    }
    try {
      opts.signal?.throwIfAborted()
      opts.signal?.addEventListener('abort', stopOnAbort, { once: true })
      for await (const event of engine.run(instruction, {
        sessionId: opts.agentSessionId ?? opts.sessionId,
        provider: provider.id,
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
        if (event.type === 'error') {
          runError = `${event.error.code}: ${event.error.message}`
        }
        if (event.type === 'done') {
          observedDone = true
          usage.inputTokens = event.usage.inputTokens
          usage.outputTokens = event.usage.outputTokens
          runStopReason = event.stopReason
        }
      }
      opts.signal?.throwIfAborted()
    } catch (error) {
      if (opts.signal?.aborted) {
        await abandonAuditSession()
        throw error
      }
      runError = error instanceof Error ? error.message : String(error)
    } finally {
      opts.signal?.removeEventListener('abort', stopOnAbort)
    }
    if (runError || !observedDone) {
      runStopReason = stopReasonProviderError({
        layer: 'scheduler',
        summary: runError ?? 'Scheduled agent ended without a terminal done event.',
      })
    }
    const failed = runStopReason?.kind === 'error'
    const finalContent = failed
      ? `INCOMPLETE: ${runStopReason?.summary ?? 'Scheduled agent failed.'}`
      : outputTracker.finalContent().trim()
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
    const headlessExit = (): JobExecutionResult | undefined => {
      const carriesUsage = hasSchedulerUsage(usage)
      if (!carriesUsage && !runStopReason) return undefined
      return scheduledAgentExecutionResult({
        rawOutput: '',
        ...(carriesUsage ? { usage } : {}),
        ...(runStopReason ? { stopReason: runStopReason } : {}),
      })
    }
    if (!failed && deliveryTracker.shouldSuppress()) {
      return headlessExit()
    }
    if (!finalContent) {
      return headlessExit()
    }

    if (opts.parentSessionId && !opts.suppressDelivery) {
      try {
        new SessionInbox().tryPublish({ sessionId: opts.parentSessionId, source: 'scheduler', sourceId: opts.sessionId,
          eventKey: opts.sessionId, title: 'Scheduled task result', body: finalContent })
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
      async notify(input) {
        const job = jobStore.get(input.jobId)
        if (job?.parentSessionId && !job.channelType && !job.channelTarget) {
          new SessionInbox().tryPublish({ sessionId: job.parentSessionId, source: 'monitor', sourceId: job.id,
            eventKey: input.id, title: input.title, body: input.body })
          await appendSchedulerChatProgress(job, input.body)
        }
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
      try {
        if (event.type === 'failed') {
          const job = jobStore.get(event.jobId)
          if (job) {
            const run = jobStore.getRun(job.id, event.runId)
            notifyRunFinished(
              job,
              {
                id: event.runId,
                durationMs: event.durationMs,
                error: event.reason,
                outputExcerpt: run?.outputExcerpt ?? null,
                stopReason: run?.stopKind ?? null,
                agentSessionId: run?.agentSessionId ?? null,
              },
              'failed',
              'scheduled',
              event.willRetry,
            )
          }
          if (event.willRetry) return
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
        } else if (event.type === 'interrupted') {
          const job = jobStore.get(event.jobId)
          if (!job) return
          notifyRunFinished(
            job,
            { id: event.runId, durationMs: null, error: event.reason, outputExcerpt: null },
            'interrupted',
            'scheduled',
          )
        } else if (event.type === 'completed') {
          const job = jobStore.get(event.jobId)
          if (!job) return
          const latestRun = jobStore.getRun(job.id, event.runId) ?? jobStore.listRuns(job.id, 1)[0]
          notifyRunFinished(
            job,
            {
              id: event.runId,
              durationMs: event.durationMs,
              error: latestRun?.error ?? null,
              outputExcerpt: latestRun?.outputExcerpt ?? null,
              stopReason: latestRun?.stopKind ?? null,
              agentSessionId: latestRun?.agentSessionId ?? null,
            },
            job.status === 'cancelled' ? 'cancelled' : 'succeeded',
            'scheduled',
          )
          if (job.instruction.startsWith('__internal:')) return
          if (scriptMonitorConfigFromMetadata(job.metadata)) return
          const seconds = Math.max(1, Math.round(event.durationMs / 1000))
          const output = latestRun?.outputExcerpt?.trim()
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
      }
    },
  })

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
      const next = nextScheduledRun(job.cron, Date.now(), timezoneForJob(job), job.metadata)
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
  ): boolean => {
    if (jobStore.get(job.id)?.status !== 'running') return false
    const scheduledFireWasDue = job.nextRunAt <= startedAt

    if (!scheduledFireWasDue) {
      if (error) jobStore.rescheduleAfterFailure(job.id, job.nextRunAt, startedAt, error)
      else jobStore.updateNextRun(job.id, job.nextRunAt, startedAt)
      return false
    }

    if (!error) {
      if (job.kind === 'oneshot') {
        jobStore.updateStatus(job.id, 'completed', { lastRunAt: startedAt })
      } else {
        advanceManualRecurringJob(job, startedAt, null)
      }
      return false
    }

    const attemptNo = job.attempt + 1
    const retryAt = nextRetryAt(Date.now(), attemptNo, {
      maxAttempts: job.maxAttempts,
      backoffMs: job.retryBackoffMs,
      maxBackoffMs: DEFAULT_RETRY_POLICY.maxBackoffMs,
    })
    if (retryAt != null) {
      jobStore.markRetry(job.id, retryAt, attemptNo, error)
      return true
    }

    if (job.kind === 'oneshot') {
      jobStore.updateStatus(job.id, 'failed', { lastRunAt: startedAt, lastError: error })
    } else {
      advanceManualRecurringJob(job, startedAt, error)
    }
    return false
  }

  const executeManualSchedulerJob = async (
    job: ScheduledJob,
    runId: string,
    startedAt: number,
    options: ManualJobRunOptions,
  ): Promise<ManualJobRunResult> => {
    try {
      const { output, outcome, failureReason, stopReason } = normalizeManualSchedulerResult(
        await executor(job, {
          runId,
          forceReport: true,
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
          ...(stopReason ? { stopReason } : {}),
        })
        const willRetry = finalizeManualSchedulerJob(job, startedAt, reason)
        {
          const finishedRun = jobStore.getRun(job.id, runId)
          notifyRunFinished(
            job,
            {
              id: runId,
              durationMs,
              error: reason,
              outputExcerpt: finishedRun?.outputExcerpt ?? null,
              stopReason: finishedRun?.stopKind ?? null,
              agentSessionId: finishedRun?.agentSessionId ?? null,
            },
            'failed',
            'manual',
            willRetry,
          )
        }
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
        ...(stopReason ? { stopReason } : {}),
      })
      finalizeManualSchedulerJob(job, startedAt, null)
      {
        const finishedRun = jobStore.getRun(job.id, runId)
        notifyRunFinished(
          job,
          {
            id: runId,
            durationMs,
            error: null,
            outputExcerpt: finishedRun?.outputExcerpt ?? null,
            stopReason: finishedRun?.stopKind ?? null,
            agentSessionId: finishedRun?.agentSessionId ?? null,
          },
          jobStore.get(job.id)?.status === 'cancelled' ? 'cancelled' : 'succeeded',
          'manual',
        )
      }
      if (!options.suppressDelivery && !job.instruction.startsWith('__internal:') && !scriptMonitorConfigFromMetadata(job.metadata)) {
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
      const willRetry = finalizeManualSchedulerJob(job, startedAt, reason)
      {
        const finishedRun = jobStore.getRun(job.id, runId)
        notifyRunFinished(
          job,
          {
            id: runId,
            durationMs,
            error: reason,
            outputExcerpt: finishedRun?.outputExcerpt ?? null,
            stopReason: finishedRun?.stopKind ?? null,
            agentSessionId: finishedRun?.agentSessionId ?? null,
          },
          'failed',
          'manual',
          willRetry,
        )
      }
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
  roleBots: RoleBotRegistry
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
  const knownEngines = () => graphRegistry.list().map((graph) => graph.id)
  if (!dataDir) {
    const roleBots = new RoleBotRegistry({ knownEngines })
    await roleBots.init()
    return { graphRegistry, roleBots }
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

  const roleBots = new RoleBotRegistry({ botsDir: join(dataDir, 'bots'), knownEngines })
  await roleBots.init()

  return {
    graphRegistry,
    graphAgentLoader,
    userAgentLoader,
    userCommandStore,
    roleBots,
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
