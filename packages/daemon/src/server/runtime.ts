import { createJournalTools } from '../tools/journal.js'
import { forgetAllOwnedMemory } from '../memory/forget-all.js'
import { readFile } from 'node:fs/promises'
import type { SepilotdConfig } from '../config/schema.js'
import { parseConfig } from '../config/loader.js'
import { applyRuntimeChannelEnvironment } from '../config/runtime-channel-env.js'
import { applyRuntimeGatewayEnvironment } from '../config/runtime-gateway-env.js'
import { ChannelRouter } from '../channels/router.js'
import { createLogger } from '../logger.js'
import { createNotificationRelayDeliveryWorker } from '../notifications/publish.js'
import { planProviderDispatcher } from '../providers/http-timeout.js'
import { ConfigWatcher } from './runtime/config-watcher.js'
import {
  cloneRuntimeConfigSnapshot,
  isSameRuntimeConfigSnapshot,
} from './runtime/config-snapshot.js'
import {
  buildProviderRegistry,
  syncDynamicProviderModels,
  createProviderFactoryRegistry,
  resolveAutonomy,
} from './runtime/providers.js'
import { buildSkillRegistry, buildSkillStore, buildMarketplaceCatalog } from './runtime/storage.js'
import { assembleStorageLayer } from './runtime/storage-layer.js'
import { assembleSecurityLayer } from './runtime/security-layer.js'
import { assembleAgentLayer } from './runtime/agent-layer.js'
import { assembleIntegrationLayer } from './runtime/integration-layer.js'
import { assembleMcpLayer } from './runtime/mcp-layer.js'
import { assembleChannelLayer } from './runtime/channel-layer.js'
import { ConfigMutationService } from './runtime/config-mutation-service.js'
import { deepFreeze } from './runtime/deep-freeze.js'
import {
  persistRuntimeConfig,
  reconfigureRuntimeAutonomy,
  reconfigureRuntimeChannelType,
  reconfigureRuntimeExtensions,
  reconfigureRuntimeSkillSources,
  restoreRuntimeNetworkPolicy,
} from './runtime/config-runtime.js'
import { UrlSource } from '../skills/sources/url.js'
import { GitSource } from '../skills/sources/git.js'
import { MarketplaceSource } from '../skills/sources/marketplace.js'
import { createInstallPipeline } from '../skills/install-pipeline.js'
import { SkillSourceUrlPolicy } from '../skills/source-url-policy.js'
import { buildToolRegistry } from './runtime/tools.js'
import {
  registerDelegationRuntimeTool,
  registerMicroAppsRuntimeTools,
  createA2aRuntime,
  createAcpRuntime,
} from '../generated/feature-registration.js'
import type { FeatureRuntimeToolDeps } from '../features/types.js'
import { createTodoWriteTool } from '../tools/todo.js'
import { TaskBoardStore } from '../memory/task-board.js'
import { createQuestionTool, PendingQuestionStore } from '../tools/question.js'
import { createCodeDiagnosticsTool, createCodeSymbolsTool } from '../tools/code-analysis.js'
import {
  createMemoryAccessHotTool,
  createMemoryAccessTouchTool,
  createMemoryAuditTool,
  createMemoryConflictsFindTool,
  createMemoryPinTool,
  createMemoryUnpinTool,
  createMemoryPinnedListTool,
  createMemoryContextSnapshotTool,
  createMemoryDailyAppendTool,
  createMemoryDailyListTool,
  createMemoryDailyReadTool,
  createMemoryDailyReplaceTool,
  createMemoryDailySearchTool,
  createMemoryDiffTool,
  createMemoryDocumentsDeleteTool,
  createMemoryDocumentsGetTool,
  createMemoryDocumentsIngestTool,
  createMemoryDocumentsListTool,
  createMemoryDocumentsPreviewTool,
  createMemoryDocumentsSearchTool,
  createMemoryDocumentsUpdateTool,
  createMemoryExportTool,
  createMemoryForgetTool,
  createMemoryGraphAuditTool,
  createMemoryGraphNeighborsTool,
  createMemoryGraphPageTool,
  createMemoryGraphRepairApplyTool,
  createMemoryGraphRepairTool,
  createMemoryGraphSearchTool,
  createMemoryImportTool,
  createMemoryListTool,
  createMemoryMaintenanceTool,
  createMemoryRememberTool,
  createMemoryRemindAtTool,
  createMemoryRemindersCancelTool,
  createMemoryRemindersListTool,
  createMemoryHistoryTool,
  createMemoryMergeTool,
  createMemorySearchByTagTool,
  createMemorySearchRelatedTool,
  createMemorySearchTool,
  createMemorySectionReplaceTool,
  createMemorySummarizeTool,
  createMemoryTagListTool,
  createMemoryTagRenameTool,
  createMemoryTagSuggestTool,
  createMemoryUpdateTool,
  createMemoryUsageTool,
} from '../tools/memory.js'
import { JsonlRemindersStore, RemindersScheduler, type Reminder } from '../memory/reminders.js'
import { ScopedFileMemoryRegistryImpl } from '../memory/scoped-file-memory.js'
import { createUsageReportTool } from '../tools/usage-report.js'
import { CustomDefsService } from '../agent/custom/service.js'
import { PrimaryAgentStore } from '../agent/primary-agent-store.js'
import { isAutoApproveActive } from '../security/auto-approve.js'
import { buildLspLayer } from '../lsp/layer.js'
import { createLspTool } from '../tools/lsp-tool.js'
import { pathToFileURL } from 'node:url'
import { join } from 'node:path'
import {
  buildDreamingEngine,
  buildGatewayClient,
  buildGraphRegistry,
  buildHookRegistry,
  buildPluginLoader,
  enableJsonLoggingIfNeeded,
} from './runtime/services.js'
import { createChannelFactoryRegistry } from './runtime/channels.js'
import { startRuntime } from './runtime/lifecycle.js'
import { assembleApprovalsLayer } from './runtime/approvals-layer.js'
import { DelegationWorker } from '../agent/delegation-worker.js'
import { AgentEngine } from '../agent/engine.js'
import { SubagentDispatcher } from '../agent/subagent-dispatcher.js'
import { createSubagentDispatchTool, createSubagentJobTool } from '../tools/subagent-dispatch.js'
import { assertCustomAgentUsable } from '../agent/custom/agents.js'
import {
  createSelfInfoTool,
  createSkillHubInstallTool,
  createSkillHubSearchTool,
} from '../tools/self-introspection.js'
import { createAssistantStatusTool } from '../tools/assistant-status.js'
import { buildAssistantRuntimeStatus } from './routes/system.js'
import { ServiceSupervisor } from '../service-supervisor/supervisor.js'
import {
  SwarmRunRegistry,
  SwarmRunStore,
  TmuxSessionPool,
  AgentLauncher,
  createDefaultSwarmAgentRuntimeAdapter,
  WorktreeManager,
  registerSwarmTools,
  driveSwarmAgent,
  type SwarmDriveInput,
} from '../agent/swarm/index.js'
import {
  createSwarmStartupFailureEvidence,
  createSwarmStartupOutputEvidence,
} from '../agent/swarm/run/startup-evidence.js'
import { runSwarmAgentPreflight, summarizeSwarmPreflight } from '../agent/swarm/tools/preflight.js'
import {
  listDaemonSwarmWorktrees,
  listTmuxSessionNames,
  reapOrphanSwarmResources,
  removeDaemonSwarmWorktree,
  removeTmuxSession,
} from '../agent/swarm/reaper.js'
import { cleanAnsi } from '../agent/swarm/tmux/capture.js'
import { buildSystemPrompt } from '../agent/system-prompt.js'
import { buildSwarmSupervisorTaskPrompt } from '../agent/swarm/system-prompt.js'
import { evaluateSwarmKeyForwarding } from '../security/swarm-key-policy.js'
import { randomBytes } from 'node:crypto'
import type { AgentContext, AgentEvent, SwarmAgentHandle, SwarmAgentName } from '@sepilotd/core'
import { registerHostSystemInfoCapability } from '../capabilities/host-system-info.js'
import { buildPendingApprovals } from './approval-state.js'
import { SessionRuntimeSnapshotStore } from './runtime/session-runtime-snapshots.js'
import { createObservabilityRepo } from '../observability/events.js'
import { ActiveRunRegistry } from './runtime/active-runs.js'
import { SessionBusyRegistry } from './runtime/session-busy.js'

export type { RuntimeServices } from './runtime/types.js'
import type { RuntimeServices } from './runtime/types.js'

const log = createLogger('runtime')

export interface BuildRuntimeOptions {
  autoApproveCliFlag?: boolean
  configLoadFailed?: boolean
  configLoadError?: string
}

export async function buildRuntime(
  config: SepilotdConfig,
  dataDir: string,
  configPath?: string,
  options: BuildRuntimeOptions = {},
): Promise<RuntimeServices> {
  let runtimeRef: RuntimeServices | null = null
  const {
    policyEngine,
    autonomy,
    auditLogger,
    extensionTokenStore,
    channelAcl,
    encryption,
    secretVault,
  } = await assembleSecurityLayer({ config, dataDir })

  // Independent I/O-bound builders — run concurrently so a slow gateway probe
  // does not serialize ahead of the skill-registry scan (both are needed before
  // the tool registry below). Promise.all preserves fail-fast: a builder that
  // throws still rejects boot, exactly as the serial awaits did.
  const [gatewayClient, skillRegistry] = await Promise.all([
    buildGatewayClient(config, dataDir),
    buildSkillRegistry(dataDir, { trustProjectSkills: config.skills?.trustProjectSkills ?? false }),
  ])
  const observability = createObservabilityRepo()
  const customDefs = new CustomDefsService()
  const serviceSupervisor = new ServiceSupervisor({ rootDir: join(dataDir, 'services') })
  const { toolRegistry, delegator, managedProcesses } = buildToolRegistry(
    config,
    gatewayClient,
    skillRegistry,
    {
      dataDir,
      serviceSupervisor,
      getWebSearchTrustedDomains: () =>
        runtimeRef?.config.webSearch.trustedDomains ?? config.webSearch.trustedDomains,
    },
  )
  // Wire validator into skill registry (after toolRegistry + policyEngine exist)
  skillRegistry.setValidator({
    toolRegistry,
    policyEngine,
    autonomy: () => resolveAutonomy(config),
  })

  // marketplace catalog and graph registry are independent of each other and
  // of the code between here and their first use; build them concurrently.
  const [marketplaceCatalog, graphBuild] = await Promise.all([
    buildMarketplaceCatalog(dataDir),
    buildGraphRegistry(dataDir),
  ])
  const { graphRegistry, graphAgentLoader, userAgentLoader, userCommandStore } = graphBuild
  const skillSourceUrlPolicy = new SkillSourceUrlPolicy(config.security.skillSources)
  const installPipeline = createInstallPipeline({
    registry: skillRegistry,
    urlSource: new UrlSource({ urlPolicy: skillSourceUrlPolicy }),
    gitSource: new GitSource({ urlPolicy: skillSourceUrlPolicy }),
    marketplaceSource: new MarketplaceSource({
      catalog: marketplaceCatalog,
      urlPolicy: skillSourceUrlPolicy,
    }),
  })
  const hookRegistry = buildHookRegistry(config, {
    auditLogger,
    deviceName: config.device.name,
  })
  // Managed children can outlive the tool call that created them, so their
  // start/exit notifications are emitted by the lifecycle owner itself.
  managedProcesses.setHookRegistry(hookRegistry)
  registerHostSystemInfoCapability({
    config,
    tools: toolRegistry,
  })
  const providerRegistryRef: { current?: ReturnType<typeof buildProviderRegistry> } = {}
  const {
    mcpManager,
    mcpPromptsRegistry,
    mcpMarketplaceCatalog,
    mcpMarketplaceSource,
    mcpConfigWriter,
  } = await assembleMcpLayer({
    dataDir,
    configPath,
    auditLogger,
    secretVault,
    deviceName: config.device.name,
    featureDeps: {
      getProviderRegistry: () => providerRegistryRef.current,
    },
  })
  const pluginLoader = await buildPluginLoader(dataDir, config)
  const providerFactoryRegistry = createProviderFactoryRegistry()
  const channelFactoryRegistry = createChannelFactoryRegistry()
  const { PluginEventBus } = await import('../plugins/event-bus.js')
  const pluginEvents = new PluginEventBus()

  await pluginLoader.loadAll({
    providers: providerFactoryRegistry,
    channels: channelFactoryRegistry,
    tools: toolRegistry,
    hooks: hookRegistry,
    skills: skillRegistry,
    graphs: graphRegistry,
    events: pluginEvents,
  })

  const providerRegistry = buildProviderRegistry(config, providerFactoryRegistry)
  providerRegistryRef.current = providerRegistry
  await syncDynamicProviderModels(providerRegistry, config)
  await mcpManager.configureServers(
    config.mcp.servers,
    toolRegistry,
    mcpPromptsRegistry,
    config.mcp.client,
  )

  const {
    sessions,
    sessionWatchBroker,
    fileMemory,
    semanticIndex,
    usageTracker,
    channelPipelineMonitor,
    channelOriginStore,
    devicePairingRegistry,
    channelReplayStore,
    channelSessionStore,
    degradedStores: storageDegraded,
  } = await assembleStorageLayer({
    config,
    dataDir,
    encryption,
    providerRegistry,
  })
  // Second-pass feature tool registration (delegation w/ session append, apps
  // tools w/ semantic index, a2a runtime). Routed through the generated
  // manifest so disabled features drop from the bundle. a2aTaskStore is null
  // when the a2a feature is disabled.
  const featureRuntimeDeps: FeatureRuntimeToolDeps = {
    dataDir,
    delegator,
    semanticIndex,
    sessions,
    deviceName: config.device.name,
  }
  registerDelegationRuntimeTool(toolRegistry, featureRuntimeDeps)
  registerMicroAppsRuntimeTools(toolRegistry, featureRuntimeDeps)
  const a2aTaskStore = createA2aRuntime(toolRegistry, featureRuntimeDeps)
  toolRegistry.register(createUsageReportTool(usageTracker))

  const questionStore = new PendingQuestionStore()
  questionStore.setOnEnqueue((q) => {
    sessionWatchBroker.emit({
      type: 'question_requested',
      sessionId: q.sessionId,
      question: { id: q.id, prompt: q.prompt, choices: q.choices },
    })
  })
  questionStore.setOnAnswer((q) => {
    sessionWatchBroker.emit({
      type: 'question_answered',
      sessionId: q.sessionId,
      questionId: q.id,
    })
  })
  const swarmStore = new SwarmRunStore(join(dataDir, 'sessions', 'swarm'))
  const swarmTmuxPool = new TmuxSessionPool()
  const swarmLauncher = new AgentLauncher(swarmTmuxPool)
  const swarmAgentRuntime = createDefaultSwarmAgentRuntimeAdapter(swarmTmuxPool, swarmLauncher)
  const swarmWorktreeManager = new WorktreeManager()
  const swarmRunRegistry = new SwarmRunRegistry(swarmStore, swarmAgentRuntime, (worktree) =>
    swarmWorktreeManager.remove(worktree),
  )
  // Track per-run worktrees so cancel/cleanup paths (which don't run the
  // engine.run finally block) can still tear them down.
  const swarmRunWorktrees = new Map<string, ReturnType<WorktreeManager['create']>>()
  void reapOrphanSwarmResources({
    listTmuxSessions: listTmuxSessionNames,
    removeTmuxSession,
    listWorktrees: () => listDaemonSwarmWorktrees(),
    removeWorktree: removeDaemonSwarmWorktree,
    activeRunIds: new Set(swarmRunRegistry.list().map((run) => run.id)),
  })
    .then((result) => {
      if (result.sessions.length || result.worktrees.length) {
        log.warn('reaped orphan swarm resources', {
          sessions: result.sessions,
          worktrees: result.worktrees,
        })
      }
    })
    .catch((err) => {
      log.warn('failed to reap orphan swarm resources', {
        error: err instanceof Error ? err.message : String(err),
      })
    })
  registerSwarmTools(toolRegistry, {
    registry: swarmRunRegistry,
    pool: swarmTmuxPool,
    launcher: swarmLauncher,
    runtime: swarmAgentRuntime,
    resolveRunId: (sid) => (typeof sid === 'string' && sid.startsWith('swarm_') ? sid : null),
  })

  // Task 20: real swarm run + key forwarding implementations.
  // The closures capture the locals built so far in buildRuntime; they're only
  // invoked from HTTP handlers after the runtime object is fully constructed.
  const startSwarmRun: RuntimeServices['startSwarmRun'] = async (run, opts) => {
    // 0) Verify the supervisor engine has a provider before doing any side
    //    effects (tmux sessions are expensive to roll back). noSupervisor
    //    sessions are user-driven, so they only need the external agent CLIs.
    const supervisorProvider = opts.noSupervisor ? null : providerRegistry.getDefault()
    if (!opts.noSupervisor && !supervisorProvider) {
      throw new Error('SWARM_NO_PROVIDER: no LLM provider configured')
    }

    // 1) Transition the run state machine into 'running' FIRST so the jsonl gets
    //    `run.started` as the first event (before any agent.spawned/active lines).
    run.start()

    let launchCwd = run.snapshot().worktree.path
    try {
      // 2) Create a worktree (git worktree on a repo, fall back to the cwd as-is on
      //    non-git directories). All warm-pool launches use the worktree path so
      //    parallel agents don't trample each other's checkouts.
      const wt = swarmWorktreeManager.create(run.snapshot().worktree.path, run.id)
      run.setWorktree(wt)
      swarmRunWorktrees.set(run.id, wt)
      launchCwd = wt.path

      // 3) Spawn warm pool agents up front so the supervisor has something to drive.
      for (const agent of opts.warmPool) {
        const handleId = `a_${randomBytes(3).toString('hex')}`
        const startedAt = Date.now()
        const preflight = runSwarmAgentPreflight({
          agent: agent as SwarmAgentName,
          cwd: launchCwd,
          env: process.env,
        })
        if (preflight.status === 'blocked') {
          const error = new Error(`SWARM_PREFLIGHT_BLOCKED: ${summarizeSwarmPreflight(preflight)}`)
          run.recordStartupEvidence(
            handleId,
            createSwarmStartupFailureEvidence({
              handle: handleId,
              agent: agent as SwarmAgentName,
              cwd: launchCwd,
              runtime: preflight.runtime,
              startedAt,
              error,
            }),
          )
          throw error
        }
        let handle: SwarmAgentHandle
        try {
          handle = await swarmAgentRuntime.launch({
            runId: run.id,
            handle: handleId,
            agent: agent as SwarmAgentName,
            cwd: launchCwd,
            autoApprove: opts.autoApproveAgents,
          })
        } catch (error) {
          run.recordStartupEvidence(
            handleId,
            createSwarmStartupFailureEvidence({
              handle: handleId,
              agent: agent as SwarmAgentName,
              cwd: launchCwd,
              startedAt,
              error,
            }),
          )
          throw error
        }
        run.addAgent(handle)
        if (!run.snapshot().activeHandle) run.setActive(handle.handle)
      }

      // 4a) "noSupervisor" mode: the user is driving via the keys endpoint
      //     directly (or via attach). Skip the engine kickoff and the
      //     accompanying cleanup-on-finish; the run lives until DELETE.
      if (opts.noSupervisor) {
        return
      }
      const provider = supervisorProvider
      if (!provider) {
        throw new Error('SWARM_NO_PROVIDER: no LLM provider configured')
      }
      const model = config.agent.defaultModel ?? provider.models[0]?.id ?? 'default'

      // 4) Kick off the supervisor agent loop asynchronously. The engine drives
      //    the goal forward via the swarm.* tools registered above; we just drain
      //    its event stream and finish the run when it completes.

      // Create a daemon session matching the run id so the swarm system-prompt
      // extension fires (it keys off `sessionId.startsWith('swarm_')`).
      const createdAt = new Date().toISOString()
      try {
        await sessions.create({
          id: run.id,
          title: `swarm: ${run.goal.slice(0, 40)}`,
          createdAt,
          updatedAt: createdAt,
          provider: provider.id,
          model,
          device: 'swarm',
          status: 'active',
          tags: ['swarm'],
        })
      } catch {
        // Session may already exist on a retry — non-fatal.
      }

      // Build the supervisor system prompt up front. The swarm extension only
      // fires when sessionId starts with `swarm_` (see swarmSystemPromptExtension).
      const systemPrompt = await buildSystemPrompt({
        config,
        tools: toolRegistry,
        skills: skillRegistry,
        fileMemory: undefined, // swarm runs have no per-session memory yet
        sessionId: run.id,
        cwd: launchCwd,
      })

      const engine = new AgentEngine({
        provider,
        tools: toolRegistry,
        policy: policyEngine,
        autonomy: resolveAutonomy(config),
        auditLogger,
        usageTracker,
        hookRegistry,
        deviceName: config.device.name,
        llmCache,
        providerCircuitBreaker,
      })

      void (async () => {
        const context: AgentContext = {
          sessionId: run.id,
          provider: provider.id,
          model,
          cwd: launchCwd,
          autoApprove: opts.autoApproveAgents,
          systemPrompt,
        }
        let sawError = false
        try {
          const supervisorTask = buildSwarmSupervisorTaskPrompt(run.goal)
          for await (const event of engine.run(
            supervisorTask,
            context,
          ) as AsyncIterable<AgentEvent>) {
            if (event.type === 'error') sawError = true
          }
          run.finish(sawError ? 'error' : 'done')
        } catch (err) {
          log.error('swarm run failed', {
            runId: run.id,
            error: err instanceof Error ? err.message : String(err),
          })
          run.finish('error')
        } finally {
          // Tear down all warm-pool tmux sessions, remove the worktree, and
          // deregister the run from the in-memory map so we don't leak.
          for (const agent of run.snapshot().agents) {
            try {
              await swarmAgentRuntime.stop(agent)
            } catch {
              /* swallow — best effort */
            }
          }
          const trackedWt = swarmRunWorktrees.get(run.id)
          if (trackedWt) {
            try {
              swarmWorktreeManager.remove(trackedWt)
            } catch {
              /* swallow */
            }
            swarmRunWorktrees.delete(run.id)
          }
          swarmRunRegistry.deregister(run.id)
        }
      })()
    } catch (err) {
      for (const agent of run.snapshot().agents) {
        try {
          await swarmAgentRuntime.stop(agent)
        } catch {
          /* best-effort */
        }
      }
      const trackedWt = swarmRunWorktrees.get(run.id)
      if (trackedWt) {
        try {
          swarmWorktreeManager.remove(trackedWt)
        } catch {
          /* best-effort */
        }
        swarmRunWorktrees.delete(run.id)
      }
      throw err
    }
  }

  const forwardSwarmKeys: RuntimeServices['forwardSwarmKeys'] = async (runId, handle, payload) => {
    const r = swarmRunRegistry.get(runId)
    if (!r) throw new Error(`run not found: ${runId}`)
    const agent = r.getAgent(handle)
    if (!agent) throw new Error(`agent not found: ${handle}`)
    const decision = evaluateSwarmKeyForwarding({
      runId,
      handle,
      keys: payload.keys,
      keyName: payload.keyName,
      enter: payload.enter,
      resize: payload.resize,
      principal: payload.principal,
      device: config.device.name,
      env: process.env,
    })
    try {
      await auditLogger.log(decision.auditEvent)
    } catch (err) {
      log.warn('failed to audit swarm key forwarding', {
        runId,
        handle,
        error: err instanceof Error ? err.message : String(err),
      })
    }
    if (!decision.allowed) {
      throw new Error(`SWARM_KEY_DENIED: ${decision.reason ?? 'denied'}`)
    }
    if (payload.resize) {
      await swarmAgentRuntime.resize(agent, payload.resize.cols, payload.resize.rows)
      return
    }
    if (typeof payload.keyName === 'string') {
      await swarmAgentRuntime.sendNamedKeys(agent, payload.keyName)
      return
    }
    if (typeof payload.keys === 'string') {
      await swarmAgentRuntime.sendKeys(agent, payload.keys, payload.enter ?? false)
    }
  }

  const captureSwarmAgent: RuntimeServices['captureSwarmAgent'] = async (
    runId,
    handle,
    options,
  ) => {
    const r = swarmRunRegistry.get(runId)
    if (!r) throw new Error(`run not found: ${runId}`)
    const agent = r.getAgent(handle)
    if (!agent) throw new Error(`agent not found: ${handle}`)
    const lines =
      typeof options?.lines === 'number'
        ? Math.max(1, Math.min(5000, Math.floor(options.lines)))
        : 200
    const text = await swarmAgentRuntime.capture(agent, lines, { raw: options?.raw })
    const evidenceText = options?.raw ? cleanAnsi(text) : text
    const evidence = createSwarmStartupOutputEvidence(agent, evidenceText)
    if (evidence) {
      r.recordStartupEvidence(agent.handle, evidence)
      r.setStatus(agent.handle, 'blocked')
    }
    return text
  }

  const killSwarmRunAgents: RuntimeServices['killSwarmRunAgents'] = async (runId) => {
    const r = swarmRunRegistry.get(runId)
    if (!r) return
    for (const agent of r.snapshot().agents) {
      try {
        await swarmAgentRuntime.stop(agent)
      } catch {
        /* best-effort */
      }
    }
    // Worktree was created in startSwarmRun and would normally be removed by
    // the engine's finally block. In noSupervisor mode (or any cancel path
    // that races the engine), this is the last hook we have to clean it up.
    const wt = swarmRunWorktrees.get(runId)
    if (wt) {
      try {
        swarmWorktreeManager.remove(wt)
      } catch {
        /* best-effort */
      }
      swarmRunWorktrees.delete(runId)
    }
  }

  const driveSwarmRunAgent: RuntimeServices['driveSwarmAgent'] = async (
    runId: string,
    input: SwarmDriveInput,
  ) =>
    driveSwarmAgent(
      {
        registry: swarmRunRegistry,
        pool: swarmTmuxPool,
        launcher: swarmLauncher,
        runtime: swarmAgentRuntime,
        resolveRunId: () => runId,
      },
      runId,
      input,
    )

  toolRegistry.register(
    createTodoWriteTool({
      appendEvent: (sessionId, event) => sessions.appendEvent(sessionId, event),
      taskBoard: new TaskBoardStore(join(dataDir, 'state')),
    }),
  )
  toolRegistry.register(createQuestionTool({ store: questionStore }))

  // FileMemory bucket factory: per-user/channel scoping. The existing
  // `fileMemory` from assembleStorageLayer reads `${dataDir}/memory` and is
  // exposed below as the registry's `global()` bucket; user/channel-scoped
  // writes land under `${dataDir}/memory/scopes/<key>/`.
  const fileMemoryRegistry = new ScopedFileMemoryRegistryImpl(`${dataDir}/memory`)
  await semanticIndex.configureFileMemory?.(fileMemoryRegistry)

  toolRegistry.register(
    createMemoryRememberTool({
      fileMemoryRegistry,
      semanticIndex,
    }),
  )
  toolRegistry.register(createMemoryListTool({ fileMemoryRegistry, semanticIndex }))
  toolRegistry.register(createMemoryForgetTool({
    fileMemoryRegistry,
    forgetAll: (scopeTags, options) => dreaming.withMemoryReset(() => forgetAllOwnedMemory(semanticIndex, fileMemoryRegistry, scopeTags, options)),
  }))
  toolRegistry.register(createMemorySearchTool({ semanticIndex }))
  toolRegistry.register(createMemoryGraphSearchTool({ semanticIndex }))
  toolRegistry.register(createMemoryGraphNeighborsTool({ semanticIndex }))
  toolRegistry.register(createMemoryGraphPageTool({ semanticIndex }))
  toolRegistry.register(createMemoryGraphAuditTool({ semanticIndex }))
  toolRegistry.register(createMemoryGraphRepairTool({ semanticIndex }))
  toolRegistry.register(createMemoryGraphRepairApplyTool({ semanticIndex }))
  toolRegistry.register(createMemoryDailyReadTool({ fileMemoryRegistry }))
  toolRegistry.register(createMemoryDailyListTool({ fileMemoryRegistry }))
  toolRegistry.register(createMemoryDailySearchTool({ fileMemoryRegistry }))
  toolRegistry.register(createMemoryDailyAppendTool({ fileMemoryRegistry }))
  toolRegistry.register(createMemoryDailyReplaceTool({ fileMemoryRegistry }))
  toolRegistry.register(createMemorySectionReplaceTool({ fileMemoryRegistry }))
  toolRegistry.register(createMemoryUpdateTool({ semanticIndex }))
  toolRegistry.register(createMemoryDocumentsIngestTool({ documentStore: semanticIndex }))
  toolRegistry.register(createMemoryDocumentsSearchTool({ documentStore: semanticIndex }))
  toolRegistry.register(createMemoryDocumentsListTool({ documentStore: semanticIndex }))
  toolRegistry.register(createMemoryDocumentsDeleteTool({ documentStore: semanticIndex }))
  toolRegistry.register(createMemoryDocumentsGetTool({ documentStore: semanticIndex }))
  toolRegistry.register(createMemoryDocumentsUpdateTool({ documentStore: semanticIndex }))
  toolRegistry.register(createMemoryDocumentsPreviewTool({ documentStore: semanticIndex }))
  toolRegistry.register(createMemoryAuditTool({ semanticIndex }))
  toolRegistry.register(createMemoryAccessTouchTool({ semanticIndex }))
  toolRegistry.register(createMemoryAccessHotTool({ semanticIndex }))
  toolRegistry.register(createMemoryPinTool({ semanticIndex }))
  toolRegistry.register(createMemoryUnpinTool({ semanticIndex }))
  toolRegistry.register(createMemoryPinnedListTool({ semanticIndex }))
  toolRegistry.register(createMemorySearchRelatedTool({ semanticIndex }))
  toolRegistry.register(createMemorySearchByTagTool({ semanticIndex }))
  toolRegistry.register(createMemoryHistoryTool({ semanticIndex }))
  toolRegistry.register(createMemoryTagListTool({ semanticIndex }))
  toolRegistry.register(createMemoryDiffTool({ semanticIndex }))
  toolRegistry.register(createMemoryTagRenameTool({ semanticIndex }))
  toolRegistry.register(createMemoryMaintenanceTool({ semanticIndex }))
  for (const tool of createJournalTools(fileMemoryRegistry)) toolRegistry.register(tool)

  const remindersStore = new JsonlRemindersStore(join(dataDir, 'memory', 'reminders.jsonl'))
  const remindersScheduler = new RemindersScheduler(remindersStore, {
    onDue: async (reminder: Reminder) => {
      try {
        await deliverReminder(reminder)
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err)
        // Re-throw so the scheduler keeps the reminder pending instead of
        // marking it fired when delivery failed.
        throw new Error(`reminder ${reminder.id} delivery failed: ${message}`)
      }
    },
  })
  remindersScheduler.start()
  toolRegistry.register(createMemoryRemindAtTool({ reminders: remindersStore }))
  toolRegistry.register(createMemoryRemindersListTool({ reminders: remindersStore }))
  toolRegistry.register(createMemoryRemindersCancelTool({ reminders: remindersStore }))
  toolRegistry.register(
    createMemoryUsageTool({
      semanticIndex,
      reminders: remindersStore,
      fileMemoryRegistry,
    }),
  )
  toolRegistry.register(
    createMemoryContextSnapshotTool({
      fileMemoryRegistry,
      semanticIndex,
      reminders: remindersStore,
    }),
  )

  // Dreaming engine is built later (depends on agentLayer); capture a
  // deferred reference here so memory.tag.suggest / memory.conflicts.find
  // can call into it.
  const dreamingRef: {
    current?: {
      suggestTags(content: string): Promise<string[]>
      findRecentContradictions(options?: { maxPairs?: number }): Promise<
        Array<{
          winnerId: string
          supersededIds: string[]
          primaryId: string
          candidateIds: string[]
        }>
      >
      summarizeMemories(
        contents: string[],
        options?: { instruction?: string; targetChars?: number },
      ): Promise<string | null>
    }
  } = {}
  toolRegistry.register(
    createMemoryTagSuggestTool({
      suggestTags: async (content: string) => dreamingRef.current?.suggestTags(content) ?? [],
    }),
  )
  toolRegistry.register(
    createMemoryConflictsFindTool({
      findRecentContradictions: async (opts) =>
        dreamingRef.current?.findRecentContradictions(opts) ?? [],
    }),
  )
  toolRegistry.register(
    createMemorySummarizeTool({
      semanticIndex,
      fileMemoryRegistry,
      summarize: async (contents, options) =>
        dreamingRef.current?.summarizeMemories(contents, options) ?? null,
    }),
  )
  toolRegistry.register(
    createMemoryMergeTool({
      semanticIndex,
      summarize: async (contents, options) =>
        dreamingRef.current?.summarizeMemories(contents, options) ?? null,
    }),
  )
  toolRegistry.register(
    createMemoryExportTool({
      fileMemoryRegistry,
      semanticIndex,
      reminders: remindersStore,
    }),
  )
  toolRegistry.register(
    createMemoryImportTool({
      fileMemoryRegistry,
      semanticIndex,
      reminders: remindersStore,
    }),
  )

  async function deliverReminder(reminder: Reminder): Promise<void> {
    if (!reminder.channelType || !reminder.chatId) {
      throw new Error('reminder has no channel routing info')
    }
    const channel = runtime.channels.find((c) => c.type === reminder.channelType)
    if (!channel || typeof channel.sendMessage !== 'function') {
      throw new Error(`no live channel for type ${reminder.channelType}`)
    }
    await channel.sendMessage(
      { id: reminder.chatId, type: 'channel' },
      {
        text: `⏰ Reminder: ${reminder.content}`,
      },
    )
  }

  const lspLayer = await buildLspLayer({
    rootUri: pathToFileURL(process.cwd()).toString(),
    enabled: true,
  })
  toolRegistry.register(createLspTool({ getClient: (lang) => lspLayer.get(lang) }))
  toolRegistry.register(
    createCodeDiagnosticsTool({
      getClient: (lang) => lspLayer.get(lang),
    }),
  )
  // Re-register code.symbols (registered LSP-unaware in buildToolRegistry,
  // before lspLayer existed) so a supplied `language` uses precise
  // workspace/symbol results, falling back to ripgrep otherwise.
  toolRegistry.register(createCodeSymbolsTool({ getClient: (lang) => lspLayer.get(lang) }))

  const { telemetry, mdns, updater } = await assembleIntegrationLayer({ config })
  const {
    llmCache,
    providerCircuitBreaker,
    modelFallbackState,
    runLimiter,
    modelRouter,
    editSnapshotStore,
    toolStatsStore,
    workspaceMutationTracker,
    sessionUndoStack,
  } = assembleAgentLayer({
    providerRegistry,
    dataDir,
    // The observability schema has always defined tool.succeeded/tool.failed and
    // aggregated them into the tool_succeeded/tool_failed metrics, but nothing
    // ever emitted them — so tool reliability read as a flat zero no matter how
    // many calls failed. Emit identity and timing only; outputs can carry user
    // content and stay in the session record.
    onToolOutcome: (outcome) => {
      observability.recordEvents([
        {
          source: 'daemon',
          surface: 'agent',
          eventType: outcome.status === 'success' ? 'tool.succeeded' : 'tool.failed',
          severity: outcome.status === 'success' ? 'debug' : 'warning',
          privacy: 'operational',
          sessionId: outcome.sessionId,
          attributes: {
            tool: outcome.tool,
            durationMs: outcome.durationMs,
          },
        },
      ])
    },
  })
  const skillStore = await buildSkillStore(dataDir)
  const dreaming = buildDreamingEngine(
    sessions,
    semanticIndex,
    providerRegistry,
    fileMemory,
    fileMemoryRegistry,
    config.agent,
  )
  dreamingRef.current = dreaming
  const {
    approvalDecisions,
    approvalRegistry,
    approvalCheckpoints,
    runCheckpoints,
    toolExecutions,
  } = await assembleApprovalsLayer({ dataDir, sessions, config })
  const activeRuns = new ActiveRunRegistry((sessionId, running) => {
    sessionWatchBroker.emit({
      type: 'run_status_changed',
      sessionId,
      running,
    })
  })
  const sessionBusy = new SessionBusyRegistry()
  const sessionRuntimeSnapshots = new SessionRuntimeSnapshotStore({
    sessions,
    getPendingState: async (session, events) => ({
      pendingApprovals: await Promise.all(
        buildPendingApprovals(session.id, events, approvalRegistry.listForSession(session.id)).map(
          async (approval) => ({
            ...approval,
            resumeAvailable: await approvalCheckpoints.has(approval.requestId),
          }),
        ),
      ),
      pendingQuestions: questionStore.list(session.id).map((question) => ({
        id: question.id,
        prompt: question.prompt,
        choices: question.choices,
      })),
    }),
  })
  sessionRuntimeSnapshots.attach(sessionWatchBroker)
  await channelSessionStore.pruneDetached((sessionId) => sessions.get(sessionId))

  await enableJsonLoggingIfNeeded(config)

  // Top-level config sections the file-watcher reload hot-applies (via
  // onConfigReloaded / onMcpChange). Changes to these do NOT require a restart.
  const FILE_WATCH_HOT_APPLIED_SECTIONS = new Set<string>([
    'network',
    'providers',
    'channels',
    'hooks',
    'mcp',
    'agent',
    'security',
    'webSearch',
    'device',
  ])

  let watcherDeps: import('./runtime/config-watcher.js').ConfigWatcherDeps | undefined
  let configWatcher: ConfigWatcher | null = null
  if (configPath) {
    watcherDeps = {
      configPath,
      readConfig: async () => applyRuntimeGatewayEnvironment(
        applyRuntimeChannelEnvironment(
          parseConfig(await readFile(configPath, 'utf-8')),
        ),
      ),
      shouldDeferReload: () => runtimeRef?.configMutationService.isApplying() ?? false,
      isCurrentConfig: (nextConfig) => isSameRuntimeConfigSnapshot(runtimeRef?.config, nextConfig),
      onMcpChange: async (mcp) => {
        await mcpManager.configureServers(mcp.servers, toolRegistry, mcpPromptsRegistry, mcp.client)
      },
      onOtherChange: (sections) => {
        // Sections reconfigured live by onConfigReloaded/onMcpChange no longer
        // need a restart; only warn for the ones that genuinely do (ports,
        // gateway URL, memory/vector backend, telemetry).
        for (const s of sections) {
          if (!FILE_WATCH_HOT_APPLIED_SECTIONS.has(s)) {
            log.warn(`Config change in '${s}' requires daemon restart`)
          }
        }
      },
    }
    configWatcher = new ConfigWatcher(watcherDeps)
    await configWatcher.start(config)
  }

  const runtime: RuntimeServices = {
    dataDir,
    config,
    configLoadFailed: options.configLoadFailed,
    configLoadError: options.configLoadError,
    storageDegraded,
    providerFactoryRegistry,
    providerRegistry,
    channelFactoryRegistry,
    toolRegistry,
    managedProcesses,
    serviceSupervisor,
    questions: questionStore,
    customDefs,
    primaryAgents: await PrimaryAgentStore.create(join(dataDir, 'sessions', 'primary-agents.json')),
    lsp: lspLayer,
    autoApprove: isAutoApproveActive({
      env: process.env,
      cliFlag: options.autoApproveCliFlag ?? false,
    }),
    policyEngine,
    sessions,
    auditLogger,
    autonomy,
    skillRegistry,
    fileMemory,
    fileMemoryRegistry,
    remindersStore,
    semanticIndex,
    usageTracker,
    hookRegistry,
    gatewayClient,
    channelAcl,
    encryption,
    secretVault,
    mcpManager,
    mcpMarketplaceCatalog,
    mcpMarketplaceSource,
    mcpConfigWriter,
    mcpPromptsRegistry,
    telemetry,
    observability,
    mdns,
    updater,
    extensionTokenStore,
    channels: [],
    channelRouter: null!,
    delegationWorker: null!,
    notificationRelayWorker: createNotificationRelayDeliveryWorker(),
    // Scheduler fields — populated by startRuntime() via buildSchedulerStack()
    jobStore: null!,
    schedulerEngine: null!,
    parseWhen: null!,
    schedulerDefaultTimezone: 'UTC',
    triggerSchedulerJob: null!,
    delegator,
    llmCache,
    providerCircuitBreaker,
    modelFallbackState,
    editSnapshotStore,
    toolStatsStore,
    workspaceMutationTracker,
    sessionUndoStack,
    pluginEvents,
    modelRouter,
    pluginLoader,
    graphRegistry,
    graphAgentLoader,
    userAgentLoader,
    userCommandStore,
    dreaming,
    skillStore,
    channelPipelineMonitor,
    channelOriginStore,
    devicePairingRegistry,
    channelReplayStore,
    channelSessionStore,
    approvalRegistry,
    approvalDecisions,
    runLimiter,
    activeRuns,
    sessionBusy,
    approvalCheckpoints,
    runCheckpoints,
    toolExecutions,
    sessionWatchBroker,
    sessionRuntimeSnapshots,
    marketplaceCatalog,
    skillSourceUrlPolicy,
    installPipeline,
    configWatcher,
    configMutationService: new ConfigMutationService({
      auditLogger,
      deviceName: config.device.name,
    }),
    subagentDispatcher: null!,
    externalAcpAgentDispatcher: null,
    a2aTaskStore,
    swarmRunRegistry,
    swarmRunStore: swarmStore,
    swarmTmuxPool,
    swarmLauncher,
    swarmAgentRuntime,
    // Task 20: real engine kickoff + tmux key forwarder (defined above so they
    // can close over the locals built in this scope).
    startSwarmRun,
    forwardSwarmKeys,
    captureSwarmAgent,
    killSwarmRunAgents,
    driveSwarmAgent: driveSwarmRunAgent,
  }
  runtimeRef = runtime

  // Subagent dispatcher — built after the runtime object exists so the
  // engine factory closure can read live providerRegistry/autonomy state.
  const subagentDispatcher = new SubagentDispatcher({
    sessions: runtime.sessions,
    onLifecycle: async (event, data) => {
      try {
        await runtime.hookRegistry.trigger({ event: event === 'started' ? 'post:subagent:start' : 'post:subagent:stop', data })
      } catch (error) {
        // Observational hooks cannot change the child's completion classification.
        log.warn('subagent lifecycle hook failed', { error: error instanceof Error ? error.message : String(error) })
      }
    },
    onRunFinished: (sessionId, options) => {
      runtime.approvalRegistry.cancelForSession(sessionId, 'Subagent run ended', options)
      runtime.activeRuns.finish(sessionId)
    },
    engineFactory: ({ tools, maxIterations, executionPolicy, sessionId, onEvent }) => {
      const provider = runtime.providerRegistry.getDefault()
      if (!provider) {
        throw new Error('SUBAGENT_NO_PROVIDER: no LLM provider configured')
      }
      const engine = new AgentEngine({
        provider,
        tools,
        policy: runtime.policyEngine,
        autonomy: executionPolicy?.autonomy ?? resolveAutonomy(runtime.config),
        maxIterations,
        hardMaxIterations: true,
        saveRunCheckpoint: checkpoint => runtime.runCheckpoints.save(checkpoint),
        clearRunCheckpoint: id => runtime.runCheckpoints.delete(id),
        saveApprovalCheckpoint: checkpoint => runtime.approvalCheckpoints.save(checkpoint),
        clearApprovalCheckpoint: id => runtime.approvalCheckpoints.delete(id),
        loadToolExecution: id => runtime.toolExecutions.get(id),
        saveToolExecution: record => runtime.toolExecutions.save(record),
        clearToolExecution: id => runtime.toolExecutions.clearActive(id),
        editSnapshotStore: runtime.editSnapshotStore,
        workspaceMutationTracker: runtime.workspaceMutationTracker,
        auditLogger: runtime.auditLogger,
        usageTracker: runtime.usageTracker,
        hookRegistry: runtime.hookRegistry,
        deviceName: runtime.config.device.name,
        llmCache: runtime.llmCache,
        providerCircuitBreaker: runtime.providerCircuitBreaker,
        activeRuns: runtime.activeRuns,
        approvalCallback: async (toolCall, requestId, options) => {
          const decision = await runtime.approvalRegistry.waitForApproval({
            sessionId, toolCall, requestId, runId: sessionId,
            forcePrompt: options?.forcePrompt, signal: options?.signal,
          })
          onEvent?.({ type: 'approval_response', requestId, decision: decision.decision, approved: decision.approved, note: decision.note })
          return decision
        },
        evaluateAutoApproval: (toolCall) => runtime.approvalRegistry.tryAutoApproval({ sessionId, toolCall, runId: sessionId }),
      })
      runtime.activeRuns.registerCanceller(sessionId, () => engine.stop())
      return engine
    },
    toolRegistry: runtime.toolRegistry,
    parentAllowedTools: () => runtime.toolRegistry.list().map((t) => t.name),
    defaultProvider: () => {
      const provider = runtime.providerRegistry.getDefault()
      if (!provider) return null
      return {
        id: provider.id,
        defaultModel: runtime.config.agent.defaultModel ?? provider.models[0]?.id ?? 'default',
      }
    },
    resolveUserAgent: async (id, cwd, workspaceRoot) => {
      const custom = (await runtime.customDefs.agentsForCwd(cwd, workspaceRoot)).find((agent) => agent.id === id)
      if (custom) {
        assertCustomAgentUsable(custom)
        return { id: custom.id, name: custom.description ?? custom.id, systemPrompt: custom.systemPrompt, model: custom.model, allowedTools: custom.allowedTools, deniedTools: custom.deniedTools, isolation: custom.isolation }
      }
      const record = runtime.userAgentLoader?.get(id)
      if (!record) return null
      return {
        id: record.spec.id,
        name: record.spec.name,
        systemPrompt: record.spec.systemPrompt,
        model: record.spec.model,
        maxIterations: record.spec.maxIterations,
      }
    },
  })
  runtime.subagentDispatcher = subagentDispatcher
  runtime.toolRegistry.register(
    createSubagentDispatchTool(subagentDispatcher, () =>
      runtime.toolRegistry.list().map((t) => t.name),
      () => runtime.backgroundSubagents,
    ),
  )
  runtime.toolRegistry.register(createSubagentJobTool(() => runtime.backgroundSubagents))
  runtime.toolRegistry.register(
    createSelfInfoTool({
      config: () => runtime.config,
      dataDir: () => runtime.dataDir,
      providerRegistry: () => runtime.providerRegistry,
      toolRegistry: () => runtime.toolRegistry,
      skillRegistry: () => runtime.skillRegistry,
      jobStore: () => runtime.jobStore,
      autonomy: () => resolveAutonomy(runtime.config),
    }),
  )
  runtime.toolRegistry.register(
    createAssistantStatusTool({
      snapshot: (options) => buildAssistantRuntimeStatus(runtime, options),
    }),
  )
  runtime.toolRegistry.register(
    createSkillHubSearchTool({
      skillRegistry: () => runtime.skillRegistry,
      toolRegistry: () => runtime.toolRegistry,
      skillStore: () => runtime.skillStore,
      marketplaceCatalog: () => runtime.marketplaceCatalog,
      sourceUrlPolicy: () => runtime.skillSourceUrlPolicy,
      autonomy: () => resolveAutonomy(runtime.config),
    }),
  )
  runtime.toolRegistry.register(
    createSkillHubInstallTool({
      skillRegistry: () => runtime.skillRegistry,
      toolRegistry: () => runtime.toolRegistry,
      skillStore: () => runtime.skillStore,
      marketplaceCatalog: () => runtime.marketplaceCatalog,
      sourceUrlPolicy: () => runtime.skillSourceUrlPolicy,
      autonomy: () => resolveAutonomy(runtime.config),
      policyEngine: () => runtime.policyEngine,
      installPipeline: () => runtime.installPipeline,
    }),
  )

  // acp runtime (dispatcher + external_acp.run tool) — built after the runtime
  // object exists so it can read live sessions/dreaming. Null when the acp
  // feature is disabled in this build.
  runtime.externalAcpAgentDispatcher = createAcpRuntime(runtime.toolRegistry, {
    ...featureRuntimeDeps,
    sessions: runtime.sessions,
    dreaming: runtime.dreaming,
  })

  runtime.configMutationService.configure({
    readRevision: () => runtime.config.configRevision,
    prepareMutableDraft: () => {
      const draft = cloneRuntimeConfigSnapshot(runtime.config)
      runtime.config = draft
      return draft
    },
    commitDraft: (draft) => {
      deepFreeze(draft)
      runtime.config = draft
      runtime.configMutationService.configure({
        deviceName: draft.device.name,
      })
    },
    snapshotConfig: () => cloneRuntimeConfigSnapshot(runtime.config),
    restoreConfig: async (snapshot) => {
      const channelTypesToRestore = new Set<string>([
        ...(snapshot.channels ?? []).map((channel) => channel.type),
        ...runtime.channels.map((channel) => channel.type),
      ])
      deepFreeze(snapshot)
      runtime.config = snapshot
      runtime.configMutationService.configure({
        deviceName: snapshot.device.name,
      })
      reconfigureRuntimeAutonomy(runtime)
      try {
        const networkDegradedReason = restoreRuntimeNetworkPolicy(runtime)
        if (networkDegradedReason) {
          log.warn('Rollback restored config with blocked provider network egress', {
            degradedReason: networkDegradedReason,
          })
        }
        await reconfigureRuntimeExtensions(
          runtime,
          new Set([
            'providers',
            'agent.autonomy',
            'agent.defaultProvider',
            'agent.defaultModel',
            'agent.disabledTools',
            'hooks.outboundWebhooks',
            'mcp.servers',
            'mcp.client',
            'security.skillSources',
          ]),
        )
        for (const channelType of channelTypesToRestore) {
          await reconfigureRuntimeChannelType(runtime, channelType)
        }
      } catch (rollbackErr) {
        log.warn('Rollback re-configuration also failed', {
          error: rollbackErr instanceof Error ? rollbackErr.message : String(rollbackErr),
        })
      }
    },
  })

  if (watcherDeps) {
    watcherDeps.onConfigReloaded = async (newConfig: SepilotdConfig) => {
      // Keep config and live services in lockstep. Previously this only swapped
      // runtime.config and reconfigured autonomy/skillSources, so a disk edit to
      // providers/channels/hooks left GET /config showing new values while the
      // running services (registry, channels, webhooks) stayed on the old
      // config — half-old/half-new until restart. Reconfigure the same services
      // the API mutation path rebuilds (mcp is handled by onMcpChange).
      // Validate filesystem-backed network inputs before publishing the new
      // config snapshot. In particular, an invalid/removed CA must leave both
      // runtime.config and the current effective dispatcher untouched.
      planProviderDispatcher(newConfig.network)

      const channelTypesToRebuild = new Set<string>([
        ...(newConfig.channels ?? []).map((channel) => channel.type),
        ...runtime.channels.map((channel) => channel.type),
      ])
      deepFreeze(newConfig)
      runtime.config = newConfig
      runtime.configMutationService.configure({
        deviceName: newConfig.device.name,
      })
      reconfigureRuntimeAutonomy(runtime)
      reconfigureRuntimeSkillSources(runtime)
      try {
        await reconfigureRuntimeExtensions(
          runtime,
          new Set([
            'network',
            'providers',
            'agent.autonomy',
            'agent.defaultProvider',
            'agent.defaultModel',
            'agent.disabledTools',
            'hooks.outboundWebhooks',
            'security.skillSources',
          ]),
        )
        for (const channelType of channelTypesToRebuild) {
          await reconfigureRuntimeChannelType(runtime, channelType)
        }
      } catch (reconfigErr) {
        log.warn(
          'File-watch reconfiguration failed; config and services may be inconsistent until restart',
          {
            error: reconfigErr instanceof Error ? reconfigErr.message : String(reconfigErr),
          },
        )
      }
    }
  }

  deepFreeze(runtime.config)

  runtime.channels = assembleChannelLayer({
    config: runtime.config,
    getConfig: () => runtime.config,
    dataDir,
    gatewayClient,
    channelFactoryRegistry,
    channelAcl: runtime.channelAcl,
    getChannelAcl: () => runtime.channelAcl,
    applyConfigMutation: (description, fn) => runtime.configMutationService.apply(description, fn),
    persistConfig: async () => {
      await persistRuntimeConfig(runtime, new Set(['channels']))
    },
  }).channels

  const channelRouter = new ChannelRouter(runtime)
  channelRouter.wireAll()
  runtime.channelRouter = channelRouter

  const delegationWorker = new DelegationWorker(runtime)
  runtime.delegationWorker = delegationWorker

  startRuntime(runtime)

  return runtime
}
