import type { SepilotdConfig } from '../../config/schema.js'
import { join } from 'node:path'
import type { GatewayClient } from '../../gateway/client.js'
import type { FileSkillRegistry } from '../../skills/registry.js'
import { TaskDelegator } from '../../agent/delegator.js'
import { buildDelegationSecurityOptions } from '../../agent/delegation-runtime-security.js'
import { createApplyPatchTool } from '../../tools/apply-patch.js'
import { createFsAppendTool } from '../../tools/fs-append.js'
import { createFsEditTool } from '../../tools/fs-edit.js'
import { createFsGlobTool } from '../../tools/fs-glob.js'
import { createFsListTool } from '../../tools/fs-list.js'
import { createFsReadTool } from '../../tools/fs-read.js'
import { createFsSearchTool } from '../../tools/fs-search.js'
import { createFsMoveTool } from '../../tools/fs-move.js'
import { createFsWriteTool } from '../../tools/fs-write.js'
import { createGitDiffTool, createGitLogTool, createGitStatusTool } from '../../tools/git.js'
import { createJpadTools } from '../../tools/jpad.js'
import { createGiteaActionsRunsTool } from '../../tools/gitea-actions-runs.js'
import { createNotificationPublishTool } from '../../tools/notification-publish.js'
import {
  createMonitorEvaluateTool,
  createMonitorReportTool,
} from '../../tools/monitoring.js'
import {
  createManagedProcessFollowTool,
  createManagedProcessReadTool,
  createManagedProcessWriteTool,
  createManagedProcessSessionsTool,
  createManagedProcessStartTool,
  createManagedProcessStopTool,
  createManagedProcessWaitTool,
  createProcessListTool,
  createProcessSignalTool,
  ManagedProcessRegistry,
} from '../../tools/process.js'
import { filterToolsByRole } from '../../tools/role-filter.js'
import { createDocTools } from '../../agent/doc/tools.js'
import { getDocRegistry } from '../../agent/doc/session.js'
import { createTerminalTool } from '../../tools/terminal.js'
import { createGitTransferTool } from '../../tools/git-transfer.js'
import { createWebFetchTool } from '../../tools/web-fetch.js'
import { createWebSearchTool } from '../../tools/web-search.js'
import { createWorkspacePrepareTool } from '../../tools/workspace-prepare.js'
import { DockerTerminalRunner } from '../../sandbox/docker-terminal-runner.js'
import { BubblewrapTerminalRunner } from '../../sandbox/bubblewrap-terminal-runner.js'
import { createSkillTool } from '../../tools/skill.js'
import { createServiceSupervisorTools } from '../../tools/services.js'
import { createNativeServiceTools } from '../../tools/native-services.js'
import type { ServiceSupervisor } from '../../service-supervisor/supervisor.js'
import type { ResourceAccessPolicy } from '@sepilotd/core'
import { checkAccessUrl } from '../../security/resource-access.js'
import { resolveAutonomy } from './providers.js'
import { ToolRegistry } from '../../tools/registry.js'
import { observeManagedProcessExits } from './process-exit-receipts.js'
import { withBackgroundExecution } from '../../tools/terminal-background.js'
import { WatchService, createWatchTools } from '../../tools/watch.js'
import {
  registerFeatureTools,
  registerPostWebFeatureTools,
  registerDelegationTools,
} from '../../generated/feature-registration.js'

export function buildToolRegistry(
  config: SepilotdConfig,
  gatewayClient: GatewayClient,
  skillRegistry: FileSkillRegistry,
  options: {
    serviceSupervisor?: ServiceSupervisor
    getAccessPolicy?: () => ResourceAccessPolicy | undefined
    dataDir?: string
    getWebSearchTrustedDomains?: () => readonly string[]
  } = {},
): {
  toolRegistry: ToolRegistry
  delegator: TaskDelegator
  managedProcesses: ManagedProcessRegistry
} {
  const toolRegistry = new ToolRegistry()
  const managedProcesses = new ManagedProcessRegistry()
  observeManagedProcessExits(managedProcesses)
  const sandboxCacheRoot = options.dataDir
    ? join(options.dataDir, 'cache', 'sandbox-workspaces')
    : undefined
  const terminalRunner =
    config.security.sandbox === 'docker'
      ? new DockerTerminalRunner(config.security.sandboxDocker ?? {})
      : config.security.sandbox === 'bubblewrap'
        ? new BubblewrapTerminalRunner({
            ...(config.security.sandboxBubblewrap ?? {}),
            cacheRoot: sandboxCacheRoot,
          })
        : undefined
  const strictWorkspaceRunner = process.platform === 'linux'
      ? new BubblewrapTerminalRunner({
        ...(config.security.sandboxBubblewrap ?? {}),
        cacheRoot: sandboxCacheRoot,
        networkMode: 'none',
        readOnlyWorkspace: true,
      })
    : undefined
  const strictWorkspaceWriteRunner = process.platform === 'linux'
      ? new BubblewrapTerminalRunner({
        ...(config.security.sandboxBubblewrap ?? {}),
        cacheRoot: sandboxCacheRoot,
        networkMode: 'none',
        readOnlyWorkspace: false,
      })
    : undefined
  const strictReadOnlyNetworkRunner = process.platform === 'linux'
      ? new BubblewrapTerminalRunner({
        ...(config.security.sandboxBubblewrap ?? {}),
        cacheRoot: sandboxCacheRoot,
        networkMode: 'host',
        readOnlyWorkspace: true,
      })
    : undefined
  const strictWorkspaceNetworkWriteRunner = process.platform === 'linux'
      ? new BubblewrapTerminalRunner({
        ...(config.security.sandboxBubblewrap ?? {}),
        cacheRoot: sandboxCacheRoot,
        networkMode: 'host',
        readOnlyWorkspace: false,
      })
    : undefined
  if (terminalRunner instanceof DockerTerminalRunner) {
    void terminalRunner.reapOrphans().catch(() => undefined)
  }
  const processStartTool = createManagedProcessStartTool(managedProcesses, {
    sandboxMode: config.security.sandbox,
    strictWorkspaceLauncher: strictWorkspaceRunner,
    strictWorkspaceWriteLauncher: strictWorkspaceWriteRunner,
  })
  toolRegistry.register(withBackgroundExecution(createTerminalTool({
    runner: terminalRunner,
    strictWorkspaceRunner,
    strictWorkspaceWriteRunner,
    strictReadOnlyNetworkRunner,
    strictWorkspaceNetworkWriteRunner,
    getManagedLoopbackConnections: (sessionId, workspaceRoot) =>
      managedProcesses.loopbackConnectionsForScope(sessionId, workspaceRoot),
  }), processStartTool))
  toolRegistry.register(createWorkspacePrepareTool({ reviewRunner: strictWorkspaceWriteRunner }))
  const registerHostFileTools =
    config.security.sandbox === 'local'
    || config.security.sandboxHostFileTools === 'allow'
  const filePosture = { sandboxMode: config.security.sandbox }
  if (registerHostFileTools) {
    toolRegistry.register(createFsReadTool())
    toolRegistry.register(createFsListTool())
    toolRegistry.register(createFsWriteTool(filePosture))
    toolRegistry.register(createFsMoveTool(filePosture))
    toolRegistry.register(createFsAppendTool(filePosture))
    toolRegistry.register(createFsGlobTool())
    toolRegistry.register(createFsSearchTool())
    toolRegistry.register(createFsEditTool(filePosture))
    toolRegistry.register(createApplyPatchTool(filePosture))
  }
  const gitToolOptions = { strictWorkspaceRunner, boundedAccess: () => options.getAccessPolicy?.()?.enabled === true }
  toolRegistry.register(createGitStatusTool(gitToolOptions))
  toolRegistry.register(createGitDiffTool(gitToolOptions))
  toolRegistry.register(createGitLogTool(gitToolOptions))
  toolRegistry.register(createProcessListTool())
  toolRegistry.register(createManagedProcessSessionsTool(managedProcesses))
  toolRegistry.register(processStartTool)
  toolRegistry.register(createManagedProcessReadTool(managedProcesses))
  toolRegistry.register(createManagedProcessWriteTool(managedProcesses))
  toolRegistry.register(createManagedProcessFollowTool(managedProcesses))
  toolRegistry.register(createManagedProcessWaitTool(managedProcesses))
  toolRegistry.register(createManagedProcessStopTool(managedProcesses))
  toolRegistry.register(createProcessSignalTool(managedProcesses))
  const watchService = new WatchService({ processes: managedProcesses, dataDir: options.dataDir })
  for (const tool of createWatchTools(watchService)) toolRegistry.register(tool)
  for (const tool of createServiceSupervisorTools(options.serviceSupervisor)) {
    toolRegistry.register(tool)
  }
  for (const tool of createNativeServiceTools()) {
    toolRegistry.register(tool)
  }
  for (const tool of createDocTools({
    registry: getDocRegistry(),
    resolve: (context) => context.writingDocId ?? getDocRegistry().getActiveId(),
  })) {
    toolRegistry.register(tool)
  }

  const isManagedLoopbackUrl = (
    sessionId: string | undefined,
    rawUrl: string,
    workspaceRoot?: string,
  ) => managedProcesses.ownsLoopbackUrl(sessionId, rawUrl, workspaceRoot)
  const managedLoopbackSocketForUrl = (
    sessionId: string | undefined,
    rawUrl: string,
    workspaceRoot?: string,
  ) => managedProcesses.loopbackConnectionForUrl(sessionId, rawUrl, workspaceRoot)?.socketPath ?? null
  const featureDeps = {
    config,
    dataDir: options.dataDir,
    gatewayClient,
    skillRegistry,
    isManagedLoopbackUrl,
    managedLoopbackSocketForUrl,
  }
  registerFeatureTools(toolRegistry, featureDeps)

  toolRegistry.register(createGitTransferTool({ egressAllowlist: config.security.egressAllowlist }))

  const egressOptions = { egressAllowlist: config.security.egressAllowlist }

  toolRegistry.register(createWebSearchTool({
    getTrustedDomains:
      options.getWebSearchTrustedDomains ?? (() => config.webSearch.trustedDomains),
    getProviderSettings: () => ({
      provider: config.webSearch.provider,
      apiKey: config.webSearch.apiKey,
      endpoint: config.webSearch.endpoint,
    }),
  }))
  toolRegistry.register(createWebFetchTool({
    ...egressOptions,
    checkUrl: url => checkAccessUrl(options.getAccessPolicy?.(), url)?.reason,
    isLocalhostAllowed: isManagedLoopbackUrl,
    managedLoopbackSocketForUrl,
  }))
  toolRegistry.register(createGiteaActionsRunsTool({
    runner: strictReadOnlyNetworkRunner,
  }))
  for (const tool of createJpadTools()) toolRegistry.register(tool)
  toolRegistry.register(createMonitorEvaluateTool())
  toolRegistry.register(createMonitorReportTool())
  toolRegistry.register(createNotificationPublishTool())
  registerPostWebFeatureTools(toolRegistry, featureDeps)
  toolRegistry.register(
    createSkillTool({
      registry: skillRegistry,
      autonomy: () => resolveAutonomy(config),
      tools: toolRegistry,
    }),
  )

  const delegationSecurity = options.dataDir
    ? buildDelegationSecurityOptions(options.dataDir, config.device.id)
    : {}
  const delegator = new TaskDelegator(gatewayClient, config, delegationSecurity)
  registerDelegationTools(toolRegistry, { ...featureDeps, delegator })

  toolRegistry.setDisabledTools(config.agent.disabledTools ?? [])

  const roleName = config.device.role as 'desktop' | 'server' | 'edge'
  const filteredTools = filterToolsByRole(toolRegistry, roleName)
  void filteredTools

  return { toolRegistry, delegator, managedProcesses }
}
