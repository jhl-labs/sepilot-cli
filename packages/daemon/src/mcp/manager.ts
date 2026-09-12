import {
  McpClient,
  type McpClientFeatureDeps,
  type McpClientHooks,
  type McpServerConfig,
  type PromptDescriptor,
  type ResourceDescriptor,
} from './client.js'
import { createLogger } from '../logger.js'
import type { ToolDefinitionRuntime, ToolRegistry } from '../tools/registry.js'
import type { McpPromptsRegistry } from './prompts-registry.js'
import { createResourceTools } from './resources-tools.js'
import { backoffFor, MAX_RECONNECT_ATTEMPTS, sleepWithAbort } from './reconnect.js'
import {
  emptyEntry,
  recordCallInto,
  snapshotEntry,
  type McpMetricsEntry,
  type McpMetricsSnapshot,
} from './metrics.js'
import type { SecretVault } from '../security/secret-vault.js'
import type { McpClientConfig } from '../config/schema.js'
import {
  buildMcpToolManifest,
  compareMcpToolManifest,
  type McpSecurityAlert,
  type McpServerProvenance,
  type McpToolManifest,
} from './security.js'

const log = createLogger('mcp.manager')

type ManagedMcpServerConfig = McpServerConfig

type McpClientLike = Pick<
  McpClient,
  'connect' | 'disconnect' | 'discoverTools' | 'isConnected'
>

type McpClientFactory = (
  config: McpServerConfig,
  hooks: McpClientHooks,
  vault?: SecretVault | null,
  clientFeatures?: McpClientConfig,
  featureDeps?: McpClientFeatureDeps,
) => McpClientLike

interface ManagedMcpServer {
  config: ManagedMcpServerConfig
  client?: McpClientLike
  toolRegistry?: ToolRegistry
  promptsRegistry?: McpPromptsRegistry
  toolNames: string[]      // registered (enabled) tools
  allToolNames: string[]   // all tools from server (pre-filter)
  toolManifest?: McpToolManifest
  expectedToolManifest?: McpToolManifest
  manifestChanged: boolean
  securityAlerts: McpSecurityAlert[]
  status: 'disabled' | 'connected' | 'error'
  error?: string
}

export interface McpServerStatus {
  name: string
  enabled: boolean
  status: 'disabled' | 'connected' | 'error'
  connected: boolean
  transport: string
  command?: string
  args?: string[]
  url?: string
  toolCount: number
  tools: string[]
  allTools: string[]
  disabledTools: string[]
  provenance?: McpServerProvenance
  toolManifest?: McpToolManifest
  toolManifestStatus: 'trusted' | 'changed' | 'untrusted-baseline'
  securityAlerts: McpSecurityAlert[]
  error?: string
}

type McpToolManifestWriter = {
  setMcpToolManifest: (
    serverName: string,
    manifest: McpToolManifest,
  ) => Promise<void>
}

export interface McpConfigureOptions {
  /**
   * The caller owns a larger atomic config transaction and will persist the
   * mutated server config after this connection succeeds. This prevents the
   * manager from racing that transaction by independently reading the old
   * config file while establishing a first-use manifest baseline.
   */
  deferInitialToolManifestPersistence?: boolean
}

export class McpManager {
  private servers = new Map<string, ManagedMcpServer>()
  private reconnectControllers = new Map<string, AbortController>()
  private metrics = new Map<string, McpMetricsEntry>()
  private clientFeatures?: McpClientConfig

  constructor(
    private readonly clientFactory: McpClientFactory = (
      config,
      hooks,
      vault,
      clientFeatures,
      featureDeps,
    ) => new McpClient(config, hooks, vault, clientFeatures, featureDeps),
    private readonly auditLogger?: { log: (entry: Record<string, unknown>) => void | Promise<void> },
    private readonly vault?: SecretVault | null,
    clientFeatures?: McpClientConfig,
    private readonly featureDeps: McpClientFeatureDeps = {},
    private readonly configWriter?: McpToolManifestWriter | null,
  ) {
    this.clientFeatures = clientFeatures
  }

  recordCall(server: string, tool: string, durationMs: number, status: 'success' | 'error'): void {
    Promise.resolve(this.auditLogger?.log({
      event: 'mcp.tool.call',
      server,
      tool,
      durationMs,
      status,
    })).catch(() => {
      // audit must not break tool execution
    })
    const entry = this.metrics.get(server) ?? emptyEntry()
    recordCallInto(entry, tool, durationMs, status)
    this.metrics.set(server, entry)
  }

  private createSecurityAlert(
    code: McpSecurityAlert['code'],
    severity: McpSecurityAlert['severity'],
    message: string,
    details?: Record<string, unknown>,
  ): McpSecurityAlert {
    return {
      code,
      severity,
      message,
      detectedAt: new Date().toISOString(),
      details,
    }
  }

  private initialSecurityAlerts(
    config: ManagedMcpServerConfig,
  ): McpSecurityAlert[] {
    const provenance = config.provenance
    if (!provenance) {
      return [
        this.createSecurityAlert(
          'MCP_PROVENANCE_MISSING',
          'warning',
          `MCP server "${config.name}" has no install provenance metadata.`,
        ),
      ]
    }
    if (!provenance.verified) {
      return [
        this.createSecurityAlert(
          'MCP_PROVENANCE_UNVERIFIED',
          'warning',
          `MCP server "${config.name}" was installed from an unverified source.`,
          { verification: provenance.verification, marketplace: provenance.marketplace },
        ),
      ]
    }
    return []
  }

  private recordSecurityAlert(
    state: ManagedMcpServer,
    alert: McpSecurityAlert,
  ): void {
    const duplicate = state.securityAlerts.some(
      (existing) =>
        existing.code === alert.code &&
        JSON.stringify(existing.details ?? {}) === JSON.stringify(alert.details ?? {}),
    )
    if (!duplicate) state.securityAlerts.push(alert)

    Promise.resolve(this.auditLogger?.log({
      event: 'mcp.security.alert',
      server: state.config.name,
      ...alert,
    })).catch(() => {
      // audit must not break MCP lifecycle
    })
  }

  private async persistInitialToolManifest(
    config: ManagedMcpServerConfig,
    manifest: McpToolManifest,
    deferPersistence = false,
  ): Promise<boolean> {
    if (config.toolManifest || !this.configWriter) return true
    if (deferPersistence) {
      config.toolManifest = manifest
      return true
    }
    try {
      await this.configWriter.setMcpToolManifest(config.name, manifest)
      config.toolManifest = manifest
      return true
    } catch (error) {
      log.warn(`Failed to persist MCP tool manifest for ${config.name}`, {
        error: error instanceof Error ? error.message : String(error),
      })
      return false
    }
  }

  private updateToolManifestState(
    state: ManagedMcpServer,
    tools: ToolDefinitionRuntime[],
  ): boolean {
    const actualManifest = buildMcpToolManifest(tools)
    const expectedManifest = state.expectedToolManifest ?? state.config.toolManifest
    state.toolManifest = actualManifest
    state.expectedToolManifest = expectedManifest ?? actualManifest

    const comparison = compareMcpToolManifest(expectedManifest, actualManifest)
    state.manifestChanged = comparison.changed
    if (comparison.changed) {
      this.recordSecurityAlert(
        state,
        this.createSecurityAlert(
          'MCP_TOOL_MANIFEST_CHANGED',
          'critical',
          `MCP server "${state.config.name}" changed its advertised tool manifest.`,
          {
            expectedDigest: expectedManifest?.digest,
            actualDigest: actualManifest.digest,
            added: comparison.added,
            removed: comparison.removed,
            changedTools: comparison.changedTools,
          },
        ),
      )
    }
    return comparison.changed
  }

  getMetrics(server?: string): McpMetricsSnapshot {
    if (server) {
      const entry = this.metrics.get(server)
      return { servers: entry ? { [server]: snapshotEntry(entry) } : {} }
    }
    const out: Record<string, McpMetricsEntry> = {}
    for (const [name, entry] of this.metrics) {
      out[name] = snapshotEntry(entry)
    }
    return { servers: out }
  }

  private isToolDisabled(config: ManagedMcpServerConfig, toolName: string): boolean {
    const disabledSet = new Set(config.disabledTools ?? [])
    const prefix = `mcp.${config.name}.`
    const shortName = toolName.startsWith(prefix) ? toolName.slice(prefix.length) : toolName
    return disabledSet.has(shortName) || disabledSet.has(toolName)
  }

  private enabledToolsForConfig(
    config: ManagedMcpServerConfig,
    tools: ToolDefinitionRuntime[],
  ): ToolDefinitionRuntime[] {
    return tools.filter((tool) => !this.isToolDisabled(config, tool.name))
  }

  private handleToolsChanged(
    name: string,
    error: Error | null,
    tools: ToolDefinitionRuntime[] | null,
  ): void {
    const state = this.servers.get(name)
    if (!state?.toolRegistry) return
    if (error || !tools) {
      state.error = error?.message ?? 'tool list changed without refreshed tools'
      return
    }
    const manifestChanged = this.updateToolManifestState(state, tools)

    if (manifestChanged) {
      for (const toolName of state.toolNames) state.toolRegistry.unregister(toolName)
      state.toolNames = []
      state.allToolNames = tools.map((tool) => tool.name)
      state.status = 'error'
      state.error = 'MCP tool manifest changed; tools are quarantined until the current manifest is explicitly trusted.'
      return
    }

    const resourcePrefix = `mcp.${name}.resources.`
    const resourceToolNames = state.toolNames.filter((toolName) => toolName.startsWith(resourcePrefix))
    const resourceAllToolNames = state.allToolNames.filter((toolName) => toolName.startsWith(resourcePrefix))

    for (const toolName of state.toolNames) {
      if (!toolName.startsWith(resourcePrefix)) {
        state.toolRegistry.unregister(toolName)
      }
    }

    const enabledTools = this.enabledToolsForConfig(state.config, tools)
    for (const tool of enabledTools) {
      state.toolRegistry.register(tool)
    }

    if (resourceToolNames.length === 0 && state.client instanceof McpClient) {
      const resourceTools = createResourceTools(state.client, name)
      const enabledResourceTools = this.enabledToolsForConfig(state.config, resourceTools)
      for (const tool of enabledResourceTools) state.toolRegistry.register(tool)
      resourceToolNames.push(...enabledResourceTools.map((tool) => tool.name))
      resourceAllToolNames.push(...resourceTools.map((tool) => tool.name))
    }

    state.toolNames = [...enabledTools.map((tool) => tool.name), ...resourceToolNames]
    state.allToolNames = [...tools.map((tool) => tool.name), ...resourceAllToolNames]
    state.status = 'connected'
    state.error = undefined
  }

  private handlePromptsChanged(
    name: string,
    error: Error | null,
    prompts: PromptDescriptor[] | null,
  ): void {
    const state = this.servers.get(name)
    if (!state?.promptsRegistry) return
    if (error || !prompts) {
      state.error = error?.message ?? 'prompt list changed without refreshed prompts'
      return
    }
    state.promptsRegistry.set(name, prompts)
    state.error = undefined
  }

  private handleResourcesChanged(
    name: string,
    error: Error | null,
    _resources: ResourceDescriptor[] | null,
  ): void {
    const state = this.servers.get(name)
    if (!state) return
    if (error) {
      state.error = error.message
      return
    }
    state.error = undefined
  }

  async connectServer(
    config: ManagedMcpServerConfig,
    toolRegistry: ToolRegistry,
    promptsRegistry?: McpPromptsRegistry,
    options: McpConfigureOptions = {},
  ): Promise<number> {
    const previousState = this.servers.get(config.name)
    const previousExpectedManifest =
      previousState?.expectedToolManifest ?? previousState?.toolManifest
    await this.disconnectServer(config.name)

    const client = this.clientFactory(config, {
      onDisconnect: (name) => this.handleDisconnect(name),
      recordCall: (tool, ms, status) => this.recordCall(config.name, tool, ms, status),
      onToolsChanged: (name, error, tools) => this.handleToolsChanged(name, error, tools),
      onPromptsChanged: (name, error, prompts) => this.handlePromptsChanged(name, error, prompts),
      onResourcesChanged: (name, error, resources) => this.handleResourcesChanged(name, error, resources),
    }, this.vault, this.clientFeatures, this.featureDeps)
    const state: ManagedMcpServer = {
      config,
      client,
      toolRegistry,
      promptsRegistry,
      toolNames: [],
      allToolNames: [],
      expectedToolManifest: config.toolManifest ?? previousExpectedManifest,
      manifestChanged: false,
      securityAlerts: this.initialSecurityAlerts(config),
      status: 'error',
    }
    this.servers.set(config.name, state)

    try {
      await client.connect()
      const tools = await client.discoverTools()
      const manifestChanged = this.updateToolManifestState(state, tools)
      let baselinePersisted = true
      if (state.toolManifest) {
        baselinePersisted = await this.persistInitialToolManifest(
          config,
          state.toolManifest,
          options.deferInitialToolManifestPersistence,
        )
      }
      const allToolNames = tools.map((t) => t.name)
      state.allToolNames = allToolNames
      if (manifestChanged) {
        state.status = 'error'
        state.error = 'MCP tool manifest changed; tools are quarantined until the current manifest is explicitly trusted.'
        return 0
      }
      if (!baselinePersisted) {
        state.status = 'error'
        state.error = 'MCP tool manifest baseline could not be persisted; tools are quarantined until durable trust can be established.'
        this.recordSecurityAlert(
          state,
          this.createSecurityAlert(
            'MCP_TOOL_MANIFEST_BASELINE_UNPERSISTED',
            'critical',
            `MCP server "${config.name}" tool manifest baseline could not be persisted.`,
            { actualDigest: state.toolManifest?.digest },
          ),
        )
        return 0
      }
      const enabledTools = this.enabledToolsForConfig(config, tools)
      for (const tool of enabledTools) {
        toolRegistry.register(tool)
      }
      state.toolNames = enabledTools.map((tool) => tool.name)
      state.allToolNames = allToolNames

      if (client instanceof McpClient) {
        // Register resource tools (universal list/templates/read)
        const resourceTools = createResourceTools(client, config.name)
        const enabledResourceTools = this.enabledToolsForConfig(config, resourceTools)
        for (const tool of enabledResourceTools) toolRegistry.register(tool)
        state.toolNames.push(...enabledResourceTools.map((t) => t.name))
        state.allToolNames.push(...resourceTools.map((t) => t.name))

        // Populate prompts registry
        if (promptsRegistry) {
          const prompts = await client.discoverPrompts()
          promptsRegistry.set(config.name, prompts)
        }
      }

      state.status = 'connected'
      state.error = undefined
      return tools.length
    } catch (error) {
      state.status = 'error'
      state.error = error instanceof Error ? error.message : String(error)
      try {
        await client.disconnect()
      } catch {
        // Ignore cleanup failures for a server that never finished starting.
      }
      throw error
    }
  }

  async configureServers(
    configs: ManagedMcpServerConfig[] | undefined,
    toolRegistry: ToolRegistry,
    promptsRegistry?: McpPromptsRegistry,
    clientFeatures?: McpClientConfig,
    options: McpConfigureOptions = {},
  ): Promise<void> {
    if (clientFeatures) {
      this.clientFeatures = clientFeatures
    }

    for (const c of this.reconnectControllers.values()) c.abort()
    this.reconnectControllers.clear()

    await this.disconnectAll(promptsRegistry)

    for (const config of configs ?? []) {
      if (config.enabled === false) {
        this.servers.set(config.name, {
          config,
          toolRegistry,
          toolNames: [],
          allToolNames: [],
          expectedToolManifest: config.toolManifest,
          toolManifest: config.toolManifest,
          manifestChanged: false,
          securityAlerts: this.initialSecurityAlerts(config),
          status: 'disabled',
        })
        continue
      }

      try {
        await this.connectServer(config, toolRegistry, promptsRegistry, options)
      } catch (error) {
        log.warn(`Failed to connect MCP server ${config.name}`, {
          error: error instanceof Error ? error.message : String(error),
        })
      }
    }
  }

  async disconnectServer(name: string, promptsRegistry?: McpPromptsRegistry): Promise<void> {
    this.reconnectControllers.get(name)?.abort()
    this.reconnectControllers.delete(name)

    const server = this.servers.get(name)
    if (!server) return

    for (const toolName of server.toolNames) {
      server.toolRegistry?.unregister(toolName)
    }

    ;(promptsRegistry ?? server.promptsRegistry)?.remove(name)

    if (server.client?.isConnected()) {
      await server.client.disconnect()
    }

    this.servers.delete(name)
  }

  async disconnectAll(promptsRegistry?: McpPromptsRegistry): Promise<void> {
    for (const name of Array.from(this.servers.keys())) {
      await this.disconnectServer(name, promptsRegistry)
    }
  }

  handleDisconnect(name: string): void {
    const server = this.servers.get(name)
    if (!server?.toolRegistry) return
    // Reset any previous loop before starting a new one
    this.reconnectControllers.get(name)?.abort()
    const controller = new AbortController()
    this.reconnectControllers.set(name, controller)
    void this.reconnectLoop(server, controller.signal, 0)
  }

  private async reconnectLoop(
    server: ManagedMcpServer,
    signal: AbortSignal,
    attempt: number,
  ): Promise<void> {
    if (signal.aborted) return
    const name = server.config.name
    if (attempt >= MAX_RECONNECT_ATTEMPTS) {
      const current = this.servers.get(name) ?? server
      current.status = 'error'
      current.error = 'reconnect exhausted'
      this.servers.set(name, current)
      this.reconnectControllers.delete(name)
      return
    }
    try {
      await sleepWithAbort(backoffFor(attempt), signal)
    } catch {
      return // aborted
    }
    if (signal.aborted) return
    // Detach our controller so connectServer -> disconnectServer doesn't abort us.
    const ours = this.reconnectControllers.get(name)
    if (ours) this.reconnectControllers.delete(name)
    try {
      await this.connectServer(server.config, server.toolRegistry!, server.promptsRegistry)
      // Success — controller already removed above, nothing more to do.
    } catch {
      if (signal.aborted) return
      // Restore controller so subsequent configureServers/disconnectServer can abort us.
      if (ours) this.reconnectControllers.set(name, ours)
      await this.reconnectLoop(server, signal, attempt + 1)
    }
  }

  getClient(name: string): McpClient | undefined {
    const server = this.servers.get(name)
    if (server?.client instanceof McpClient) return server.client
    return undefined
  }

  listServers(): McpServerStatus[] {
    return Array.from(this.servers.entries())
      .map(([name, server]) => {
        const cfg = server.config
        const toolManifestStatus: McpServerStatus['toolManifestStatus'] = server.manifestChanged
          ? 'changed'
          : cfg.toolManifest
            ? 'trusted'
            : 'untrusted-baseline'
        return {
          name,
          enabled: cfg.enabled,
          status: server.status,
          connected: server.client?.isConnected() ?? false,
          transport: cfg.transport,
          command: cfg.transport === 'stdio' ? cfg.command : undefined,
          args: cfg.transport === 'stdio' ? cfg.args : undefined,
          url: cfg.transport === 'stdio' ? undefined : cfg.url,
          toolCount: server.toolNames.length,
          tools: [...server.toolNames],
          allTools: [...(server.allToolNames ?? [])],
          disabledTools: [...(cfg.disabledTools ?? [])],
          provenance: cfg.provenance,
          toolManifest: server.toolManifest,
          toolManifestStatus,
          securityAlerts: [...server.securityAlerts],
          error: server.error,
        }
      })
      .sort((a, b) => a.name.localeCompare(b.name))
  }
}
