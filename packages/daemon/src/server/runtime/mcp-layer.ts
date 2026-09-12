import { ConfigWriter } from '../../mcp/config-writer.js'
import { McpMarketplaceCatalog } from '../../mcp/marketplace-catalog.js'
import { McpMarketplaceSource } from '../../mcp/marketplace-source.js'
import { McpPromptsRegistry } from '../../mcp/prompts-registry.js'
import { buildMcpManager } from './services.js'
import type { buildAuditLogger } from './storage.js'
import type { SecretVault } from '../../security/secret-vault.js'
import type { McpClientFeatureDeps } from '../../mcp/client.js'

type AuditLoggerInstance = Awaited<ReturnType<typeof buildAuditLogger>>

export interface McpLayer {
  mcpManager: ReturnType<typeof buildMcpManager>
  mcpPromptsRegistry: McpPromptsRegistry
  mcpMarketplaceCatalog: McpMarketplaceCatalog
  mcpMarketplaceSource: McpMarketplaceSource
  mcpConfigWriter: ConfigWriter | null
}

/**
 * MCP subsystem: the manager that spawns/connects servers, the
 * per-server prompt registry, the marketplace catalog + its query
 * source, and the optional config writer that persists new MCP
 * server entries back to config.yaml.
 */
export async function assembleMcpLayer(args: {
  dataDir: string
  configPath: string | undefined
  auditLogger: AuditLoggerInstance
  secretVault: SecretVault
  deviceName: string
  featureDeps?: McpClientFeatureDeps
}): Promise<McpLayer> {
  const mcpConfigWriter = args.configPath ? new ConfigWriter(args.configPath) : null
  const mcpManager = buildMcpManager(
    args.auditLogger,
    args.secretVault,
    args.deviceName,
    args.featureDeps,
    mcpConfigWriter,
  )
  const mcpPromptsRegistry = new McpPromptsRegistry()
  const mcpMarketplaceCatalog = new McpMarketplaceCatalog(args.dataDir)
  await mcpMarketplaceCatalog.init()
  const mcpMarketplaceSource = new McpMarketplaceSource({ catalog: mcpMarketplaceCatalog })

  return {
    mcpManager,
    mcpPromptsRegistry,
    mcpMarketplaceCatalog,
    mcpMarketplaceSource,
    mcpConfigWriter,
  }
}
