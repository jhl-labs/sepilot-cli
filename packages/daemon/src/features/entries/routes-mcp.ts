import type { FastifyInstance } from 'fastify'
import { mcpRoutes } from '../../server/routes/mcp.js'
import { mcpPromptsRoutes } from '../../server/routes/mcp-prompts.js'
import { mcpResourcesRoutes } from '../../server/routes/mcp-resources.js'
import { mcpMetricsRoutes } from '../../server/routes/mcp-metrics.js'
import { mcpMarketplaceRoutes } from '../../server/routes/mcp-marketplace.js'
import { mcpToolsRoutes } from '../../server/routes/mcp-tools.js'
import { mcpUtilitiesRoutes } from '../../server/routes/mcp-utilities.js'
import type { FeatureRouteDeps } from '../types.js'

export async function registerRoutes(app: FastifyInstance, deps: FeatureRouteDeps): Promise<void> {
  await app.register(mcpRoutes, { prefix: deps.prefix })
  await app.register(mcpPromptsRoutes, { prefix: deps.prefix })
  await app.register(mcpResourcesRoutes, { prefix: deps.prefix })
  await app.register(mcpMetricsRoutes, { prefix: deps.prefix })
  await app.register(mcpMarketplaceRoutes, { prefix: deps.prefix })
  await app.register(mcpToolsRoutes, { prefix: deps.prefix })
  await app.register(mcpUtilitiesRoutes, { prefix: deps.prefix })
}
