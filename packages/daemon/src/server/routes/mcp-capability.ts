import type { FastifyInstance } from 'fastify'
import { bindCapability } from '../capabilities/bind.js'
import { configRoutes } from './config.js'
import { mcpMarketplaceRoutes } from './mcp-marketplace.js'
import { mcpMetricsRoutes } from './mcp-metrics.js'
import { mcpPromptsRoutes } from './mcp-prompts.js'
import { mcpResourcesRoutes } from './mcp-resources.js'
import { mcpToolsRoutes } from './mcp-tools.js'
import { mcpUtilitiesRoutes } from './mcp-utilities.js'
import { mcpRoutes } from './mcp.js'

export async function registerMcpCapabilityRoutes(app: FastifyInstance): Promise<void> {
  await bindCapability(
    app,
    {
      name: 'mcp',
      version: '1',
      description: 'Manage MCP servers, tools, prompts, resources, marketplaces, and client capabilities.',
      methods: [
        { method: 'GET', path: '/mcp/servers' },
        { method: 'GET', path: '/mcp/servers/:name/tools' },
        { method: 'POST', path: '/mcp/servers/:name/tools/:tool/disable' },
        { method: 'POST', path: '/mcp/servers/:name/tools/:tool/enable' },
        { method: 'POST', path: '/mcp/servers/:name/tools/:tool/call' },
        { method: 'GET', path: '/mcp/servers/:name/prompts' },
        { method: 'POST', path: '/mcp/servers/:name/prompts/:prompt' },
        { method: 'GET', path: '/mcp/servers/:name/resources' },
        { method: 'GET', path: '/mcp/servers/:name/resources/templates' },
        { method: 'POST', path: '/mcp/servers/:name/resources/read' },
        { method: 'GET', path: '/mcp/servers/:name/resources/subscriptions' },
        { method: 'POST', path: '/mcp/servers/:name/resources/subscribe' },
        { method: 'POST', path: '/mcp/servers/:name/resources/unsubscribe' },
        { method: 'GET', path: '/mcp/servers/:name/resources/updates' },
        { method: 'POST', path: '/mcp/servers/:name/completion' },
        { method: 'POST', path: '/mcp/servers/:name/logging/level' },
        { method: 'GET', path: '/mcp/servers/:name/logs' },
        { method: 'GET', path: '/mcp/metrics' },
        { method: 'GET', path: '/mcp/metrics/:server' },
        { method: 'GET', path: '/mcp/marketplace/search' },
        { method: 'POST', path: '/mcp/install' },
        { method: 'GET', path: '/mcp/marketplaces' },
        { method: 'POST', path: '/mcp/marketplaces' },
        { method: 'DELETE', path: '/mcp/marketplaces/:name' },
        { method: 'GET', path: '/config' },
        { method: 'PUT', path: '/config' },
        { method: 'POST', path: '/config/mcp/servers' },
        { method: 'DELETE', path: '/config/mcp/servers/:name' },
        { method: 'POST', path: '/config/mcp/servers/:name/enable' },
        { method: 'POST', path: '/config/mcp/servers/:name/disabled-tools' },
      ],
    },
    async (a) => {
      await configRoutes(a)
      await mcpRoutes(a)
      await mcpPromptsRoutes(a)
      await mcpResourcesRoutes(a)
      await mcpMetricsRoutes(a)
      await mcpMarketplaceRoutes(a)
      await mcpToolsRoutes(a)
      await mcpUtilitiesRoutes(a)
    },
  )
}
