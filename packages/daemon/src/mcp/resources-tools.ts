import type { ToolDefinitionRuntime, ToolResult } from '../tools/registry.js'
import type { McpClient } from './client.js'

export function createResourceTools(client: McpClient, serverName: string): ToolDefinitionRuntime[] {
  const prefix = `mcp.${serverName}.resources`

  const listTool: ToolDefinitionRuntime = {
    name: `${prefix}.list`,
    description: `List resources exposed by MCP server "${serverName}"`,
    inputSchema: { type: 'object', properties: {} },
    async execute(): Promise<ToolResult> {
      const start = Date.now()
      try {
        const resources = await client.discoverResources()
        return { output: JSON.stringify(resources), status: 'success', durationMs: Date.now() - start }
      } catch (err: unknown) {
        return { output: err instanceof Error ? err.message : String(err), status: 'error', durationMs: Date.now() - start }
      }
    },
  }

  const templatesTool: ToolDefinitionRuntime = {
    name: `${prefix}.templates`,
    description: `List resource URI templates exposed by MCP server "${serverName}"`,
    inputSchema: { type: 'object', properties: {} },
    async execute(): Promise<ToolResult> {
      const start = Date.now()
      try {
        const templates = await client.discoverResourceTemplates()
        return { output: JSON.stringify(templates), status: 'success', durationMs: Date.now() - start }
      } catch (err: unknown) {
        return { output: err instanceof Error ? err.message : String(err), status: 'error', durationMs: Date.now() - start }
      }
    },
  }

  const readTool: ToolDefinitionRuntime = {
    name: `${prefix}.read`,
    description: `Read a resource by URI from MCP server "${serverName}"`,
    inputSchema: {
      type: 'object',
      properties: { uri: { type: 'string', description: 'Resource URI (e.g. file:///path)' } },
      required: ['uri'],
    },
    async execute(input: Record<string, unknown>): Promise<ToolResult> {
      const start = Date.now()
      const uri = input.uri
      if (typeof uri !== 'string' || !uri) {
        return { output: 'uri argument is required', status: 'error', durationMs: Date.now() - start }
      }
      try {
        const { contents } = await client.readResource(uri)
        const text = contents
          .map((c) => c.text ?? (c.blob ? `<blob:${c.mimeType ?? 'application/octet-stream'}>` : ''))
          .join('\n')
        return { output: text, status: 'success', durationMs: Date.now() - start }
      } catch (err: unknown) {
        return { output: err instanceof Error ? err.message : String(err), status: 'error', durationMs: Date.now() - start }
      }
    },
  }

  return [listTool, templatesTool, readTool]
}
