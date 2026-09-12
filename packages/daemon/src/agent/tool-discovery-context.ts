import type { ToolDefinitionRuntime, ToolRegistry } from '../tools/registry.js'

/** Per-turn cache; projection is recomputed after each narrowing of authority. */
export function createToolDiscoveryContext(input: string) {
  const cache = new Map<ToolDefinitionRuntime, Promise<string>>()
  return async (tools: ToolRegistry, context: { cwd?: string; workspaceRoot?: string }): Promise<string> => {
    const parts = await Promise.all(tools.list(context).filter(tool => tool.discoveryContext).map(tool => {
      let entry = cache.get(tool)
      if (!entry) {
        entry = Promise.resolve().then(() => tool.discoveryContext!(input))
          .catch(() => `Resource discovery for ${tool.name} is unavailable. Use permitted discovery tools if needed; do not invent resources.`)
        cache.set(tool, entry)
      }
      return entry
    }))
    return parts.filter(Boolean).join('\n\n')
  }
}
