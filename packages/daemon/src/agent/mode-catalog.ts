import type { GraphAgentInfo, GraphAgentRegistry } from './graph/registry.js'

export interface SemanticModeDescriptor {
  id: string
  description: string
  capabilities?: readonly string[]
}

export const ENGINE_MODE_DESCRIPTORS: Readonly<Record<'instant' | 'react', SemanticModeDescriptor>> = {
  instant: {
    id: 'instant',
    description: 'Direct answers from current context, focused memory recall and memory persistence; no workspace, web or process tools. Transfers when the goal needs more.',
    capabilities: ['application-state'],
  },
  react: {
    id: 'react',
    description: 'General-purpose tool loop without a planning graph: one-shot questions, focused investigation, code search and inspection, a single precise edit, native attachments.',
  },
}

const AUTO_ROUTER_EXCLUDED_GRAPH_MODES = new Set<string>([
  'persona-panel',
])

export function isAutoRouterExcludedGraphMode(
  mode: string,
  platform: NodeJS.Platform = process.platform,
): boolean {
  if (AUTO_ROUTER_EXCLUDED_GRAPH_MODES.has(mode)) return true
  return mode === 'computer-use' && platform !== 'win32'
}

export function isAutoRoutableGraph(
  graph: Pick<GraphAgentInfo, 'id' | 'source' | 'capabilities'>,
  platform: NodeJS.Platform = process.platform,
): boolean {
  if (graph.id === 'react' || graph.id === 'instant') return false
  if (isAutoRouterExcludedGraphMode(graph.id, platform)) return false
  if (graph.capabilities?.direct) return false
  if (graph.capabilities?.autoRoutable !== undefined) return graph.capabilities.autoRoutable
  return graph.source !== 'builtin' && graph.source !== 'capability'
}

export function listSemanticModes(
  registry: Pick<GraphAgentRegistry, 'list'> | undefined,
  options: { platform?: NodeJS.Platform } = {},
): SemanticModeDescriptor[] {
  const graphs = (registry?.list() ?? [])
    .filter((graph) => isAutoRoutableGraph(graph, options.platform))
    .map((graph) => ({
      id: graph.id,
      description: graph.description ?? '',
      ...(graph.capabilities?.executionCapabilities
        ? { capabilities: graph.capabilities.executionCapabilities }
        : {}),
    }))
  return [ENGINE_MODE_DESCRIPTORS.instant, ENGINE_MODE_DESCRIPTORS.react, ...graphs]
}

export const WORKSPACE_SHELL_SURFACES: ReadonlySet<string> = new Set(['cli', 'desktop', 'web'])

export function resolveAutoFallbackMode(input: {
  surface?: string
  activeRemoteBrowser?: boolean
}): 'react' | 'instant' {
  if (input.activeRemoteBrowser) return 'react'
  const surface = input.surface?.trim().toLowerCase()
  return surface && WORKSPACE_SHELL_SURFACES.has(surface) ? 'react' : 'instant'
}
