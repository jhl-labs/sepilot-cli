import type { GraphAgentInfo, GraphAgentRegistry } from './graph/registry.js'

/**
 * Single source of truth for the execution modes that semantic routing may
 * choose between. Both the advisory intent router (explicit `auto`) and the
 * in-run `agent.transfer` catalog read this list, so the cheap classifier and
 * the executing model are told the same modes with the same descriptions.
 *
 * Only the two engine modes plus graphs that opt in with
 * `capabilities.autoRoutable` are offered. Every other registered graph
 * (legacy orchestration, narrow specialists, experimental reasoning shapes)
 * stays reachable through an explicit mode selection, where the caller knows
 * what it asked for, but never competes in a classifier's candidate list.
 */

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
  // persona-panel needs an explicit roster from the caller. Auto-selecting it
  // with an empty roster degrades to a bare LLM turn without the normal system
  // prompt, memory, or tool surface.
  'persona-panel',
])

/** Platform/roster constraints that make a graph unusable under auto routing. */
export function isAutoRouterExcludedGraphMode(
  mode: string,
  platform: NodeJS.Platform = process.platform,
): boolean {
  if (AUTO_ROUTER_EXCLUDED_GRAPH_MODES.has(mode)) return true
  // The computer-use graph depends on the Windows automation backend. Keep
  // explicit mode available, but do not let auto routing select it elsewhere.
  return mode === 'computer-use' && platform !== 'win32'
}

/**
 * Whether a registered graph belongs in a semantic candidate list.
 * Builtin graphs must opt in; user, YAML and plugin graphs are routable by
 * default because their author registered them to be chosen.
 */
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

/**
 * Surfaces that are workspace shells: the user sits in a repository or
 * project directory, so a turn whose intent could not be classified is more
 * likely to need tools than not. Everywhere else (channels, mobile, bare
 * HTTP) the lean instant controller remains the safe entry point.
 */
export const WORKSPACE_SHELL_SURFACES: ReadonlySet<string> = new Set(['cli', 'desktop', 'web'])

export function resolveAutoFallbackMode(input: {
  surface?: string
  activeRemoteBrowser?: boolean
}): 'react' | 'instant' {
  if (input.activeRemoteBrowser) return 'react'
  const surface = input.surface?.trim().toLowerCase()
  return surface && WORKSPACE_SHELL_SURFACES.has(surface) ? 'react' : 'instant'
}
