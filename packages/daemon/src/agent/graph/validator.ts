import type { StateGraph } from './engine.js'
import type { GraphRuntimeState } from './types.js'

export interface ValidationResult {
  valid: boolean
  errors: string[]
  warnings: string[]
}

export function validateGraph<
  State extends GraphRuntimeState,
  Context,
>(
  graph: StateGraph<State, Context>,
): ValidationResult {
  const errors: string[] = []
  const warnings: string[] = []

  const nodes = graph.getNodes()
  const edges = graph.getEdges()
  const start = graph.getStartNode()

  // Check start node exists
  if (!nodes.has(start)) {
    errors.push(`Start node '${start}' not found in graph`)
  }

  // Check all edge targets exist
  for (const [from, edge] of edges) {
    if (edge.type === 'direct') {
      if (edge.to !== '__end__' && !nodes.has(edge.to)) {
        errors.push(`Edge from '${from}' targets non-existent node '${edge.to}'`)
      }
      continue
    }

    for (const target of edge.targets) {
      if (target !== '__end__' && !nodes.has(target)) {
        errors.push(`Conditional edge from '${from}' targets non-existent node '${target}'`)
      }
    }
  }

  // Check for unreachable nodes
  const reachable = new Set<string>()
  const queue = [start]
  while (queue.length > 0) {
    const node = queue.shift()!
    if (reachable.has(node)) continue
    reachable.add(node)
    const edge = edges.get(node)
    if (!edge) {
      continue
    }

    if (edge.type === 'direct') {
      if (edge.to !== '__end__') {
        queue.push(edge.to)
      }
      continue
    }

    for (const target of edge.targets) {
      if (target !== '__end__') {
        queue.push(target)
      }
    }
  }

  for (const name of nodes.keys()) {
    if (!reachable.has(name)) {
      warnings.push(`Node '${name}' is unreachable from start`)
    }
  }

  // Check nodes with no outgoing edges (should only be terminal nodes)
  for (const name of nodes.keys()) {
    if (!edges.has(name) && name !== '__end__') {
      warnings.push(`Node '${name}' has no outgoing edge (implicit terminal)`)
    }
  }

  return { valid: errors.length === 0, errors, warnings }
}
