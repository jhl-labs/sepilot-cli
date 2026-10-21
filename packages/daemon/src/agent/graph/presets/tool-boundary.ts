import type { AgentGraph } from '../engine.js'
import type { Deps } from '../nodes.js'

export function withGraphToolBoundary(graph: AgentGraph, deps: Deps): AgentGraph {
  for (const [name, definition] of graph.getNodes()) {
    const run = definition.run
    graph.addNode(name, { ...definition, run: async function* (state, context) {
      const scoped = context ? { ...context, tools: deps.tools, autonomy: deps.autonomy,
        modeControl: undefined } : undefined
      const result = run(state, scoped)
      if (Symbol.asyncIterator in result) return yield* result
      return await result
    } })
  }
  return graph
}
