import type { AgentGraph } from '../engine.js'
import type { Deps } from '../nodes.js'

/** Carry a fixed tool capability through every node and checkpoint entry. */
export function withGraphToolBoundary(graph: AgentGraph, deps: Deps): AgentGraph {
  for (const [name, definition] of graph.getNodes()) {
    const run = definition.run
    graph.addNode(name, { ...definition, run: async function* (state, context) {
      // Control tools are injected independently of the ToolRegistry and can
      // discover/transfer into a broader mode. Fixed-boundary graphs opt out.
      const scoped = context ? { ...context, tools: deps.tools, autonomy: deps.autonomy,
        modeControl: undefined } : undefined
      const result = run(state, scoped)
      if (Symbol.asyncIterator in result) return yield* result
      return await result
    } })
  }
  return graph
}
