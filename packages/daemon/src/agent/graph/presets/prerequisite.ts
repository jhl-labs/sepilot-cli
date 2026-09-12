import { stopReasonUserActionRequired } from '../../stop-reason.js'
import type { AgentGraph } from '../engine.js'
import type { AgentState, GraphExecutionContext } from '../types.js'

/** Guard each entry point, including checkpoint resumes and subgraph dispatch. */
export function withGraphPrerequisite(
  graph: AgentGraph,
  check: (state: AgentState, context: GraphExecutionContext | undefined, node: string) => Promise<string | undefined> | string | undefined,
): AgentGraph {
  const blocked = new WeakSet<AgentState>()
  for (const [name, definition] of graph.getNodes()) {
    const edge = graph.getEdges().get(name)
    const run = definition.run
    graph.addNode(name, { ...definition, run: async function* (state, context) {
      const reason = await check(state, context, name)
      if (reason) {
        blocked.add(state)
        state.output = reason
        state.shouldStop = true
        state.toolCalls = []
        state.stopReason = stopReasonUserActionRequired(reason)
        yield { type: 'message', content: reason }
        return state
      }
      blocked.delete(state)
      const result = run(state, context)
      if (Symbol.asyncIterator in result) return yield* result
      return await result
    } })
    graph.addConditionalEdge(name, (state, context) => {
      if (blocked.has(state)) return '__end__'
      return edge?.type === 'conditional' ? edge.router(state, context) : edge?.to ?? '__end__'
    }, [...new Set(['__end__', ...(edge?.type === 'conditional' ? edge.targets : edge ? [edge.to] : [])])])
  }
  return graph
}
