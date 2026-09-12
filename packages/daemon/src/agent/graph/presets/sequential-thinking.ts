import type { Deps } from '../nodes.js'
import { AgentGraph } from '../engine.js'
import type { AgentState } from '../types.js'
import * as N from '../nodes.js'

const sequentialThinkingPrompt = [
  'You are a sequential-thinking agent.',
  'First produce concise, user-visible intermediate conclusions, then answer by following them in order.',
  'Prefer a single best next step over branching unless evidence forces reconsideration.',
  'When using tools, keep each tool call tied to the current step and summarize what changed before continuing.',
  'Do not reveal private chain-of-thought; surface only compact step summaries and conclusions.',
].join(' ')

export function buildSequentialThinkingGraph(deps: Deps) {
  return new AgentGraph()
    .setStart('context_manager')
    .addNode('context_manager', N.contextManager(deps), {
      lifecycleState: 'thinking',
    })
    .addNode('sequential_deliberation', N.sequentialDeliberation(deps), {
      lifecycleState: 'thinking',
    })
    .addNode('agent', N.agent({
      ...deps,
      systemPrompt: [deps.systemPrompt ?? '', sequentialThinkingPrompt]
        .filter(Boolean)
        .join('\n\n'),
    }), {
      lifecycleState: 'thinking',
    })
    .addNode('tools', N.toolExecutor(deps), {
      lifecycleState: 'acting',
      resumeStage: 'acting',
      pendingToolExecutionNode: true,
    })
    .addNode('reflection', N.reflection({ critiqueDeps: deps }), {
      lifecycleState: 'observing',
      resumeStage: 'observing',
    })
    .addNode('iteration_guard', N.iterationGuard(), {
      lifecycleState: 'thinking',
    })
    .addNode('reporter', N.reporter({ enableSkillExtraction: true }), {
      lifecycleState: 'done',
    })
    .addEdge('context_manager', 'sequential_deliberation')
    .addEdge('sequential_deliberation', 'agent')
    .addConditionalEdge(
      'agent',
      (s: AgentState) => s.toolCalls.length > 0 ? 'tools' : 'reporter',
      ['tools', 'reporter'],
    )
    .addEdge('tools', 'reflection')
    .addEdge('reflection', 'iteration_guard')
    .addConditionalEdge(
      'iteration_guard',
      (s: AgentState) => s.shouldStop ? 'reporter' : 'agent',
      ['reporter', 'agent'],
    )
    .addEdge('reporter', '__end__')
}
