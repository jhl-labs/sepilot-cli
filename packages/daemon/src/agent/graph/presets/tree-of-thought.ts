import { AgentGraph } from '../engine.js'
import type { Deps } from '../nodes.js'
import type { AgentState } from '../types.js'
import * as N from '../nodes.js'

const treeOfThoughtPrompt = [
  'You are a tree-of-thought reasoning agent.',
  'Generate and compare distinct candidate approaches before committing to one.',
  'After a branch is selected, carry it out faithfully and validate with tools when uncertain.',
  'Do not reveal private chain-of-thought; surface only compact branch summaries, selection rationale, and the final answer.',
].join(' ')

/**
 * Real Tree-of-Thought:
 * 1. `treeOfThought` node generates N branches in parallel, judges them,
 *    and pushes the winning branch into the message history as a system note.
 * 2. The downstream agent loop carries out that approach (with tool access).
 *
 * If branching produces nothing usable the agent simply runs as normal,
 * so this graph is robust against provider hiccups.
 */
export function buildTreeOfThoughtGraph(deps: Deps): AgentGraph {
  return new AgentGraph()
    .setStart('context_manager')
    .addNode('context_manager', N.contextManager(deps), { lifecycleState: 'thinking' })
    .addNode(
      'tot_branch',
      N.treeOfThought(deps, { branches: 3, systemPrompt: treeOfThoughtPrompt }),
      { lifecycleState: 'thinking' },
    )
    .addNode(
      'agent',
      N.agent({
        ...deps,
        systemPrompt: [deps.systemPrompt ?? '', treeOfThoughtPrompt]
          .filter(Boolean)
          .join('\n\n'),
      }),
      { lifecycleState: 'thinking' },
    )
    .addNode('tools', N.toolExecutor(deps), {
      lifecycleState: 'acting',
      resumeStage: 'acting',
      pendingToolExecutionNode: true,
    })
    .addNode('reflection', N.reflection({ critiqueDeps: deps }), {
      lifecycleState: 'observing',
      resumeStage: 'observing',
    })
    .addNode('iteration_guard', N.iterationGuard(), { lifecycleState: 'thinking' })
    .addNode('reporter', N.reporter({ enableSkillExtraction: true }), { lifecycleState: 'done' })
    .addEdge('context_manager', 'tot_branch')
    .addEdge('tot_branch', 'agent')
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
