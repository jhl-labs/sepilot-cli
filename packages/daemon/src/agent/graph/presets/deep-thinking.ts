import { AgentGraph } from '../engine.js'
import type { Deps } from '../nodes.js'
import type { AgentState } from '../types.js'
import * as N from '../nodes.js'

const deepThinkingPrompt = [
  'You are a deep-thinking agent.',
  'The deliberation nodes already emit separate summaries of assumptions, checks, and risks. Respect the user’s exact final-answer format; do not append a verification summary or explanation when it was not requested.',
  'Pressure-test critical claims and revise the final answer when the verifier finds a gap.',
  'Favor depth over speed, but stop once the answer is well-supported rather than looping aimlessly.',
  'Do not reveal private chain-of-thought; surface only compact deliberation and verification summaries.',
].join(' ')

export function buildDeepThinkingGraph(deps: Deps): AgentGraph {
  return new AgentGraph()
    .setStart('context_manager')
    .addNode('context_manager', N.contextManager(deps), {
      lifecycleState: 'thinking',
    })
    // Keep the long-task fan-out hook that existing deployments expect, but
    // route it into a real deep deliberation pass instead of a generic loop.
    .addNode('auto_decompose', N.autoDecompose(deps), {
      lifecycleState: 'thinking',
    })
    .addNode('decompose_tools', N.toolExecutor(deps), {
      lifecycleState: 'acting',
      resumeStage: 'acting',
      pendingToolExecutionNode: true,
    })
    .addNode('deep_deliberation', N.deepDeliberation(deps), {
      lifecycleState: 'thinking',
    })
    .addNode('agent', N.agent({
      ...deps,
      systemPrompt: [deps.systemPrompt ?? '', deepThinkingPrompt]
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
    .addNode('deep_answer_verifier', N.deepAnswerVerifier(deps), {
      lifecycleState: 'thinking',
    })
    .addNode('reporter', N.reporter({ enableSkillExtraction: true }), {
      lifecycleState: 'done',
    })
    .addEdge('context_manager', 'auto_decompose')
    .addConditionalEdge(
      'auto_decompose',
      (s: AgentState) => s.toolCalls.length > 0 ? 'decompose_tools' : 'deep_deliberation',
      ['decompose_tools', 'deep_deliberation'],
    )
    .addEdge('decompose_tools', 'deep_deliberation')
    .addEdge('deep_deliberation', 'agent')
    .addConditionalEdge(
      'agent',
      (s: AgentState) => s.toolCalls.length > 0 ? 'tools' : 'deep_answer_verifier',
      ['tools', 'deep_answer_verifier'],
    )
    .addEdge('tools', 'reflection')
    .addEdge('reflection', 'iteration_guard')
    .addConditionalEdge(
      'iteration_guard',
      (s: AgentState) => s.shouldStop ? 'deep_answer_verifier' : 'agent',
      ['deep_answer_verifier', 'agent'],
    )
    .addConditionalEdge(
      'deep_answer_verifier',
      // A `revise` verdict with no corrected answer routes back to the agent for
      // one bounded revision pass instead of shipping the known-bad answer.
      (s: AgentState) => s.deepAnswerReviseRequested ? 'agent' : 'reporter',
      ['agent', 'reporter'],
    )
    .addEdge('reporter', '__end__')
}
