import { AgentGraph } from '../engine.js'
import type { Deps } from '../nodes.js'
import type { AgentState } from '../types.js'
import * as N from '../nodes.js'
import { openEditCheckpointNode } from './edit-checkpoint-node.js'
import { debateNode } from '../debate.js'
import { withOptionalToolNameAllowlist } from '../../../tools/role-filter.js'

export interface ReviewerGraphOptions {
  enableDebate?: boolean
  preserveProtocolOutput?: boolean
  /** The caller must run its own authoritative quality-conclusion gate. */
  outcomeReviewOwner?: N.AgentNodeOptions['outcomeReviewOwner']
}

export function buildReviewerGraph(
  rawDeps: Deps,
  options: ReviewerGraphOptions = {},
): AgentGraph {
  // The review contract applies to every entry point, including coder/cowork
  // subgraphs. Registry metadata alone does not constrain the executor.
  const deps = {
    ...rawDeps,
    tools: withOptionalToolNameAllowlist(rawDeps.tools, rawDeps.tools.list()
      .filter((tool) => rawDeps.tools.securityDescriptor(tool.name).effect === 'observe')
      .map((tool) => tool.name)),
  }
  const debateEnabled = options.enableDebate === true
  // Debate produces a synthesized resolution that still needs the normal
  // reporter. Direct protocol preservation applies only to the non-debate
  // reviewer used as an internal quality phase.
  const preserveProtocolOutput = options.preserveProtocolOutput === true && !debateEnabled
  const graph = new AgentGraph()
    .setStart('context_manager')
    .addNode('context_manager', N.contextManager(deps), { lifecycleState: 'thinking' })
    .addNode('open_edit_checkpoint', openEditCheckpointNode('reviewer-agent'), {
      lifecycleState: 'thinking',
    })
    .addNode('agent', N.agent({
      ...deps,
      systemPrompt: (deps.systemPrompt ?? '') + '\n\nYou are a code reviewer. Read the code carefully, identify bugs, security issues, performance problems, and style violations. Be specific with line references.',
    }, {
      outcomeReviewOwner: preserveProtocolOutput ? options.outcomeReviewOwner : 'node',
    }), { lifecycleState: 'thinking' })
    .addNode('tools', N.toolExecutor(deps), {
      lifecycleState: 'acting',
      resumeStage: 'acting',
      pendingToolExecutionNode: true,
    })
    .addNode('iteration_guard', N.iterationGuard(), { lifecycleState: 'thinking' })
    .addNode('reflection', N.reflection({ critiqueDeps: deps }), {
      lifecycleState: 'observing',
      resumeStage: 'observing',
    })

  if (!preserveProtocolOutput) {
    graph.addNode('reporter', N.reporter({ enableSkillExtraction: true }), {
      lifecycleState: 'done',
    })
  }

  graph
    .addEdge('context_manager', 'open_edit_checkpoint')
    .addEdge('open_edit_checkpoint', 'agent')
    .addEdge('tools', 'reflection')
    .addEdge('reflection', 'iteration_guard')

  if (debateEnabled) {
    graph
      .addNode('debate', debateNode({ provider: deps.provider }), {
        lifecycleState: 'thinking',
      })
      .addNode('review_quality_gate', N.qualityGate({
        phase: 'review',
        maxBacktracks: 1,
      }), {
        lifecycleState: 'observing',
      })
      .addConditionalEdge(
        'agent',
        (s: AgentState) => s.toolCalls.length > 0 ? 'tools' : 'debate',
        ['tools', 'debate'],
      )
      .addConditionalEdge(
        'iteration_guard',
        (s: AgentState) => s.shouldStop ? 'debate' : 'agent',
        ['debate', 'agent'],
      )
      .addEdge('debate', 'review_quality_gate')
      .addConditionalEdge(
        'review_quality_gate',
        (s: AgentState) => s.qualityGateDecision === 'retry' ? 'agent' : 'reporter',
        ['agent', 'reporter'],
      )
  } else if (preserveProtocolOutput) {
    graph
      .addConditionalEdge(
        'agent',
        (s: AgentState) => s.toolCalls.length > 0 ? 'tools' : '__end__',
        ['tools', '__end__'],
      )
      .addConditionalEdge(
        'iteration_guard',
        (s: AgentState) => s.shouldStop ? '__end__' : 'agent',
        ['__end__', 'agent'],
      )
  } else {
    graph
      .addConditionalEdge(
        'agent',
        (s: AgentState) => s.toolCalls.length > 0 ? 'tools' : 'reporter',
        ['tools', 'reporter'],
      )
      .addConditionalEdge(
        'iteration_guard',
        (s: AgentState) => s.shouldStop ? 'reporter' : 'agent',
        ['reporter', 'agent'],
      )
  }

  if (!preserveProtocolOutput) {
    graph.addEdge('reporter', '__end__')
  }
  return graph
}
