import { AgentGraph } from '../engine.js'
import type { Deps } from '../nodes.js'
import type { ToolCall } from '@sepilotd/core'
import type { AgentState, GraphExecutionContext } from '../types.js'
import * as N from '../nodes.js'
import { withToolNameAllowlist } from '../../../tools/role-filter.js'
import { deterministicFocusedActionResult } from '../../focused-action-result.js'

export interface FocusedLoopOptions {
  systemPrompt: string
  executionControl?: 'model-directed'
  toolAllowlist?: readonly string[]
  enablePostEditAnalysis?: boolean
  enableAutoDecompose?: boolean
  requireCurrentTurnToolEvidence?: boolean
  enableReflectionCritique?: boolean
  initialToolCall?: (
    state: AgentState,
    context?: GraphExecutionContext,
  ) => ToolCall | undefined
  completeAfterInitialToolResult?: boolean
}

function toolNameMatchesAllowlist(name: string, patterns: readonly string[]): boolean {
  return patterns.some((pattern) => (
    pattern.endsWith('.*')
      ? name.startsWith(pattern.slice(0, -1))
      : name === pattern
  ))
}

export function scopeDepsToolset(deps: Deps, allowlist?: readonly string[]): Deps {
  if (allowlist === undefined) return deps
  if (process.env.SEPILOTD_PRESET_TOOL_SCOPING === '0') return deps
  if (typeof (deps.tools as { list?: unknown }).list !== 'function') return deps
  const allowedNames = new Set<string>()
  for (const tool of deps.tools.list()) {
    if (toolNameMatchesAllowlist(tool.name, allowlist)) {
      allowedNames.add(tool.name)
    }
  }
  return { ...deps, tools: withToolNameAllowlist(deps.tools, allowedNames) }
}

function hasDeterministicInitialToolResult(state: AgentState): boolean {
  const toolCallId = state.deterministicInitialToolCallId
  if (!toolCallId) return false
  return Boolean(
    state.recentToolResults?.some((result) => result.toolCallId === toolCallId)
    || state.messages.some((message) =>
      message.role === 'tool' && message.toolCallId === toolCallId),
  )
}

export function buildFocusedLoopGraph(
  rawDeps: Deps,
  options: FocusedLoopOptions,
): AgentGraph {
  const deps = scopeDepsToolset(rawDeps, options.toolAllowlist)
  const graph = new AgentGraph()
    .setStart('context_manager')
    .addNode('context_manager', N.contextManager(deps), {
      lifecycleState: 'thinking',
    })

  if (options.initialToolCall) {
    graph
      .addNode('initial_action', async (s: AgentState, context) => {
        const toolCall = options.initialToolCall?.(s, context)
        s.deterministicInitialToolCallId = toolCall?.id
        s.toolCalls = toolCall ? [toolCall] : []
        return s
      }, {
        lifecycleState: 'thinking',
      })
      .addNode('initial_tools', N.toolExecutor(deps), {
        lifecycleState: 'acting',
        resumeStage: 'acting',
        pendingToolExecutionNode: true,
      })
  }

  const afterInitialAction = options.enableAutoDecompose ? 'auto_decompose' : 'agent'
  if (options.initialToolCall) {
    graph
      .addEdge('context_manager', 'initial_action')
      .addConditionalEdge(
        'initial_action',
        (s: AgentState) => s.toolCalls.length > 0 ? 'initial_tools' : afterInitialAction,
        ['initial_tools', afterInitialAction],
      )
      .addEdge(
        'initial_tools',
        options.completeAfterInitialToolResult
          ? 'initial_action_completion_guard'
          : 'agent',
      )
  }
  if (options.enableAutoDecompose) {
    graph
      .addNode('auto_decompose', N.autoDecompose(deps), {
        lifecycleState: 'thinking',
      })
      .addNode('decompose_tools', N.toolExecutor(deps), {
        lifecycleState: 'acting',
        resumeStage: 'acting',
        pendingToolExecutionNode: true,
      })
  }

  graph
    .addNode('agent', N.agent({
      ...deps,
      systemPrompt: [deps.systemPrompt ?? '', options.systemPrompt]
        .filter(Boolean)
        .join('\n\n'),
    }, { executionControl: options.executionControl }), {
      lifecycleState: 'thinking',
    })
    .addNode('tools', N.toolExecutor(deps, { executionControl: options.executionControl }), {
      lifecycleState: 'acting',
      resumeStage: 'acting',
      pendingToolExecutionNode: true,
    })

  if (options.executionControl === 'model-directed') {
    graph
      .addNode('reporter', N.reporter(), { lifecycleState: 'done' })
      .addEdge('context_manager', 'agent')
      .addConditionalEdge('agent', (s) => s.modelResponseRecoveryPending ? 'agent' : s.toolCalls.length > 0 ? 'tools' : 'reporter', ['agent', 'tools', 'reporter'])
      .addConditionalEdge('tools', (s) => s.shouldStop ? 'reporter' : 'agent', ['reporter', 'agent'])
      .addEdge('reporter', '__end__')
    return graph
  }

  if (options.enablePostEditAnalysis) {
    graph.addNode('post_edit_analysis', N.postEditAnalysis(deps), {
      lifecycleState: 'observing',
    })
  }

  graph
    .addNode('reflection', N.reflection(
      options.enableReflectionCritique === false ? {} : { critiqueDeps: deps },
    ), {
      lifecycleState: 'observing',
      resumeStage: 'observing',
    })
    .addNode('iteration_guard', N.iterationGuard(), {
      lifecycleState: 'thinking',
    })
    .addNode('reporter', N.reporter({ enableSkillExtraction: true }), {
      lifecycleState: 'done',
    })
  if (options.requireCurrentTurnToolEvidence) {
    graph.addNode('current_turn_tool_evidence_guard', N.currentTurnToolEvidenceGuard(), {
      lifecycleState: 'thinking',
    })
  }
  if (options.completeAfterInitialToolResult) {
    graph.addNode('initial_action_completion_guard', async (s: AgentState) => {
      if (!hasDeterministicInitialToolResult(s)) return s
      s.toolCalls = []
      s.shouldStop = true
      if (!s.output.trim()) {
        const toolCallId = s.deterministicInitialToolCallId
        const recent = [...(s.recentToolResults ?? [])]
          .reverse()
          .find((result) => result.toolCallId === toolCallId)
        const message = [...s.messages]
          .reverse()
          .find((entry) => entry.role === 'tool' && entry.toolCallId === toolCallId)
        const messageOutput = typeof message?.content === 'string'
          ? message.content.trim()
          : ''
        const observed = recent?.output.trim() ?? messageOutput
        const formatted = recent
          ? deterministicFocusedActionResult(s.input, {
              toolName: recent.toolName ?? '',
              status: recent.status,
              output: recent.output,
            })
          : undefined
        s.output = formatted
          ?? (recent?.status === 'error'
            ? `INCOMPLETE: The requested action failed.\n${observed}`
            : observed
              ? `Observed result:\n${observed}`
          : 'INCOMPLETE: The requested command completed without a reportable result.'
          )
      }
      return s
    }, {
      lifecycleState: 'thinking',
    })
  }
  if (options.enableAutoDecompose) {
    if (!options.initialToolCall) graph.addEdge('context_manager', 'auto_decompose')
    graph
      .addConditionalEdge(
        'auto_decompose',
        (s: AgentState) => s.toolCalls.length > 0 ? 'decompose_tools' : 'agent',
        ['decompose_tools', 'agent'],
      )
      .addEdge('decompose_tools', 'agent')
  } else {
    if (!options.initialToolCall) graph.addEdge('context_manager', 'agent')
  }
  graph
    .addConditionalEdge(
      'agent',
      (s: AgentState) => options.completeAfterInitialToolResult
          && hasDeterministicInitialToolResult(s)
        ? 'initial_action_completion_guard'
        : s.toolCalls.length > 0
        ? 'tools'
        : options.requireCurrentTurnToolEvidence
          ? 'current_turn_tool_evidence_guard'
          : 'reporter',
      options.completeAfterInitialToolResult
        ? options.requireCurrentTurnToolEvidence
          ? ['initial_action_completion_guard', 'tools', 'current_turn_tool_evidence_guard']
          : ['initial_action_completion_guard', 'tools', 'reporter']
        : options.requireCurrentTurnToolEvidence
          ? ['tools', 'current_turn_tool_evidence_guard']
          : ['tools', 'reporter'],
    )

  if (options.completeAfterInitialToolResult) {
    graph.addEdge('initial_action_completion_guard', 'reporter')
  }

  if (options.requireCurrentTurnToolEvidence) {
    graph.addConditionalEdge(
      'current_turn_tool_evidence_guard',
      (s: AgentState) => s.shouldStop ? 'reporter' : 'agent',
      ['reporter', 'agent'],
    )
  }

  if (options.enablePostEditAnalysis) {
    graph
      .addEdge('tools', 'post_edit_analysis')
      .addEdge('post_edit_analysis', 'reflection')
  } else {
    graph.addEdge('tools', 'reflection')
  }

  graph
    .addEdge('reflection', 'iteration_guard')
    .addConditionalEdge(
      'iteration_guard',
      (s: AgentState) => s.shouldStop ? 'reporter' : 'agent',
      ['reporter', 'agent'],
    )
    .addEdge('reporter', '__end__')

  return graph
}
