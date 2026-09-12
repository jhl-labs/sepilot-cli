import { AgentGraph } from '../engine.js'
import type { Deps } from '../nodes.js'
import type { ToolCall } from '@sepilotd/core'
import type { AgentState, GraphExecutionContext } from '../types.js'
import * as N from '../nodes.js'
import { withToolNameAllowlist } from '../../../tools/role-filter.js'
import { deterministicFocusedActionResult } from '../../focused-action-result.js'

export interface FocusedLoopOptions {
  systemPrompt: string
  /**
   * Domain tool allowlist for this preset. Patterns support a trailing `.*`
   * wildcard (e.g. `code.*`, `browser.*`). When set, the loop rebuilds
   * `deps.tools` so the agent node only *sees* these tools and the tool
   * executor only *runs* them — this is what makes an operational preset an
   * actual capability boundary rather than a prompt-only "specialization".
   * Global policy/autonomy still gates every call on top of this. Set
   * `SEPILOTD_PRESET_TOOL_SCOPING=0` to disable scoping (backward compat).
   */
  toolAllowlist?: readonly string[]
  /**
   * When true, a `post_edit_analysis` node runs immediately after each
   * `tools` step. It directly invokes `code.dependencies`,
   * `code.diagnostics`, and `code.symbols` for any file edited in the
   * batch and folds the impact radius + outstanding diagnostics into
   * `reflectionMemo` for the next agent turn.
   *
   * Off by default — turn it on for editing-heavy wrappers like
   * `editor-agent` where the extra analytic round-trip is worth it.
   */
  enablePostEditAnalysis?: boolean
  /**
   * When true, an `auto_decompose` node sits at the start of the loop.
   * On long inputs (and only when `SEPILOTD_AUTO_DECOMPOSE=1` is set),
   * it asks the LLM for sub-tasks and pushes one `subagent.dispatch`
   * tool call per sub-task. The downstream tools node executes the
   * subagents in isolation and the parent agent then synthesises a
   * final answer from their results.
   *
   * Off by default — graphs that genuinely benefit from long-horizon
   * fan-out (e.g. complex multi-component refactors) opt in.
   */
  enableAutoDecompose?: boolean
  /** Require at least one tool result after the active user-turn boundary. */
  requireCurrentTurnToolEvidence?: boolean
  /**
   * Optional deterministic first action derived from trusted structured run
   * state. This is used for an explicit single argv command: the graph runs
   * exactly what the user supplied before asking the model to summarize it,
   * avoiding speculative inventory calls or argument mutation.
   */
  initialToolCall?: (
    state: AgentState,
    context?: GraphExecutionContext,
  ) => ToolCall | undefined
  /**
   * Treat the deterministic initial action as the only authorized execution.
   * Once its result is present, format grounded evidence without another
   * provider turn, discard any pending follow-up calls, and finish.
   */
  completeAfterInitialToolResult?: boolean
}

function toolNameMatchesAllowlist(name: string, patterns: readonly string[]): boolean {
  return patterns.some((pattern) => (
    pattern.endsWith('.*')
      ? name.startsWith(pattern.slice(0, -1))
      : name === pattern
  ))
}

/**
 * Rebuild `deps.tools` to the preset's domain allowlist. Returns `deps`
 * unchanged when no allowlist is declared or scoping is disabled, so callers
 * that opt out keep the full registry. Exported for preset-scope tests.
 */
export function scopeDepsToolset(deps: Deps, allowlist?: readonly string[]): Deps {
  if (allowlist === undefined) return deps
  if (process.env.SEPILOTD_PRESET_TOOL_SCOPING === '0') return deps
  // Only a real ToolRegistry can be re-scoped; loosely-typed callers that hand a
  // partial tools object keep it untouched.
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
    }), {
      lifecycleState: 'thinking',
    })
    .addNode('tools', N.toolExecutor(deps), {
      lifecycleState: 'acting',
      resumeStage: 'acting',
      pendingToolExecutionNode: true,
    })

  if (options.enablePostEditAnalysis) {
    graph.addNode('post_edit_analysis', N.postEditAnalysis(deps), {
      lifecycleState: 'observing',
    })
  }

  graph
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
