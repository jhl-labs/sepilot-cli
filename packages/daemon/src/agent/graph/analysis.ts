import type { AgentState as LifecycleState, ILLMProvider } from '@sepilotd/core'
import type { AutonomyLevel } from '@sepilotd/core'
import type { PolicyEngine } from '../../security/policy-engine.js'
import type { ToolRegistry } from '../../tools/registry.js'
import type { GraphAgentInfo } from './registry.js'
import type {
  AgentState,
  GraphEdgeDefinition,
  GraphEdgeTarget,
  GraphExecutionContext,
  GraphNodeDefinition,
  GraphNodeMeta,
} from './types.js'

export type GraphNodePromptKind =
  | 'system'
  | 'routing'
  | 'agent'
  | 'tool'
  | 'guard'
  | 'subgraph'
  | 'report'
  | 'internal'

export type GraphNodeRecommendedModel =
  | 'default'
  | 'fast'
  | 'strong'
  | 'none'

export interface GraphNodeModelOverride {
  model?: string
}

export type GraphNodeModelOverrides = Record<
  string,
  Record<string, GraphNodeModelOverride>
>

export interface AgentGraphAnalysisNode {
  id: string
  label: string
  summary: string
  lifecycleState: string
  resumeStage: string
  promptKind: GraphNodePromptKind
  prompt: string
  modelConfigurable: boolean
  recommendedModel: GraphNodeRecommendedModel
  activeModel?: string
  notes: string[]
}

export interface AgentGraphAnalysisEdge {
  from: string
  to: string
  type: 'direct' | 'conditional'
  label?: string
}

export interface AgentGraphAnalysisSnapshot {
  id: string
  name: string
  description: string
  source?: GraphAgentInfo['source']
  startNode: string
  nodeCount: number
  edgeCount: number
  nodes: AgentGraphAnalysisNode[]
  edges: AgentGraphAnalysisEdge[]
  modelOverrides: Record<string, GraphNodeModelOverride>
}

interface AnalysisDeps {
  provider: ILLMProvider
  tools: ToolRegistry
  policy: PolicyEngine
  autonomy: AutonomyLevel
}

interface NodePromptInfo {
  summary: string
  prompt: string
  promptKind: GraphNodePromptKind
  modelConfigurable: boolean
  recommendedModel: GraphNodeRecommendedModel
  notes?: string[]
}

const DETERMINISTIC_PROMPT = 'No LLM prompt. This node moves state, executes tools, captures summaries, or routes by deterministic graph state.'

const NODE_PROMPTS: Record<string, NodePromptInfo> = {
  auto_decompose: {
    summary: 'Splits large composite requests into independent subagent prompts.',
    prompt: 'You are a task decomposer. Break the user task into 2-4 independent subtasks only when the task is genuinely composite. Return strict JSON with a subtasks array.',
    promptKind: 'routing',
    modelConfigurable: true,
    recommendedModel: 'fast',
    notes: ['Fast JSON-capable models are usually enough here.'],
  },
  triage: {
    summary: 'Classifies the task as simple, complex, code, or creative.',
    prompt: DETERMINISTIC_PROMPT,
    promptKind: 'routing',
    modelConfigurable: false,
    recommendedModel: 'none',
  },
  capability_scout: {
    summary: 'Runs a small discovery pass when the graph needs capability context.',
    prompt: 'When capability scouting is useful, the node asks the model for focused discovery work and may emit tool calls before specialist routing.',
    promptKind: 'routing',
    modelConfigurable: true,
    recommendedModel: 'fast',
  },
  specialist_router: {
    summary: 'Chooses the specialist route for the enhanced graph.',
    prompt: 'You are routing a task inside the enhanced orchestrator graph. Choose exactly one route from simple, generalist, creative, reviewer, coder, researcher. Return JSON only with route, reason, and brief.',
    promptKind: 'routing',
    modelConfigurable: true,
    recommendedModel: 'fast',
    notes: ['Good place for a cheap/fast model because the output is short JSON.'],
  },
  memory_retriever: {
    summary: 'Loads relevant memories before planning or execution.',
    prompt: DETERMINISTIC_PROMPT,
    promptKind: 'internal',
    modelConfigurable: false,
    recommendedModel: 'none',
  },
  planner: {
    summary: 'Creates a compact execution plan.',
    prompt: 'Break this task into 2-5 steps. Return a JSON array of strings only. Relevant memories, routing briefs, and codebase exploration can be included as system context.',
    promptKind: 'system',
    modelConfigurable: true,
    recommendedModel: 'fast',
  },
  coding_planner: {
    summary: 'Builds the coding run contract and validation plan.',
    prompt: 'You are planning a coding execution graph. Return JSON with summary, acceptance criteria, constraints, out-of-scope items, plan, and validation plan.',
    promptKind: 'system',
    modelConfigurable: true,
    recommendedModel: 'fast',
  },
  codebase_exploration: {
    summary: 'Asks the agent to inspect the repository before implementation.',
    prompt: 'Focused codebase exploration prompt: inspect relevant files, symbols, and project structure before planning edits. Tool use is allowed.',
    promptKind: 'agent',
    modelConfigurable: true,
    recommendedModel: 'default',
  },
  large_codebase_scout: {
    summary: 'Dispatches read-only scout subagents for large repositories.',
    prompt: 'Generate targeted scout prompts for independent read-only subagents, then summarize the resulting codebase map.',
    promptKind: 'routing',
    modelConfigurable: true,
    recommendedModel: 'fast',
  },
  context_manager: {
    summary: 'Compresses and trims conversation context when needed.',
    prompt: 'Context manager prompt is used only when semantic compression is needed. It preserves task-critical history while fitting the active model context window.',
    promptKind: 'internal',
    modelConfigurable: true,
    recommendedModel: 'fast',
  },
  tool_recommender: {
    summary: 'Suggests the smallest useful tool set for the next phase.',
    prompt: 'Recommend focused tools for the current phase based on the task, plan, recent results, and available tool catalog.',
    promptKind: 'routing',
    modelConfigurable: true,
    recommendedModel: 'fast',
  },
  agent: {
    summary: 'Main agent LLM turn.',
    prompt: 'The node assembles the daemon system prompt, answer protocol, relevant memories, routing brief, run contract, current plan step, prior messages, and user input. It may stream text and tool calls.',
    promptKind: 'agent',
    modelConfigurable: true,
    recommendedModel: 'default',
  },
  implement: {
    summary: 'Main coding implementation LLM turn.',
    prompt: 'You are a coding agent. Implement the requested change with focused edits, use tools deliberately, and leave validation-ready work behind.',
    promptKind: 'agent',
    modelConfigurable: true,
    recommendedModel: 'strong',
  },
  validator: {
    summary: 'Validation-phase agent turn.',
    prompt: 'You are a coding validation agent. Verify the implementation with tests, diagnostics, build output, and concrete evidence.',
    promptKind: 'agent',
    modelConfigurable: true,
    recommendedModel: 'default',
  },
  search_agent: {
    summary: 'Research evidence-gathering agent.',
    prompt: 'You are the research evidence gathering specialist. Search, fetch, inspect, and collect grounded findings before synthesis.',
    promptKind: 'agent',
    modelConfigurable: true,
    recommendedModel: 'default',
  },
  verify_agent: {
    summary: 'Research verification agent.',
    prompt: 'You are the research verification specialist. Cross-check findings, source quality, and uncertainty before final synthesis.',
    promptKind: 'agent',
    modelConfigurable: true,
    recommendedModel: 'default',
  },
  reflection: {
    summary: 'Turns recent failures into a short recovery hint.',
    prompt: 'You are a critic helping an agent recover from failures. Output one short paragraph with the likely cause and concrete next step.',
    promptKind: 'system',
    modelConfigurable: true,
    recommendedModel: 'fast',
  },
  implementation_reflection: {
    summary: 'Implementation failure recovery hint.',
    prompt: 'You are a critic helping a coding agent recover from failures. Output one short paragraph with the likely cause and concrete next step.',
    promptKind: 'system',
    modelConfigurable: true,
    recommendedModel: 'fast',
  },
  validation_reflection: {
    summary: 'Validation failure recovery hint.',
    prompt: 'You are a critic helping a validation agent recover from failures. Output one short paragraph with the likely cause and next validation step.',
    promptKind: 'system',
    modelConfigurable: true,
    recommendedModel: 'fast',
  },
  search_reflection: {
    summary: 'Research search recovery hint.',
    prompt: 'You are a critic helping a research agent recover from weak or failed search results. Output one short next step.',
    promptKind: 'system',
    modelConfigurable: true,
    recommendedModel: 'fast',
  },
  verify_reflection: {
    summary: 'Research verification recovery hint.',
    prompt: 'You are a critic helping a verification agent recover from uncertainty or source conflicts. Output one short next step.',
    promptKind: 'system',
    modelConfigurable: true,
    recommendedModel: 'fast',
  },
  quality_gate: {
    summary: 'Judges whether the current phase should pass or retry.',
    prompt: 'Judge the phase evidence against the acceptance criteria and decide pass or retry. Return a concise decision with concrete reasons.',
    promptKind: 'routing',
    modelConfigurable: true,
    recommendedModel: 'fast',
  },
  validation_quality_gate: {
    summary: 'Judges validation evidence.',
    prompt: 'Judge validation evidence against the run contract. Prefer real command output over self-report and decide pass or retry.',
    promptKind: 'routing',
    modelConfigurable: true,
    recommendedModel: 'fast',
  },
  review_quality_gate: {
    summary: 'Judges review evidence.',
    prompt: 'Judge review findings and implementation evidence. Decide whether another implementation pass is required.',
    promptKind: 'routing',
    modelConfigurable: true,
    recommendedModel: 'fast',
  },
  debate: {
    summary: 'Runs debate-style review before the final reviewer decision.',
    prompt: 'Multiple review perspectives critique the work and compare evidence before a final review decision.',
    promptKind: 'agent',
    modelConfigurable: true,
    recommendedModel: 'strong',
  },
  finalizer: {
    summary: 'Produces the user-facing final response.',
    prompt: 'Produce a compact outcome-first response in the user\'s language. Internal run contracts, criterion verdicts, evidence-ledger labels, quality gates, phase names, and token usage remain structured diagnostics rather than conversational text.',
    promptKind: 'report',
    modelConfigurable: true,
    recommendedModel: 'default',
  },
  reporter: {
    summary: 'Emits the final graph output and optional skill extraction.',
    prompt: DETERMINISTIC_PROMPT,
    promptKind: 'report',
    modelConfigurable: false,
    recommendedModel: 'none',
  },
  decompose: {
    summary: 'Breaks cowork tasks into specialist subtasks.',
    prompt: 'Break the task into 2-5 specialist subtasks for coder, reviewer, and researcher roles. Return JSON only as an array of role/instruction objects.',
    promptKind: 'routing',
    modelConfigurable: true,
    recommendedModel: 'fast',
  },
  delegate: {
    summary: 'Runs cowork specialist turns.',
    prompt: 'You are the selected cowork specialist. Complete the assigned subtask concisely and preserve implementation details.',
    promptKind: 'agent',
    modelConfigurable: true,
    recommendedModel: 'default',
  },
  synthesize: {
    summary: 'Synthesizes cowork specialist outputs.',
    prompt: 'Synthesize coder, reviewer, and researcher outputs into a single useful response with conflicts and remaining risks called out.',
    promptKind: 'report',
    modelConfigurable: true,
    recommendedModel: 'default',
  },
}

function humanizeNodeId(id: string): string {
  return id
    .split('_')
    .filter(Boolean)
    .map((part) => part.slice(0, 1).toUpperCase() + part.slice(1))
    .join(' ')
}

function defaultResumeStage(
  lifecycleState: LifecycleState,
): NonNullable<GraphNodeMeta['resumeStage']> {
  if (lifecycleState === 'acting') return 'acting'
  if (lifecycleState === 'observing') return 'observing'
  return 'thinking'
}

function isToolLikeNode(id: string): boolean {
  return id === 'tools'
    || id.endsWith('_tools')
    || id.includes('tool_executor')
    || id === 'decompose_tools'
}

function isGuardLikeNode(id: string): boolean {
  return id.includes('guard')
    || id.startsWith('capture_')
    || id.startsWith('mark_')
    || id.startsWith('open_edit_checkpoint')
    || id === 'observation_context'
    || id === 'post_action_observation'
}

function inferNodeInfo(id: string): NodePromptInfo {
  if (NODE_PROMPTS[id]) {
    return NODE_PROMPTS[id]
  }
  if (isToolLikeNode(id)) {
    return {
      summary: 'Executes pending tool calls and records observations.',
      prompt: DETERMINISTIC_PROMPT,
      promptKind: 'tool',
      modelConfigurable: false,
      recommendedModel: 'none',
    }
  }
  if (isGuardLikeNode(id)) {
    return {
      summary: 'Applies deterministic graph control, checkpoint, capture, or phase bookkeeping.',
      prompt: DETERMINISTIC_PROMPT,
      promptKind: 'guard',
      modelConfigurable: false,
      recommendedModel: 'none',
    }
  }
  if (id.endsWith('_subgraph') || id.includes('subgraph')) {
    return {
      summary: 'Runs a nested specialist graph and maps its result back into the parent graph.',
      prompt: 'Subgraph wrapper. The prompt comes from the nested graph nodes that run inside this node.',
      promptKind: 'subgraph',
      modelConfigurable: false,
      recommendedModel: 'none',
      notes: ['Configure the nested node names that appear in that specialist graph.'],
    }
  }
  if (id.includes('agent') || id.includes('moderator') || id.includes('panel')) {
    return {
      summary: 'LLM agent node.',
      prompt: 'Agent node prompt is assembled from the graph preset system prompt, daemon system prompt, conversation state, and user task.',
      promptKind: 'agent',
      modelConfigurable: true,
      recommendedModel: 'default',
    }
  }
  return {
    summary: 'Graph runtime node.',
    prompt: DETERMINISTIC_PROMPT,
    promptKind: 'internal',
    modelConfigurable: false,
    recommendedModel: 'none',
  }
}

function analyzeNode(
  id: string,
  definition: GraphNodeDefinition<AgentState, GraphExecutionContext>,
  overrides: Record<string, GraphNodeModelOverride>,
): AgentGraphAnalysisNode {
  const lifecycleState = definition.meta?.lifecycleState ?? 'thinking'
  const meta: Required<GraphNodeMeta> = {
    lifecycleState,
    resumeStage: definition.meta?.resumeStage ?? defaultResumeStage(lifecycleState),
    pendingToolExecutionNode: definition.meta?.pendingToolExecutionNode ?? false,
  }
  const info = inferNodeInfo(id)
  const activeModel = overrides[id]?.model?.trim() || undefined
  return {
    id,
    label: humanizeNodeId(id),
    summary: info.summary,
    lifecycleState: meta.lifecycleState,
    resumeStage: meta.resumeStage,
    promptKind: info.promptKind,
    prompt: info.prompt,
    modelConfigurable: info.modelConfigurable,
    recommendedModel: info.recommendedModel,
    activeModel,
    notes: [
      ...(info.notes ?? []),
      ...(meta.pendingToolExecutionNode ? ['Resume checkpoint can restart inside this tool node.'] : []),
      ...(activeModel ? [`Configured model override: ${activeModel}`] : []),
    ],
  }
}

function edgeTargets(
  edge: GraphEdgeDefinition<AgentState, GraphExecutionContext>,
): GraphEdgeTarget[] {
  return edge.type === 'direct'
    ? [edge.to]
    : edge.targets.length > 0
      ? edge.targets
      : ['__end__']
}

export function createAgentGraphAnalysisSnapshots(
  agents: GraphAgentInfo[],
  deps: AnalysisDeps,
  overrides: GraphNodeModelOverrides = {},
): AgentGraphAnalysisSnapshot[] {
  return agents.map((agent) => {
    const graph = agent.builder(deps)
    const graphOverrides = overrides[agent.id] ?? {}
    const nodes = Array.from(graph.getNodes().entries()).map(([id, definition]) => (
      analyzeNode(id, definition, graphOverrides)
    ))
    const edges = Array.from(graph.getEdges().entries()).flatMap(([from, edge]) => (
      edgeTargets(edge).map((to) => ({
        from,
        to,
        type: edge.type,
        label: edge.type === 'conditional' ? 'condition' : undefined,
      }))
    ))

    return {
      id: agent.id,
      name: agent.name,
      description: agent.description,
      source: agent.source,
      startNode: graph.getStartNode(),
      nodeCount: nodes.length,
      edgeCount: edges.length,
      nodes,
      edges,
      modelOverrides: graphOverrides,
    }
  })
}
