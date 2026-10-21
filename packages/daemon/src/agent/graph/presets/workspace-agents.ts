import { realpath, stat } from 'node:fs/promises'
import { isAbsolute, relative } from 'node:path'
import { AutonomyLevel } from '@sepilotd/core'
import type { AgentGraph } from '../engine.js'
import type { Deps } from '../nodes.js'
import type { AgentState, GraphExecutionContext } from '../types.js'
import type { ToolRegistry } from '../../../tools/registry.js'
import { toolExposureGroupForTool, withToolNameAllowlist } from '../../../tools/role-filter.js'
import { buildCoderGraph } from './coder.js'
import { withGraphToolBoundary } from './tool-boundary.js'
import { withGraphPrerequisite } from './prerequisite.js'
import { buildFocusedLoopGraph } from './focused-loop.js'

export async function prepareWorkspaceAgent(state: AgentState, context?: GraphExecutionContext) {
  try {
    const cwd = context?.agentContext.cwd?.trim()
    if (!context || !cwd || !isAbsolute(cwd)) throw new Error('작업 디렉터리를 먼저 선택하세요.')
    const root = await realpath(cwd)
    if (!(await stat(root)).isDirectory()) throw new Error('작업 디렉터리가 아닙니다.')
    if (context.agentContext.workspaceRoot) {
      const previous = await realpath(context.agentContext.workspaceRoot)
      if (!isAbsolute(context.agentContext.workspaceRoot) || relative(context.agentContext.workspaceRoot, previous) !== '') {
        throw new Error('작업공간 경로가 변경되었습니다. 작업 폴더를 다시 선택하세요.')
      }
      const path = relative(previous, root)
      if (path === '..' || path.startsWith('../') || path.startsWith('..\\') || isAbsolute(path)) {
        throw new Error('기존 작업공간 밖으로 이동할 수 없습니다.')
      }
    }
    context.agentContext.cwd = root
    context.agentContext.workspaceRoot ??= root
    context.agentContext.autoApprove = false
    context.agentContext.requireToolApproval = true
    context.requireToolApproval = true
  } catch (error) {
    state.output = `INCOMPLETE: ${error instanceof Error ? error.message : String(error)}`
    state.shouldStop = true
    state.toolCalls = []
  }
  return state
}

function guard(graph: AgentGraph, entry: string, deps: Deps): AgentGraph {
  graph.setStart('workspace_boundary')
    .addNode('workspace_boundary', async (state) => state)
    .addEdge('workspace_boundary', entry)
  return withGraphToolBoundary(withGraphPrerequisite(graph, async (state, context) => {
    const checked = await prepareWorkspaceAgent({ ...state, shouldStop: false }, context)
    return checked.shouldStop ? checked.output : undefined
  }), deps)
}

const CODER_TOOL_GROUPS: ReadonlySet<string> = new Set([
  'files', 'code', 'web', 'browser', 'process', 'services', 'delegation',
])
const CODER_GENERAL_TOOL_NAMES: ReadonlySet<string> = new Set([
  'question', 'skill', 'todowrite', 'system.info',
  'memory.search', 'knowledge.search', 'knowledge.read',
  'agent.tools', 'agent.transfer',
])

export function coderPolicyToolNames(tools: Pick<ToolRegistry, 'list'>): Set<string> {
  return new Set(tools.list().map((tool) => tool.name).filter((name) => {
    if (CODER_GENERAL_TOOL_NAMES.has(name)) return true
    const group = toolExposureGroupForTool(name)
    return group !== null && CODER_TOOL_GROUPS.has(group)
  }))
}

function withWorkspaceProfiles(
  raw: Deps,
  build: (deps: Deps) => AgentGraph,
  entry: string,
  policyToolNames: (tools: Pick<ToolRegistry, 'list'>) => Set<string>,
): AgentGraph {
  const allowed = new Set(raw.tools.list().filter((tool) =>
    /^(fs\.|code\.|git\.|process\.)/.test(tool.name)
    || ['terminal.run', 'apply_patch', 'todowrite', 'question'].includes(tool.name),
  ).map((tool) => tool.name))
  const deps = { ...raw, tools: withToolNameAllowlist(raw.tools, allowed),
    autonomy: raw.autonomy === AutonomyLevel.ReadOnly ? AutonomyLevel.ReadOnly : AutonomyLevel.Supervised }
  const graph = guard(build(deps), entry, deps)
  const policyGraph = build({
    ...raw,
    tools: withToolNameAllowlist(raw.tools, policyToolNames(raw.tools)),
  })
  for (const [name, node] of [...graph.getNodes()]) {
    const policyNode = policyGraph.getNodes().get(name)
    graph.addNode(name, (state, context) => {
      if (context?.agentContext.workspaceIsolation === 'policy'
        && !context.agentContext.workspaceRoot) {
        return policyNode ? policyNode.run(state, context) : Promise.resolve(state)
      }
      return node.run(state, context)
    }, node.meta)
  }
  return graph
}

export function buildWorkspaceCoderGraph(raw: Deps): AgentGraph {
  return withWorkspaceProfiles(raw, buildCoderGraph, 'memory_retriever', coderPolicyToolNames)
}

export const FAST_CODER_SYSTEM_PROMPT = [
  'You are a fast, precise coding agent working directly in the user\'s repository.',
  'Work in a tight loop: locate the relevant code with search and reads, make the smallest correct change with the edit tools, then run the most focused check that proves it (a single test file, a typecheck, a build of the touched package).',
  'Do not write plans, inventories or design documents unless asked; do not explore beyond what the change needs; do not run repository-wide suites when a focused check exists.',
  'Read a file before editing it and re-read edited regions when an edit tool reports a mismatch.',
  'Finish with a short report: what changed (file paths), what you verified and how, and anything you did not verify.',
  'Batch independent observations in one response when useful. Choose your own next action from tool results; do not repeat unchanged observations without a reason.',
  'If you cannot finish within the available budget or encounter a blocker you cannot resolve, start your final answer with "INCOMPLETE:" followed by what you tried and what remains. Never claim an edit, command or check you did not perform.',
].join(' ')

const FAST_CODER_EXTRA_TOOL_GROUPS: ReadonlySet<string> = new Set(['integrations'])

export function fastCoderPolicyToolNames(tools: Pick<ToolRegistry, 'list'>): Set<string> {
  const names = coderPolicyToolNames(tools)
  for (const tool of tools.list()) {
    const group = toolExposureGroupForTool(tool.name)
    if (group !== null && FAST_CODER_EXTRA_TOOL_GROUPS.has(group)) names.add(tool.name)
  }
  return names
}

export function buildFastCoderGraph(raw: Deps): AgentGraph {
  return withWorkspaceProfiles(
    raw,
    (deps) => buildFocusedLoopGraph(deps, {
      systemPrompt: FAST_CODER_SYSTEM_PROMPT,
      enableReflectionCritique: false,
      executionControl: 'model-directed',
    }),
    'context_manager',
    fastCoderPolicyToolNames,
  )
}

export function buildCodebaseScoutGraph(raw: Deps): AgentGraph {
  const names = new Set(['fs.read', 'fs.list', 'fs.glob', 'fs.search', 'code.symbols', 'code.dependencies'])
  const allowed = new Set(raw.tools.list().filter((tool) => names.has(tool.name)
    && raw.tools.securityDescriptor(tool.name).effect === 'observe').map((tool) => tool.name))
  const deps = { ...raw, tools: withToolNameAllowlist(raw.tools, allowed), autonomy: AutonomyLevel.ReadOnly }
  return guard(buildFocusedLoopGraph(deps, {
    systemPrompt: 'Explore and analyze the selected codebase without changing it. Map architecture, entry points, module boundaries, dependencies and risks. Work incrementally: search and inspect relevant files, avoid dumping the whole repository, cite file paths and symbols for findings, and state gaps in evidence. Do not execute commands, edit files, access outside the workspace, or delegate work.',
    requireCurrentTurnToolEvidence: true,
  }), 'context_manager', deps)
}
