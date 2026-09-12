import { buildChatOnlyGraph } from './chat-only.js'
import type { GraphAgentRegistry } from '../registry.js'
import { buildEnhancedGraph } from '../builder.js'
import { buildCoworkGraph } from '../../cowork/orchestrator.js'
import { buildBrowserAgentGraph } from './browser-agent.js'
import { buildComputerUseGraph } from './computer-use.js'
import { buildWorkspaceCoderGraph, buildCodebaseScoutGraph } from './workspace-agents.js'
import { buildDeepThinkingGraph } from './deep-thinking.js'
import { buildDeepWebResearchGraph } from './deep-web-research.js'
import { buildEditorAgentGraph } from './editor-agent.js'
import { buildResearcherGraph } from './researcher.js'
import { buildReviewerGraph } from './reviewer.js'
import { buildPersonaPanelGraph } from './persona-panel.js'
import { buildSequentialThinkingGraph } from './sequential-thinking.js'
import { buildTerminalAgentGraph } from './terminal-agent.js'
import { buildTreeOfThoughtGraph } from './tree-of-thought.js'

export const builtinGraphBuilders = {
  'chat-only': buildChatOnlyGraph,
  enhanced: buildEnhancedGraph,
  reviewer: (deps: Parameters<typeof buildReviewerGraph>[0]) =>
    buildReviewerGraph(deps, { enableDebate: true }),
  coder: buildWorkspaceCoderGraph,
  'codebase-scout': buildCodebaseScoutGraph,
  researcher: buildResearcherGraph,
  cowork: buildCoworkGraph,
  'sequential-thinking': buildSequentialThinkingGraph,
  'tree-of-thought': buildTreeOfThoughtGraph,
  'deep-thinking': buildDeepThinkingGraph,
  'deep-web-research': buildDeepWebResearchGraph,
  'persona-panel': buildPersonaPanelGraph,
  'browser-agent': buildBrowserAgentGraph,
  'computer-use': buildComputerUseGraph,
  'editor-agent': buildEditorAgentGraph,
  'terminal-agent': buildTerminalAgentGraph,
} as const

export type BuiltinGraphId = keyof typeof builtinGraphBuilders
export const builtinGraphIds = Object.keys(builtinGraphBuilders) as BuiltinGraphId[]

const GENERAL_EXECUTION_CAPABILITIES = [
  'process', 'service', 'terminal', 'browser',
  'filesystem-read', 'filesystem-write', 'network', 'application-state',
] as const

const RESEARCH_EXECUTION_CAPABILITIES = [
  'browser', 'filesystem-read', 'network',
] as const

export function registerBuiltinGraphs(registry: GraphAgentRegistry): void {
  registry.register({
    id: 'chat-only', name: 'Chat Only (ChatGPT)',
    description: 'Conversation with read-only RAG and stored knowledge. No actions, writes, delegation, or mode transfers.',
    builder: buildChatOnlyGraph, source: 'builtin', limits: { maxIterations: 8 },
    capabilities: { readOnly: true, durableRunContract: false, nativeMultimodalInput: true, executionCapabilities: [] },
  })
  registry.register({
    id: 'codebase-scout', name: 'Codebase Scout',
    description: 'Read-only workspace exploration, dependency mapping, and architecture analysis. No edits, commands, or delegation.',
    builder: buildCodebaseScoutGraph, source: 'builtin', limits: { maxIterations: 24 },
    capabilities: { readOnly: true, nativeMultimodalInput: true, executionCapabilities: ['filesystem-read'] },
  })
  registry.register({
    id: 'enhanced',
    name: 'Enhanced Agent',
    description: 'General-purpose orchestration for mixed or unclear requests; triages internally and routes to simple, coder, researcher, creative, or reviewer specialists',
    builder: buildEnhancedGraph,
    source: 'builtin',
    capabilities: {
      artifactWrite: true,
      executionCapabilities: [...GENERAL_EXECUTION_CAPABILITIES],
    },
  })

  registry.register({
    id: 'reviewer',
    name: 'Code Reviewer',
    description: 'Review code for bugs, security issues, and improvements with debate quality gate',
    builder: (deps) => buildReviewerGraph(deps, { enableDebate: true }),
    source: 'builtin',
    capabilities: {
      readOnly: true,
      executionCapabilities: ['filesystem-read'],
    },
  })

  registry.register({
    id: 'coder',
    name: 'Coder Agent',
    description: 'Code implementation, validation, debugging and review. CLI sessions use configured tool policy; strict workspace sessions require a sandbox runner for commands and managed processes.',
    builder: buildWorkspaceCoderGraph,
    source: 'builtin',
    limits: { maxIterations: 36 },
    capabilities: {
      artifactWrite: true,
      autoRoutable: true,
      nativeMultimodalInput: true,
      executionCapabilities: ['process', 'terminal', 'filesystem-read', 'filesystem-write', 'network'],
    },
  })

  registry.register({
    id: 'researcher',
    name: 'Research Agent',
    description: 'Evidence gathering and synthesis across many sources, including broad repository or document investigation when no write is required',
    builder: buildResearcherGraph,
    source: 'builtin',
    capabilities: {
      readOnly: true,
      autoRoutable: true,
      executionCapabilities: [...RESEARCH_EXECUTION_CAPABILITIES],
    },
  })

  registry.register({
    id: 'cowork',
    name: 'Cowork Team',
    description: 'Team orchestration that decomposes one run across multiple specialist roles (planner, coder, reviewer, researcher); reserve for large ambiguous multi-step work that genuinely needs several roles — prefer coder or researcher when one role covers the task',
    builder: buildCoworkGraph,
    source: 'builtin',
    capabilities: {
      artifactWrite: true,
      executionCapabilities: [...GENERAL_EXECUTION_CAPABILITIES],
    },
  })

  registry.register({
    id: 'sequential-thinking',
    name: 'Sequential Thinking',
    description: 'Linear multi-step reasoning with explicit intermediate conclusions',
    builder: buildSequentialThinkingGraph,
    source: 'builtin',
    limits: { maxIterations: 8 },
  })

  registry.register({
    id: 'tree-of-thought',
    name: 'Tree Of Thought',
    description: 'Compare multiple candidate reasoning branches before converging',
    builder: buildTreeOfThoughtGraph,
    source: 'builtin',
    limits: { maxIterations: 9 },
  })

  registry.register({
    id: 'deep-thinking',
    name: 'Deep Thinking',
    description: 'Deliberate reasoning with explicit verification of critical assumptions',
    builder: buildDeepThinkingGraph,
    source: 'builtin',
    limits: { maxIterations: 12 },
  })

  registry.register({
    id: 'deep-web-research',
    name: 'Deep Web Research',
    description: 'Verification-heavy research workflow with broader search and stronger source checking',
    builder: buildDeepWebResearchGraph,
    source: 'builtin',
    limits: { maxIterations: 10 },
    capabilities: {
      readOnly: true,
      executionCapabilities: [...RESEARCH_EXECUTION_CAPABILITIES],
    },
  })

  registry.register({
    id: 'persona-panel',
    name: 'Persona Panel',
    description: 'Multiple personas answer the same prompt; a moderator pass weaves the replies together',
    builder: buildPersonaPanelGraph,
    source: 'builtin',
    limits: { maxIterations: 1 },
  })

  registry.register({
    id: 'browser-agent',
    name: 'Browser Agent',
    description: 'Specialized for browser navigation, extraction, and page inspection',
    builder: buildBrowserAgentGraph,
    source: 'builtin',
    limits: { maxIterations: 8 },
    capabilities: { executionCapabilities: ['browser', 'network'] },
  })

  registry.register({
    id: 'computer-use',
    name: 'Computer Use Agent',
    description: 'Windows desktop GUI automation with observe, hover, click, drag, type, hotkey, scroll, and wait tools',
    builder: buildComputerUseGraph,
    source: 'builtin',
    limits: { maxIterations: 32 },
  })

  registry.register({
    id: 'editor-agent',
    name: 'Editor Agent',
    description: 'Specialized for precise source editing with file and patch tools',
    builder: buildEditorAgentGraph,
    source: 'builtin',
    limits: { maxIterations: 10 },
    capabilities: {
      artifactWrite: true,
      executionCapabilities: ['filesystem-read', 'filesystem-write'],
    },
  })

  registry.register({
    id: 'terminal-agent',
    name: 'Terminal Agent',
    description: 'Specialized for command-driven workflows with tighter shell discipline',
    builder: buildTerminalAgentGraph,
    source: 'builtin',
    limits: { maxIterations: 9 },
    capabilities: {
      durableRunContract: false,
      executionCapabilities: ['process', 'terminal', 'filesystem-read', 'network'],
    },
  })
}
