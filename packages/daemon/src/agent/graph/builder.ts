import { AgentGraph } from './engine.js'
import type { Deps } from './nodes.js'
import type { AgentState } from './types.js'
import * as N from './nodes.js'
import { agentSubgraphNode } from './subgraph.js'
import { buildCoderGraph } from './presets/coder.js'
import { buildResearcherGraph } from './presets/researcher.js'
import { buildReviewerGraph } from './presets/reviewer.js'
import { cloneMessage } from '../tool-execution.js'
import { cloneEvidenceLedger } from './evidence-ledger.js'

const simpleSystemPrompt = [
  'You are the fast-response specialist inside an orchestrated graph.',
  'Answer directly and only use tools when they materially improve correctness.',
  'Keep the response concise and task-focused.',
].join(' ')

const generalistSystemPrompt = [
  'You are the generalist task execution specialist inside an orchestrated graph.',
  'Analyze the task, make a compact plan, use tools deliberately when useful, and converge on a concrete answer.',
  'Prefer clear structure over verbosity.',
].join(' ')

const creativeSystemPrompt = [
  'You are the creative execution specialist inside an orchestrated graph.',
  'Produce intentional, useful creative output that fits the requested audience, constraints, and tone.',
  'Use tools only when they materially improve the final result.',
].join(' ')

function appendSystemPrompt(
  base: string | undefined,
  extra: string,
): string {
  return [base ?? '', extra].filter(Boolean).join('\n\n')
}

function createChildState(
  parent: AgentState,
  taskType: AgentState['taskType'],
): AgentState {
  return {
    input: parent.input,
    currentUserContent: parent.currentUserContent,
    effectiveContextWindowTokens: parent.effectiveContextWindowTokens,
    messages: parent.messages.map(cloneMessage),
    currentStep: '',
    planIndex: 0,
    specialistRoute: parent.specialistRoute,
    specialistReason: parent.specialistReason,
    specialistBrief: parent.specialistBrief,
    seedContract: parent.seedContract
      ? {
          ...parent.seedContract,
          acceptanceCriteria: parent.seedContract.acceptanceCriteria.map((criterion) => ({
            ...criterion,
          })),
          constraints: [...parent.seedContract.constraints],
          outOfScope: [...parent.seedContract.outOfScope],
          requiredArtifacts: parent.seedContract.requiredArtifacts
            ? parent.seedContract.requiredArtifacts.map((artifact) => ({ ...artifact }))
            : undefined,
          evidenceRequirements: parent.seedContract.evidenceRequirements
            ? parent.seedContract.evidenceRequirements.map((requirement) => ({ ...requirement }))
            : undefined,
          artifactSections: parent.seedContract.artifactSections
            ? parent.seedContract.artifactSections.map((section) => ({ ...section }))
            : undefined,
          executionIntent: parent.seedContract.executionIntent
            ? {
                ...parent.seedContract.executionIntent,
                capabilities: [...parent.seedContract.executionIntent.capabilities],
                allowedTools: parent.seedContract.executionIntent.allowedTools
                  ? [...parent.seedContract.executionIntent.allowedTools]
                  : undefined,
                toolSequence: parent.seedContract.executionIntent.toolSequence
                  ? [...parent.seedContract.executionIntent.toolSequence]
                  : undefined,
                authorizedWriteTargets: parent.seedContract.executionIntent.authorizedWriteTargets
                  ? [...parent.seedContract.executionIntent.authorizedWriteTargets]
                  : undefined,
                protectedWriteTargets: parent.seedContract.executionIntent.protectedWriteTargets
                  ? [...parent.seedContract.executionIntent.protectedWriteTargets]
                  : undefined,
                requestedTerminalCommand: parent.seedContract.executionIntent.requestedTerminalCommand
                  ? {
                      executable: parent.seedContract.executionIntent.requestedTerminalCommand.executable,
                      args: [...parent.seedContract.executionIntent.requestedTerminalCommand.args],
                    }
                  : undefined,
                requestedProcessStart: parent.seedContract.executionIntent.requestedProcessStart
                  ? structuredClone(parent.seedContract.executionIntent.requestedProcessStart)
                  : undefined,
                constrainedProcessStart: parent.seedContract.executionIntent.constrainedProcessStart
                  ? structuredClone(parent.seedContract.executionIntent.constrainedProcessStart)
                  : undefined,
              }
            : undefined,
        }
      : undefined,
    codebaseExploration: parent.codebaseExploration,
    codebaseMap: parent.codebaseMap
      ? {
          generatedAt: parent.codebaseMap.generatedAt,
          scoutPrompts: [...parent.codebaseMap.scoutPrompts],
          summaries: parent.codebaseMap.summaries.map((summary) => ({ ...summary })),
        }
      : undefined,
    evidenceLedger: cloneEvidenceLedger(parent.evidenceLedger),
    toolRecommendationSummary: parent.toolRecommendationSummary,
    toolCalls: [],
    toolResults: [],
    recentToolResults: [...(parent.recentToolResults ?? [])],
    memories: [...parent.memories],
    output: '',
    totalUsage: { ...parent.totalUsage },
    iteration: 0,
    maxIterations: parent.maxIterations,
    shouldStop: false,
    taskType,
    toolCallHistory: [...(parent.toolCallHistory ?? [])],
    // Transport-flip capability facts must reach the specialist subgraph so a
    // known-bad-native model does not re-detect the flip (and repeatedly fail
    // native) inside every specialist. Structural capability, not model-name.
    preferPromptReact: parent.preferPromptReact,
    emptyNativeTurnsCount: parent.emptyNativeTurnsCount,
    contentOnlyNativeTurnsCount: parent.contentOnlyNativeTurnsCount,
  }
}

function mapSpecialistStateOut(parent: AgentState, child: AgentState): AgentState {
  parent.messages = child.messages.map(cloneMessage)
  parent.memories = [...child.memories]
  parent.totalUsage = { ...child.totalUsage }
  parent.effectiveContextWindowTokens = child.effectiveContextWindowTokens
    ?? parent.effectiveContextWindowTokens
  parent.plan = child.plan ? [...child.plan] : undefined
  parent.validationPlan = child.validationPlan ? [...child.validationPlan] : undefined
  parent.seedContract = child.seedContract
    ? {
        ...child.seedContract,
        acceptanceCriteria: child.seedContract.acceptanceCriteria.map((criterion) => ({
          ...criterion,
        })),
        constraints: [...child.seedContract.constraints],
        outOfScope: [...child.seedContract.outOfScope],
        requiredArtifacts: child.seedContract.requiredArtifacts
          ? child.seedContract.requiredArtifacts.map((artifact) => ({ ...artifact }))
          : undefined,
        evidenceRequirements: child.seedContract.evidenceRequirements
          ? child.seedContract.evidenceRequirements.map((requirement) => ({ ...requirement }))
          : undefined,
        artifactSections: child.seedContract.artifactSections
          ? child.seedContract.artifactSections.map((section) => ({ ...section }))
          : undefined,
        executionIntent: child.seedContract.executionIntent
          ? {
              ...child.seedContract.executionIntent,
              capabilities: [...child.seedContract.executionIntent.capabilities],
              allowedTools: child.seedContract.executionIntent.allowedTools
                ? [...child.seedContract.executionIntent.allowedTools]
                : undefined,
              toolSequence: child.seedContract.executionIntent.toolSequence
                ? [...child.seedContract.executionIntent.toolSequence]
                : undefined,
              authorizedWriteTargets: child.seedContract.executionIntent.authorizedWriteTargets
                ? [...child.seedContract.executionIntent.authorizedWriteTargets]
                : undefined,
              protectedWriteTargets: child.seedContract.executionIntent.protectedWriteTargets
                ? [...child.seedContract.executionIntent.protectedWriteTargets]
                : undefined,
              requestedTerminalCommand: child.seedContract.executionIntent.requestedTerminalCommand
                ? {
                    executable: child.seedContract.executionIntent.requestedTerminalCommand.executable,
                    args: [...child.seedContract.executionIntent.requestedTerminalCommand.args],
                  }
                : undefined,
              requestedProcessStart: child.seedContract.executionIntent.requestedProcessStart
                ? structuredClone(child.seedContract.executionIntent.requestedProcessStart)
                : undefined,
              constrainedProcessStart: child.seedContract.executionIntent.constrainedProcessStart
                ? structuredClone(child.seedContract.executionIntent.constrainedProcessStart)
                : undefined,
            }
          : undefined,
      }
    : undefined
  parent.specialistRoute = child.specialistRoute
  parent.specialistReason = child.specialistReason
  parent.specialistBrief = child.specialistBrief
  parent.codebaseExploration = child.codebaseExploration
  parent.codebaseMap = child.codebaseMap
    ? {
        generatedAt: child.codebaseMap.generatedAt,
        scoutPrompts: [...child.codebaseMap.scoutPrompts],
        summaries: child.codebaseMap.summaries.map((summary) => ({ ...summary })),
      }
    : undefined
  parent.toolRecommendationSummary = child.toolRecommendationSummary
  parent.evidenceLedger = cloneEvidenceLedger(child.evidenceLedger)
  parent.planIndex = child.planIndex
  parent.toolCalls = []
  parent.toolResults = []
  parent.recentToolResults = [...(child.recentToolResults ?? [])]
  parent.output = child.output
  parent.analysisSummary = child.analysisSummary
  parent.findingsSummary = child.findingsSummary
  parent.implementationSummary = child.implementationSummary
  parent.implementationMutationEvidence = child.implementationMutationEvidence
    ? child.implementationMutationEvidence.map((evidence) => ({ ...evidence }))
    : undefined
  parent.verificationSummary = child.verificationSummary
  parent.validationSummary = child.validationSummary
  parent.reviewSummary = child.reviewSummary
  parent.qualityGateDecision = child.qualityGateDecision
  parent.qualityGateSummary = child.qualityGateSummary
  parent.completionDiagnostics = child.completionDiagnostics
    ? structuredClone(child.completionDiagnostics)
    : undefined
  parent.completionGateBlocks = child.completionGateBlocks
  parent.completionGateRejectedDraft = child.completionGateRejectedDraft
  parent.backtrackCount = child.backtrackCount
  parent.backtrackReason = child.backtrackReason
  parent.coworkArtifactRetryCount = child.coworkArtifactRetryCount
  parent.iteration = child.iteration
  // A user-owned prerequisite closes the whole turn, not just the specialist.
  // Preserve its structured outcome so the outer done event cannot claim success.
  parent.userActionRequired = child.userActionRequired
  parent.approvalDenied = child.approvalDenied
  if (child.userActionRequired || child.approvalDenied) parent.stopReason = child.stopReason
  parent.shouldStop = Boolean(child.userActionRequired || child.approvalDenied)
  parent.toolCallHistory = [...(child.toolCallHistory ?? [])].slice(-200)
  // Carry the observed transport-flip back to the parent so a flip detected
  // inside the specialist persists for the rest of the parent run (and, via
  // session persistence, subsequent turns).
  parent.preferPromptReact = child.preferPromptReact
  parent.emptyNativeTurnsCount = child.emptyNativeTurnsCount
  parent.contentOnlyNativeTurnsCount = child.contentOnlyNativeTurnsCount
  return parent
}

function buildSimpleGraph(deps: Deps): AgentGraph {
  return new AgentGraph()
    .setStart('iteration_guard')
    .addNode('iteration_guard', N.iterationGuard(), { lifecycleState: 'thinking' })
    .addNode('context_manager', N.contextManager(deps), { lifecycleState: 'thinking' })
    .addNode('agent', N.agent({
      ...deps,
      systemPrompt: appendSystemPrompt(deps.systemPrompt, simpleSystemPrompt),
    }), { lifecycleState: 'thinking' })
    .addNode('tools', N.toolExecutor(deps), {
      lifecycleState: 'acting',
      resumeStage: 'acting',
      pendingToolExecutionNode: true,
    })
    .addNode('reflection', N.reflection({ critiqueDeps: deps }), {
      lifecycleState: 'observing',
      resumeStage: 'observing',
    })
    .addNode('reporter', N.reporter({ enableSkillExtraction: true }), { lifecycleState: 'done' })
    .addConditionalEdge(
      'iteration_guard',
      (s: AgentState) => s.shouldStop ? 'reporter' : 'context_manager',
      ['reporter', 'context_manager'],
    )
    .addEdge('context_manager', 'agent')
    .addConditionalEdge(
      'agent',
      (s: AgentState) => s.toolCalls.length > 0 ? 'tools' : 'reporter',
      ['tools', 'reporter'],
    )
    .addEdge('tools', 'reflection')
    // Tool results are evidence, not the user-facing answer. Give the model a
    // bounded follow-up turn to synthesize them instead of sending reporter's
    // generic "tool work completed without a final answer" fallback.
    .addEdge('reflection', 'iteration_guard')
    .addEdge('reporter', '__end__')
}

// Generalist / simple specialists get Reflexion critiques too (errors AND
// low-yield). Previously only the coder/research paths passed critiqueDeps, so
// a generalist run that stalled on unhelpful results never revised strategy.
function buildIterativeSpecialistGraph(deps: Deps, specialistSystemPrompt: string): AgentGraph {
  return new AgentGraph()
    .setStart('memory_retriever')
    .addNode('memory_retriever', N.memoryRetriever(deps), { lifecycleState: 'thinking' })
    .addNode('planner', N.planner(deps), { lifecycleState: 'thinking' })
    .addNode('iteration_guard', N.iterationGuard(), { lifecycleState: 'thinking' })
    .addNode('context_manager', N.contextManager(deps), { lifecycleState: 'thinking' })
    .addNode('agent', N.agent({
      ...deps,
      systemPrompt: appendSystemPrompt(deps.systemPrompt, specialistSystemPrompt),
    }), { lifecycleState: 'thinking' })
    .addNode('tools', N.toolExecutor(deps), {
      lifecycleState: 'acting',
      resumeStage: 'acting',
      pendingToolExecutionNode: true,
    })
    .addNode('reflection', N.reflection({ critiqueDeps: deps }), {
      lifecycleState: 'observing',
      resumeStage: 'observing',
    })
    .addNode('reporter', N.reporter({ enableSkillExtraction: true }), { lifecycleState: 'done' })
    .addEdge('memory_retriever', 'planner')
    .addEdge('planner', 'iteration_guard')
    .addConditionalEdge(
      'iteration_guard',
      (s: AgentState) => s.shouldStop ? 'reporter' : 'context_manager',
      ['reporter', 'context_manager'],
    )
    .addEdge('context_manager', 'agent')
    .addConditionalEdge(
      'agent',
      (s: AgentState) => s.toolCalls.length > 0 ? 'tools' : 'reporter',
      ['tools', 'reporter'],
    )
    .addEdge('tools', 'reflection')
    .addConditionalEdge(
      'reflection',
      (s: AgentState) => !s.shouldStop && (
        !s.output?.trim()
        || Boolean(s.plan && s.planIndex < s.plan.length)
      ) ? 'iteration_guard' : 'reporter',
      ['iteration_guard', 'reporter'],
    )
    .addEdge('reporter', '__end__')
}

function buildGeneralistGraph(deps: Deps): AgentGraph {
  return buildIterativeSpecialistGraph(deps, generalistSystemPrompt)
}

function buildCreativeGraph(deps: Deps): AgentGraph {
  return buildIterativeSpecialistGraph(deps, creativeSystemPrompt)
}

export interface EnhancedGraphOptions {
  /**
   * When true, an `auto_decompose` stage runs before triage. With
   * `SEPILOTD_AUTO_DECOMPOSE=1` and a long enough input, the parent
   * fans out to subagent.dispatch calls and only then routes the
   * synthesised result through the specialist subgraph. Off by default.
   */
  enableAutoDecompose?: boolean
}

export function buildEnhancedGraph(
  deps: Deps,
  options: EnhancedGraphOptions = {},
): AgentGraph {
  const simpleGraph = buildSimpleGraph(deps)
  const generalistGraph = buildGeneralistGraph(deps)
  const creativeGraph = buildCreativeGraph(deps)
  const coderGraph = buildCoderGraph(deps)
  const researcherGraph = buildResearcherGraph(deps)
  const reviewerGraph = buildReviewerGraph(deps)

  const graph = new AgentGraph()
    .setStart(options.enableAutoDecompose ? 'auto_decompose' : 'triage')

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
    .addNode('triage', N.triage(deps), { lifecycleState: 'thinking' })
    .addNode('capability_scout', N.capabilityScout(deps), { lifecycleState: 'thinking' })
    .addNode('capability_scout_tools', N.toolExecutor(deps), {
      lifecycleState: 'acting',
      resumeStage: 'acting',
      pendingToolExecutionNode: true,
    })
    .addNode('specialist_router', N.specialistRouter(deps), { lifecycleState: 'thinking' })
    .addNode('simple_subgraph', agentSubgraphNode({
      nodeId: 'simple_subgraph',
      graph: simpleGraph,
      forwardMessages: false,
      mapIn: (parent) => createChildState(parent, 'simple'),
      mapOut: (parent, child) => mapSpecialistStateOut(parent, child),
    }), {
      lifecycleState: 'thinking',
    })
    .addNode('generalist_subgraph', agentSubgraphNode({
      nodeId: 'generalist_subgraph',
      graph: generalistGraph,
      forwardMessages: false,
      mapIn: (parent) => createChildState(parent, 'complex'),
      mapOut: (parent, child) => mapSpecialistStateOut(parent, child),
    }), {
      lifecycleState: 'thinking',
    })
    .addNode('creative_subgraph', agentSubgraphNode({
      nodeId: 'creative_subgraph',
      graph: creativeGraph,
      forwardMessages: false,
      mapIn: (parent) => createChildState(parent, 'creative'),
      mapOut: (parent, child) => mapSpecialistStateOut(parent, child),
    }), {
      lifecycleState: 'thinking',
    })
    .addNode('reviewer_subgraph', agentSubgraphNode({
      nodeId: 'reviewer_subgraph',
      graph: reviewerGraph,
      forwardMessages: false,
      mapIn: (parent) => createChildState(parent, 'complex'),
      mapOut: (parent, child) => mapSpecialistStateOut(parent, child),
    }), {
      lifecycleState: 'thinking',
    })
    .addNode('coder_subgraph', agentSubgraphNode({
      nodeId: 'coder_subgraph',
      graph: coderGraph,
      forwardMessages: false,
      mapIn: (parent) => createChildState(parent, 'code'),
      mapOut: (parent, child) => mapSpecialistStateOut(parent, child),
    }), {
      lifecycleState: 'thinking',
    })
    .addNode('researcher_subgraph', agentSubgraphNode({
      nodeId: 'researcher_subgraph',
      graph: researcherGraph,
      forwardMessages: false,
      mapIn: (parent) => createChildState(parent, 'complex'),
      mapOut: (parent, child) => mapSpecialistStateOut(parent, child),
    }), {
      lifecycleState: 'thinking',
    })
    .addNode('reporter', N.reporter({ enableSkillExtraction: true }), { lifecycleState: 'done' })

  // Edges
  if (options.enableAutoDecompose) {
    graph
      .addConditionalEdge(
        'auto_decompose',
        (s: AgentState) => s.toolCalls.length > 0 ? 'decompose_tools' : 'triage',
        ['decompose_tools', 'triage'],
      )
      .addEdge('decompose_tools', 'triage')
  }

  graph
    .addEdge('triage', 'capability_scout')
    .addConditionalEdge(
      'capability_scout',
      (s: AgentState) => s.toolCalls.length > 0 ? 'capability_scout_tools' : 'specialist_router',
      ['capability_scout_tools', 'specialist_router'],
    )
    .addEdge('capability_scout_tools', 'specialist_router')
    .addConditionalEdge(
      'specialist_router',
      (s: AgentState) => {
        switch (s.specialistRoute) {
          case 'simple':
            return 'simple_subgraph'
          case 'creative':
            return 'creative_subgraph'
          case 'reviewer':
            return 'reviewer_subgraph'
          case 'coder':
            return 'coder_subgraph'
          case 'researcher':
            return 'researcher_subgraph'
          case 'generalist':
          default:
            return 'generalist_subgraph'
        }
      },
      ['simple_subgraph', 'generalist_subgraph', 'creative_subgraph', 'reviewer_subgraph', 'coder_subgraph', 'researcher_subgraph'],
    )
    .addEdge('simple_subgraph', 'reporter')
    .addEdge('generalist_subgraph', 'reporter')
    .addEdge('creative_subgraph', 'reporter')
    .addEdge('reviewer_subgraph', 'reporter')
    .addEdge('coder_subgraph', 'reporter')
    .addEdge('researcher_subgraph', 'reporter')
    .addEdge('reporter', '__end__')

  return graph
}

export const __testables = {
  buildSimpleGraph,
  buildIterativeSpecialistGraph,
  createChildState,
  mapSpecialistStateOut,
}
