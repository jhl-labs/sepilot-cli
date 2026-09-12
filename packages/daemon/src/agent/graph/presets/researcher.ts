import { createHash } from 'node:crypto'
import { AgentGraph } from '../engine.js'
import type { Deps } from '../nodes.js'
import type { AgentState } from '../types.js'
import * as N from '../nodes.js'
import { agentSubgraphNode } from '../subgraph.js'
import { childIterationBudget } from '../iteration-budget.js'
import { childStateFrom, mergeChildInto } from '../subgraph-state.js'
import { withOptionalToolNameAllowlist } from '../../../tools/role-filter.js'
import type { ToolRegistry } from '../../../tools/registry.js'
import { signatureOf } from '../../stuck-tool-repeat.js'

export type ResearchDepth = 'standard' | 'deep'

type ResearchPhase = 'search' | 'verification'

// Verification consumes the evidence packet produced by search. Keep only
// tools that can inspect a named source directly; broad discovery belongs to
// the search phase, which the parent graph can re-enter when verification
// reports a gap. This is intentionally a conservative capability boundary:
// an unclassified integration is not silently promoted into verification.
const RESEARCH_VERIFICATION_TOOL_NAMES = new Set([
  'webfetch',
  'browser.navigate',
  'browser.evaluate',
  'browser.extract',
  'fs.read',
  'code.symbols',
  'code.dependencies',
  'git.diff',
  'memory.context.snapshot',
  'memory.documents.preview',
  'memory.documents.get',
  'memory.daily.read',
])

export function isResearchVerificationTool(toolName: string): boolean {
  return RESEARCH_VERIFICATION_TOOL_NAMES.has(toolName)
}

/**
 * Resolve the verification surface from capability metadata as well as the
 * built-in generic readers. An integration's exact authenticated source
 * reader must remain visible here; otherwise the phase-local projection also
 * removes canonical-route ownership and permits a generic unauthenticated
 * transport for that same URL. Only observe tools may opt in.
 */
export function researchVerificationToolNames(tools: ToolRegistry): string[] {
  return tools.list()
    .filter((tool) => (
      isResearchVerificationTool(tool.name)
      || (
        tool.researchVerification === 'direct-source'
        && tools.securityDescriptor(tool.name).effect === 'observe'
      )
    ))
    .map((tool) => tool.name)
}

export function resolveResearchPhaseIterationBudget(
  parent: Pick<AgentState, 'maxIterations' | 'seedContract'>,
  phase: ResearchPhase,
  depth: ResearchDepth = 'standard',
): number {
  if (depth === 'deep') {
    return childIterationBudget(parent, phase === 'search'
      ? {
          min: 4,
          legacyMax: 8,
          contractMin: 14,
          parentShare: 0.35,
          hardCap: 36,
        }
      : {
          min: 3,
          legacyMax: 6,
          contractMin: 8,
          parentShare: 0.18,
          hardCap: 20,
        })
  }

  return childIterationBudget(parent, phase === 'search'
    ? {
        min: 4,
        legacyMax: 4,
        contractMin: 8,
        parentShare: 0.2,
        hardCap: 8,
      }
    : {
        min: 3,
        legacyMax: 3,
        contractMin: 4,
        parentShare: 0.1,
        hardCap: 4,
      })
}

const RESEARCH_PHASE_BATCH_LIMITS: Record<ResearchPhase, number> = {
  search: 4,
  verification: 3,
}

/**
 * A graph iteration can contain a model-generated batch, so iteration limits
 * alone do not bound evidence volume. Keep each phase to one compact packet
 * while retaining a larger explicit deep-research allowance.
 */
export function resolveResearchPhaseToolBudget(
  parent: Pick<AgentState, 'maxIterations' | 'maxToolCallsPerTurn' | 'seedContract'>,
  phase: ResearchPhase,
  depth: ResearchDepth = 'standard',
): number {
  const iterations = resolveResearchPhaseIterationBudget(parent, phase, depth)
  const batchLimit = resolveResearchPhaseBatchLimit(parent.maxToolCallsPerTurn, phase)
  return Math.max(1, iterations * batchLimit)
}

export function resolveResearchPhaseBatchLimit(
  configuredLimit: number | undefined,
  phase: ResearchPhase,
): number {
  const phaseLimit = RESEARCH_PHASE_BATCH_LIMITS[phase]
  if (
    typeof configuredLimit !== 'number'
    || !Number.isFinite(configuredLimit)
    || configuredLimit <= 0
  ) return phaseLimit
  return Math.min(Math.max(1, Math.floor(configuredLimit)), phaseLimit)
}

export function researchPhaseActualToolCallCount(s: AgentState): number {
  if (
    typeof s.researchPhaseToolCallCount === 'number'
    && Number.isFinite(s.researchPhaseToolCallCount)
  ) {
    return Math.max(0, Math.floor(s.researchPhaseToolCallCount))
  }
  const baseline = Math.max(0, s.researchPhaseToolHistoryBaseline ?? 0)
  return (s.toolCallHistory ?? [])
    .slice(baseline)
    .filter((entry) => entry.executionObserved !== false)
    .length
}

export function researchRunActualToolCallCount(s: AgentState): number {
  if (
    typeof s.researchRunToolCallCount === 'number'
    && Number.isFinite(s.researchRunToolCallCount)
  ) {
    return Math.max(0, Math.floor(s.researchRunToolCallCount))
  }
  // Resume compatibility for checkpoints written before the run counter was
  // introduced. Counting retained executions is conservative and auditable;
  // subsequent phase handoffs switch to the exact accumulated counter.
  return (s.toolCallHistory ?? [])
    .filter((entry) => entry.executionObserved !== false)
    .length
}

/**
 * Convert the run's already allocated reasoning effort into an execution
 * ceiling. This is a runaway guard, not a completion target: evidence novelty
 * decides whether another round is useful. The 200-call upper bound matches
 * the retained tool-history window, so the controller never continues after
 * it can no longer audit every execution used by its own decision.
 */
export function resolveResearchRunToolBudget(
  parent: Pick<
    AgentState,
    'maxIterations' | 'maxToolCallsPerTurn' | 'researchRunToolCallBudget' | 'seedContract'
  >,
  depth: ResearchDepth = 'standard',
): number {
  if (
    typeof parent.researchRunToolCallBudget === 'number'
    && Number.isFinite(parent.researchRunToolCallBudget)
    && parent.researchRunToolCallBudget > 0
  ) {
    return Math.floor(parent.researchRunToolCallBudget)
  }
  const phaseFloor = resolveResearchPhaseToolBudget(parent, 'search', depth)
    + resolveResearchPhaseToolBudget(parent, 'verification', depth)
  const allocatedIterations = Math.max(1, Math.floor(parent.maxIterations || 1))
  const perIteration = resolveResearchPhaseBatchLimit(
    parent.maxToolCallsPerTurn,
    'search',
  )
  const depthMultiplier = depth === 'deep' ? 2 : 1
  return Math.min(
    200,
    Math.max(phaseFloor, allocatedIterations * perIteration * depthMultiplier),
  )
}

/**
 * Close a research phase after its bounded evidence packet is full. This is a
 * normal handoff to the phase reporter, not global task-budget exhaustion.
 */
export function enforceResearchPhaseToolBudget(
  s: AgentState,
  phase: ResearchPhase,
  depth: ResearchDepth = 'standard',
): boolean {
  const phaseBudget = resolveResearchPhaseToolBudget(s, phase, depth)
  const phaseCalls = researchPhaseActualToolCallCount(s)
  const runBudget = resolveResearchRunToolBudget(s, depth)
  const runCalls = researchRunActualToolCallCount(s)
  if (phaseCalls < phaseBudget && runCalls < runBudget) {
    // Bound the next generated batch by the remaining packet capacity. Without
    // this dynamic cap, a four-call batch after six recorded calls could make
    // an eight-call phase execute ten observations before the next guard.
    s.maxToolCallsPerTurn = Math.min(
      resolveResearchPhaseBatchLimit(s.maxToolCallsPerTurn, phase),
      phaseBudget - phaseCalls,
      runBudget - runCalls,
    )
    return false
  }

  s.shouldStop = true
  if (runCalls >= runBudget) {
    appendResearchRunBudgetMessage(s, runBudget, runCalls)
    return true
  }
  const prefix = '[Research phase evidence budget]'
  if (!s.messages.some((message) => (
    message.role === 'system'
    && typeof message.content === 'string'
    && message.content.startsWith(prefix)
  ))) {
    s.messages.push({
      role: 'system',
      content: [
        prefix,
        `The ${phase} phase reached its ${phaseBudget}-call actual execution budget (${phaseCalls} recorded).`,
        'Do not request more tools in this phase. Report the strongest retained evidence, contradictions, and the exact remaining gap for the next phase or finalizer.',
      ].join(' '),
    })
  }
  return true
}

function appendResearchRunBudgetMessage(
  s: AgentState,
  runBudget = resolveResearchRunToolBudget(s),
  runCalls = researchRunActualToolCallCount(s),
): void {
  const prefix = '[Research run execution budget]'
  const messages = s.messages ?? (s.messages = [])
  if (messages.some((message) => (
    message.role === 'system'
    && typeof message.content === 'string'
    && message.content.startsWith(prefix)
  ))) return
  messages.push({
    role: 'system',
    content: [
      prefix,
      `The run used its allocated ${runBudget}-call research execution budget (${runCalls} recorded).`,
      'Finalize from retained evidence and identify unresolved claims as incomplete; do not imply that an arbitrary source-count target was satisfied.',
    ].join(' '),
  })
}

function researchPhaseGuard(
  phase: ResearchPhase,
  depth: ResearchDepth,
): ReturnType<typeof N.iterationGuard> {
  const ordinaryGuard = N.iterationGuard()
  return async (s, context) => {
    const next = await ordinaryGuard(s, context)
    enforceResearchPhaseToolBudget(next, phase, depth)
    return next
  }
}

/**
 * Route the researcher graph after the verification subgraph: re-search when the
 * verification flagged unresolved gaps and the run still has re-search budget,
 * otherwise finalize. Exported for direct unit testing of the routing contract.
 */
export function routeAfterVerification(
  s: AgentState,
  depth: ResearchDepth = 'standard',
): 'search_subgraph' | 'finalizer' {
  const rounds = s.researchRounds ?? 0
  const configuredRoundCeiling = N.resolveMaxResearchRounds()
  const runToolBudget = resolveResearchRunToolBudget(s, depth)
  const actualRunCalls = researchRunActualToolCallCount(s)
  if (
    (configuredRoundCeiling == null || rounds < configuredRoundCeiling)
    && !s.budgetExhausted
    && s.researchSearchAddedEvidence !== false
    && actualRunCalls < runToolBudget
    && N.verificationFlagsUnresolvedGaps(s.verificationSummary)
  ) {
    return 'search_subgraph'
  }
  if (
    N.verificationFlagsUnresolvedGaps(s.verificationSummary)
    && actualRunCalls >= runToolBudget
  ) {
    appendResearchRunBudgetMessage(s, runToolBudget, actualRunCalls)
  }
  return 'finalizer'
}

function declaredToolInput(
  tools: ToolRegistry,
  toolName: string,
  input: Record<string, unknown>,
): Record<string, unknown> {
  const schema = tools.get(toolName)?.inputSchema
  const properties = schema?.properties
  if (!properties || typeof properties !== 'object' || Array.isArray(properties)) {
    return input
  }
  return Object.fromEntries(
    Object.keys(properties)
      .filter((key) => Object.hasOwn(input, key))
      .map((key) => [key, input[key]]),
  )
}

/**
 * Set fingerprint of successful observe results. Call ids, timestamps,
 * duplicated observations, and undeclared presentation arguments are not
 * evidence novelty; a different source identity or different complete result
 * fingerprint is.
 */
export function researchObservationFingerprint(
  history: AgentState['toolCallHistory'],
  tools: ToolRegistry,
): string {
  const identities = new Set<string>()
  for (const entry of history ?? []) {
    if (
      entry.status !== 'success'
      || entry.executionObserved === false
      || tools.securityDescriptor(entry.tool).effect !== 'observe'
    ) continue
    const inputIdentity = signatureOf({
      tool: entry.tool,
      input: declaredToolInput(tools, entry.tool, entry.input),
    })
    const outputIdentity = entry.outputFingerprint
      ?? createHash('sha256').update(entry.output ?? '').digest('hex')
    identities.add(`${inputIdentity}:${outputIdentity}`)
  }
  return createHash('sha256')
    .update([...identities].sort().join('\n'))
    .digest('hex')
}

/** Skip a second verification pass when re-search added no successful evidence. */
export function routeAfterSearch(
  s: AgentState,
  depth: ResearchDepth = 'standard',
): 'verification_subgraph' | 'finalizer' {
  // The executor has already closed an explicitly exact tool workflow after
  // recording every permitted outcome. A verification subgraph cannot add
  // evidence because the user's closed capability boundary exposes no
  // remaining source tool; entering it only spends model turns and can replace
  // a valid findings summary with a progress-only verification draft.
  if (
    s.stuckRepeatForcedFinal === true
    && s.forcedFinalSynthesisReason === 'exact-tool-budget'
  ) {
    return 'finalizer'
  }
  const runBudget = resolveResearchRunToolBudget(s, depth)
  const runCalls = researchRunActualToolCallCount(s)
  if (runCalls >= runBudget) {
    appendResearchRunBudgetMessage(s, runBudget, runCalls)
    return 'finalizer'
  }
  if (
    (s.researchRounds ?? 0) > 1
    && s.researchSearchAddedEvidence === false
    && N.verificationFlagsUnresolvedGaps(s.verificationSummary)
  ) {
    return 'finalizer'
  }
  return 'verification_subgraph'
}

const searchSystemPrompt = [
  'You are the research evidence gathering specialist.',
  'Search broadly, gather concrete findings, and keep track of source details.',
  'A selected skill\'s dependency-ready required stage is reserved inside this bounded evidence phase. Complete that required stage before optional observations; unrelated calls may be deferred until its successful result is available.',
  'When a successful webfetch result says relevant middle content was omitted, read the same URL with webfetch query set to one concrete literal term from the active claim; do not refresh the full page or guess that the omitted section is absent.',
  'When fetched HTML names a relevant document or navigation destination but the stripped text does not show its href, call webfetch on that page with linkQuery set to literal anchor text or a URL fragment, then follow the resolved link. Do not invent a likely path when the source page can expose it.',
  'Use tools when needed, then summarize the strongest findings with source-aware language.',
].join(' ')

const verificationSystemPrompt = [
  'You are the research verification specialist.',
  'Treat the gathered findings as a fixed evidence packet: inspect named sources directly when needed, but do not perform broad discovery searches.',
  'Start with the retained successful tool excerpts. Do not fetch an already observed passage again merely because this is a separate verification phase.',
  'Verify claims needed for the user-requested deliverable. Remove unsupported optional detail instead of adding new version, quotation, or breadth requirements that the user and contract did not request.',
  'A findings summary may paraphrase its source. Literal extraction queries must use terms actually observed in the source, not guessed quotations of the paraphrase. Finish with VERIFIED as soon as the required claims are supported.',
  'For a named long source whose middle was omitted, use webfetch query with one claim-relevant literal term to obtain a bounded exact excerpt before declaring the source insufficient.',
  'If static HTML exposes the title of a relevant linked source without its destination, use webfetch linkQuery to resolve that anchor before guessing URLs or declaring the source undiscoverable.',
  'If evidence is missing or contradictory, report the concrete gap so the parent graph can return to its search phase.',
  'Finish with a concise verification summary and make the last non-empty line exactly VERIFIED: followed by the support summary, or UNVERIFIED: followed by the remaining evidence gap.',
].join(' ')

function buildSearchSubgraph(deps: Deps, depth: ResearchDepth): AgentGraph {
  return new AgentGraph()
    .setStart('context_manager')
    .addNode('context_manager', N.contextManager(deps), {
      lifecycleState: 'thinking',
    })
    .addNode('search_agent', N.agent({
      ...deps,
      systemPrompt: (deps.systemPrompt ?? '') + '\n\n' + searchSystemPrompt,
    }), {
      lifecycleState: 'thinking',
    })
    .addNode('search_tools', N.toolExecutor(deps), {
      lifecycleState: 'acting',
      resumeStage: 'acting',
      pendingToolExecutionNode: true,
    })
    .addNode('search_reflection', N.reflection({ critiqueDeps: deps }), {
      lifecycleState: 'observing',
      resumeStage: 'observing',
    })
    .addNode('search_guard', researchPhaseGuard('search', depth), {
      lifecycleState: 'thinking',
    })
    .addNode('search_reporter', N.reporter({ enableSkillExtraction: true }), {
      lifecycleState: 'done',
    })
    .addEdge('context_manager', 'search_agent')
    .addConditionalEdge(
      'search_agent',
      (s: AgentState) => s.toolCalls.length > 0 ? 'search_tools' : 'search_reporter',
      ['search_tools', 'search_reporter'],
    )
    .addEdge('search_tools', 'search_reflection')
    .addEdge('search_reflection', 'search_guard')
    .addConditionalEdge(
      'search_guard',
      (s: AgentState) => s.shouldStop ? 'search_reporter' : 'search_agent',
      ['search_reporter', 'search_agent'],
    )
    .addEdge('search_reporter', '__end__')
}

function buildVerificationSubgraph(deps: Deps, depth: ResearchDepth): AgentGraph {
  const verificationDeps = {
    ...deps,
    tools: withOptionalToolNameAllowlist(
      deps.tools,
      researchVerificationToolNames(deps.tools),
    ),
  }
  return new AgentGraph()
    .setStart('context_manager')
    .addNode('context_manager', N.contextManager(verificationDeps), {
      lifecycleState: 'thinking',
    })
    .addNode('verify_agent', N.agent({
      ...verificationDeps,
      systemPrompt: (deps.systemPrompt ?? '') + '\n\n' + verificationSystemPrompt,
    }), {
      lifecycleState: 'thinking',
    })
    .addNode('verify_tools', N.toolExecutor(verificationDeps), {
      lifecycleState: 'acting',
      resumeStage: 'acting',
      pendingToolExecutionNode: true,
    })
    .addNode('verify_reflection', N.reflection({
      advancePlanOnSuccess: false,
      critiqueDeps: verificationDeps,
    }), {
      lifecycleState: 'observing',
      resumeStage: 'observing',
    })
    .addNode('verify_guard', researchPhaseGuard('verification', depth), {
      lifecycleState: 'thinking',
    })
    .addNode('verify_reporter', N.reporter({ enableSkillExtraction: true }), {
      lifecycleState: 'done',
    })
    .addEdge('context_manager', 'verify_agent')
    .addConditionalEdge(
      'verify_agent',
      (s: AgentState) => s.toolCalls.length > 0 ? 'verify_tools' : 'verify_reporter',
      ['verify_tools', 'verify_reporter'],
    )
    .addEdge('verify_tools', 'verify_reflection')
    .addEdge('verify_reflection', 'verify_guard')
    .addConditionalEdge(
      'verify_guard',
      (s: AgentState) => s.shouldStop ? 'verify_reporter' : 'verify_agent',
      ['verify_reporter', 'verify_agent'],
    )
    .addEdge('verify_reporter', '__end__')
}

export function buildResearcherGraph(
  deps: Deps,
  options: { depth?: ResearchDepth } = {},
): AgentGraph {
  const depth = options.depth ?? 'standard'
  const searchGraph = buildSearchSubgraph(deps, depth)
  const verificationGraph = buildVerificationSubgraph(deps, depth)

  return new AgentGraph()
    .setStart('memory_retriever')
    .addNode('memory_retriever', N.memoryRetriever(deps), {
      lifecycleState: 'thinking',
    })
    .addNode('planner', N.planner(deps), {
      lifecycleState: 'thinking',
    })
    .addNode('search_subgraph', agentSubgraphNode({
      nodeId: 'search_subgraph',
      graph: searchGraph,
      nodePrefix: 'search',
      forwardMessages: false,
      mapContext: (_parent, context) => context
        ? {
            ...context,
            toolSecurityEffectBoundary: 'observe-only',
          }
        : context,
      mapIn: (parent) => {
        if (parent.researchRunToolCallBudget == null) {
          parent.researchRunToolCallBudget = resolveResearchRunToolBudget(parent, depth)
        }
        if (parent.researchRunToolCallCount == null) {
          parent.researchRunToolCallCount = (parent.researchRounds ?? 0) > 0
            ? researchRunActualToolCallCount(parent)
            : 0
        }
        return childStateFrom(parent, {
          input: parent.input,
          planIndex: parent.planIndex,
          researchPhaseToolCallCount: 0,
          researchPhaseToolHistoryBaseline: (parent.toolCallHistory ?? []).length,
          researchSearchBaselineFingerprint: researchObservationFingerprint(
            parent.toolCallHistory,
            deps.tools,
          ),
          maxIterations: resolveResearchPhaseIterationBudget(parent, 'search', depth),
          maxToolCallsPerTurn: Math.min(
            resolveResearchPhaseBatchLimit(parent.maxToolCallsPerTurn, 'search'),
            parent.researchRunToolCallBudget - parent.researchRunToolCallCount,
          ),
          taskType: 'complex',
        })
      },
      mapOut: (parent, child) => {
        const findingsSummary = child.output.trim()
        const runCalls = researchRunActualToolCallCount(child)
        const searchEvidenceFingerprint = researchObservationFingerprint(
          child.toolCallHistory,
          deps.tools,
        )
        const searchAddedEvidence = child.researchSearchBaselineFingerprint !== undefined
          && child.researchSearchBaselineFingerprint !== searchEvidenceFingerprint
        mergeChildInto(parent, child)
        parent.researchRunToolCallCount = runCalls
        parent.planIndex = child.planIndex
        // Count each completed search pass for diagnostics and an optional
        // operator-provided ceiling. Default convergence is evidence-driven.
        parent.researchRounds = (parent.researchRounds ?? 0) + 1
        parent.researchSearchBaselineFingerprint = child.researchSearchBaselineFingerprint
        parent.researchSearchAddedEvidence = searchAddedEvidence
        if (findingsSummary) {
          parent.findingsSummary = findingsSummary
        }
        return parent
      },
    }), {
      lifecycleState: 'thinking',
    })
    .addNode('verification_subgraph', agentSubgraphNode({
      nodeId: 'verification_subgraph',
      graph: verificationGraph,
      nodePrefix: 'verify',
      forwardMessages: false,
      mapContext: (_parent, context) => context
        ? { ...context, toolSecurityEffectBoundary: 'observe-only' }
        : context,
      mapIn: (parent) => childStateFrom(parent, {
        input: [
          'Verify the current research findings. Identify what is supported, weak, missing, or contradictory.',
          `Original task:\n${parent.input}`,
          parent.findingsSummary
            ? `Findings summary:\n${parent.findingsSummary}`
            : '',
        ].filter(Boolean).join('\n\n'),
        researchPhaseToolCallCount: 0,
        researchPhaseToolHistoryBaseline: (parent.toolCallHistory ?? []).length,
        maxIterations: resolveResearchPhaseIterationBudget(parent, 'verification', depth),
        maxToolCallsPerTurn: Math.min(
          resolveResearchPhaseBatchLimit(parent.maxToolCallsPerTurn, 'verification'),
          resolveResearchRunToolBudget(parent, depth) - researchRunActualToolCallCount(parent),
        ),
        taskType: 'complex',
      }),
      mapOut: (parent, child) => {
        const verificationSummary = child.output.trim()
        const runCalls = researchRunActualToolCallCount(child)
        mergeChildInto(parent, child)
        parent.researchRunToolCallCount = runCalls
        if (verificationSummary) {
          parent.verificationSummary = verificationSummary
        }
        return parent
      },
    }), {
      lifecycleState: 'thinking',
    })
    .addNode('finalizer', N.researchFinalizer(deps), {
      lifecycleState: 'done',
    })
    .addEdge('memory_retriever', 'planner')
    .addEdge('planner', 'search_subgraph')
    .addConditionalEdge(
      'search_subgraph',
      (s: AgentState) => routeAfterSearch(s, depth),
      ['verification_subgraph', 'finalizer'],
    )
    .addConditionalEdge(
      'verification_subgraph',
      (s: AgentState) => routeAfterVerification(s, depth),
      ['search_subgraph', 'finalizer'],
    )
    .addEdge('finalizer', '__end__')
}
