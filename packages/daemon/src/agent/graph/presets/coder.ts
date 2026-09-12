import { randomUUID } from 'node:crypto'
import type { ToolCall } from '@sepilotd/core'
import { AgentGraph } from '../engine.js'
import type { Deps } from '../nodes.js'
import type { AgentState } from '../types.js'
import * as N from '../nodes.js'
import { buildReviewerGraph } from './reviewer.js'
import { agentSubgraphNode } from '../subgraph.js'
import { openEditCheckpointNode } from './edit-checkpoint-node.js'
import { childIterationBudget } from '../iteration-budget.js'
import { childStateFrom, mergeChildInto } from '../subgraph-state.js'
import { buildFocusedLoopGraph } from './focused-loop.js'
import { logAgentDebugTrace } from '../../../observability/agent-trace.js'
import { inputLimitsCurrentTurnToDocumentArtifact } from '../../task-contract.js'

const operationalToolAllowlist = [
  'process.*',
  'service.*',
  'todowrite',
  'terminal.run',
  'browser.*',
  'webfetch',
  'fs.read',
  'fs.list',
  'fs.glob',
  'fs.search',
  'git.status',
  'git.diff',
  'git.log',
  'system.info',
] as const

const operationalSystemPrompt = [
  'You are executing a bounded operational action inside a coding session.',
  'Fulfill the requested runtime, process, service, browser, HTTP, log, or observation action directly.',
  'The structured run contract forbids product/source mutation: do not create, edit, append, delete, or generate application source, configuration, or build inputs, and do not use terminal commands to do so.',
  'Validation artifacts explicitly requested by the assigned subtask, such as browser screenshots at the requested artifact paths and todowrite QA evidence, are allowed; they do not grant authority to change the product under test.',
  'Use existing application entry points and runtime options instead of changing source code or inventing a replacement application.',
  'For a user-requested development process that should remain available in later turns, prefer process.start lifetime=session. For host/LAN exposure or daemon-restart survival, prefer service.start.',
  'When leaving a process or service running, report its id, lifecycle, endpoint/reachability, status, and the commands or tools for reading logs and stopping it.',
  'Stop only bounded validation helpers; do not stop the requested long-lived runtime before reporting completion.',
].join(' ')

function routesToOperationalSubgraph(state: AgentState): boolean {
  const intent = state.seedContract?.executionIntent
  return intent?.workspaceMutation === 'forbidden'
    && (intent.kind === 'operational-action' || intent.kind === 'inspection')
}

function requestedTerminalToolCall(state: AgentState): ToolCall | undefined {
  const processStart = state.seedContract?.executionIntent?.requestedProcessStart
  if (processStart) {
    return {
      id: `requested-process-${randomUUID()}`,
      name: 'process.start',
      arguments: {
        ...processStart,
        args: [...processStart.args],
        ...(typeof processStart.network === 'object'
          ? {
              network: {
                ...processStart.network,
                ...(processStart.network.ports ? { ports: [...processStart.network.ports] } : {}),
              },
            }
          : {}),
      },
    }
  }
  const command = state.seedContract?.executionIntent?.requestedTerminalCommand
  if (!command) return undefined
  return {
    id: `requested-terminal-${randomUUID()}`,
    name: 'terminal.run',
    arguments: {
      executable: command.executable,
      args: [...command.args],
    },
  }
}

/**
 * A staged "document first" turn is complete once its requested artifact was
 * written. Sending it through the code validation + reviewer pipeline expands
 * the user's current phase into build/runtime work and adds several unrelated
 * provider calls. The next implementation phase remains a later user turn.
 */
function shouldFinalizeCurrentDocumentPhase(state: AgentState): boolean {
  // The edge calls this only after the successful-action guard immediately
  // above it. Keep phase detection independent from artifact-kind heuristics:
  // an "architecture plan" and a "design document" are the same current-turn
  // boundary even when their filenames or wording differ.
  return inputLimitsCurrentTurnToDocumentArtifact(state.input)
}

function routesDirectlyToCurrentDocumentPhase(state: AgentState): boolean {
  return inputLimitsCurrentTurnToDocumentArtifact(state.input)
}

function routeImplementationAgentResult(
  state: AgentState,
): 'tools' | 'implementation_guard' | 'implementation_file_edit_guard' | 'capture_implementation' {
  if (state.toolCalls.length > 0) return 'tools'
  if (state.implementationModelRecoveryRequested) return 'implementation_guard'
  return state.shouldStop ? 'capture_implementation' : 'implementation_file_edit_guard'
}

function routeImplementationGuardResult(
  state: AgentState,
): 'tools' | 'implement' | 'finalizer' | 'implementation_file_edit_guard' | 'capture_implementation' {
  if (state.toolCalls.length > 0) return 'tools'
  if (state.implementationCompleteRequested) return 'capture_implementation'
  if (!state.shouldStop) return 'implement'
  return N.shouldRouteStoppedImplementationToEditGuard(state)
    ? 'implementation_file_edit_guard'
    : 'finalizer'
}

function routeValidationAgentResult(
  state: AgentState,
): 'validation_tools' | 'capture_validation' | 'validation_completion_guard' {
  if (state.toolCalls.length > 0) return 'validation_tools'
  // A transport/protocol exhaustion after inherited validation evidence is
  // not itself a quality verdict. Send it through the bounded convergence
  // controller, which can obtain one structured VERIFIED/UNVERIFIED result,
  // instead of skipping directly to the finalizer.
  return state.shouldStop ? 'validation_completion_guard' : 'capture_validation'
}

function routePostEditAnalysisResult(
  state: AgentState,
): 'mark_finalize_phase' | 'mark_validation_phase' {
  // A concrete blocker/transport stop still finishes immediately. A
  // structured semantic completion decision is different: the independent
  // implementation audit has concluded that the current workspace already
  // satisfies the implementation phase, so validation must be allowed to
  // confirm or reject that conclusion even when this turn needed no new edit.
  const semanticallyComplete = state.implementationCompleteRequested === true
  if (state.shouldStop && !semanticallyComplete) return 'mark_finalize_phase'
  // Ablation toggle: skip the post-edit validation + review phases.
  if (process.env.SEPILOTD_CODER_SKIP_VALIDATION_REVIEW === '1') {
    return 'mark_finalize_phase'
  }
  if (!N.stateHasSuccessfulImplementationAction(state) && !semanticallyComplete) {
    return 'mark_finalize_phase'
  }
  if (shouldFinalizeCurrentDocumentPhase(state)) {
    return 'mark_finalize_phase'
  }
  return 'mark_validation_phase'
}

function routeCodingFinalizerResult(state: AgentState): 'implementation_guard' | '__end__' {
  return state.completionDiagnostics?.gate?.decision === 'block'
    ? 'implementation_guard'
    : '__end__'
}

function currentDocumentInventoryArguments(cwd: string | undefined): Record<string, unknown> {
  return {
    ...(cwd ? { cwd } : {}),
    // A greenfield decision needs the complete top-level inventory once.
    // Omitting dotfiles invites a second, nearly identical fs.list call before
    // the model is willing to write the requested document.
    hidden: true,
  }
}

const coderSystemPrompt = [
  'You are a coding agent.',
  'Focus on reading the relevant code, making targeted changes, and using tools deliberately.',
  'For large codebases, map the likely package and symbol ownership first: scoped search/glob, code.symbols or LSP references for identifiers, code.dependencies for imports/callers, then only read the narrow files you need.',
  'Use a read-only explore/research subagent for noisy subsystem surveys or verbose validation output; bring back compact evidence instead of flooding the implementation context.',
  'Prefer the smallest coherent patch that solves the task.',
  // Edit-strategy resilience: weaker models often emit malformed apply_patch
  // hunks and loop on the failure, producing an empty/garbled file.
  'If a patch/edit tool (apply_patch or fs.edit) fails more than once on the same file, STOP retrying it — write the COMPLETE corrected file in one fs.write call instead. For a brand-new file, use fs.write directly rather than apply_patch.',
  'When the intended change is to delete an obsolete file, delete it with apply_patch (or another permitted deletion capability). Never replace it with a tombstone, error-exit stub, or “removed” placeholder unless the user explicitly requested that compatibility behavior; a failed deletion is a blocker to repair or report, not authorization to preserve junk under the same path.',
  // Robustness + testability guidance (general, helps every CLI/TUI/service task):
  'Write code that adapts to its real runtime, not just the happy path: handle variable counts (e.g. many CPUs/cores, many rows/items), the actual terminal width and height, empty inputs, and missing files. Do not assume a small fixed number of anything or that everything fits on one screen.',
  'Structure the program so its data-gathering and computation logic can be exercised independently of any interactive/full-screen UI (small pure functions you can call and check directly).',
  'For browser-rendered frontend work, do a short, concrete design pass before coding and record it as a completed todowrite item before the first implementation edit: identify the actual target user workflow, named primary screens/states, responsive layout strategy, visual style direction, expected controls/interactions, and required visual assets/media. Do not satisfy this with placeholder labels only. Build the actual usable product/tool/game as the first screen unless the user specifically asked for a landing page.',
  "Treat 'start/run/serve this application' as a runtime action, not implicit authorization to hard-code a host or port in source: prefer existing CLI/config options and edit source only when the requested behavior genuinely requires a code change. Choose server lifetime from purpose. For an agent-internal validation server, use process.start with lifetime='bounded', a positive ttlMs, and network { mode: loopback, ports: [PORT] }. For a user-requested development/watch server that should remain available during later turns, use process.start with lifetime='session' and leave it running. If the user explicitly needs host/LAN exposure such as 0.0.0.0 or survival across daemon restarts, use service.start instead. Managed loopback exposes only the declared same-session localhost ports inside the agent bridge; it is not host-network exposure. Use browser tools or webfetch for GET requests; use terminal.run with network=loopback for curl or other API calls requiring methods, headers, or bodies.",
  'Keep repository inspection proportional to the next decision. Search for the relevant symbol or assertion first, then fs.read only the narrow line range around the hit. Do not speculatively read complete source and test files in parallel, and do not request an unchanged file view again after a successful read or cache hit.',
  'For browser-rendered frontend work (web apps, static sites, dashboards, landing pages, games, HTML/CSS/UI design), plan for real visual validation: start or locate a local server when needed, preferably with process.start/process.stop for long-running dev or static servers instead of backgrounding terminal.run, capture rendered screenshots at desktop and mobile viewports with PNG paths that produce `Screenshot image attachment: attached`, inspect the actual image and read browser.screenshot layout-audit warnings, compare the rendered result against the design pass, record a completed visual QA todowrite after the latest browser audit with concrete issues found or explicitly none while checking layout/spacing, text wrapping/overflow, contrast/readability, overlap/collision, controls/touch targets, and assets/media completeness, include representative interactive/active states at desktop and mobile viewports for interactive artifacts using browser.click/browser.evaluate with saved screenshot/layout audits whose images are attached, collect dynamic browser.evaluate evidence for games or animated canvas/WebGL work showing frame, pixel, position, or game-state changes over time at desktop and mobile viewports, and fix visible layout, spacing, overflow, text wrapping, low contrast, overlapping text/controls, blank-band/cutoff warnings, and polish issues before claiming completion. Any visual fix requires a fresh screenshot after the fix.',
  'When you finish the implementation phase, summarize what changed and why.',
  'If a process or service remains running, include its id, lifecycle scope, status, endpoint/reachability scope, and later log/status/stop controls in the summary. Do not bury those details in tool output.',
].join(' ')

const validatorSystemPrompt = [
  'You are a coding validation agent.',
  'Inspect the implementation, run the smallest meaningful verification step, and report what is verified versus still unverified.',
  'Prefer concrete validation over vague claims.',
  'Do not report something as working unless you actually ran it this turn and read the output.',
  'Match evidence to each acceptance criterion instead of treating one green command as proof of every criterion. A criterion requiring a newly added or updated regression test needs corresponding test-artifact mutation evidence plus a relevant passing check; a pre-existing passing test alone does not satisfy it.',
  'If the task expects a built executable, a generated/output file, or a passing test, produce it and then verify it exists at the expected path with the expected content — a clean compile is not proof the program runs or that the artifact landed where the task wants it.',
  "For an agent-internal workspace-local web validation server, use process.start with lifetime='bounded', a positive ttlMs, and network { mode: loopback, ports: [PORT] } so browser and HTTP validation can reach the same managed server. Reuse a user-requested session-lifetime process or durable service available to the current strict workspace instead of starting a duplicate. Use terminal.run with network=loopback for curl requests that need custom methods, headers, or bodies; managed loopback proves only capability-bound workspace localhost reachability, not a host/LAN 0.0.0.0 bind.",
  'For browser-rendered frontend work (web apps, static sites, dashboards, landing pages, games, HTML/CSS/UI design), syntax checks, DOM counts, and build success are not design validation. Run the UI in a real browser, use process.start/process.stop for long-running local servers when available, capture and inspect screenshots at desktop and mobile viewports whose browser output says `Screenshot image attachment: attached`, read the browser.screenshot layout audit, compare the result against the design plan, record a completed visual QA todowrite after the latest browser audit with issues found or explicitly none while checking layout/spacing, text wrapping/overflow, contrast/readability, overlap/collision, controls/touch targets, and assets/media completeness, use browser.click/browser.evaluate with saved screenshot/layout audits and attached images for representative interaction/active states at desktop and mobile viewports when interaction exists instead of modifying the app only to make validation convenient, and collect dynamic browser.evaluate evidence for games or animated canvas/WebGL work showing frame, pixel, position, or game-state changes over time at desktop and mobile viewports. Check console/page errors when possible, then report VERIFIED only if screenshots were actually inspected, dynamic evidence was gathered where applicable, and the summary names the visual QA issues found or explicitly says none were found. A clean audit is useful evidence but not a substitute for visual inspection when visual inspection is required. A home screen alone is insufficient for games or tools. If screenshots or layout audit show broken wrapping, edge-hugging layouts, giant controls, empty-looking game boards, low contrast, overlapping text/controls, blank lower bands, clipped content, or other obvious visual defects, return UNVERIFIED so implementation can fix and re-screenshot. If browser screenshots cannot be captured, attached, or viewed, end with UNVERIFIED and the blocker.',
  "After validation evidence is collected, do not create a cleanup loop. Stop bounded validation servers, but keep a lifetime='session' process or durable service running when it exists for the user's ongoing development. Do not treat an intentionally stopped validation server as a new failure and do not restart it just to re-run equivalent checks. When handing off a running server, report its id, lifecycle, status, reachability scope, and log/status/stop controls.",
  // TUI/interactive verification: do not skip checking just because the UI cannot render headlessly.
  'If the program is interactive or full-screen (a TUI/curses/GUI app) and cannot be run headlessly, do NOT skip verification. Exercise an existing non-interactive interface to its underlying logic: import the module or call its data/computation functions and check the outputs are actually correct — e.g. values are non-zero where the system clearly has activity, lists have the expected number of entries, and nothing is a placeholder. Always also run a syntax/import smoke check. Validation is workspace-read-only; if the implementation provides no testable non-interactive interface, report that as an implementation defect instead of adding or changing source during validation.',
  'If verification fails or you could not run it, say so plainly and list what is still unverified — do not paper over it in the summary.',
  'End your summary with exactly one of these two lines, on its own line, as the final line:',
  'VERIFIED: <one-line evidence — name the artifact and the check that confirmed it>',
  'UNVERIFIED: <one-line blocker — what failed or could not be checked>',
  'The orchestrator reads this stem to decide whether to backtrack. Hedging earlier in the summary does not trigger a backtrack; only the UNVERIFIED stem does. Pick one stem and emit it verbatim.',
].join(' ')

export const __testables = {
  coderSystemPrompt,
  validatorSystemPrompt,
  operationalSystemPrompt,
  operationalToolAllowlist,
  requestedTerminalToolCall,
  routesToOperationalSubgraph,
  routesDirectlyToCurrentDocumentPhase,
  routeImplementationAgentResult,
  routeImplementationGuardResult,
  routeValidationAgentResult,
  routePostEditAnalysisResult,
  routeCodingFinalizerResult,
  shouldFinalizeCurrentDocumentPhase,
  currentDocumentInventoryArguments,
}

function stripToolCatalog(systemPrompt: string | undefined): string {
  if (!systemPrompt) return ''
  // Drop only the single-line `Available tools: a, b, c, ...` enumeration
  // emitted by buildSystemPrompt — prompt-react re-presents the same list
  // with descriptions and JSON schemas, so the bare comma-separated copy
  // is duplicate noise. Keep the surrounding tool-usage guidance lines
  // ("Read before write", "Failure recovery", error-code hints, etc.)
  // because they are not redundant with prompt-react.
  return systemPrompt.replace(/^Available tools:[^\n]*\n?/m, '').trim()
}

export function buildCoderGraph(deps: Deps): AgentGraph {
  const coderBaseSystemPrompt = [
    stripToolCatalog(deps.systemPrompt),
    coderSystemPrompt,
  ].filter(Boolean).join('\n\n')
  const validatorBaseSystemPrompt = [
    stripToolCatalog(deps.systemPrompt),
    validatorSystemPrompt,
  ].filter(Boolean).join('\n\n')
  const reviewerGraph = buildReviewerGraph({
    ...deps,
    systemPrompt: [
      stripToolCatalog(deps.systemPrompt),
      'You are reviewing code inside a coding workflow after implementation and validation already happened.',
      'Focus on correctness, regressions, missing verification, and any remaining risk.',
      'This internal review is evidence-only. Inspect current source, diffs, diagnostics, and retained validation results as needed, but do not rerun tests/builds/formatters, mutate files, or manage processes. If current evidence is insufficient, return UNVERIFIED with the concrete missing evidence so the workflow can route back to validation.',
      'Match implementation and validation evidence to each acceptance criterion independently. In particular, do not accept a requirement to add or update regression coverage when the change evidence contains no matching test artifact, even if an existing suite passes.',
      'End your summary with exactly one of these two lines, on its own line, as the final line: "VERIFIED: <evidence>" if the review found no blocking issues, or "UNVERIFIED: <one-line blocker>" if a backtrack is needed. The orchestrator reads only this stem; hedging earlier in the review does not trigger a backtrack.',
    ].filter(Boolean).join('\n\n'),
  }, { preserveProtocolOutput: true })
  const operationalGraph = buildFocusedLoopGraph({
    ...deps,
    systemPrompt: stripToolCatalog(deps.systemPrompt),
  }, {
    systemPrompt: operationalSystemPrompt,
    toolAllowlist: operationalToolAllowlist,
    requireCurrentTurnToolEvidence: true,
    initialToolCall: requestedTerminalToolCall,
    completeAfterInitialToolResult: true,
  })

  return new AgentGraph()
    .setStart('memory_retriever')
    .addNode('memory_retriever', N.memoryRetriever(deps), {
      lifecycleState: 'thinking',
    })
    .addNode('execution_intent_router', async (s: AgentState, context) => {
      const intent = s.seedContract?.executionIntent
      await logAgentDebugTrace({
        event: 'supervisor.execution-intent-route',
        source: 'coder-graph',
        sessionId: context?.agentContext.sessionId,
        runId: context?.agentContext.sessionId,
        status: intent?.kind ?? 'default-coder',
        data: {
          workspaceMutation: intent?.workspaceMutation,
          capabilities: intent?.capabilities ?? [],
          route: routesToOperationalSubgraph(s)
            ? 'operational_subgraph'
            : routesDirectlyToCurrentDocumentPhase(s)
              ? 'current_document_inventory'
            : 'codebase_exploration',
        },
      })
      return s
    }, {
      lifecycleState: 'thinking',
    })
    .addNode('operational_subgraph', agentSubgraphNode({
      nodeId: 'operational_subgraph',
      graph: operationalGraph,
      forwardMessages: true,
      mapIn: (parent) => childStateFrom(parent, {
        taskType: 'simple',
        maxIterations: childIterationBudget(parent, {
          min: 3,
          legacyMax: 8,
          contractMin: 8,
          parentShare: 0.5,
          hardCap: 16,
        }),
      }),
      mapOut: (parent, child) => {
        const output = child.output
        mergeChildInto(parent, child)
        parent.output = output
        return parent
      },
    }), {
      lifecycleState: 'thinking',
    })
    .addNode('current_document_inventory', async (s: AgentState, context) => {
      s.currentStep = 'current_document_inventory'
      s.toolCalls = deps.tools.isEnabled('fs.list')
        ? [{
            id: `current-document-inventory-${randomUUID()}`,
            name: 'fs.list',
            arguments: currentDocumentInventoryArguments(context?.agentContext.cwd),
          }]
        : []
      return s
    }, {
      lifecycleState: 'thinking',
    })
    .addNode('current_document_inventory_tools', N.toolExecutor(deps), {
      lifecycleState: 'acting',
      resumeStage: 'acting',
    })
    .addNode('codebase_exploration', N.codebaseExplorer(deps), {
      lifecycleState: 'thinking',
    })
    .addNode('codebase_exploration_tools', N.toolExecutor(deps), {
      lifecycleState: 'acting',
      resumeStage: 'acting',
    })
    .addNode('capture_codebase_exploration', N.captureCodebaseExplorationResults(), {
      lifecycleState: 'observing',
      resumeStage: 'observing',
    })
    .addNode('large_codebase_scout', N.largeCodebaseScout(deps), {
      lifecycleState: 'thinking',
    })
    .addNode('large_codebase_scout_tools', N.toolExecutor(deps), {
      lifecycleState: 'acting',
      resumeStage: 'acting',
    })
    .addNode('capture_large_codebase_scout', N.captureLargeCodebaseScoutResults(), {
      lifecycleState: 'observing',
      resumeStage: 'observing',
    })
    .addNode('coding_planner', N.codingPlanner(deps), {
      lifecycleState: 'thinking',
    })
    .addNode('plan_revision', N.planRevision(deps), {
      lifecycleState: 'thinking',
    })
    .addNode('context_manager', N.contextManager(deps), {
      lifecycleState: 'thinking',
    })
    .addNode('tool_recommender', N.toolRecommender(deps), {
      lifecycleState: 'thinking',
    })
    .addNode('open_edit_checkpoint', openEditCheckpointNode('coder-implement'), {
      lifecycleState: 'thinking',
    })
    .addNode('implement', N.agent({
      ...deps,
      systemPrompt: coderBaseSystemPrompt,
    }), {
      lifecycleState: 'thinking',
    })
    .addNode('tools', N.toolExecutor(deps), {
      lifecycleState: 'acting',
      resumeStage: 'acting',
      pendingToolExecutionNode: true,
    })
    .addNode('implementation_reflection', N.reflection({ critiqueDeps: deps }), {
      lifecycleState: 'observing',
      resumeStage: 'observing',
    })
    .addNode('implementation_completion_guard', N.implementationCompletionGuard(), {
      lifecycleState: 'thinking',
    })
    // Adaptive recovery for a flailing implement loop. The checkpoint remains
    // bounded and preserves the full contextual tool surface; the independent
    // no-progress judgment below decides whether another concrete action is
    // still available instead of terminating solely from a loop counter.
    .addNode('implementation_scaffold', N.implementationActionCheckpoint(), {
      lifecycleState: 'thinking',
    })
    .addNode('implementation_guard', N.iterationGuard({ recoveryDeps: deps }), {
      lifecycleState: 'thinking',
    })
    .addNode('implementation_file_edit_guard', N.implementationFileEditGuard(), {
      lifecycleState: 'thinking',
    })
    .addNode('capture_implementation', N.captureImplementationSummary(), {
      lifecycleState: 'thinking',
    })
    .addNode('post_edit_analysis', N.postEditAnalysis(deps), {
      lifecycleState: 'observing',
    })
    .addNode('validation_brief', N.validationBrief(), {
      lifecycleState: 'thinking',
    })
    .addNode('validation_context_manager', N.contextManager(deps), {
      lifecycleState: 'thinking',
    })
    .addNode('validation_tool_recommender', N.toolRecommender(deps), {
      lifecycleState: 'thinking',
    })
    .addNode('validator', N.agent({
      ...deps,
      systemPrompt: validatorBaseSystemPrompt,
    }), {
      lifecycleState: 'thinking',
    })
    .addNode('validation_tools', N.toolExecutor(deps), {
      lifecycleState: 'acting',
      resumeStage: 'acting',
      pendingToolExecutionNode: true,
    })
    .addNode('validation_reflection', N.reflection({ advancePlanOnSuccess: false, critiqueDeps: deps }), {
      lifecycleState: 'observing',
      resumeStage: 'observing',
    })
    .addNode('validation_completion_guard', N.validationCompletionGuard(deps), {
      lifecycleState: 'thinking',
    })
    .addNode('validation_guard', N.iterationGuard(), {
      lifecycleState: 'thinking',
    })
    .addNode('capture_validation', N.captureValidationSummary(), {
      lifecycleState: 'thinking',
    })
    .addNode('enforce_validation_cmd', N.enforceValidationCommand(deps), {
      lifecycleState: 'thinking',
    })
    .addNode('enforce_validation_tools', N.toolExecutor(deps), {
      lifecycleState: 'acting',
      resumeStage: 'acting',
      pendingToolExecutionNode: true,
    })
    .addNode('capture_validation_outcome', N.captureValidationOutcome(), {
      lifecycleState: 'observing',
      resumeStage: 'observing',
    })
    .addNode('validation_quality_gate', N.qualityGate({
      phase: 'validation',
      maxBacktracks: N.resolveValidationMaxBacktracks(),
    }), {
      lifecycleState: 'observing',
    })
    .addNode('review_brief', N.reviewBrief(), {
      lifecycleState: 'thinking',
    })
    .addNode('review_subgraph', agentSubgraphNode({
      nodeId: 'review_subgraph',
      graph: reviewerGraph,
      forwardMessages: false,
      mapIn: (parent) => childStateFrom(parent, {
        input: [
          'Review the completed coding work for correctness, regressions, missing validation, and remaining risks.',
          `Original task:\n${parent.input}`,
          parent.seedContract
            ? `Run contract:\n${[
                `Goal: ${parent.seedContract.summary}`,
                'Acceptance criteria:',
                ...parent.seedContract.acceptanceCriteria.map((criterion) =>
                  `- ${criterion.id}: ${criterion.text}`),
              ].join('\n')}`
            : '',
          parent.implementationSummary
            ? `Implementation summary:\n${parent.implementationSummary}`
            : '',
          parent.validationSummary
            ? `Validation summary:\n${parent.validationSummary}`
            : '',
        ].filter(Boolean).join('\n\n'),
        maxIterations: childIterationBudget(parent, {
          min: 4,
          legacyMax: 8,
          contractMin: 16,
          parentShare: 0.35,
          hardCap: 48,
        }),
        taskType: 'code',
      }),
      mapOut: (parent, child) => {
        const reviewSummary = child.output.trim()
        mergeChildInto(parent, child)
        if (reviewSummary) {
          parent.reviewSummary = reviewSummary
        }
        return parent
      },
      mapContext: (_parent, context) => context
        ? { ...context, toolSecurityEffectBoundary: 'observe-only' }
        : undefined,
    }), {
      lifecycleState: 'thinking',
    })
    .addNode('capture_review', N.captureReviewSummary(), {
      lifecycleState: 'thinking',
    })
    .addNode('review_completion_guard', N.reviewCompletionGuard(deps), {
      lifecycleState: 'thinking',
    })
    .addNode('review_quality_gate', N.qualityGate({
      phase: 'review',
      maxBacktracks: 1,
    }), {
      lifecycleState: 'observing',
    })
    .addNode('mark_implementation_phase', N.markPhase('implementation'), {
      lifecycleState: 'thinking',
    })
    .addNode('mark_validation_phase', N.markPhase('validation'), {
      lifecycleState: 'thinking',
    })
    .addNode('mark_review_phase', N.markPhase('review'), {
      lifecycleState: 'thinking',
    })
    .addNode('mark_finalize_phase', N.markPhase('finalize'), {
      lifecycleState: 'thinking',
    })
    .addNode('finalizer', N.codingFinalizer(deps), {
      lifecycleState: 'done',
    })
    .addEdge('memory_retriever', 'execution_intent_router')
    .addConditionalEdge(
      'execution_intent_router',
      (s: AgentState) => {
        if (routesToOperationalSubgraph(s)) return 'operational_subgraph'
        if (routesDirectlyToCurrentDocumentPhase(s)) return 'current_document_inventory'
        return 'codebase_exploration'
      },
      ['operational_subgraph', 'current_document_inventory', 'codebase_exploration'],
    )
    .addEdge('operational_subgraph', '__end__')
    .addConditionalEdge(
      'current_document_inventory',
      (s: AgentState) => s.toolCalls.length > 0
        ? 'current_document_inventory_tools'
        : 'mark_implementation_phase',
      ['current_document_inventory_tools', 'mark_implementation_phase'],
    )
    .addEdge('current_document_inventory_tools', 'mark_implementation_phase')
    .addConditionalEdge(
      'codebase_exploration',
      (s: AgentState) => s.toolCalls.length > 0 ? 'codebase_exploration_tools' : 'large_codebase_scout',
      ['codebase_exploration_tools', 'large_codebase_scout'],
    )
    .addEdge('codebase_exploration_tools', 'capture_codebase_exploration')
    .addEdge('capture_codebase_exploration', 'large_codebase_scout')
    .addConditionalEdge(
      'large_codebase_scout',
      (s: AgentState) => s.toolCalls.length > 0 ? 'large_codebase_scout_tools' : 'coding_planner',
      ['large_codebase_scout_tools', 'coding_planner'],
    )
    .addEdge('large_codebase_scout_tools', 'capture_large_codebase_scout')
    .addEdge('capture_large_codebase_scout', 'coding_planner')
    .addEdge('coding_planner', 'context_manager')
    .addEdge('context_manager', 'tool_recommender')
    .addEdge('tool_recommender', 'mark_implementation_phase')
    .addEdge('mark_implementation_phase', 'open_edit_checkpoint')
    .addEdge('open_edit_checkpoint', 'implement')
    .addConditionalEdge(
      'implement',
      routeImplementationAgentResult,
      ['tools', 'implementation_guard', 'implementation_file_edit_guard', 'capture_implementation'],
    )
    .addEdge('tools', 'implementation_reflection')
    .addEdge('implementation_reflection', 'implementation_completion_guard')
    .addConditionalEdge(
      'implementation_completion_guard',
      (s: AgentState) => {
        if (s.implementationCompleteRequested) return 'capture_implementation'
        // Flailing model asked for a plan mid-run — give it one, then resume.
        if (s.implementationScaffoldingRequested && !s.implementationScaffoldingApplied) {
          return 'implementation_scaffold'
        }
        return 'implementation_guard'
      },
      ['capture_implementation', 'implementation_scaffold', 'implementation_guard'],
    )
    .addEdge('implementation_scaffold', 'implement')
    .addConditionalEdge(
      'implementation_guard',
      routeImplementationGuardResult,
      ['finalizer', 'implement', 'tools', 'implementation_file_edit_guard', 'capture_implementation'],
    )
    .addConditionalEdge(
      'implementation_file_edit_guard',
      (s: AgentState) => s.toolCalls.length > 0
        ? 'tools'
        : s.implementationRetryRequested
          ? 'implement'
          : 'capture_implementation',
      ['tools', 'implement', 'capture_implementation'],
    )
    .addEdge('capture_implementation', 'post_edit_analysis')
    .addConditionalEdge(
      'post_edit_analysis',
      routePostEditAnalysisResult,
      ['mark_finalize_phase', 'mark_validation_phase'],
    )
    .addEdge('mark_validation_phase', 'validation_brief')
    .addEdge('validation_brief', 'validation_context_manager')
    .addEdge('validation_context_manager', 'validation_tool_recommender')
    .addEdge('validation_tool_recommender', 'validator')
    .addConditionalEdge(
      'validator',
      routeValidationAgentResult,
      ['validation_tools', 'capture_validation', 'validation_completion_guard'],
    )
    .addEdge('validation_tools', 'validation_reflection')
    .addEdge('validation_reflection', 'validation_completion_guard')
    .addConditionalEdge(
      'validation_completion_guard',
      (s: AgentState) => /^(?:VERIFIED|UNVERIFIED):/u.test(s.output.trim())
        ? 'capture_validation'
        : 'validation_guard',
      ['capture_validation', 'validation_guard'],
    )
    .addConditionalEdge(
      'validation_guard',
      (s: AgentState) => s.shouldStop ? 'finalizer' : 'validator',
      ['finalizer', 'validator'],
    )
    .addConditionalEdge(
      'capture_validation',
      (s: AgentState) => s.qualityConclusionRecoveryRequested === 'validation'
        ? 'validation_completion_guard'
        : 'enforce_validation_cmd',
      ['validation_completion_guard', 'enforce_validation_cmd'],
    )
    .addConditionalEdge(
      'enforce_validation_cmd',
      (s: AgentState) => s.toolCalls.length > 0
        ? 'enforce_validation_tools'
        : 'validation_quality_gate',
      ['enforce_validation_tools', 'validation_quality_gate'],
    )
    .addEdge('enforce_validation_tools', 'capture_validation_outcome')
    .addEdge('capture_validation_outcome', 'validation_quality_gate')
    .addConditionalEdge(
      'validation_quality_gate',
      (s: AgentState) => {
        if (s.qualityGateDecision === 'incomplete') return 'mark_finalize_phase'
        if (s.qualityGateDecision !== 'retry') return 'mark_review_phase'
        if (s.qualityConclusionRetryTarget === 'validation') return 'validation_brief'
        return (s.backtrackCount ?? 0) >= 2 ? 'plan_revision' : 'context_manager'
      },
      ['context_manager', 'plan_revision', 'validation_brief', 'mark_review_phase', 'mark_finalize_phase'],
    )
    .addEdge('plan_revision', 'context_manager')
    .addEdge('mark_review_phase', 'review_brief')
    .addEdge('review_brief', 'review_subgraph')
    .addEdge('review_subgraph', 'capture_review')
    .addEdge('capture_review', 'review_completion_guard')
    .addEdge('review_completion_guard', 'review_quality_gate')
    .addConditionalEdge(
      'review_quality_gate',
      (s: AgentState) => {
        if (s.qualityGateDecision !== 'retry') return 'mark_finalize_phase'
        return s.qualityConclusionRetryTarget === 'validation'
          ? 'mark_validation_phase'
          : 'context_manager'
      },
      ['context_manager', 'mark_validation_phase', 'mark_finalize_phase'],
    )
    .addEdge('mark_finalize_phase', 'finalizer')
    .addConditionalEdge(
      'finalizer',
      routeCodingFinalizerResult,
      ['implementation_guard', '__end__'],
    )
}
