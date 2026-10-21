import type {
  ChatRequest,
  AgentContext,
  AgentEvent,
  AgentExecutionCapability,
  AgentRunContract,
  AgentStateBoardSnapshot,
  ApprovalDecision,
  AutonomyLevel,
  IAuditLogger,
  IDocumentMemoryStore,
  ILLMProvider,
  ISemanticIndex,
  MemoryContextItem,
  MemoryDocumentChunk,
  MemoryEntry,
  Message,
  TokenUsage,
} from '@sepilotd/core'
import { randomUUID } from 'node:crypto'
import { recallObservedMaxOutputTokens, resolveModelOutputAllowance } from './max-output-token-recovery.js'
import { hasBinaryThinkingControl, resolveControlThinkingLevel, type ThinkingLevelSource } from '../providers/thinking-policy.js'

const CONTROL_PLANNER_REASONING_HEADROOM_CAP_TOKENS = 32_000
import { fitControlRequest } from './context-manager.js'
import { createToolDiscoveryContext } from './tool-discovery-context.js'
import { createAbortError, getAbortError, isAbortError } from '../abort.js'
import type { HookRegistry } from '../hook/registry.js'
import { formatDocumentChunkForPrompt } from '../memory/document-citations.js'
import { filterAppIndexEntries } from '../memory/internal-app-index.js'
import { selectRelevantMemories } from '../memory/relevance.js'
import {
  isMemoryArchived,
  isMemorySuperseded,
  isMemoryVisibleInScope,
} from '../memory/scope.js'
import type { UsageTracker } from '../memory/usage-tracker.js'
import { logAgentDebugTrace } from '../observability/agent-trace.js'
import type { LLMCache } from '../providers/cache.js'
import {
  toProviderApiError,
  type ProviderCircuitBreaker,
} from '../providers/circuit-breaker.js'
import type { PolicyEngine } from '../security/policy-engine.js'
import type { ActiveRunRegistry, SessionRunContextUpdate } from '../server/runtime/active-runs.js'
import { isApprovalParkedError } from '../server/runtime/approvals.js'
import {
  cloneCheckpointExecutionSkillIds,
  cloneCheckpointScopeTags,
  cloneCheckpointSkillExecutionPolicies,
  cloneCheckpointSkillToolNames,
  cloneCheckpointMemoryPolicy,
  cloneCheckpointToolAllowlist,
  type ApprovalRunCheckpoint,
} from '../server/runtime/checkpoints.js'
import type { SessionRunCheckpoint } from '../server/runtime/runs.js'
import type { ToolExecutionRecord } from '../server/runtime/tool-executions.js'
import type { ToolRegistry } from '../tools/registry.js'
import {
  withContextualToolExposure,
  withOptionalToolNameAllowlist,
  withToolNameAllowlist,
  type ToolExposureGroup,
} from '../tools/role-filter.js'
import {
  AuxiliaryLlmBudgetExhaustedError,
  AuxiliaryLlmTimeoutError,
  AuxiliaryStageCircuit,
  createAuxiliaryLlmTurnBudget,
  getDefaultAuxiliaryStageCircuit,
  runAuxiliaryLlmChat,
  type AuxiliaryLlmTurnBudget,
} from './auxiliary-llm.js'
import { AgentEngine, type ApprovalCallback } from './engine.js'
import { buildEnhancedGraph } from './graph/builder.js'
import { cloneGraphState } from './graph/checkpoints.js'
import type { AgentGraph } from './graph/engine.js'
import {
  resolveRunIterationBudget,
  type RunBudgetSurface,
} from './iteration-budget.js'
import type { GraphAgentRegistry, GraphBuilder } from './graph/registry.js'
import type { AgentState, GraphExecutionContext } from './graph/types.js'
import { stopReasonUserAbort, stopReasonWallClock } from './stop-reason.js'
import { resolveRunWallClock } from './progress-signal.js'
import { withEscalationOffer } from './graph/run-escalation.js'
import { resolveInstantModeToolNames } from './instant-mode-tool-intent.js'
import { latestUserText } from './memory-write-completion.js'
import {
  isAutoRouterExcludedGraphMode,
  listSemanticModes,
  resolveAutoFallbackMode,
} from './mode-catalog.js'
import {
  createModeControl,
  INSTANT_MAX_ITERATIONS,
  mergeModeTransferContract,
  type ExecutionHandoff,
  type ModeControl,
  type ModeTransfer,
} from './mode-control.js'
import {
  decodeMultimodalInput,
  primaryInstructionFromContentParts,
  requiresNativeMultimodalInput,
} from './multimodal-input.js'
import { buildRetrievalQuery } from './query-context.js'
import { checkSpendBudget, type SpendBudgetConfig } from './spend-guard.js'
import {
  acceptsRunContractAuthority,
  buildDurableRunContractGroundingAuditRequest,
  buildDurableRunContractPlanningRequest,
  continueRunContractForTurn,
  contractHasDocumentArtifactWork,
  DURABLE_RUN_CONTRACT_TOOL_NAME,
  formatRunContractForPrompt,
  parseDurableRunContractPlan,
  RUN_CONTRACT_AUTHORITY_TOOL_NAME,
} from './task-contract.js'
import {
  cloneMessage,
  cloneToolCall,
  type AutoApprovalEvaluator,
  type PendingToolExecution,
} from './tool-execution.js'
import {
  beginCurrentAgentTurnUserMessage,
  preparePreviousMessagesForTurn,
} from './turn-context.js'
import { isFocusedSingleFileReplacement } from './request-shape.js'
import {
  triggerPostAgentRun,
  triggerPreAgentRun,
  type AgentRunHookData,
} from './turn-hooks.js'

export type AgentMode = 'react' | 'enhanced' | 'auto' | (string & {})

interface AgentRunHookObservation {
  startedAt: number
  status: 'success' | 'error' | 'aborted' | 'incomplete'
  output: string
  streamedOutput: string
  usage?: { inputTokens: number; outputTokens: number }
  error?: string
}

function createAgentRunHookObservation(): AgentRunHookObservation {
  return {
    startedAt: Date.now(),
    status: 'incomplete',
    output: '',
    streamedOutput: '',
  }
}

function observeAgentRunHookEvent(
  observation: AgentRunHookObservation,
  event: AgentEvent,
): void {
  if (event.type === 'text_delta') {
    observation.streamedOutput = `${observation.streamedOutput}${event.text}`.slice(-64_000)
  } else if (event.type === 'message') {
    observation.output = event.content.slice(-64_000)
  } else if (event.type === 'done') {
    const kind = event.stopReason?.kind
    if (kind === 'incomplete' || kind === 'blocked') observation.status = 'incomplete'
    else if (kind === 'cancelled') observation.status = 'aborted'
    else if (kind === 'error') observation.status = 'error'
    else observation.status = 'success'
    observation.usage = { ...event.usage }
  } else if (event.type === 'error') {
    observation.status = 'error'
    observation.error = event.error.message ?? event.error.code
  } else if (event.type === 'state_change' && event.state === 'error') {
    observation.status = 'error'
  }
}

async function* observeAgentRunEvents(
  events: AsyncIterable<AgentEvent>,
  observation: AgentRunHookObservation,
): AsyncIterable<AgentEvent> {
  for await (const event of events) {
    observeAgentRunHookEvent(observation, event)
    yield event
  }
}

function completeAgentRunHookData(
  base: AgentRunHookData,
  observation: AgentRunHookObservation,
): AgentRunHookData {
  const output = observation.output || observation.streamedOutput
  return {
    ...base,
    status: observation.status,
    durationMs: Date.now() - observation.startedAt,
    ...(observation.usage ? { usage: observation.usage } : {}),
    ...(output ? { output } : {}),
    ...(observation.error ? { error: observation.error } : {}),
  }
}

function callerDeadlineOutcome(
  deadlineAt: number | undefined,
  usage: TokenUsage = { inputTokens: 0, outputTokens: 0 },
): AgentEvent[] | null {
  if (deadlineAt === undefined || resolveRunWallClock(Date.now(), deadlineAt, '').budgetMs !== 0) return null
  const stopReason = stopReasonWallClock(0)
  return [{ type: 'message', content: stopReason.summary ?? 'Run deadline reached.' }, { type: 'done', usage, stopReason }]
}

function safeFallbackPending(probe: () => boolean): boolean {
  try {
    return probe() === true
  } catch {
    return false
  }
}

export { isAutoRouterExcludedGraphMode } from './mode-catalog.js'

const MEMORY_SEARCH_LIMIT = 6
const SCOPED_MEMORY_SEARCH_LIMIT = 24
const MEMORY_PROMPT_SNIPPET_LIMIT = 280

type MemoryContextSemanticIndex = ISemanticIndex & Partial<IDocumentMemoryStore> & {
  listRecent?: (limit: number) => Promise<Array<Omit<MemoryEntry, 'score'>>>
  recordAccess?: (ids: string[]) => Promise<void>
}

export interface ModeRouterOptions {
  provider: ILLMProvider
  tools: ToolRegistry
  authorizedTools?: ToolRegistry
  delegationTools?: ToolRegistry
  activeRemoteBrowser?: boolean
  policy: PolicyEngine
  autonomy: AutonomyLevel
  semanticIndex?: MemoryContextSemanticIndex
  systemPrompt?: string
  systemToolGuidance?: import('./system-prompt.js').SystemToolGuidanceOptions
  maxIterations?: number
  previousMessages?: Message[]
  defaultMode?: AgentMode
  auditLogger?: IAuditLogger
  usageTracker?: UsageTracker
  spendBudget?: SpendBudgetConfig
  hookRegistry?: HookRegistry
  deviceName?: string
  thinkingLevel?: string
  thinkingLevelSource?: ThinkingLevelSource
  maxTokens?: number
  temperature?: number
  auxModel?: string
  textDeltaMode?: 'buffered' | 'live'
  ragEnabled?: boolean
  panelPersonas?: import('./personas.js').Persona[]
  panelStrategy?: 'sequential' | 'moderated'
  llmCache?: LLMCache
  providerCircuitBreaker?: ProviderCircuitBreaker
  auxiliaryLlmBudget?: AuxiliaryLlmTurnBudget
  auxiliaryStageCircuit?: AuxiliaryStageCircuit | null
  graphRegistry?: GraphAgentRegistry
  graphNodeModelOverrides?: import('../config/schema.js').SepilotdConfig['agent']['graphNodeModelOverrides']
  approvalCallback?: ApprovalCallback
  evaluateAutoApproval?: AutoApprovalEvaluator
  requestQuestion?: (input: {
    sessionId: string
    prompt: string
    choices?: string[]
  }) => Promise<string>
  saveApprovalCheckpoint?: (checkpoint: ApprovalRunCheckpoint) => Promise<void>
  clearApprovalCheckpoint?: (requestId: string) => Promise<void>
  saveRunCheckpoint?: (checkpoint: SessionRunCheckpoint) => Promise<void>
  clearRunCheckpoint?: (sessionId: string) => Promise<void>
  journalStateBoard?: (sessionId: string, board: AgentStateBoardSnapshot) => Promise<void>
  journalSteeringConsumed?: (sessionId: string, noteId: string) => Promise<void>
  activeRuns?: ActiveRunRegistry
  loadToolExecution?: (sessionId: string) => Promise<ToolExecutionRecord | null>
  saveToolExecution?: (record: ToolExecutionRecord) => Promise<void>
  clearToolExecution?: (sessionId: string) => Promise<void>
  editSnapshotStore?: import('./edit-rollback/store.js').EditSnapshotStore
  toolStatsStore?: import('./tool-learning/store.js').ToolStatsStore
  workspaceMutationTracker?: import('./workspace-mutation/tracker.js').WorkspaceMutationTracker
  pluginEvents?: import('../plugins/event-bus.js').PluginEventBus
  strictFinalAnswerProtocol?: boolean
  reviewToollessFinals?: boolean
  durableRunContracts?: boolean
  stagePolicy?: import('./bots/types.js').RoleBotStagePolicy
  memoryRetrieval?: boolean
  maxContinuationCycles?: number
  hardMaxIterations?: boolean
  budgetSurface?: RunBudgetSurface
  intentModeHint?: {
    mode: string
    confidence: 'high' | 'medium' | 'low'
    fallback: boolean
    executionIntent?: import('@sepilotd/core').AgentExecutionIntent
    contractRelation?: 'continue' | 'new' | 'review'
  }
  persistTransportPreference?: (
    sessionId: string,
    preferPromptReact: boolean | undefined,
  ) => void | Promise<void>
}

function raceWithAbort<T>(operation: Promise<T>, signal: AbortSignal): Promise<T> {
  if (signal.aborted) {
    return Promise.reject(getAbortError(signal))
  }

  return new Promise<T>((resolve, reject) => {
    const onAbort = () => {
      cleanup()
      reject(getAbortError(signal))
    }
    const cleanup = () => {
      signal.removeEventListener('abort', onAbort)
    }

    signal.addEventListener('abort', onAbort, { once: true })
    operation.then(
      (value) => {
        cleanup()
        resolve(value)
      },
      (error: unknown) => {
        cleanup()
        reject(error)
      },
    )
  })
}

function requestAuthorityForRecord(
  context: AgentContext,
): Pick<SessionRunContextUpdate, 'toolNames' | 'bot' | 'surface'> {
  const none = context.requestAuthorityComplete === true ? null : undefined
  return {
    toolNames: context.requestedToolNames ?? none,
    bot: context.roleBotId ?? none,
    surface: context.surface ?? none,
  }
}

export class AgentModeRouter {
  private options: ModeRouterOptions
  private transferredGraphState?: AgentState
  private modeControl?: ModeControl
  private graphCache = new Map<string, AgentGraph>()
  private activeReactEngine: AgentEngine | null = null
  private activeGraphState: AgentState | null = null
  private activeGraphContext: GraphExecutionContext | null = null
  private stopRequested = false
  private readonly graphDeps = new Map<string, { autonomy: AutonomyLevel }>()
  private activeGraphAbortController: AbortController | null = null
  private activeRunAbortController: AbortController | null = null
  private lastRunSessionId?: string

  constructor(options: ModeRouterOptions) {
    this.options = {
      ...options,
      auxiliaryLlmBudget: options.auxiliaryLlmBudget ?? createAuxiliaryLlmTurnBudget(),
    }
  }

  setAutonomy(level: AutonomyLevel): void {
    this.options = { ...this.options, autonomy: level }
    this.activeReactEngine?.setAutonomy(level)
    if (this.activeGraphContext) this.activeGraphContext.autonomy = level
    for (const deps of this.graphDeps.values()) deps.autonomy = level
    if (this.lastRunSessionId) this.options.activeRuns?.updateRecordedAutonomy(this.lastRunSessionId, level)
  }

  private createReactEngine(
    textDeltaMode = this.options.textDeltaMode,
    overrides: {
      strictFinalAnswerProtocol?: boolean
      reviewOutcomes?: boolean
      reviewToollessFinals?: boolean
      tools?: ToolRegistry
      maxIterations?: number
      hardMaxIterations?: boolean
    } = {},
  ): AgentEngine {
    return new AgentEngine({
      provider: this.options.provider,
      tools: overrides.tools ?? this.options.tools,
      policy: this.options.policy,
      autonomy: this.options.autonomy,
      maxIterations: overrides.maxIterations ?? this.options.maxIterations,
      hardMaxIterations: overrides.hardMaxIterations ?? this.options.hardMaxIterations,
      modeControl: this.modeControl,
      semanticRouting: true,
      systemToolGuidance: this.options.systemToolGuidance,
      auditLogger: this.options.auditLogger,
      usageTracker: this.options.usageTracker,
      spendBudget: this.options.spendBudget,
      hookRegistry: this.options.hookRegistry,
      deviceName: this.options.deviceName,
      thinkingLevel: this.options.thinkingLevel,
      thinkingLevelSource: this.options.thinkingLevelSource,
      maxTokens: this.options.maxTokens,
      temperature: this.options.temperature,
      textDeltaMode,
      llmCache: this.options.llmCache,
      providerCircuitBreaker: this.options.providerCircuitBreaker,
      auxiliaryLlmBudget: this.options.auxiliaryLlmBudget,
      approvalCallback: this.options.approvalCallback,
      evaluateAutoApproval: this.options.evaluateAutoApproval,
      saveApprovalCheckpoint: this.options.saveApprovalCheckpoint,
      clearApprovalCheckpoint: this.options.clearApprovalCheckpoint,
      saveRunCheckpoint: this.options.saveRunCheckpoint,
      clearRunCheckpoint: this.options.clearRunCheckpoint,
      loadToolExecution: this.options.loadToolExecution,
      saveToolExecution: this.options.saveToolExecution,
      clearToolExecution: this.options.clearToolExecution,
      toolStats: this.options.toolStatsStore,
      workspaceMutationTracker: this.options.workspaceMutationTracker,
      editSnapshotStore: this.options.editSnapshotStore,
      journalStateBoard: this.options.journalStateBoard,
      activeRuns: this.options.activeRuns,
      journalSteeringConsumed: this.options.journalSteeringConsumed,
      strictFinalAnswerProtocol:
        overrides.strictFinalAnswerProtocol ?? this.options.strictFinalAnswerProtocol,
      reviewOutcomes: overrides.reviewOutcomes,
      reviewToollessFinals: overrides.reviewToollessFinals ?? this.options.reviewToollessFinals,
      maxContinuationCycles: this.maxContinuationCycles(),
      emitAgentRunHooks: false,
    })
  }

  private resolveAgentContext(
    context: AgentContext,
    input?: string,
  ): AgentContext {
    const previousMessages = context.previousMessages ?? this.options.previousMessages
    return {
      ...context,
      systemPrompt: this.options.systemPrompt ?? context.systemPrompt,
      previousMessages: input
        ? preparePreviousMessagesForTurn(input, previousMessages)
        : previousMessages,
    }
  }

  private beginTurnAgentContext(context: AgentContext, input: string): AgentContext {
    const restoredFromPreviousTurn = context.runContractScope === 'previous-turn'
    const contractRelation = this.confidentContractRelationHint()
    let continuingContract = !restoredFromPreviousTurn
      ? context.runContract
      : context.completionRequest === undefined && contractRelation === 'continue'
        ? continueRunContractForTurn(context.runContract, input)
        : undefined
    const executionIntentHint = context.completionRequest === undefined
      ? this.confidentExecutionIntentHint()
      : undefined
    if (contractRelation === 'continue' && continuingContract && executionIntentHint) {
      continuingContract = {
        ...continuingContract,
        executionIntent: executionIntentHint,
      }
    }
    const resolved = this.resolveAgentContext(context, input)
    if (restoredFromPreviousTurn && context.runContract && !continuingContract && contractRelation !== 'new' && context.completionRequest === undefined) {
      resolved.previousMessages = [
        { role: 'system', content: '[Previous task context: reference only; determine whether the current request continues this goal.]\n' + formatRunContractForPrompt(context.runContract) },
        ...(resolved.previousMessages ?? []),
      ]
    }
    return {
      ...resolved,
      runContract: continuingContract,
      runContractScope: 'current-turn',
    }
  }

  private durableRunContractsEnabled(): boolean {
    return this.options.durableRunContracts
      ?? process.env.SEPILOTD_DURABLE_RUN_CONTRACTS !== '0'
  }

  private durableRunContractPlannerEnabled(): boolean {
    return this.options.stagePolicy?.contractPlanner !== false
      && process.env.SEPILOTD_RUN_CONTRACT_PLANNER !== '0'
  }

  private shouldPlanDurableRunContract(mode: AgentMode): boolean {
    return mode !== 'instant' && mode !== 'react'
      && this.options.graphRegistry?.get(mode)?.capabilities?.durableRunContract !== false
  }

  private shouldAttachDurableRunContract(
    mode: AgentMode,
    contract: AgentRunContract,
  ): boolean {
    if (mode === 'instant') return false
    if (mode === 'react') {
      return Boolean(
        contract.executionIntent
        && contract.executionIntent.kind !== 'conversation',
      )
    }
    return this.options.graphRegistry?.get(mode)?.capabilities?.durableRunContract !== false
  }

  private async withDurableRunContract(
    input: string,
    context: AgentContext,
    mode: AgentMode,
    signal?: AbortSignal,
  ): Promise<AgentContext> {
    if (mode === 'instant' || !this.durableRunContractsEnabled() || context.runContract) {
      return context
    }
    const completionInput = context.completionRequest ?? input
    const executionIntentHint = context.completionRequest === undefined
      ? this.confidentExecutionIntentHint()
      : undefined
    const fallback: AgentRunContract = {
      source: 'fallback', summary: completionInput,
      acceptanceCriteria: [{ id: 'outcome', text: context.completionRequest ?? 'Complete the current request and substantiate any actions with tool evidence.' }],
      constraints: [], outOfScope: [],
      ...(executionIntentHint ? { executionIntent: executionIntentHint } : {}),
    }
    if (!this.shouldAttachDurableRunContract(mode, fallback) && context.completionRequest === undefined) {
      return context
    }
    if (
      mode !== 'auto'
      && this.contractHasDurableArtifactWork(fallback)
      && this.artifactWriteSupport(mode) !== 'supported'
    ) {
      return context
    }
    if (isFocusedSingleFileReplacement(completionInput)) {
      return {
        ...context,
        runContract: fallback,
      }
    }
    if (
      !this.durableRunContractPlannerEnabled()
      || !this.shouldPlanDurableRunContract(mode)
    ) {
      return {
        ...context,
        runContract: fallback,
      }
    }

    const stageCircuit = this.options.auxiliaryStageCircuit === null
      ? null
      : this.options.auxiliaryStageCircuit ?? getDefaultAuxiliaryStageCircuit()
    const circuitKey = AuxiliaryStageCircuit.key(
      'durable-run-contract-planner',
      this.options.provider.id,
      context.model,
    )
    if (stageCircuit?.isOpen(circuitKey)) {
      await logAgentDebugTrace({
        event: 'aux.stage.skipped',
        source: 'mode-router',
        sessionId: context.sessionId,
        runId: context.sessionId,
        status: 'skipped',
        data: { stage: 'durable-run-contract-planner', provider: this.options.provider.id, model: context.model },
      })
      return {
        ...context,
        runContract: fallback,
      }
    }

    try {
      const nativeToolUse = this.options.provider.models.find(
        (candidate) => candidate.id === context.model,
      )?.capabilities.toolUse !== false
      const modelInfo = this.options.provider.models.find(model => model.id === context.model)
      const buildPlanningRequest = (native: boolean): ChatRequest => {
        const planning = buildDurableRunContractPlanningRequest({
          model: context.model,
          userRequest: completionInput,
          cwd: context.cwd,
          fallback,
          nativeToolUse: native,
          availableToolNames: this.options.tools.list().map((tool) => tool.name),
        })
        const plannerDecisionTokens = planning.maxTokens!
        const plannerAllowance = modelInfo?.capabilities.thinking
          ? Math.min(Math.max(plannerDecisionTokens * 12, 8_000), CONTROL_PLANNER_REASONING_HEADROOM_CAP_TOKENS)
          : plannerDecisionTokens
        planning.maxTokens = Math.min(plannerAllowance, resolveModelOutputAllowance({
          requested: this.options.maxTokens,
          modelMaximum: modelInfo?.maxOutputTokens,
          observedMaximum: recallObservedMaxOutputTokens(this.options.provider.id, context.model),
          fallback: plannerAllowance,
        }))
        planning.thinkingLevel = resolveControlThinkingLevel(this.options.thinkingLevel ?? 'auto', {
          binaryThinking: hasBinaryThinkingControl(modelInfo),
          source: this.options.thinkingLevelSource,
        })
        Object.assign(planning, fitControlRequest(planning, modelInfo))
        return planning
      }
      const runPlanner = (planning: ChatRequest, label: string) => runAuxiliaryLlmChat({
        provider: this.options.provider,
        request: planning,
        label,
        breaker: this.options.providerCircuitBreaker,
        signal,
        sessionId: context.sessionId,
        budget: createAuxiliaryLlmTurnBudget(),
      })
      let plannerNativeToolUse = nativeToolUse
      let request = buildPlanningRequest(plannerNativeToolUse)
      let response: Awaited<ReturnType<typeof runAuxiliaryLlmChat>>
      try {
        response = await runPlanner(request, 'Durable run contract planner')
      } catch (error) {
        if (
          !plannerNativeToolUse
          || !(error instanceof AuxiliaryLlmTimeoutError || error instanceof AuxiliaryLlmBudgetExhaustedError)
        ) throw error
        await logAgentDebugTrace({
          event: 'aux.stage.retried',
          source: 'mode-router',
          sessionId: context.sessionId,
          runId: context.sessionId,
          status: 'retry',
          data: {
            stage: 'durable-run-contract-planner',
            provider: this.options.provider.id,
            model: context.model,
            reason: error instanceof Error ? error.message : String(error),
            transport: 'prompt-json',
          },
        })
        plannerNativeToolUse = false
        request = buildPlanningRequest(plannerNativeToolUse)
        response = await runPlanner(request, 'Durable run contract planner (prompt retry)')
      }
      stageCircuit?.recordSuccess(circuitKey)
      const structuredContract = response.message.toolCalls?.find(
        (toolCall) => toolCall.name === DURABLE_RUN_CONTRACT_TOOL_NAME,
      )?.arguments
      const candidate = parseDurableRunContractPlan(
        structuredContract
          ? JSON.stringify(structuredContract)
          : this.extractAutoModeResponse(response.message.content),
        fallback,
        {
          userRequest: completionInput,
          availableToolNames: this.options.tools.list().map((tool) => tool.name),
        },
      )
      if (candidate === fallback) {
        return {
          ...context,
          runContract: fallback,
        }
      }
      const auditRequest = buildDurableRunContractGroundingAuditRequest({
        model: context.model,
        userRequest: completionInput,
        cwd: context.cwd,
        candidate,
        fallback,
        nativeToolUse: plannerNativeToolUse,
      })
      auditRequest.maxTokens = request.maxTokens
      auditRequest.thinkingLevel = request.thinkingLevel
      Object.assign(auditRequest, fitControlRequest(auditRequest, this.options.provider.models.find(model => model.id === context.model)))
      const auditResponse = await runAuxiliaryLlmChat({
        provider: this.options.provider,
        request: auditRequest,
        label: 'Durable run contract authority audit',
        breaker: this.options.providerCircuitBreaker,
        signal,
        budget: createAuxiliaryLlmTurnBudget(),
        sessionId: context.sessionId,
      })
      const auditCalls = auditResponse.message.toolCalls ?? []
      const validAuditSurface = plannerNativeToolUse
        ? auditCalls.length === 1 && auditCalls[0]?.name === RUN_CONTRACT_AUTHORITY_TOOL_NAME
        : auditCalls.length === 0
      const auditedStructuredContract = validAuditSurface ? auditCalls[0]?.arguments : undefined
      const accepted = validAuditSurface && auditResponse.finishReason !== 'length' && acceptsRunContractAuthority(
        auditedStructuredContract
          ? JSON.stringify(auditedStructuredContract)
          : this.extractAutoModeResponse(auditResponse.message.content),
        completionInput,
        candidate,
      )
      return {
        ...context,
        runContract: accepted ? candidate : fallback,
      }
    } catch (error) {
      if (isAbortError(error)) {
        throw error
      }
      if (
        error instanceof AuxiliaryLlmTimeoutError
        || error instanceof AuxiliaryLlmBudgetExhaustedError
      ) {
        stageCircuit?.recordTimeout(circuitKey)
      }
      return {
        ...context,
        runContract: fallback,
      }
    }
  }

  private maxContinuationCycles(): number {
    return resolveRunIterationBudget({
      surface: this.options.budgetSurface ?? 'chat',
      requested: this.options.maxIterations,
      hardMaxIterations: this.options.hardMaxIterations,
      maxContinuationCycles: this.options.maxContinuationCycles,
    }).maxContinuationCycles
  }

  private memoryContextTitle(source: MemoryEntry['source']): string {
    switch (source) {
      case 'conversation':
        return 'Conversation memory'
      case 'user':
        return 'User memory'
      case 'skill':
        return 'Skill memory'
      case 'document':
        return 'Document memory'
    }
  }

  private normalizeContextSnippet(content: string, limit = 220): string {
    const normalized = content.trim().replace(/\s+/g, ' ')
    if (!normalized) {
      return ''
    }
    return normalized.length > limit
      ? `${normalized.slice(0, limit - 1)}…`
      : normalized
  }

  private createMemoryContextItem(result: MemoryEntry): MemoryContextItem {
    const title = this.memoryContextTitle(result.source)
    return {
      id: result.id,
      kind: 'memory',
      source: result.source,
      title,
      snippet: this.normalizeContextSnippet(result.content),
      citationLabel: title,
      score: result.score,
    }
  }

  private createDocumentContextItem(chunk: MemoryDocumentChunk): MemoryContextItem {
    const title = chunk.documentTitle || chunk.chunkTitle || 'Document'
    return {
      id: chunk.id,
      kind: 'document',
      source: chunk.source,
      title,
      snippet: this.normalizeContextSnippet(chunk.snippet ?? chunk.content),
      citationLabel: chunk.citationLabel ?? title,
      score: chunk.score,
      documentId: chunk.documentId,
      documentTitle: chunk.documentTitle,
      documentPath: chunk.documentPath,
    }
  }

  private memorySearchLimitForContext(context: AgentContext): number {
    return context.scopeTags?.length ? SCOPED_MEMORY_SEARCH_LIMIT : MEMORY_SEARCH_LIMIT
  }

  private filterMemoryEntriesForContext(
    entries: MemoryEntry[],
    context: AgentContext,
  ): MemoryEntry[] {
    return entries.filter((entry) => {
      if (isMemorySuperseded(entry.tags) || isMemoryArchived(entry.tags)) {
        return false
      }
      if (!context.scopeTags?.length) {
        return true
      }
      return isMemoryVisibleInScope(entry.tags, context.scopeTags)
    })
  }

  private selectRelevantMemoryEntries(
    query: string,
    entries: MemoryEntry[],
    limit: number,
  ): MemoryEntry[] {
    return selectRelevantMemories(query, entries, {
      limit,
      threshold: 0.1,
    })
  }

  private async resolveKeywordMemoryFallback(
    query: string,
    context: AgentContext,
    limit: number,
  ): Promise<MemoryEntry[]> {
    if (!this.options.semanticIndex) {
      return []
    }
    const raw = await this.options.semanticIndex.search(query, {
      type: 'keyword',
      limit: this.memorySearchLimitForContext(context),
      sources: ['conversation', 'user'],
    })
    const filtered = this.filterMemoryEntriesForContext(filterAppIndexEntries(raw, false), context)
    return this.selectRelevantMemoryEntries(query, filtered, limit)
  }

  private async resolveRelevantMemoryEntries(
    query: string,
    context: AgentContext,
    limit: number,
  ): Promise<MemoryEntry[]> {
    if (!this.options.semanticIndex) {
      return []
    }

    const raw = await this.options.semanticIndex.search(query, {
      type: 'hybrid',
      limit: this.memorySearchLimitForContext(context),
      minScore: 0.08,
    })
    const filtered = this.filterMemoryEntriesForContext(filterAppIndexEntries(raw, false), context)
    let selected = this.selectRelevantMemoryEntries(query, filtered, limit)

    if (selected.length === 0) {
      selected = await this.resolveKeywordMemoryFallback(query, context, limit)
    }

    this.recordMemoryAccess(selected)
    return selected
  }

  private recordMemoryAccess(entries: MemoryEntry[]): void {
    if (entries.length === 0 || !this.options.semanticIndex?.recordAccess) {
      return
    }
    void this.options.semanticIndex.recordAccess(entries.map((entry) => entry.id)).catch(() => {})
  }

  private formatMemoryForPrompt(result: MemoryEntry): string {
    const content = result.content.trim().replace(/\s+/g, ' ')
    return content.length > MEMORY_PROMPT_SNIPPET_LIMIT
      ? `${content.slice(0, MEMORY_PROMPT_SNIPPET_LIMIT - 3)}...`
      : content
  }

  private async resolveRelevantContext(
    input: string,
    context: AgentContext,
  ): Promise<{ relevantMemories: string[]; relevantContextItems: MemoryContextItem[] }> {
    if (context.memoryPolicy?.read === false || context.memoryPolicy?.automaticRead === false) return { relevantMemories: [], relevantContextItems: [] }
    const query = buildRetrievalQuery(
      context.memoryQuery ?? input,
      context.previousMessages ?? [],
    )
    if (!this.options.semanticIndex || query.length < 3) {
      return { relevantMemories: [], relevantContextItems: [] }
    }

    try {
      const fallbackEntries = await this.resolveRelevantMemoryEntries(query, context, 3)
      const fallback = fallbackEntries
        .map((result) => this.formatMemoryForPrompt(result))
        .filter((content) => Boolean(content))
      const fallbackItems = fallbackEntries.map((result) => this.createMemoryContextItem(result))

      if (
        this.options.ragEnabled === false
        || typeof this.options.semanticIndex.searchDocuments !== 'function'
      ) {
        return {
          relevantMemories: fallback,
          relevantContextItems: fallbackItems,
        }
      }

      let documentResults: MemoryDocumentChunk[] = []
      try {
        documentResults = filterAppIndexEntries(
          await this.options.semanticIndex.searchDocuments(query, {
            type: 'hybrid',
            limit: context.scopeTags?.length ? SCOPED_MEMORY_SEARCH_LIMIT : 3,
            minScore: 0.08,
          }),
          false,
        ).filter((chunk) => (
          !context.scopeTags?.length
          || isMemoryVisibleInScope(chunk.tags, context.scopeTags)
        ))
      } catch {
        return {
          relevantMemories: fallback,
          relevantContextItems: fallbackItems,
        }
      }
      const general = fallbackEntries
        .filter((result) => result.source !== 'document')
        .slice(0, 2)
      const generalMemories = general
        .map((result) => this.formatMemoryForPrompt(result))
        .filter((content) => Boolean(content))
      const generalItems = general.map((result) => this.createMemoryContextItem(result))

      const documents = documentResults
        .map((chunk) => formatDocumentChunkForPrompt(chunk))
        .filter(Boolean)
        .slice(0, 2)
      const documentItems = documentResults
        .map((chunk) => this.createDocumentContextItem(chunk))
        .filter((item) => Boolean(item.snippet))
        .slice(0, 2)

      const contextualMemories = [...generalMemories, ...documents]
      const contextualItems = [...generalItems, ...documentItems]
      if (contextualMemories.length > 0) {
        return {
          relevantMemories: contextualMemories.slice(0, 4),
          relevantContextItems: contextualItems.slice(0, 4),
        }
      }
      return {
        relevantMemories: fallback,
        relevantContextItems: fallbackItems,
      }
    } catch {
      return { relevantMemories: [], relevantContextItems: [] }
    }
  }

  private async hydrateAgentContext(input: string, context: AgentContext): Promise<AgentContext> {
    return this.hydrateResolvedAgentContext(input, this.resolveAgentContext(context, input))
  }

  private async hydrateResolvedAgentContext(input: string, resolved: AgentContext): Promise<AgentContext> {
    if (resolved.memoryPolicy?.read === false || resolved.memoryPolicy?.automaticRead === false) return { ...resolved, relevantMemories: [], relevantContextItems: [] }
    if (resolved.relevantMemories?.length || resolved.relevantContextItems?.length) {
      return resolved
    }
    if (this.options.memoryRetrieval === false) return resolved

    const { relevantMemories, relevantContextItems } = await this.resolveRelevantContext(
      input,
      resolved,
    )
    if (relevantMemories.length === 0 && relevantContextItems.length === 0) {
      return resolved
    }

    return {
      ...resolved,
      relevantMemories,
      relevantContextItems,
    }
  }

  private resolveGraphBuilder(mode: string): GraphBuilder | null {
    const registered = this.options.graphRegistry?.get(mode)?.builder
    if (registered) {
      return registered
    }
    if (mode === 'enhanced') {
      return buildEnhancedGraph
    }
    return null
  }

  private resolveGraphMaxIterations(mode: string): number | undefined {
    return this.options.graphRegistry?.get(mode)?.limits?.maxIterations
  }

  private getGraph(mode: string, builder: GraphBuilder): AgentGraph {
    let cached = this.graphCache.get(mode)
    if (!cached) {
      const deps = {
        provider: this.options.provider,
        tools: this.options.tools,
        policy: this.options.policy,
        autonomy: this.options.autonomy,
        semanticIndex: this.options.semanticIndex,
        systemPrompt: this.options.systemPrompt,
        systemToolGuidance: this.options.systemToolGuidance,
        usageTracker: this.options.usageTracker,
        providerCircuitBreaker: this.options.providerCircuitBreaker,
      }
      this.graphDeps.set(mode, deps)
      cached = builder(deps)
      this.graphCache.set(mode, cached)
    }
    return cached
  }

  private getRequiredGraph(mode: string): AgentGraph {
    const builder = this.resolveGraphBuilder(mode)
    if (!builder) {
      throw new Error(`Graph mode '${mode}' is not available`)
    }
    return this.getGraph(mode, builder)
  }

  private extractAutoModeResponse(content: Message['content']): string {
    if (typeof content === 'string') {
      return content
    }
    return content
      .filter((part): part is { type: 'text'; text: string } => part.type === 'text')
      .map((part) => part.text)
      .join('\n')
  }

  private isDirectCapabilityMode(mode: AgentMode): boolean {
    const graph = this.options.graphRegistry?.get(mode)
    return Boolean(graph?.source === 'capability' && graph.capabilities?.direct)
  }

  private confidentIntentModeHint(): string | null {
    const hint = this.options.intentModeHint
    if (!hint || hint.fallback || hint.confidence !== 'high') {
      return null
    }
    if (hint.mode === 'instant' || hint.mode === 'react') return hint.mode
    const graph = this.options.graphRegistry?.get(hint.mode)
    return graph && !isAutoRouterExcludedGraphMode(graph.id) && !graph.capabilities?.direct ? hint.mode : null
  }

  private confidentExecutionIntentHint(): import('@sepilotd/core').AgentExecutionIntent | undefined {
    const hint = this.options.intentModeHint
    if (!hint || hint.fallback || hint.confidence !== 'high' || !hint.executionIntent) {
      return undefined
    }
    return {
      ...hint.executionIntent,
      capabilities: [...hint.executionIntent.capabilities],
      capabilityPolicy: 'closed',
      ...(hint.executionIntent.allowedTools
        ? { allowedTools: [...hint.executionIntent.allowedTools] }
        : {}),
      ...(hint.executionIntent.toolSequence
        ? { toolSequence: [...hint.executionIntent.toolSequence] }
        : {}),
      ...(hint.executionIntent.authorizedWriteTargets
        ? { authorizedWriteTargets: [...hint.executionIntent.authorizedWriteTargets] }
        : {}),
      ...(hint.executionIntent.protectedWriteTargets
        ? { protectedWriteTargets: [...hint.executionIntent.protectedWriteTargets] }
        : {}),
    }
  }

  private confidentContractRelationHint(): 'continue' | 'new' | 'review' | undefined {
    const hint = this.options.intentModeHint
    if (!hint || hint.fallback || hint.confidence !== 'high') {
      return undefined
    }
    return hint.contractRelation
  }

  private contractHasDurableArtifactWork(contract: AgentContext['runContract']): boolean {
    return contractHasDocumentArtifactWork(contract)
  }

  private artifactWriteSupport(mode: AgentMode): 'supported' | 'unsupported' | 'unknown' {
    if (mode === 'react') {
      return 'supported'
    }
    const graph = this.options.graphRegistry?.get(mode)
    if (!graph) {
      return 'unknown'
    }
    if (graph.capabilities?.artifactWrite) {
      return 'supported'
    }
    if (graph.capabilities?.readOnly) {
      return 'unsupported'
    }
    return 'unknown'
  }

  private resolveStagePolicy(mode: string): import('./bots/types.js').RoleBotStagePolicy | undefined {
    if (this.options.graphRegistry?.get(mode)?.capabilities?.auxiliaryStages === 'none') {
      return { contractPlanner: false, outcomeReview: false, completionGate: false, recoveryController: false }
    }
    return this.options.stagePolicy
  }

  private buildGraphContext(
    input: string | undefined,
    context: AgentContext,
    mode: string,
    pendingToolExecution?: PendingToolExecution,
    signal?: AbortSignal,
    textDeltaMode = this.options.textDeltaMode,
  ): GraphExecutionContext {
    return {
      graphId: mode,
      stagePolicy: this.resolveStagePolicy(mode),
      executionControl: this.options.graphRegistry?.get(mode)?.capabilities?.executionControl,
      modeControl: this.modeControl,
      agentContext: this.resolveAgentContext(context, input),
      toolsForbiddenByUser: this.options.tools.list().length === 0 && !this.modeControl?.tools.length,
      requireToolApproval: context.requireToolApproval,
      signal,
      provider: this.options.provider,
      tools: this.options.tools,
      policy: this.options.policy,
      autonomy: this.options.autonomy,
      semanticIndex: this.options.semanticIndex,
      systemPrompt: this.options.systemPrompt ?? context.systemPrompt,
      usageTracker: this.options.usageTracker,
      systemToolGuidance: this.options.systemToolGuidance,
      auditLogger: this.options.auditLogger,
      hookRegistry: this.options.hookRegistry,
      deviceName: this.options.deviceName,
      thinkingLevel: this.options.thinkingLevel,
      thinkingLevelSource: this.options.thinkingLevelSource,
      maxTokens: this.options.maxTokens,
      temperature: this.options.temperature,
      auxModel: this.options.auxModel,
      textDeltaMode,
      panelPersonas: this.options.panelPersonas,
      panelStrategy: this.options.panelStrategy,
      llmCache: this.options.llmCache,
      providerCircuitBreaker: this.options.providerCircuitBreaker,
      auxiliaryLlmBudget: this.options.auxiliaryLlmBudget,
      graphNodeModelOverrides: this.options.graphNodeModelOverrides,
      approvalCallback: this.options.approvalCallback,
      evaluateAutoApproval: this.options.evaluateAutoApproval,
      saveApprovalCheckpoint: this.options.saveApprovalCheckpoint,
      clearApprovalCheckpoint: this.options.clearApprovalCheckpoint,
      saveRunCheckpoint: async (checkpoint) => {
        await this.options.saveRunCheckpoint?.(checkpoint)
        if (!checkpoint.pendingToolExecution?.currentExecutionId) {
          await this.options.clearToolExecution?.(checkpoint.sessionId)
        }
      },
      clearRunCheckpoint: this.options.clearRunCheckpoint,
      journalStateBoard: this.options.journalStateBoard
        ? (board) => this.options.journalStateBoard!(context.sessionId, board)
        : undefined,
      journalSteeringConsumed: this.options.journalSteeringConsumed
        ? (noteId) => this.options.journalSteeringConsumed!(context.sessionId, noteId)
        : undefined,
      activeRuns: this.options.activeRuns,
      loadToolExecution: this.options.loadToolExecution,
      saveToolExecution: this.options.saveToolExecution,
      clearToolExecution: this.options.clearToolExecution,
      pendingToolExecution,
      editSnapshotStore: this.options.editSnapshotStore,
      toolStatsStore: this.options.toolStatsStore,
      workspaceMutationTracker: this.options.workspaceMutationTracker,
      pluginEvents: this.options.pluginEvents,
      requestQuestion: this.options.requestQuestion,
      strictFinalAnswerProtocol:
        this.options.strictFinalAnswerProtocol
        ?? (process.env.SEPILOTD_STRICT_ANSWER_PROTOCOL === '1'),
      maxContinuationCycles: this.options.graphRegistry?.get(mode)?.capabilities?.executionControl === 'model-directed'
        ? 0 : this.maxContinuationCycles(),
    }
  }

  private buildInitialGraphState(
    input: string,
    context: AgentContext,
    mode: string,
  ): AgentState {
    const modelDirected = this.options.graphRegistry?.get(mode)?.capabilities?.executionControl === 'model-directed'
    return {
      input,
      currentUserContent: context.currentUserContent,
      messages: context.executionHandoff?.messages
        .filter((message) => !message.metadata?.runtimeContext)
        .map(cloneMessage) ?? beginCurrentAgentTurnUserMessage(
        context.previousMessages?.map(cloneMessage) ?? [],
        context.currentUserContent ?? input,
        context.currentUserContent ? input : undefined,
      ),
      currentStep: '',
      planIndex: 0,
      toolCalls: [],
      toolResults: [],
      recentToolResults: [],
      memories: [...(context.relevantMemories ?? [])],
      output: '',
      totalUsage: { ...(context.executionHandoff?.usage ?? { inputTokens: 0, outputTokens: 0 }) },
      iteration: context.executionHandoff?.iterations ?? 0,
      maxIterations: resolveRunIterationBudget({
        surface: this.options.budgetSurface ?? 'chat',
        requested: this.options.maxIterations ?? (modelDirected ? this.resolveGraphMaxIterations(mode) : undefined),
        graphPreset: this.resolveGraphMaxIterations(mode),
        contract: context.runContract,
        hardMaxIterations: this.options.hardMaxIterations || modelDirected,
      }).maxIterations,
      shouldStop: false,
      seedContract: context.runContract,
      taskType: 'simple',
      ...(this.transferredGraphState ? {
        evidenceLedger: this.transferredGraphState.evidenceLedger,
        failedAttempts: this.transferredGraphState.failedAttempts,
        openQuestions: this.transferredGraphState.openQuestions,
        toolCallHistory: this.transferredGraphState.toolCallHistory,
        recentToolResults: this.transferredGraphState.recentToolResults,
      } : {}),
      preferPromptReact: context.preferPromptReact,
    }
  }

  private resolvePendingToolGraphNode(graph: AgentGraph): string {
    return graph.findNodeByMeta((meta) => meta.pendingToolExecutionNode)
      ?? (graph.getNodes().has('tools') ? 'tools' : '')
  }

  private normalizeLegacyGraphResumeState(
    graph: AgentGraph,
    state: AgentState,
  ): AgentState {
    if (!state.currentStep || graph.getNodes().has(state.currentStep)) {
      return state
    }

    const legacyGenericNodes = new Set([
      'memory_retriever',
      'planner',
      'iteration_guard',
      'context_manager',
      'agent',
      'tools',
      'reflection',
    ])
    if (!legacyGenericNodes.has(state.currentStep)) {
      return state
    }

    const targetSubgraph =
      state.taskType === 'simple' && graph.getNodes().has('simple_subgraph')
        ? 'simple_subgraph'
        : state.taskType === 'creative' && graph.getNodes().has('creative_subgraph')
          ? 'creative_subgraph'
          : graph.getNodes().has('generalist_subgraph')
            ? 'generalist_subgraph'
            : ''
    if (!targetSubgraph) {
      return state
    }

    state.subgraphState = {
      node: targetSubgraph,
      state: cloneGraphState(state),
    }
    state.currentStep = targetSubgraph
    state.toolCalls = []
    state.toolResults = []
    state.output = ''
    return state
  }

  private graphStateFromRunCheckpoint(
    graph: AgentGraph,
    checkpoint: SessionRunCheckpoint,
  ): AgentState | null {
    if (!checkpoint.graphState) {
      return null
    }

    const state = cloneGraphState(checkpoint.graphState)
    state.messages = checkpoint.messages.map(cloneMessage)
    state.totalUsage = { ...checkpoint.totalUsage }
    state.iteration = checkpoint.iteration
    state.maxIterations = checkpoint.maxIterations
    state.shouldStop = false
    if (
      checkpoint.pendingToolExecution?.toolCalls.length
      && state.toolCalls.length === 0
    ) {
      state.toolCalls = checkpoint.pendingToolExecution.toolCalls.map(cloneToolCall)
    }
    this.normalizeLegacyGraphResumeState(graph, state)
    if (!state.currentStep) {
      state.currentStep = checkpoint.pendingToolExecution
        ? this.resolvePendingToolGraphNode(graph)
        : ''
    }
    return state
  }

  private graphStateFromApprovalCheckpoint(
    graph: AgentGraph,
    checkpoint: ApprovalRunCheckpoint,
  ): AgentState | null {
    if (!checkpoint.graphState) {
      return null
    }

    const state = cloneGraphState(checkpoint.graphState)
    state.messages = checkpoint.messages.map(cloneMessage)
    state.toolCalls = checkpoint.toolCalls.map(cloneToolCall)
    state.totalUsage = { ...checkpoint.totalUsage }
    state.iteration = checkpoint.iteration
    state.maxIterations = checkpoint.maxIterations
    state.shouldStop = false
    this.normalizeLegacyGraphResumeState(graph, state)
    if (!state.currentStep) {
      state.currentStep = this.resolvePendingToolGraphNode(graph)
    }
    return state
  }

  async *run(input: string, context: AgentContext, mode?: AgentMode): AsyncIterable<AgentEvent> {
    if (this.stopRequested) {
      yield { type: 'done', usage: { inputTokens: 0, outputTokens: 0 }, stopReason: stopReasonUserAbort() }
      return
    }
    const expired = callerDeadlineOutcome(context.deadlineAt)
    if (expired) {
      yield* expired
      return
    }
    const decodedContentParts = decodeMultimodalInput(input)
    const nativeMultimodalInput = decodedContentParts
      ? requiresNativeMultimodalInput(decodedContentParts)
      : false
    const nativeGraphInput = this.options.graphRegistry
      ?.get(mode ?? this.options.defaultMode ?? 'instant')?.capabilities?.nativeMultimodalInput
    if (decodedContentParts && (!nativeMultimodalInput || nativeGraphInput)) {
      const primaryInstruction = primaryInstructionFromContentParts(decodedContentParts)
      if (primaryInstruction !== null) {
        context = { ...context, currentUserContent: decodedContentParts }
        input = primaryInstruction
      }
    }
    const runHookData: AgentRunHookData = {
      runId: randomUUID(),
      sessionId: context.sessionId,
      mode: mode ?? this.options.defaultMode ?? 'instant',
      provider: context.provider,
      model: context.model,
      input,
      ...(context.backgroundJobId ? {
        backgroundJobId: context.backgroundJobId,
        detached: true,
      } : {}),
      ...(context.surface ? { surface: context.surface } : {}),
      ...(context.roleBotId ? { roleBotId: context.roleBotId } : {}),
      ...(context.providerAttempt ? { attempt: context.providerAttempt.attempt } : {}),
    }
    const isFallbackPending = context.providerAttempt?.isFallbackPending
    const hookObservation = createAgentRunHookObservation()
    const runAbortController = new AbortController()
    this.activeRunAbortController = runAbortController
    const unregisterCanceller = this.options.activeRuns?.registerCanceller(context.sessionId, () => this.stop())
    const unregisterPrivilege = this.options.activeRuns?.registerPrivilegeSetter(context.sessionId, (level) => this.setAutonomy(level))
    this.lastRunSessionId = context.sessionId
    this.options.activeRuns?.recordRunContext(context.sessionId, {
      mode: mode ?? this.options.defaultMode,
      cwd: context.cwd,
      workspaceRoot: context.workspaceRoot,
      provider: context.provider,
      model: context.model,
      autonomy: this.options.autonomy,
      ...requestAuthorityForRecord(context),
    })
    this.options.activeRuns?.upsert({ sessionId: context.sessionId, currentNode: 'routing',
      iteration: 0, maxIterations: this.options.maxIterations ?? 10 })
    const { signal } = runAbortController
    let triggerPostRun = false
    try {
      const preRun = await raceWithAbort(
        triggerPreAgentRun(this.options.hookRegistry, runHookData),
        signal,
      )
      if (preRun.aborted) {
        yield {
          type: 'error',
          error: { code: 'FORBIDDEN', message: preRun.reason ?? 'Agent run blocked by hook' },
        }
        return
      }
      if (this.options.usageTracker && this.options.spendBudget) {
        const spend = checkSpendBudget(this.options.usageTracker, this.options.spendBudget, {
          sessionId: context.sessionId,
        })
        if (!spend.allowed) {
          yield {
            type: 'error',
            error: { code: 'SPEND_BUDGET_EXCEEDED', message: spend.reason ?? 'spend budget exceeded' },
          }
          return
        }
      }
      triggerPostRun = true
      const expiredAfterHooks = callerDeadlineOutcome(context.deadlineAt)
      if (expiredAfterHooks) {
        for (const event of expiredAfterHooks) {
          observeAgentRunHookEvent(hookObservation, event)
          yield event
        }
        return
      }
    const executionPolicy = context.executionPolicy ?? {
      configuredAutonomy: this.options.autonomy,
      effectiveAutonomy: this.options.autonomy,
      clamped: false,
      agentMode: mode ?? this.options.defaultMode ?? 'instant',
      primaryAgentId: context.primaryAgentId,
      workspaceBoundary: context.workspaceRoot ? 'strict' as const : 'unrestricted' as const,
      freshApprovalRequired: context.requireToolApproval === true,
    }
    await logAgentDebugTrace({
      event: 'policy.execution-context',
      source: 'mode-router',
      sessionId: context.sessionId,
      runId: context.sessionId,
      mode: executionPolicy.agentMode,
      status: executionPolicy.clamped ? 'clamped' : 'effective',
      data: { ...executionPolicy },
    })
    yield { type: 'execution_policy', policy: executionPolicy }
    context = this.beginTurnAgentContext(context, input)
    const requestedMode = mode ?? this.options.defaultMode ?? 'instant'
    let nonAutoRouteReason: string | undefined
    let nonAutoRouteFallback = false
    const isMultimodal = nativeMultimodalInput
    const autoFallbackMode = resolveAutoFallbackMode({
      surface: context.surface,
      activeRemoteBrowser: this.options.activeRemoteBrowser,
    })
    let resolvedMode: AgentMode = requestedMode === 'auto'
      ? this.confidentIntentModeHint() ?? autoFallbackMode
      : requestedMode
    if (isMultimodal && resolvedMode !== 'instant' && resolvedMode !== 'react'
      && !this.options.graphRegistry?.get(resolvedMode)?.capabilities?.nativeMultimodalInput) {
      resolvedMode = 'react'
      nonAutoRouteReason = 'Native multimodal input requires the content-aware tool loop.'
      nonAutoRouteFallback = true
    }
    if (requestedMode === 'auto') {
      nonAutoRouteReason = this.confidentIntentModeHint()
        ? 'The semantic router selected the execution mode.'
        : this.options.activeRemoteBrowser
          ? 'A selected browser is active; start the tool loop with its authorized capabilities.'
          : autoFallbackMode === 'react'
            ? 'No confident semantic decision; a workspace shell starts the tool loop, which can answer directly or transfer.'
            : 'The bounded controller will answer or select an execution mode from the request and available capabilities.'
    }

    const turnContext: AgentContext = context

    if (this.isDirectCapabilityMode(resolvedMode)) {
      yield {
        type: 'mode_route_decision',
        chosen: resolvedMode,
        candidates: [...new Set([requestedMode, resolvedMode])],
        reason: nonAutoRouteReason ?? 'Direct capability mode selected.',
        fallback: nonAutoRouteFallback,
      }
      yield* observeAgentRunEvents(
        this.dispatch(input, this.resolveAgentContext(turnContext, input), resolvedMode),
        hookObservation,
      )
      return
    }

    const hydratedContext = await raceWithAbort(
      this.hydrateAgentContext(input, turnContext),
      signal,
    )
    const expiredAfterHydration = callerDeadlineOutcome(context.deadlineAt)
    if (expiredAfterHydration) {
      for (const event of expiredAfterHydration) {
        observeAgentRunHookEvent(hookObservation, event)
        yield event
      }
      return
    }
    const resolvedContext = await raceWithAbort(
      this.withDurableRunContract(input, hydratedContext, resolvedMode, signal),
      signal,
    )
    yield {
      type: 'mode_route_decision',
      chosen: resolvedMode,
      candidates: [...new Set([requestedMode, resolvedMode])],
      reason: nonAutoRouteReason ?? 'Explicit mode selected.',
      fallback: nonAutoRouteFallback,
    }
    if (resolvedContext.relevantContextItems?.length) {
      yield {
        type: 'memory_context',
        id: randomUUID(),
        items: resolvedContext.relevantContextItems,
      }
    }

    yield* observeAgentRunEvents(
      this.dispatch(input, resolvedContext, resolvedMode),
      hookObservation,
    )
    } catch (error) {
      if (signal.aborted && isAbortError(error)) {
        hookObservation.status = 'aborted'
        hookObservation.error = getAbortError(signal).message
      } else {
        hookObservation.status = 'error'
        hookObservation.error = error instanceof Error ? error.message : String(error)
        throw error
      }
    } finally {
      if (this.activeRunAbortController === runAbortController) {
        unregisterCanceller?.()
        unregisterPrivilege?.()
        this.options.activeRuns?.finish(context.sessionId)
        this.activeRunAbortController = null
      }
      if (triggerPostRun) {
        if (signal.aborted) hookObservation.status = 'aborted'
        await triggerPostAgentRun(
          this.options.hookRegistry,
          {
            ...completeAgentRunHookData(runHookData, hookObservation),
            ...(isFallbackPending ? { fallbackPending: safeFallbackPending(isFallbackPending) } : {}),
          },
        )
        if (hookObservation.status !== 'incomplete') {
          await this.options.workspaceMutationTracker?.cleanupScratch(context.sessionId).catch(() => {})
        }
      }
    }
  }

  async *resumeFromApprovalCheckpoint(checkpoint: ApprovalRunCheckpoint, approved: boolean | ApprovalDecision): AsyncIterable<AgentEvent> {
    yield* this.resumeCheckpoint(checkpoint, approved)
  }

  async *resumeFromRunCheckpoint(checkpoint: SessionRunCheckpoint): AsyncIterable<AgentEvent> {
    yield* this.resumeCheckpoint(checkpoint)
  }

  private async *resumeCheckpoint(checkpoint: ApprovalRunCheckpoint | SessionRunCheckpoint, approved?: boolean | ApprovalDecision): AsyncIterable<AgentEvent> {
    if (this.stopRequested) {
      yield { type: 'done', usage: { inputTokens: 0, outputTokens: 0 }, stopReason: stopReasonUserAbort() }
      return
    }
    const expired = callerDeadlineOutcome(checkpoint.deadlineAt, checkpoint.totalUsage)
    if (expired) {
      yield* expired
      return
    }
    const mode = checkpoint.modeControlState?.mode ?? checkpoint.mode ?? 'react'
    const hookData: AgentRunHookData = {
      runId: randomUUID(),
      sessionId: checkpoint.sessionId,
      mode,
      provider: checkpoint.provider,
      model: checkpoint.model,
      resumed: true,
    }
    const observation = createAgentRunHookObservation()
    const preRun = await triggerPreAgentRun(this.options.hookRegistry, hookData)
    if (preRun.aborted) {
      yield { type: 'error', error: { code: 'FORBIDDEN', message: preRun.reason ?? 'Agent run blocked by hook' } }
      return
    }
    const controller = new AbortController()
    this.activeRunAbortController = controller
    const unregisterCanceller = this.options.activeRuns?.registerCanceller(checkpoint.sessionId, () => this.stop())
    try {
      const context: AgentContext = {
        sessionId: checkpoint.sessionId, provider: checkpoint.provider, model: checkpoint.model,
        cwd: checkpoint.cwd, workspaceRoot: checkpoint.workspaceRoot,
        workspaceIsolation: checkpoint.workspaceIsolation,
        scopeTags: cloneCheckpointScopeTags(checkpoint.scopeTags),
        executionSkillIds: cloneCheckpointExecutionSkillIds(checkpoint.executionSkillIds),
        skillToolNames: cloneCheckpointSkillToolNames(checkpoint.skillToolNames),
        deadlineAt: checkpoint.deadlineAt,
        memoryPolicy: cloneCheckpointMemoryPolicy(checkpoint.memoryPolicy),
        toolAllowlist: cloneCheckpointToolAllowlist(checkpoint.toolAllowlist),
        delegationToolAllowlist: cloneCheckpointToolAllowlist(checkpoint.delegationToolAllowlist),
        skillExecutionPolicies: cloneCheckpointSkillExecutionPolicies(checkpoint.skillExecutionPolicies),
        requireToolApproval: checkpoint.requireToolApproval,
        systemPrompt: checkpoint.systemPrompt, previousMessages: checkpoint.messages.map(cloneMessage),
        runContract: checkpoint.runContract, modeControlState: checkpoint.modeControlState,
      }
      yield* observeAgentRunEvents(this.dispatch(latestUserText(checkpoint.messages), context, mode, { checkpoint, approved }), observation)
    } catch (error) {
      if (isApprovalParkedError(error)) throw error
      const event: AgentEvent = { type: 'error', error: toProviderApiError(error) }
      observeAgentRunHookEvent(observation, event)
      yield event
    } finally {
      if (controller.signal.aborted) observation.status = 'aborted'
      unregisterCanceller?.()
      if (this.activeRunAbortController === controller) this.activeRunAbortController = null
      await triggerPostAgentRun(this.options.hookRegistry, completeAgentRunHookData(hookData, observation))
      if (observation.status !== 'incomplete') {
        await this.options.workspaceMutationTracker?.cleanupScratch(checkpoint.sessionId).catch(() => {})
      }
    }
  }

  private async *runResumedEngine(
    checkpoint: ApprovalRunCheckpoint | SessionRunCheckpoint,
    context: AgentContext,
    mode: string,
    approved?: boolean | ApprovalDecision,
  ): AsyncIterable<AgentEvent> {
    if (mode === 'react' || mode === 'instant') {
      const engine = this.createReactEngine(checkpoint.textDeltaMode, {
        maxIterations: checkpoint.maxIterations, hardMaxIterations: true,
        ...(mode === 'instant' ? { reviewOutcomes: false, strictFinalAnswerProtocol: false, reviewToollessFinals: false } : {}),
      })
      this.activeReactEngine = engine
      try {
        if ('requestId' in checkpoint) yield* engine.resumeFromCheckpoint(checkpoint, approved ?? false)
        else yield* engine.resumeFromRunCheckpoint(checkpoint)
      } finally {
        if (this.activeReactEngine === engine) this.activeReactEngine = null
      }
      return
    }
    const graph = this.getRequiredGraph(mode)
    const state = 'requestId' in checkpoint
      ? this.graphStateFromApprovalCheckpoint(graph, checkpoint)
      : this.graphStateFromRunCheckpoint(graph, checkpoint)
    if (!state) throw new Error('Graph checkpoint is missing graph state')
    const pending = 'requestId' in checkpoint ? {
      toolCalls: checkpoint.toolCalls.map(cloneToolCall), startIndex: checkpoint.currentToolIndex,
      batchSize: 1, initialApprovalDecision: approved ?? false, skipToolCallEventForStart: true,
    } : checkpoint.pendingToolExecution
    const controller = new AbortController()
    this.activeGraphState = state
    this.activeGraphAbortController = controller
    try {
      yield* graph.run(state, this.buildGraphContext(undefined, context, mode, pending, controller.signal, checkpoint.textDeltaMode))
    } finally {
      if (this.activeGraphState === state) this.activeGraphState = null
      if (this.activeGraphAbortController === controller) this.activeGraphAbortController = null
    }
  }

  private async *dispatch(
    input: string,
    context: AgentContext,
    mode: string,
    resume?: { checkpoint: ApprovalRunCheckpoint | SessionRunCheckpoint; approved?: boolean | ApprovalDecision },
  ): AsyncIterable<AgentEvent> {
    const originalOptions = this.options
    const ceiling = withOptionalToolNameAllowlist(
      originalOptions.authorizedTools ?? originalOptions.tools,
      context.toolAllowlist,
    )
    const delegationBase = originalOptions.delegationTools ?? originalOptions.authorizedTools ?? originalOptions.tools
    const turnMaxIterations = context.modeControlState?.turnMaxIterations ?? originalOptions.maxIterations ?? 50
    if (resume) {
      this.options = { ...originalOptions, maxIterations: turnMaxIterations,
        tools: withOptionalToolNameAllowlist(ceiling, context.modeControlState?.visibleToolNames) }
      this.graphCache.clear()
    }
    let currentMode = mode
    let currentContext = context
    const discover = createToolDiscoveryContext(primaryInstructionFromContentParts(decodeMultimodalInput(input) ?? []) ?? input)
    try {
      for (let transfers = context.modeControlState?.transferCount ?? 0; transfers <= 4; transfers += 1) {
        if (this.activeRunAbortController?.signal.aborted) return
        const expired = callerDeadlineOutcome(
          currentContext.deadlineAt, resume?.checkpoint.totalUsage ?? currentContext.executionHandoff?.usage,
        )
        if (expired) {
          yield* expired
          return
        }
        let pending: { transfer: ModeTransfer; snapshot: ExecutionHandoff } | undefined
        const multimodalTurn = requiresNativeMultimodalInput(Array.isArray(currentContext.currentUserContent) ? currentContext.currentUserContent : decodeMultimodalInput(input) ?? [])
        const modes = listSemanticModes(this.options.graphRegistry)
          .filter((mode) => !multimodalTurn || mode.id === 'instant' || mode.id === 'react')
        const contractCeiling = withOptionalToolNameAllowlist(withOptionalToolNameAllowlist(ceiling, currentContext.toolAllowlist), currentContext.runContract?.executionIntent?.allowedTools)
        const delegationCeiling = withOptionalToolNameAllowlist(delegationBase, currentContext.delegationToolAllowlist ?? currentContext.toolAllowlist)
        const delegationContractCeiling = withOptionalToolNameAllowlist(delegationCeiling, currentContext.runContract?.executionIntent?.allowedTools)
        const discoveryContext = await raceWithAbort(discover(contractCeiling, currentContext), this.activeRunAbortController?.signal ?? new AbortController().signal)
        this.modeControl = createModeControl({
          currentMode, modes, tools: contractCeiling, context: currentContext,
          delegationTools: delegationContractCeiling,
          discoveryContext,
          transferCount: transfers, turnMaxIterations,
          activeToolNames: currentMode === 'instant'
            ? [...resolveInstantModeToolNames(input, 'instant', this.options.tools.list().map((tool) => tool.name), undefined, undefined, this.options.activeRemoteBrowser) ?? []]
            : this.options.tools.list().map((tool) => tool.name),
          canTransfer: transfers < 4 && contractCeiling.list().length > 0,
          onTransfer: (transfer, snapshot) => {
            this.transferredGraphState = snapshot.graphState ? cloneGraphState(snapshot.graphState) : undefined
            const { graphState: _graphState, ...handoff } = snapshot
            pending = { transfer, snapshot: structuredClone(handoff) }
          },
        })
        if (resume) {
          const initialResume = resume
          resume = undefined
          yield* this.runResumedEngine(initialResume.checkpoint, currentContext, currentMode, initialResume.approved)
        } else if (currentMode === 'react' || currentMode === 'instant') {
          yield* this.runReact(input, currentContext, currentMode)
        } else {
          const builder = this.resolveGraphBuilder(currentMode)
          if (builder) yield* this.runGraph(input, currentContext, currentMode, builder)
          else yield* this.runReact(input, currentContext)
        }
        if (!pending) return
        const { transfer, snapshot } = pending
        const remaining = turnMaxIterations - snapshot.iterations
        if (remaining <= 0) {
          yield { type: 'error', error: { code: 'INTERNAL_ERROR', message: 'The turn budget was exhausted before the requested mode transition.' } }
          return
        }
        const nextCeiling = withOptionalToolNameAllowlist(contractCeiling, transfer.executionIntent.allowedTools)
        const capabilityGroups: Partial<Record<AgentExecutionCapability, ToolExposureGroup>> = {
          'filesystem-read': 'files', 'filesystem-write': 'files', terminal: 'files',
          process: 'process', service: 'services', browser: 'browser', network: 'web',
        }
        const selectedGroups = transfer.executionIntent.capabilities.flatMap((capability) => capabilityGroups[capability] ? [capabilityGroups[capability]!] : [])
        const projected = withContextualToolExposure(nextCeiling, { routedMode: transfer.mode, semanticRouting: true, selectedGroups, activeRemoteBrowser: originalOptions.activeRemoteBrowser })
        const carried = transfer.mode === 'instant' ? [] : this.options.tools.list().map((tool) => tool.name)
        const selectedNames = new Set([...projected.list().map((tool) => tool.name), ...transfer.tools, ...carried])
        this.options = {
          ...originalOptions,
          autonomy: this.options.autonomy,
          tools: withToolNameAllowlist(nextCeiling, selectedNames),
          maxIterations: turnMaxIterations,
          hardMaxIterations: true,
        }
        this.graphCache.clear()
        currentMode = transfer.mode
        currentContext = {
          ...currentContext,
          executionHandoff: snapshot,
          toolAllowlist: nextCeiling.list().map((tool) => tool.name),
          delegationToolAllowlist: withOptionalToolNameAllowlist(delegationContractCeiling, transfer.executionIntent.allowedTools).list().map((tool) => tool.name),
          runContract: mergeModeTransferContract(
            this.transferredGraphState?.seedContract ?? currentContext.runContract,
            transfer,
            input,
          ),
        }
        yield {
          type: 'mode_route_decision', chosen: currentMode,
          candidates: modes.map((candidate) => candidate.id),
          reason: transfer.reason, fallback: false,
        }
      }
    } finally {
      this.options = { ...originalOptions, autonomy: this.options.autonomy }
      this.transferredGraphState = undefined
      this.modeControl = undefined
      this.graphCache.clear()
    }
  }

  private async *runReact(
    input: string,
    context: AgentContext,
    mode: AgentMode = 'react',
  ): AsyncIterable<AgentEvent> {
    const instantTools = mode === 'instant'
      ? withOptionalToolNameAllowlist(this.options.tools, resolveInstantModeToolNames(input, mode, undefined, undefined, undefined, this.options.activeRemoteBrowser))
      : undefined
    const engine = this.createReactEngine(undefined, mode === 'instant' ? {
      strictFinalAnswerProtocol: false,
      reviewOutcomes: false,
      reviewToollessFinals: false,
      tools: instantTools,
      maxIterations: Math.min(
        (context.executionHandoff?.iterations ?? 0) + INSTANT_MAX_ITERATIONS,
        this.options.maxIterations ?? 50,
      ),
      hardMaxIterations: true,
    } : this.options.stagePolicy?.outcomeReview === false ? {
      reviewOutcomes: false,
      reviewToollessFinals: false,
    } : {})
    this.activeReactEngine = engine
    try {
      yield* engine.run(input, this.resolveAgentContext(context, input))
    } finally {
      if (this.activeReactEngine === engine) {
        this.activeReactEngine = null
      }
    }
  }

  private async *runGraph(
    input: string,
    context: AgentContext,
    mode: string,
    builder: GraphBuilder,
  ): AsyncIterable<AgentEvent> {
    const abortController = new AbortController()
    const graphContext = this.buildGraphContext(
      input,
      context,
      mode,
      undefined,
      abortController.signal,
    )
    const state = this.buildInitialGraphState(input, graphContext.agentContext, mode)
    const seededPreferPromptReact = state.preferPromptReact
    this.activeGraphState = state
    this.activeGraphContext = graphContext
    this.activeGraphAbortController = abortController
    const escalation = this.options.graphRegistry?.get(mode)?.capabilities?.escalation
    try {
      try {
        for await (const event of this.getGraph(mode, builder).run(state, graphContext)) {
          if (event.type === 'done' && event.stopReason && escalation) {
            yield {
              ...event,
              stopReason: withEscalationOffer(event.stopReason, state, escalation),
            }
            continue
          }
          yield event
        }
      } catch (error) {
        yield {
          type: 'error',
          error: toProviderApiError(error),
        }
      }
    } finally {
      if (this.activeGraphState === state) {
        this.activeGraphState = null
      }
      if (this.activeGraphAbortController === abortController) {
        this.activeGraphAbortController = null
      }
      if (
        this.options.persistTransportPreference
        && state.preferPromptReact !== seededPreferPromptReact
      ) {
        try {
          await this.options.persistTransportPreference(
            graphContext.agentContext.sessionId,
            state.preferPromptReact,
          )
        } catch {
        }
      }
    }
  }

  async stop(): Promise<void> {
    this.stopRequested = true
    this.activeRunAbortController?.abort(createAbortError('Agent execution stopped'))
    if (this.activeGraphState) {
      this.activeGraphState.shouldStop = true
    }
    this.activeGraphAbortController?.abort(createAbortError('Agent execution stopped'))
    if (this.activeReactEngine) {
      await this.activeReactEngine.stop()
    }
  }
}
