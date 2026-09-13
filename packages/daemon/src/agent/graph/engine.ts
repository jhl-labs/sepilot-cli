import type { AgentEvent } from '@sepilotd/core'
import { isAbortError } from '../../abort.js'
import {
  logAgentDebugTrace,
  logAgentRunTrace,
} from '../../observability/agent-trace.js'
import { toProviderApiError } from '../../providers/circuit-breaker.js'
import { buildLlmRequestEvent } from '../../observability/llm-request-event.js'
import { IrreducibleContextOverflowError } from '../context-manager.js'
import type { RunResumeStage } from '../../server/runtime/runs.js'
import { cloneGraphState, createGraphRunCheckpoint } from './checkpoints.js'
import { buildStateBoard, type AgentSteeringNote } from './state-board.js'
import {
  captureContinuationProgress,
  hasContinuationProgress,
  resolveRunMaxWallMs,
} from './continuation-progress.js'
import type {
  AgentState,
  AgentSeedContract,
  GraphEdgeDefinition,
  GraphEdgeTarget,
  GraphExecutionContext,
  GraphNodeDefinition,
  GraphNodeMeta,
  GraphRuntimeState,
  NodeFn,
  RouterFn,
  StreamingNodeFn,
} from './types.js'
import {
  buildBudgetExhaustedMessage,
  buildContinuationPrompt,
  resolveMaxContinuationCycles,
} from '../task-contract.js'
import {
  isIncompleteOutput,
  stopReasonApprovalDenied,
  stopReasonBudget,
  stopReasonCompleted,
  stopReasonCompletionGate,
  stopReasonNoProgress,
  stopReasonObservationBudget,
  stopReasonProviderError,
  stopReasonStuckRepeat,
  stopReasonUserAbort,
  stopReasonWallClock,
} from '../stop-reason.js'
import type { RunStopReason } from '@sepilotd/core'

export interface StateGraphRunOptions {
  maxNodeExecutions?: number
}

export interface GraphChildErrorInfo {
  code: string
  message: string
}

/**
 * Thrown by a subgraph node after its child graph already surfaced a terminal
 * error event. The parent engine treats it as a halt-without-duplicate: the
 * child's error was forwarded once, so the parent stops as an error without
 * re-emitting a second error event.
 */
export class GraphChildHaltError extends Error {
  constructor(readonly childError: GraphChildErrorInfo) {
    super(childError.message)
    this.name = 'GraphChildHaltError'
  }
}

/** Thrown when a conditional edge router returns a target it never declared. */
export class GraphRoutingError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'GraphRoutingError'
  }
}

// Bounds for the item snapshots included in the `state_board` stream event so
// a large board can never bloat the live stream; the full board stays in the
// journal and the /state endpoint.
const MAX_STATE_BOARD_EVENT_ITEMS = 20
const MAX_STATE_BOARD_EVENT_TEXT_CHARS = 120

function truncateBoardEventText(text: string): string {
  const normalized = text.trim().replace(/\s+/g, ' ')
  return normalized.length > MAX_STATE_BOARD_EVENT_TEXT_CHARS
    ? `${normalized.slice(0, MAX_STATE_BOARD_EVENT_TEXT_CHARS - 1)}…`
    : normalized
}

function isStreamingNodeExecution<
  State extends GraphRuntimeState,
  Context,
>(
  execution: ReturnType<NodeFn<State, Context>>,
): execution is ReturnType<StreamingNodeFn<State, Context>> {
  return typeof execution === 'object'
    && execution !== null
    && Symbol.asyncIterator in execution
}

function lifecycleStateToRunStage(
  lifecycleState: import('@sepilotd/core').AgentState,
): RunResumeStage {
  if (lifecycleState === 'acting') {
    return 'acting'
  }
  if (lifecycleState === 'observing') {
    return 'observing'
  }
  return 'thinking'
}

function runContractKey(contract: AgentSeedContract | undefined): string | undefined {
  return contract ? JSON.stringify(contract) : undefined
}

function buildNodeTraceEvent(
  node: string,
  startedAt: number,
  nextEdge?: GraphEdgeTarget | null,
): Extract<AgentEvent, { type: 'node_trace' }> {
  return {
    type: 'node_trace',
    node,
    durationMs: Math.max(0, Date.now() - startedAt),
    ...(nextEdge ? { nextEdge } : {}),
  }
}

async function* observeAuxiliaryLlmRequests<T>(
  promise: Promise<T>,
  context: unknown,
): AsyncGenerator<AgentEvent, T> {
  if (!isGraphExecutionContext(context)) {
    return await promise
  }

  // A node starts executing synchronously until its first await before this
  // observer receives the returned promise. That lets a convergence node
  // lazily create its bounded control budget and still expose the request on
  // the same live event stream as ordinary auxiliary work.
  const budgets = [context.auxiliaryLlmBudget, context.controlLlmBudget]
    .filter((budget, index, all): budget is NonNullable<typeof budget> => (
      Boolean(budget) && all.indexOf(budget) === index
    ))
  if (budgets.length === 0) {
    return await promise
  }
  context.emittedLlmRequestObjects ??= new WeakSet()
  let settled = false
  let value: T | undefined
  let failure: unknown
  const completion = promise.then(
    (result) => {
      value = result
      settled = true
    },
    (error) => {
      failure = error
      settled = true
    },
  )

  while (!settled) {
    for (const budget of budgets) {
      for (const active of budget.snapshotActiveRequests()) {
        if (context.emittedLlmRequestObjects.has(active.request)) continue
        context.emittedLlmRequestObjects.add(active.request)
        yield buildLlmRequestEvent({
          sessionId: context.agentContext.sessionId,
          iteration: context.activeGraphIteration,
          source: `aux:${active.label}`,
          request: active.request,
          providerId: active.providerId,
          startedAt: active.startedAt,
          timeoutMs: active.timeoutMs,
          auxiliary: true,
        })
      }
    }
    if (settled) break
    await new Promise<void>((resolve) => {
      let unsubscribes: Array<() => void> = []
      const finish = () => {
        for (const unsubscribe of unsubscribes) unsubscribe()
        unsubscribes = []
        resolve()
      }
      unsubscribes = budgets.map((budget) => budget.subscribe(finish))
      void completion.then(finish, finish)
    })
  }

  await completion
  if (failure !== undefined) throw failure
  return value as T
}

function resolveNodeModelOverride(
  context: GraphExecutionContext,
  nodeId: string,
): string | undefined {
  const model = context.graphNodeModelOverrides?.[context.graphId]?.[nodeId]?.model?.trim()
  if (!model) {
    return undefined
  }
  const providerModels = context.provider?.models ?? []
  return providerModels.some((candidate) => candidate.id === model)
    ? model
    : undefined
}

export class StateGraph<
  State extends GraphRuntimeState,
  Context,
> {
  private nodes = new Map<string, GraphNodeDefinition<State, Context>>()
  private edges = new Map<string, GraphEdgeDefinition<State, Context>>()
  private startNode = 'start'

  addNode(
    name: string,
    fn: NodeFn<State, Context>,
    meta?: GraphNodeMeta,
  ): this
  addNode(
    name: string,
    definition: GraphNodeDefinition<State, Context>,
  ): this
  addNode(
    name: string,
    fnOrDefinition: NodeFn<State, Context> | GraphNodeDefinition<State, Context>,
    meta?: GraphNodeMeta,
  ): this {
    const definition =
      typeof fnOrDefinition === 'function'
        ? { run: fnOrDefinition, meta }
        : fnOrDefinition
    this.nodes.set(name, definition)
    return this
  }

  addEdge(from: string, to: GraphEdgeTarget): this {
    this.edges.set(from, { type: 'direct', to })
    return this
  }

  addConditionalEdge(
    from: string,
    router: RouterFn<State, Context>,
    targets: readonly GraphEdgeTarget[] = [],
  ): this {
    this.edges.set(from, {
      type: 'conditional',
      router,
      targets: [...targets],
    })
    return this
  }

  setStart(name: string): this {
    this.startNode = name
    return this
  }

  getNodes(): Map<string, GraphNodeDefinition<State, Context>> {
    return this.nodes
  }

  getEdges(): Map<string, GraphEdgeDefinition<State, Context>> {
    return this.edges
  }

  getStartNode(): string {
    return this.startNode
  }

  protected resolveNodeMeta(
    definition: GraphNodeDefinition<State, Context>,
  ): Required<GraphNodeMeta> {
    const lifecycleState = definition.meta?.lifecycleState ?? 'thinking'
    return {
      lifecycleState,
      resumeStage:
        definition.meta?.resumeStage
        ?? lifecycleStateToRunStage(lifecycleState),
      pendingToolExecutionNode:
        definition.meta?.pendingToolExecutionNode ?? false,
    }
  }

  findNodeByMeta(
    predicate: (
      meta: Required<GraphNodeMeta>,
      name: string,
      definition: GraphNodeDefinition<State, Context>,
    ) => boolean,
  ): string | null {
    for (const [name, definition] of this.nodes.entries()) {
      if (predicate(this.resolveNodeMeta(definition), name, definition)) {
        return name
      }
    }
    return null
  }

  private resolveNextNode(
    currentNode: string,
    state: State,
    context?: Context,
  ): GraphEdgeTarget | null {
    const edge = this.edges.get(currentNode)
    if (!edge) {
      return null
    }

    if (edge.type === 'direct') {
      return edge.to
    }

    const next = edge.router(state, context)
    // A conditional router that declared its targets must return one of them
    // (or the terminal '__end__'). Anything else is a router bug — surface it
    // instead of letting an undefined/typo target break to a silent "success".
    if (edge.targets.length > 0) {
      if (next === undefined || next === null) {
        throw new GraphRoutingError(
          `conditional edge from '${currentNode}' returned no target`,
        )
      }
      if (next !== '__end__' && !edge.targets.includes(next)) {
        throw new GraphRoutingError(
          `conditional edge from '${currentNode}' returned undeclared target '${String(next)}'`,
        )
      }
    }
    return next
  }

  async *run(
    state: State,
    context?: Context,
    options?: StateGraphRunOptions,
  ): AsyncIterable<AgentEvent> {
    let current = state.currentStep && this.nodes.has(state.currentStep)
      ? state.currentStep
      : this.startNode
    let nodeExecutions = 0
    const maxNodeExecutions = options?.maxNodeExecutions ?? state.maxIterations * 10
    // No-progress cycle detection: count how often the run revisits the exact
    // same (node, progress-signature). A run that makes progress mints fresh
    // signatures; a tight A→B oscillation revisits the same handful and trips
    // the limit long before it burns the whole (possibly multiplied) budget.
    const noProgressLimit = Math.max(6, state.maxIterations * 2)
    const signatureCounts = new Map<string, number>()
    const maxContinuationCycles = isGraphExecutionContext(context)
      ? resolveMaxContinuationCycles(context.maxContinuationCycles)
      : resolveMaxContinuationCycles()
    let continuationCycle = 0
    // Once a control/guard node explicitly routes an exhausted run into a
    // lifecycleState='done' reporter, that terminal handoff is authoritative.
    // The reporter must run exactly once and the generic continuation
    // supervisor must not reset the graph behind its back.
    let terminalHandoffCommitted = false
    let terminalStatus: 'success' | 'error' | 'aborted' | 'incomplete' = 'success'
    let terminalError: string | undefined
    let lastRunContractKey: string | undefined
    // A board describes durable semantic run state, not the graph cursor.
    // Persisting and streaming the same board at every graph node made long
    // runs grow linearly with internal orchestration (often hundreds of
    // duplicate events). Keep node lifecycle in state_change/node_trace and
    // emit the board only when its semantic payload actually changes.
    let lastStateBoardFingerprint: string | undefined
    // A graph can stop from an acting/observing node (for example after an
    // explicit approval denial) without reaching a lifecycleState='done'
    // node. Track message payloads already placed on the stream so the
    // terminal output is never lost, while preserving exactly-once delivery
    // when a streaming node already surfaced the same text.
    const emittedMessageContents = new Set<string>()
    const wallDeadlineMs = resolveRunMaxWallMs()
    const wallStartedAt = Date.now()
    // Progress-gated continuation: another continuation cycle is granted only
    // when the cycle that just ended advanced the run (verified evidence,
    // executed tool call, artifact mutation). The baseline is taken at run
    // start so a first cycle that did nothing stops as no_progress too.
    if (maxContinuationCycles > 0) {
      const agentState = state as Partial<AgentState>
      agentState.continuationProgressSnapshot = captureContinuationProgress(
        agentState as AgentState,
      )
    }
    const continuationProgressed = (): boolean => {
      const agentState = state as Partial<AgentState>
      const current = captureContinuationProgress(agentState as AgentState)
      const progressed = hasContinuationProgress(
        agentState.continuationProgressSnapshot,
        current,
      )
      agentState.continuationProgressSnapshot = current
      return progressed
    }
    const stopWithoutContinuationProgress = (
      layerBudget: number,
    ): string => {
      const agentState = state as Partial<AgentState>
      state.shouldStop = true
      terminalStatus = 'incomplete'
      agentState.budgetExhausted = true
      agentState.stopReason = stopReasonNoProgress({
        layer: 'continuation',
        budget: maxContinuationCycles,
        used: continuationCycle,
        contract: agentState.seedContract,
      })
      const output = buildBudgetExhaustedMessage({
        mode: isGraphExecutionContext(context) ? context.graphId : 'graph',
        layer: 'continuation',
        iterationBudget: layerBudget,
        detail: `cycle ${continuationCycle + 1}/${maxContinuationCycles + 1}, ${layerBudget} steps, no new verified evidence, tool execution or artifact change`,
        contract: agentState.seedContract,
      })
      agentState.output = output
      return output
    }

    if (context && isGraphExecutionContext(context)) {
      await logAgentDebugTrace({
        event: 'graph.run.start',
        source: 'graph',
        sessionId: context.agentContext.sessionId,
        runId: context.agentContext.sessionId,
        mode: context.graphId,
        graphId: context.graphId,
        iteration: state.iteration,
        data: {
          startNode: current,
          maxNodeExecutions,
          maxContinuationCycles,
          state: graphDebugState(state),
        },
      })
    }

    try {
    while (current && current !== '__end__') {
      // Honor an abort at the very top of the loop so a stop request takes
      // effect immediately — before running the next (possibly bookkeeping)
      // node — instead of leaking one more node execution past the abort.
      if (contextAbortSignal(context)?.aborted) {
        terminalStatus = 'aborted'
        yield { type: 'done', usage: state.totalUsage, stopReason: stopReasonUserAbort() }
        return
      }
      if (wallDeadlineMs != null && Date.now() - wallStartedAt >= wallDeadlineMs) {
        terminalStatus = 'incomplete'
        state.shouldStop = true
        const agentState = state as Partial<AgentState>
        agentState.budgetExhausted = true
        agentState.stopReason = stopReasonWallClock(wallDeadlineMs)
        agentState.output = agentState.output
          || buildBudgetExhaustedMessage({
            mode: isGraphExecutionContext(context) ? context.graphId : 'graph',
            layer: 'wall_clock',
            iterationBudget: wallDeadlineMs,
            contract: agentState.seedContract,
          })
        emittedMessageContents.add(agentState.output)
        yield {
          type: 'message',
          content: agentState.output,
        }
        break
      }
      // Honor shouldStop, but always allow a terminal 'done' node (e.g.
      // reporter / finalizer) to run once before exiting. Otherwise a
      // node like iteration_guard that signals stop while routing to the
      // reporter would skip the reporter entirely and starve any
      // post-run hook (skill auto-discovery, final summary writes, etc.)
      // attached there.
      if (state.shouldStop) {
        const definition = this.nodes.get(current)
        const meta = definition ? this.resolveNodeMeta(definition) : null
        if (!meta || meta.lifecycleState !== 'done') {
          const output = (state as Partial<AgentState>).output
          if (
            typeof output === 'string'
            && output.length > 0
            && !emittedMessageContents.has(output)
          ) {
            emittedMessageContents.add(output)
            yield { type: 'message', content: output }
          }
          break
        }
      }

      nodeExecutions++
      if (nodeExecutions > maxNodeExecutions) {
        if (continuationCycle < maxContinuationCycles && !continuationProgressed()) {
          const output = stopWithoutContinuationProgress(maxNodeExecutions)
          emittedMessageContents.add(output)
          yield { type: 'message', content: output }
          break
        }
        if (continuationCycle < maxContinuationCycles) {
          continuationCycle++
          const agentState = state as Partial<AgentState>
          const prompt = buildContinuationPrompt({
            mode: isGraphExecutionContext(context) ? context.graphId : 'graph',
            cycle: continuationCycle,
            maxCycles: maxContinuationCycles,
            contract: agentState.seedContract,
          })
          agentState.messages = [
            ...(agentState.messages ?? []),
            { role: 'system', content: prompt },
          ]
          agentState.budgetExhausted = false
          agentState.internalGraphContinuation = true
          agentState.output = ''
          state.shouldStop = false
          state.iteration = 0
          state.currentStep = ''
          nodeExecutions = 0
          current = this.startNode
          yield {
            type: 'thinking',
            content: `[continuation] Node execution budget reached; continuing automatically (${continuationCycle}/${maxContinuationCycles}).`,
          }
          if (context && isGraphExecutionContext(context)) {
            await context.saveRunCheckpoint?.(
              createGraphRunCheckpoint(state as unknown as AgentState, context, 'observing'),
            )
          }
          continue
        }
        state.shouldStop = true
        terminalStatus = 'incomplete'
        const agentState = state as Partial<AgentState>
        agentState.budgetExhausted = true
        agentState.stopReason = stopReasonBudget(
          'node',
          maxNodeExecutions,
          nodeExecutions,
          agentState.seedContract,
        )
        const output = buildBudgetExhaustedMessage({
          mode: isGraphExecutionContext(context) ? context.graphId : 'graph',
          layer: 'node',
          iterationBudget: maxNodeExecutions,
          contract: agentState.seedContract,
        })
        agentState.output = output
        emittedMessageContents.add(output)
        yield {
          type: 'message',
          content: output,
        }
        break
      }

      // Expose the remaining budget so a nested subgraph node can descend from
      // it instead of restarting a full budget (shared, not multiplied).
      if (isGraphExecutionContext(context)) {
        context.remainingNodeBudget = Math.max(0, maxNodeExecutions - nodeExecutions + 1)
      }

      const definition = this.nodes.get(current)
      if (!definition) {
        terminalStatus = 'error'
        terminalError = `Unknown node: ${current}`
        yield {
          type: 'error',
          error: {
            code: 'INTERNAL_ERROR',
            message: terminalError,
          },
        }
        if (context && isGraphExecutionContext(context)) {
          await logAgentRunTrace({
            source: 'graph',
            status: terminalStatus,
            mode: context.graphId,
            graphId: context.graphId,
            sessionId: context.agentContext.sessionId,
            provider: context.agentContext.provider,
            model: context.agentContext.model,
            usage: state.totalUsage,
            error: terminalError,
            meta: { currentStep: current, iteration: state.iteration },
          })
        }
        return
      }

      const nodeName = current
      const meta = this.resolveNodeMeta(definition)
      yield { type: 'state_change', state: meta.lifecycleState }

      let restoreGraphContext: (() => void) | null = null
      if (context && isGraphExecutionContext(context)) {
        const previousNodeId = context.activeGraphNodeId
        const previousGraphIteration = context.activeGraphIteration
        const previousAgentContext = context.agentContext
        const overrideModel = resolveNodeModelOverride(context, nodeName)
        context.activeGraphNodeId = nodeName
        context.activeGraphIteration = state.iteration
        if (overrideModel) {
          context.agentContext = {
            ...context.agentContext,
            model: overrideModel,
          }
        }
        restoreGraphContext = () => {
          context.activeGraphNodeId = previousNodeId
          context.activeGraphIteration = previousGraphIteration
          context.agentContext = previousAgentContext
        }
      }

      state.currentStep = nodeName
      updateActiveRun(context, state, nodeName)
      if (context && isGraphExecutionContext(context)) {
        const agentState = state as unknown as AgentState
        const pendingToolExecution =
          meta.pendingToolExecutionNode
          && Array.isArray(agentState.toolCalls)
          && agentState.toolCalls.length > 0
            ? {
                toolCalls: agentState.toolCalls,
                startIndex: context.pendingToolExecution?.startIndex ?? 0,
                batchSize: context.pendingToolExecution?.batchSize,
                currentExecutionId: context.pendingToolExecution?.currentExecutionId,
              }
            : undefined
        await context.saveRunCheckpoint?.(
          createGraphRunCheckpoint(
            agentState,
            context,
            meta.resumeStage,
            pendingToolExecution,
          ),
        )
        // Journal a lightweight board snapshot at the same cadence as the
        // heavy checkpoint when its semantic content changes, so
        // restart/resume can replay the latest board without storing one
        // identical copy per internal graph node.
        const stateBoard = buildStateBoard(agentState, context)
        const stateBoardFingerprint = JSON.stringify(stateBoard)
        if (stateBoardFingerprint !== lastStateBoardFingerprint) {
          lastStateBoardFingerprint = stateBoardFingerprint
          await context.journalStateBoard?.(stateBoard)
          // Also surface changed board content so CLI/GUI surfaces can update
          // their durable plan/todo view. Node progress has its own events.
          yield {
            type: 'state_board',
            criteriaTotal: stateBoard.completionCriteria.length,
            planTotal: stateBoard.plan.length,
            planDone: stateBoard.plan.filter((step) => step.status === 'done').length,
            todosTotal: stateBoard.todos.length,
            todosDone: stateBoard.todos.filter((todo) => todo.status === 'completed').length,
            currentNode: nodeName,
            nodeState: meta.lifecycleState,
            iteration: agentState.iteration,
            // Bounded item snapshots so surfaces can render a live todo/plan
            // list (not just counts) — the user-facing "what is it doing now".
            todos: stateBoard.todos.length > 0
              ? stateBoard.todos.slice(0, MAX_STATE_BOARD_EVENT_ITEMS).map((todo) => ({
                  content: truncateBoardEventText(todo.content),
                  status: todo.status,
                }))
              : undefined,
            plan: stateBoard.plan.length > 0
              ? stateBoard.plan.slice(0, MAX_STATE_BOARD_EVENT_ITEMS).map((step) => ({
                  title: truncateBoardEventText(step.title),
                  status: step.status,
                  depth: step.depth,
                }))
              : undefined,
          }
        }
      }

      const nodeStartedAt = Date.now()
      if (context && isGraphExecutionContext(context)) {
        await logAgentDebugTrace({
          event: 'graph.node.start',
          source: 'graph',
          sessionId: context.agentContext.sessionId,
          runId: context.agentContext.sessionId,
          mode: context.graphId,
          graphId: context.graphId,
          node: nodeName,
          iteration: state.iteration,
          data: {
            nodeExecution: nodeExecutions,
            lifecycleState: meta.lifecycleState,
            state: graphDebugState(state),
          },
        })
      }
      try {
        const execution = definition.run(state, context)
        let emittedToolCalls = false
        let emittedToolResults = false
        if (isStreamingNodeExecution(execution)) {
          try {
            let observed = observeAuxiliaryLlmRequests(execution.next(), context)
            let observedResult = await observed.next()
            while (!observedResult.done) {
              yield observedResult.value
              observedResult = await observed.next()
            }
            let result = observedResult.value
            while (!result.done) {
              if (result.value.type === 'tool_call') emittedToolCalls = true
              if (result.value.type === 'tool_result') emittedToolResults = true
              if (result.value.type === 'message') {
                emittedMessageContents.add(result.value.content)
              }
              yield result.value
              observed = observeAuxiliaryLlmRequests(execution.next(), context)
              observedResult = await observed.next()
              while (!observedResult.done) {
                yield observedResult.value
                observedResult = await observed.next()
              }
              result = observedResult.value
            }
            state = result.value
          } finally {
            // If this run generator is abandoned (client close / abort) while
            // suspended at a `yield`, close the node's generator so its finally
            // blocks run and provider streams / subprocesses / locks are freed.
            // A no-op once the node has already completed.
            await execution.return?.(undefined as never).catch(() => {})
          }
        } else {
          const observed = observeAuxiliaryLlmRequests(execution, context)
          let observedResult = await observed.next()
          while (!observedResult.done) {
            yield observedResult.value
            observedResult = await observed.next()
          }
          state = observedResult.value
        }
        updateActiveRun(context, state, nodeName)

        for (const event of takePendingAgentEvents(context)) {
          if (event.type === 'message') emittedMessageContents.add(event.content)
          yield event
        }

        const agentState = state as Partial<AgentState>
        const nextRunContractKey = runContractKey(agentState.seedContract)
        if (
          agentState.seedContract
          && nextRunContractKey
          && nextRunContractKey !== lastRunContractKey
        ) {
          lastRunContractKey = nextRunContractKey
          yield {
            type: 'run_contract',
            contract: agentState.seedContract,
          }
        }
        if (
          !emittedToolCalls
          && meta.lifecycleState === 'acting'
          && Array.isArray(agentState.toolCalls)
          && agentState.toolCalls.length > 0
        ) {
          for (const toolCall of agentState.toolCalls) {
            yield { type: 'tool_call', toolCall }
          }
        }
        if (
          !emittedToolResults
          && Array.isArray(agentState.toolResults)
          && agentState.toolResults.length > 0
        ) {
          for (const result of agentState.toolResults) {
            yield {
              type: 'tool_result',
              toolCallId: result.toolCallId,
              output: result.output,
              status: result.status as 'success' | 'error',
            }
          }
          agentState.toolResults = []
        }
      } catch (error) {
        for (const event of takePendingAgentEvents(context)) {
          if (event.type === 'message') emittedMessageContents.add(event.content)
          yield event
        }
        yield buildNodeTraceEvent(nodeName, nodeStartedAt)
        if (error instanceof GraphChildHaltError) {
          // A child subgraph already surfaced its terminal error event; halt the
          // parent as an error without emitting a duplicate error frame.
          terminalStatus = 'error'
          terminalError = error.childError.message
        } else if (
          (isGraphExecutionContext(context) && (context.signal?.aborted ?? false))
          || isAbortError(error)
        ) {
          terminalStatus = 'aborted'
          yield { type: 'done', usage: state.totalUsage, stopReason: stopReasonUserAbort() }
        } else {
          terminalStatus = 'error'
          const mappedError = error instanceof IrreducibleContextOverflowError
            ? { code: 'CONTEXT_LENGTH' as const, message: error.message }
            : toProviderApiError(error)
          terminalError = mappedError.message
          yield {
            type: 'error',
            error: mappedError,
          }
        }
        if (context && isGraphExecutionContext(context)) {
          await logAgentRunTrace({
            source: 'graph',
            status: terminalStatus,
            mode: context.graphId,
            graphId: context.graphId,
            sessionId: context.agentContext.sessionId,
            provider: context.agentContext.provider,
            model: context.agentContext.model,
            usage: state.totalUsage,
            output: (state as Partial<AgentState>).output,
            error: terminalError,
            meta: { currentStep: current, iteration: state.iteration },
          })
          if (terminalStatus !== 'aborted') {
            await context.clearRunCheckpoint?.(context.agentContext.sessionId)
          }
        }
        return
      } finally {
        restoreGraphContext?.()
      }

      if (isGraphExecutionContext(context) && context.modeControl?.pending()) return
      if (state.modeControlContinue) {
        state.modeControlContinue = false
        continue
      }

      // No-progress cycle guard: bail early if the run keeps revisiting the same
      // (node, progress-signature) far past the limit — an oscillation that
      // would otherwise burn the entire budget without advancing.
      const signature = progressSignature(nodeName, state)
      const repeated = (signatureCounts.get(signature) ?? 0) + 1
      signatureCounts.set(signature, repeated)
      if (repeated > noProgressLimit) {
        terminalStatus = 'incomplete'
        state.shouldStop = true
        const agentState = state as Partial<AgentState>
        agentState.budgetExhausted = true
        agentState.stopReason = stopReasonNoProgress({
          layer: 'cycle',
          contract: agentState.seedContract,
        })
        agentState.output =
          agentState.output
          || 'Run stopped: detected a no-progress cycle (the graph kept revisiting the same state without advancing). Partial progress is preserved in the session; resume from the checkpoint to continue.'
        if (context && isGraphExecutionContext(context)) {
          await logAgentDebugTrace({
            event: 'graph.guard',
            source: 'graph',
            sessionId: context.agentContext.sessionId,
            runId: context.agentContext.sessionId,
            mode: context.graphId,
            graphId: context.graphId,
            node: nodeName,
            iteration: state.iteration,
            status: 'blocked',
            data: {
              guard: 'no-progress-cycle',
              signature,
              repeated,
              limit: noProgressLimit,
              state: graphDebugState(state),
            },
          })
        }
        yield buildNodeTraceEvent(nodeName, nodeStartedAt)
        emittedMessageContents.add(agentState.output)
        yield { type: 'message', content: agentState.output }
        break
      }

      let next: GraphEdgeTarget | null
      try {
        next = this.resolveNextNode(nodeName, state, context)
      } catch (routingError) {
        terminalStatus = 'error'
        terminalError =
          routingError instanceof Error ? routingError.message : String(routingError)
        yield buildNodeTraceEvent(nodeName, nodeStartedAt)
        yield {
          type: 'error',
          error: { code: 'INTERNAL_ERROR', message: terminalError },
        }
        if (context && isGraphExecutionContext(context)) {
          await logAgentRunTrace({
            source: 'graph',
            status: terminalStatus,
            mode: context.graphId,
            graphId: context.graphId,
            sessionId: context.agentContext.sessionId,
            provider: context.agentContext.provider,
            model: context.agentContext.model,
            usage: state.totalUsage,
            error: terminalError,
            meta: { currentStep: current, iteration: state.iteration },
          })
          await context.clearRunCheckpoint?.(context.agentContext.sessionId)
        }
        return
      }
      const routedAgentState = state as Partial<AgentState>
      const nextDefinition = next && next !== '__end__' ? this.nodes.get(next) : undefined
      const nextMeta = nextDefinition ? this.resolveNodeMeta(nextDefinition) : undefined
      if (
        routedAgentState.budgetExhausted
        && meta.lifecycleState !== 'done'
        && nextMeta?.lifecycleState === 'done'
      ) {
        terminalHandoffCommitted = true
      }
      if (
        routedAgentState.budgetExhausted
        && !terminalHandoffCommitted
        && continuationCycle < maxContinuationCycles
        && !continuationProgressed()
      ) {
        const output = stopWithoutContinuationProgress(state.maxIterations)
        yield buildNodeTraceEvent(nodeName, nodeStartedAt)
        emittedMessageContents.add(output)
        yield { type: 'message', content: output }
        break
      }
      if (
        routedAgentState.budgetExhausted
        && !terminalHandoffCommitted
        && continuationCycle < maxContinuationCycles
      ) {
        continuationCycle++
        const prompt = buildContinuationPrompt({
          mode: isGraphExecutionContext(context) ? context.graphId : 'graph',
          cycle: continuationCycle,
          maxCycles: maxContinuationCycles,
          contract: routedAgentState.seedContract,
        })
        routedAgentState.messages = [
          ...(routedAgentState.messages ?? []),
          { role: 'system', content: prompt },
        ]
        routedAgentState.budgetExhausted = false
        routedAgentState.internalGraphContinuation = true
        routedAgentState.output = ''
        state.shouldStop = false
        state.iteration = 0
        state.currentStep = ''
        nodeExecutions = 0
        current = this.startNode
        yield {
          type: 'thinking',
          content: `[continuation] Iteration budget reached; continuing automatically (${continuationCycle}/${maxContinuationCycles}).`,
        }
        yield buildNodeTraceEvent(nodeName, nodeStartedAt, this.startNode)
        if (context && isGraphExecutionContext(context)) {
          await context.saveRunCheckpoint?.(
            createGraphRunCheckpoint(state as unknown as AgentState, context, 'observing'),
          )
        }
        continue
      }
      if (
        meta.lifecycleState === 'done'
        && typeof routedAgentState.output === 'string'
        && routedAgentState.output.length > 0
        && !emittedMessageContents.has(routedAgentState.output)
      ) {
        emittedMessageContents.add(routedAgentState.output)
        yield { type: 'message', content: routedAgentState.output }
      }
      if (context && isGraphExecutionContext(context)) {
        await logAgentDebugTrace({
          event: 'graph.node.end',
          source: 'graph',
          sessionId: context.agentContext.sessionId,
          runId: context.agentContext.sessionId,
          mode: context.graphId,
          graphId: context.graphId,
          node: nodeName,
          iteration: state.iteration,
          durationMs: Math.max(0, Date.now() - nodeStartedAt),
          status: 'success',
          data: {
            nextEdge: next,
            state: graphDebugState(state),
          },
        })
      }
      yield buildNodeTraceEvent(nodeName, nodeStartedAt, next)
      if (!next) {
        break
      }
      current = next
    }

    if (context && isGraphExecutionContext(context)) {
      // The regular checkpoint cadence journals the board before each node.
      // Finalizer/reporter nodes populate completion diagnostics during their
      // execution and then terminate, so without this post-node projection the
      // accepted criterion verdicts exist only in debug traces. Journal one
      // final snapshot when semantic state changed; the fingerprint keeps
      // ordinary runs and budget stops from writing duplicate boards.
      const terminalStateBoard = buildStateBoard(state as unknown as AgentState, context)
      const terminalStateBoardFingerprint = JSON.stringify(terminalStateBoard)
      if (terminalStateBoardFingerprint !== lastStateBoardFingerprint) {
        lastStateBoardFingerprint = terminalStateBoardFingerprint
        await context.journalStateBoard?.(terminalStateBoard)
      }
    }
    yield {
      type: 'done',
      usage: state.totalUsage,
      stopReason: resolveGraphStopReason(
        state as Partial<AgentState>,
        terminalStatus,
        terminalError,
        contextAbortSignal(context)?.aborted ?? false,
      ),
    }
    if (context && isGraphExecutionContext(context)) {
      const aborted = context.signal?.aborted ?? false
      const agentState = state as Partial<AgentState>
      const finalStatus = aborted
        ? 'aborted'
        : agentState.budgetExhausted
          ? 'incomplete'
          : terminalStatus
      await logAgentRunTrace({
        source: 'graph',
        status: finalStatus,
        mode: context.graphId,
        graphId: context.graphId,
        sessionId: context.agentContext.sessionId,
        provider: context.agentContext.provider,
        model: context.agentContext.model,
        usage: state.totalUsage,
        output: agentState.output,
        error: terminalError,
        meta: {
          currentStep: state.currentStep,
          iteration: state.iteration,
          ...(agentState.completionDiagnostics
            ? { completion: agentState.completionDiagnostics }
            : {}),
        },
      })
      if (finalStatus === 'incomplete') {
        const resumeState = cloneGraphState(state as unknown as AgentState)
        resumeState.currentStep = ''
        resumeState.iteration = 0
        resumeState.shouldStop = false
        resumeState.budgetExhausted = undefined
        resumeState.output = ''
        await context.saveRunCheckpoint?.(
          createGraphRunCheckpoint(resumeState, context, 'observing'),
        )
      } else if (finalStatus === 'success') {
        await context.clearRunCheckpoint?.(context.agentContext.sessionId)
      }
    }
    } finally {
      finishActiveRun(context)
    }
  }
}

/**
 * Derive the structured stop reason for a graph run's terminal `done` event.
 * A node that ended the run explicitly records `state.stopReason`; otherwise
 * the reason is derived from the graph's own loop-control state so every
 * termination path is classified without parsing the assistant prose.
 */
export function resolveGraphStopReason(
  state: Partial<AgentState>,
  terminalStatus: string,
  terminalError: string | undefined,
  aborted: boolean,
): RunStopReason {
  if (aborted || terminalStatus === 'aborted') return stopReasonUserAbort()
  if (terminalStatus === 'error') {
    return stopReasonProviderError({ layer: 'graph', summary: terminalError })
  }
  if (state.stopReason) return state.stopReason
  // Tool execution already attested an operator denial. Ending the graph
  // successfully is not evidence that the requested work was completed.
  if (state.approvalDenied) return stopReasonApprovalDenied(state.approvalDenied.toolName ?? 'tool')
  const contract = state.seedContract
  const incomplete = isIncompleteOutput(state.output)
  if (state.completionDiagnostics?.gate?.budgetExhausted === true) {
    return stopReasonCompletionGate({ unmet: state.completionDiagnostics.gate.unmet })
  }
  const forced = state.forcedFinalSynthesisReason
  const gateUnmet = (state.completionDiagnostics?.gate?.unmet?.length ?? 0) > 0
  if (forced && !incomplete && !state.budgetExhausted && !gateUnmet) {
    // Tool access was closed for a final synthesis turn and the model still
    // produced a complete answer: the run completed, the closure is context.
    // Unmet acceptance criteria recorded by the completion gate keep a forced
    // final classified by its forcing layer below, never as completed.
    return stopReasonCompleted({ layer: forced })
  }
  if (state.budgetExhausted || forced) {
    switch (forced) {
      case 'stuck-repeat':
        return stopReasonStuckRepeat({ contract })
      case 'inspection-observation-budget':
        return stopReasonObservationBudget({ incomplete, contract })
      case 'recovery-exhausted':
        return stopReasonNoProgress({ layer: 'recovery', contract })
      case 'provider-no-progress':
        return stopReasonNoProgress({ layer: 'provider', contract })
      case 'completion-gate-evidence-closure':
        return stopReasonCompletionGate({
          unmet: state.completionDiagnostics?.gate?.unmet,
        })
      case 'exact-tool-budget':
        return stopReasonBudget('exact_tool', undefined, undefined, contract)
      case 'iteration-budget':
        return stopReasonBudget('iteration', state.maxIterations, state.iteration, contract)
      default:
        break
    }
    if (state.budgetExhausted) {
      return stopReasonBudget('iteration', state.maxIterations, state.iteration, contract)
    }
  }
  return stopReasonCompleted()
}

function updateActiveRun<State extends GraphRuntimeState>(
  context: unknown,
  state: State,
  currentNode: string,
): void {
  if (!isGraphExecutionContext(context)) {
    return
  }
  context.activeRuns?.upsert({
    sessionId: context.agentContext.sessionId,
    graphId: context.graphId,
    currentNode,
    iteration: state.iteration,
    maxIterations: state.maxIterations,
    tokensInput: state.totalUsage.inputTokens,
    tokensOutput: state.totalUsage.outputTokens,
  })
  // Keep a reference to the LIVE state object (not a clone) so routes such as
  // `POST /sessions/:id/steer` can append to `steeringNotes` and have this
  // in-flight run observe it on a later turn.
  context.activeRuns?.registerLiveState?.(
    context.agentContext.sessionId,
    state as unknown as { steeringNotes?: AgentSteeringNote[] },
  )
}

function finishActiveRun(context: unknown): void {
  if (!isGraphExecutionContext(context)) {
    return
  }
  context.activeRuns?.finish(context.agentContext.sessionId)
}

function progressSignature<State extends GraphRuntimeState>(
  nodeName: string,
  state: State,
): string {
  const s = state as Partial<AgentState>
  return [
    nodeName,
    state.iteration,
    Array.isArray(s.toolCalls) ? s.toolCalls.length : 0,
    Array.isArray(s.toolResults) ? s.toolResults.length : 0,
    Array.isArray(s.messages) ? s.messages.length : 0,
    typeof s.output === 'string' ? s.output.length : 0,
  ].join('|')
}

function graphDebugState<State extends GraphRuntimeState>(state: State): Record<string, unknown> {
  const s = state as Partial<AgentState>
  const ledger = s.evidenceLedger
  return {
    currentStep: state.currentStep,
    iteration: state.iteration,
    maxIterations: state.maxIterations,
    shouldStop: state.shouldStop,
    budgetExhausted: s.budgetExhausted === true,
    messages: s.messages?.length ?? 0,
    pendingToolCalls: s.toolCalls?.length ?? 0,
    toolHistory: s.toolCallHistory?.length ?? 0,
    outputChars: s.output?.length ?? 0,
    phase: s.phaseUsageStart?.phase,
    backtracks: s.backtrackCount ?? 0,
    completionGateBlocks: s.completionGateBlocks ?? 0,
    artifactRevisionDrafts: s.artifactRevisionDrafts ?? 0,
    todos: {
      total: s.todoList?.length ?? 0,
      completed: s.todoList?.filter((item) => item.status === 'completed').length ?? 0,
    },
    evidence: ledger
      ? {
          sourceReads: ledger.sourceReads.length,
          sourceSearches: ledger.sourceSearches.length,
          artifactWrites: ledger.artifactWrites.length,
          artifactReadBacks: ledger.artifactReadBacks.length,
          validationRuns: ledger.validationRuns.length,
        }
      : undefined,
    usage: { ...state.totalUsage },
  }
}

function contextAbortSignal(context: unknown): AbortSignal | undefined {
  if (context && typeof context === 'object' && 'signal' in context) {
    return (context as { signal?: AbortSignal }).signal
  }
  return undefined
}

function isGraphExecutionContext(
  context: unknown,
): context is GraphExecutionContext {
  return typeof context === 'object'
    && context !== null
    && 'agentContext' in context
    && 'graphId' in context
}

function takePendingAgentEvents(context: unknown): AgentEvent[] {
  if (!isGraphExecutionContext(context) || !context.pendingAgentEvents?.length) {
    return []
  }
  return context.pendingAgentEvents.splice(0)
}

export class AgentGraph extends StateGraph<AgentState, GraphExecutionContext> {}
