import type { AgentEvent } from '@sepilotd/core'
import type { AgentGraph, StateGraph, StateGraphRunOptions, GraphChildErrorInfo } from './engine.js'
import { GraphChildHaltError } from './engine.js'
import {
  cloneGraphState,
  createGraphApprovalCheckpoint,
  createGraphRunCheckpoint,
} from './checkpoints.js'
import type {
  AgentState,
  GraphExecutionContext,
  GraphRuntimeState,
  NodeFn,
} from './types.js'
import { cloneMessage, cloneToolCall } from '../tool-execution.js'

export interface SubgraphNodeOptions<
  ParentState extends GraphRuntimeState,
  ParentContext,
  ChildState extends GraphRuntimeState,
  ChildContext = ParentContext,
> {
  graph: StateGraph<ChildState, ChildContext>
  mapIn: (
    parentState: ParentState,
    context?: ParentContext,
  ) => ChildState | Promise<ChildState>
  mapOut: (
    parentState: ParentState,
    childState: ChildState,
    context?: ParentContext,
  ) => ParentState | Promise<ParentState>
  mapContext?: (
    parentState: ParentState,
    context?: ParentContext,
  ) => ChildContext | Promise<ChildContext>
  nodePrefix?: string
  forwardMessages?: boolean
  runOptions?: StateGraphRunOptions
}

export interface AgentSubgraphNodeOptions {
  nodeId: string
  graph: AgentGraph
  mapIn: (
    parentState: AgentState,
    context?: GraphExecutionContext,
  ) => AgentState | Promise<AgentState>
  mapOut: (
    parentState: AgentState,
    childState: AgentState,
    context?: GraphExecutionContext,
  ) => AgentState | Promise<AgentState>
  mapContext?: (
    parentState: AgentState,
    context?: GraphExecutionContext,
  ) => GraphExecutionContext | undefined | Promise<GraphExecutionContext | undefined>
  nodePrefix?: string
  forwardMessages?: boolean
  runOptions?: StateGraphRunOptions
}

export function subgraphNode<
  ParentState extends GraphRuntimeState,
  ParentContext,
  ChildState extends GraphRuntimeState,
  ChildContext = ParentContext,
>(
  options: SubgraphNodeOptions<ParentState, ParentContext, ChildState, ChildContext>,
): NodeFn<ParentState, ParentContext> {
  return async function* (
    parentState: ParentState,
    context?: ParentContext,
  ): AsyncGenerator<AgentEvent, ParentState, void> {
    const childState = await options.mapIn(parentState, context)
    const childContext = options.mapContext
      ? await options.mapContext(parentState, context)
      : context as unknown as ChildContext
    let prefixedTextStream = false
    let childError: GraphChildErrorInfo | null = null

    for await (const event of options.graph.run(
      childState,
      childContext,
      withSharedBudget(childContext, options.runOptions),
    )) {
      if (event.type === 'done') {
        continue
      }

      const forwarded = prefixSubgraphEvent(
        event,
        options.nodePrefix,
        prefixedTextStream,
        options.forwardMessages ?? true,
      )
      if (!forwarded) {
        continue
      }

      if (forwarded.type === 'error') {
        childError = { ...forwarded.error }
      }
      if (forwarded.type === 'text_delta' && options.nodePrefix) {
        prefixedTextStream = true
      }
      yield forwarded
    }

    // A failed child must not let the parent finalize as success: surface the
    // terminal status instead of swallowing it. The child already emitted its
    // error frame, so GraphChildHaltError halts the parent without a duplicate.
    if (childError) {
      throw new GraphChildHaltError(childError)
    }
    const childAborted = readAbortSignal(childContext)?.aborted ?? false
    const nextParent = await options.mapOut(parentState, childState, context)
    if (childAborted) {
      nextParent.shouldStop = true
    }
    return nextParent
  }
}

export function agentSubgraphNode(
  options: AgentSubgraphNodeOptions,
): NodeFn<AgentState, GraphExecutionContext> {
  return async function* (
    parentState: AgentState,
    context?: GraphExecutionContext,
  ): AsyncGenerator<AgentEvent, AgentState, void> {
    const resumedChildState =
      parentState.subgraphState?.node === options.nodeId
        ? cloneGraphState(parentState.subgraphState.state)
        : null
    const childState = resumedChildState ?? await options.mapIn(parentState, context)
    const baseContext = options.mapContext
      ? await options.mapContext(parentState, context)
      : context
    const childContext = buildAgentSubgraphContext(
      parentState,
      childState,
      options.nodeId,
      baseContext,
    )
    let prefixedTextStream = false
    let childError: GraphChildErrorInfo | null = null

    syncParentStateFromChild(parentState, childState, options.nodeId)

    for await (const event of options.graph.run(
      childState,
      childContext,
      withSharedBudget(childContext, options.runOptions),
    )) {
      if (event.type === 'done') {
        continue
      }

      const forwarded = prefixSubgraphEvent(
        event,
        options.nodePrefix,
        prefixedTextStream,
        options.forwardMessages ?? true,
      )
      if (!forwarded) {
        continue
      }

      if (forwarded.type === 'error') {
        childError = { ...forwarded.error }
      }
      if (forwarded.type === 'text_delta' && options.nodePrefix) {
        prefixedTextStream = true
      }
      yield forwarded
    }

    // Propagate a failed child to the parent instead of finalizing as success.
    if (childError) {
      throw new GraphChildHaltError(childError)
    }
    if (childContext) {
      await childContext.saveRunCheckpoint?.(
        createGraphRunCheckpoint(childState, childContext, 'thinking'),
      )
    }
    const childAborted = readAbortSignal(childContext)?.aborted ?? false
    const nextParent = await options.mapOut(parentState, childState, context)
    nextParent.subgraphState = undefined
    if (childAborted) {
      nextParent.shouldStop = true
    }
    return nextParent
  }
}

function readAbortSignal(context: unknown): AbortSignal | undefined {
  if (context && typeof context === 'object' && 'signal' in context) {
    return (context as { signal?: AbortSignal }).signal
  }
  return undefined
}

// Cap the child's node budget by the parent's remaining budget so nesting
// shares one descending budget instead of restarting a full budget per level.
function withSharedBudget(
  context: unknown,
  runOptions: StateGraphRunOptions | undefined,
): StateGraphRunOptions | undefined {
  const remaining =
    context && typeof context === 'object' && 'remainingNodeBudget' in context
      ? (context as { remainingNodeBudget?: number }).remainingNodeBudget
      : undefined
  if (typeof remaining !== 'number' || remaining <= 0) {
    return runOptions
  }
  const declared = runOptions?.maxNodeExecutions
  const maxNodeExecutions =
    typeof declared === 'number' ? Math.min(declared, remaining) : remaining
  return { ...runOptions, maxNodeExecutions }
}

function buildAgentSubgraphContext(
  parentState: AgentState,
  childState: AgentState,
  nodeId: string,
  context?: GraphExecutionContext,
): GraphExecutionContext | undefined {
  if (!context) {
    return undefined
  }

  return {
    ...context,
    agentSubgraphNodeId: nodeId,
    agentContext: {
      ...context.agentContext,
      runContract: childState.seedContract,
    },
    saveRunCheckpoint: async (checkpoint) => {
      syncParentStateFromChild(parentState, childState, nodeId)
      await context.saveRunCheckpoint?.(
        createGraphRunCheckpoint(
          parentState,
          context,
          checkpoint.stage,
          checkpoint.pendingToolExecution
            ? {
                toolCalls: checkpoint.pendingToolExecution.toolCalls.map(cloneToolCall),
                startIndex: checkpoint.pendingToolExecution.startIndex,
                batchSize: checkpoint.pendingToolExecution.batchSize,
                currentExecutionId: checkpoint.pendingToolExecution.currentExecutionId,
              }
            : undefined,
        ),
      )
    },
    saveApprovalCheckpoint: async (checkpoint) => {
      syncParentStateFromChild(parentState, childState, nodeId)
      await context.saveApprovalCheckpoint?.(
        createGraphApprovalCheckpoint(
          checkpoint.requestId,
          parentState,
          context,
          checkpoint.toolCalls.map(cloneToolCall),
          checkpoint.currentToolIndex,
        ),
      )
    },
  }
}

function syncParentStateFromChild(
  parentState: AgentState,
  childState: AgentState,
  nodeId: string,
): void {
  parentState.messages = childState.messages.map(cloneMessage)
  parentState.effectiveContextWindowTokens = childState.effectiveContextWindowTokens
    ?? parentState.effectiveContextWindowTokens
  parentState.toolCalls = childState.toolCalls.map(cloneToolCall)
  parentState.toolResults = childState.toolResults.map((result) => ({ ...result }))
  parentState.memories = [...childState.memories]
  parentState.output = childState.output
  parentState.totalUsage = { ...childState.totalUsage }
  parentState.subgraphState = {
    node: nodeId,
    state: cloneGraphState(childState),
  }
}

export const __testables = {
  buildAgentSubgraphContext,
  withSharedBudget,
}

function prefixSubgraphEvent(
  event: AgentEvent,
  nodePrefix: string | undefined,
  prefixedTextStream: boolean,
  forwardMessages: boolean,
): AgentEvent | null {
  if (!nodePrefix) {
    if (event.type === 'message' && !forwardMessages) {
      return null
    }
    return event
  }

  const prefix = `[${nodePrefix}] `
  switch (event.type) {
    case 'thinking':
      return { ...event, content: `${prefix}${event.content}` }
    case 'message':
      if (!forwardMessages) {
        return null
      }
      return { ...event, content: `${prefix}${event.content}` }
    case 'text_delta':
      return {
        ...event,
        text: prefixedTextStream ? event.text : `${prefix}${event.text}`,
      }
    case 'error':
      return {
        ...event,
        error: {
          ...event.error,
          message: `${prefix}${event.error.message}`,
        },
      }
    default:
      return event
  }
}
