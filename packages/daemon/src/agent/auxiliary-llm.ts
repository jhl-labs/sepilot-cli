import type {
  ChatRequest,
  ChatResponse,
  ILLMProvider,
  TokenUsage,
  ToolCall,
} from '@sepilotd/core'
import { createAbortError } from '../abort.js'
import { parseToolCallArguments } from './tool-call-args.js'
import { extractContent } from '../providers/utils.js'
import {
  guardedProviderChat,
  guardedProviderStream,
  type ProviderCircuitBreaker,
} from '../providers/circuit-breaker.js'
import { logLlmCallTraceDetached } from '../observability/agent-trace.js'
import { resolveCallDeadline } from './control-call-policy.js'

const DEFAULT_AUXILIARY_LLM_TIMEOUT_MS = 25_000
export const DEFAULT_AUXILIARY_LLM_TURN_BUDGET_MS = 25_000
const REASONING_RETRY_MAX_TOKENS = 8_000

export type AuxiliaryLlmTransport = 'chat' | 'stream' | 'auto'

export class AuxiliaryLlmBudgetExhaustedError extends Error {
  constructor(readonly budgetMs: number | null) {
    super(budgetMs === null
      ? 'Auxiliary LLM turn budget is exhausted'
      : `Auxiliary LLM turn budget exhausted after ${budgetMs}ms`)
    this.name = 'AuxiliaryLlmBudgetExhaustedError'
  }
}

export class AuxiliaryLlmTimeoutError extends Error {
  constructor(
    readonly label: string,
    readonly timeoutMs: number,
  ) {
    super(`${label} timed out after ${timeoutMs}ms`)
    this.name = 'AuxiliaryLlmTimeoutError'
  }
}

/**
 * One active wall-clock budget shared by every optional LLM call in a turn.
 * Main-model generation and tool execution do not consume this budget: a
 * completion review must still be possible after a long implementation step.
 * A timed-out auxiliary request exhausts the budget and aborts its peers so
 * independent planners/reviewers cannot serially add their full timeouts.
 */
export class AuxiliaryLlmTurnBudget {
  readonly startedAt = Date.now()
  private consumedMs = 0
  private activeSince: number | null = null
  private exhausted = false
  private readonly controllers = new Set<AbortController>()
  private readonly activeRequests = new Map<number, AuxiliaryLlmActiveRequest>()
  private readonly listeners = new Set<() => void>()
  private nextRequestId = 1

  constructor(readonly budgetMs: number | null = resolveAuxiliaryLlmTurnBudgetMs()) {}

  remainingMs(now = Date.now()): number | null {
    if (this.exhausted) return 0
    if (this.budgetMs === null) return null
    const activeMs = this.activeSince === null ? 0 : Math.max(0, now - this.activeSince)
    return Math.max(0, this.budgetMs - this.consumedMs - activeMs)
  }

  isExhausted(now = Date.now()): boolean {
    const remaining = this.remainingMs(now)
    return remaining !== null && remaining <= 0
  }

  register(controller: AbortController): () => void {
    if (this.isExhausted()) {
      throw new AuxiliaryLlmBudgetExhaustedError(this.budgetMs)
    }
    if (this.controllers.size === 0) this.activeSince = Date.now()
    this.controllers.add(controller)
    return () => {
      if (!this.controllers.delete(controller)) return
      if (this.controllers.size === 0 && this.activeSince !== null) {
        this.consumedMs += Math.max(0, Date.now() - this.activeSince)
        this.activeSince = null
      }
    }
  }

  beginRequest(input: Omit<AuxiliaryLlmActiveRequest, 'id'>): () => void {
    const id = this.nextRequestId++
    this.activeRequests.set(id, { id, ...input })
    this.notify()
    return () => {
      if (this.activeRequests.delete(id)) this.notify()
    }
  }

  snapshotActiveRequests(): AuxiliaryLlmActiveRequest[] {
    return [...this.activeRequests.values()]
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  private notify(): void {
    for (const listener of this.listeners) listener()
  }

  exhaust(reason?: unknown): void {
    if (this.exhausted) return
    this.exhausted = true
    const abortReason = reason instanceof Error
      ? reason
      : new AuxiliaryLlmBudgetExhaustedError(this.budgetMs)
    for (const controller of this.controllers) {
      controller.abort(abortReason)
    }
    this.controllers.clear()
    this.notify()
  }
}

export interface AuxiliaryLlmActiveRequest {
  id: number
  providerId: string
  request: ChatRequest
  label: string
  startedAt: number
  timeoutMs: number | null
}

function reasoningRetryMaxTokens(
  provider: ILLMProvider,
  request: ChatRequest,
): number | null {
  const current = request.maxTokens ?? 0
  const configuredMaximum = provider.models.find(
    (model) => model.id === request.model,
  )?.maxOutputTokens
  const ceiling = Number.isFinite(configuredMaximum) && configuredMaximum
    ? Math.max(1, Math.floor(configuredMaximum))
    : REASONING_RETRY_MAX_TOKENS
  const next = Math.min(
    ceiling,
    Math.max(REASONING_RETRY_MAX_TOKENS, current * 12),
  )
  return next > current ? next : null
}

function needsReasoningRetry(response: ChatResponse): boolean {
  return response.finishReason === 'length'
    && extractContent(response.message).trim().length === 0
    && (response.message.toolCalls?.length ?? 0) === 0
    && Boolean(response.thinking?.trim())
}

function combineRetryUsage(
  first: ChatResponse,
  retry: ChatResponse,
): ChatResponse {
  return {
    ...retry,
    usage: {
      inputTokens: first.usage.inputTokens + retry.usage.inputTokens,
      outputTokens: first.usage.outputTokens + retry.usage.outputTokens,
      thinkingTokens: (first.usage.thinkingTokens ?? 0)
        + (retry.usage.thinkingTokens ?? 0) || undefined,
    },
  }
}

export function resolveAuxiliaryLlmTimeoutMs(): number | null {
  const raw = process.env.SEPILOTD_AUX_LLM_TIMEOUT_MS
  if (raw === '0') {
    return null
  }
  if (raw === undefined) {
    return DEFAULT_AUXILIARY_LLM_TIMEOUT_MS
  }
  const parsed = Number.parseInt(raw, 10)
  return Number.isFinite(parsed) && parsed > 0
    ? parsed
    : DEFAULT_AUXILIARY_LLM_TIMEOUT_MS
}

export function resolveAuxiliaryLlmTurnBudgetMs(): number | null {
  const raw = process.env.SEPILOTD_AUX_LLM_TURN_BUDGET_MS
  if (raw === '0') return null
  if (raw === undefined) return DEFAULT_AUXILIARY_LLM_TURN_BUDGET_MS
  const parsed = Number.parseInt(raw, 10)
  return Number.isFinite(parsed) && parsed > 0
    ? parsed
    : DEFAULT_AUXILIARY_LLM_TURN_BUDGET_MS
}

/**
 * Bounded cooldown for an optional auxiliary stage (contract planner,
 * grounding audit) on one provider/model. An auxiliary call that times out
 * consumed its whole wall-clock allowance and produced nothing; a second
 * consecutive timeout for the same stage on the same model is evidence that
 * this model cannot answer the stage inside its budget right now, so paying
 * the full timeout again on every turn only adds latency before the fallback
 * the stage would have returned anyway. The circuit is in-memory, keyed by
 * stage + provider + model, closes on the first success, and expires on its
 * own — it is a runtime breaker, not a persisted compatibility profile.
 */
export class AuxiliaryStageCircuit {
  private readonly entries = new Map<string, { consecutiveTimeouts: number; openUntil: number }>()

  constructor(
    private readonly options: {
      openAfterConsecutiveTimeouts?: number
      cooldownMs?: number
      now?: () => number
    } = {},
  ) {}

  static key(stage: string, providerId: string, model: string): string {
    return `${stage}|${providerId}|${model}`
  }

  private now(): number {
    return (this.options.now ?? Date.now)()
  }

  private threshold(): number {
    return this.options.openAfterConsecutiveTimeouts ?? 2
  }

  isOpen(key: string): boolean {
    const entry = this.entries.get(key)
    if (!entry) return false
    if (entry.openUntil > this.now()) return true
    if (entry.openUntil > 0) {
      // Cooldown elapsed: allow one probe; another timeout re-opens at once.
      entry.openUntil = 0
      entry.consecutiveTimeouts = Math.max(0, this.threshold() - 1)
    }
    return false
  }

  recordTimeout(key: string): void {
    const entry = this.entries.get(key) ?? { consecutiveTimeouts: 0, openUntil: 0 }
    entry.consecutiveTimeouts += 1
    if (entry.consecutiveTimeouts >= this.threshold()) {
      entry.openUntil = this.now() + (this.options.cooldownMs ?? 10 * 60_000)
    }
    this.entries.set(key, entry)
  }

  recordSuccess(key: string): void {
    this.entries.delete(key)
  }
}

let defaultAuxiliaryStageCircuit: AuxiliaryStageCircuit | null = null

export function getDefaultAuxiliaryStageCircuit(): AuxiliaryStageCircuit {
  defaultAuxiliaryStageCircuit ??= new AuxiliaryStageCircuit()
  return defaultAuxiliaryStageCircuit
}

export function createAuxiliaryLlmTurnBudget(): AuxiliaryLlmTurnBudget {
  return new AuxiliaryLlmTurnBudget()
}

function formatAuxiliaryTraceError(error: unknown): string | undefined {
  if (error === undefined) return undefined
  if (error instanceof Error) return error.message
  if (typeof error === 'string') return error
  if (error !== null && typeof error === 'object') {
    const message = (error as { message?: unknown }).message
    if (typeof message === 'string') return message
    return 'Unknown auxiliary LLM error'
  }
  return String(error)
}

export async function runAuxiliaryLlmChat(input: {
  provider: ILLMProvider
  request: ChatRequest
  label: string
  breaker?: ProviderCircuitBreaker
  signal?: AbortSignal
  timeoutMs?: number | null
  budget?: AuxiliaryLlmTurnBudget
  allowReasoningRetry?: boolean
  maxRetries?: number
  transport?: AuxiliaryLlmTransport
  sessionId?: string
}): Promise<ChatResponse> {
  const first = await runSingleAuxiliaryLlmChat(input)
  if (input.allowReasoningRetry === false || !needsReasoningRetry(first)) {
    return first
  }

  const maxTokens = reasoningRetryMaxTokens(input.provider, input.request)
  if (maxTokens === null) {
    return first
  }

  const retry = await runSingleAuxiliaryLlmChat({
    ...input,
    request: {
      ...input.request,
      maxTokens,
      messages: [
        ...input.request.messages,
        {
          role: 'system',
          content: 'The previous attempt used its output budget on hidden reasoning. Return the requested compact answer now, with no preamble or analysis.',
        },
      ],
    },
  })
  return combineRetryUsage(first, retry)
}

async function runSingleAuxiliaryLlmChat(input: {
  provider: ILLMProvider
  request: ChatRequest
  label: string
  breaker?: ProviderCircuitBreaker
  signal?: AbortSignal
  timeoutMs?: number | null
  budget?: AuxiliaryLlmTurnBudget
  maxRetries?: number
  transport?: AuxiliaryLlmTransport
  sessionId?: string
}): Promise<ChatResponse> {
  const remainingBudgetMs = input.budget?.remainingMs() ?? null
  if (remainingBudgetMs !== null && remainingBudgetMs <= 0) {
    input.budget?.exhaust()
    throw new AuxiliaryLlmBudgetExhaustedError(input.budget?.budgetMs ?? null)
  }
  const timeoutMs = resolveCallDeadline(input.timeoutMs, resolveAuxiliaryLlmTimeoutMs(), remainingBudgetMs)
  const controller = new AbortController()
  const unregisterBudget = input.budget?.register(controller) ?? (() => {})
  const finishActiveRequest = input.budget?.beginRequest({
    providerId: input.provider.id,
    request: input.request,
    label: input.label,
    startedAt: Date.now(),
    timeoutMs,
  }) ?? (() => {})
  const abortFromParent = () => {
    controller.abort(createAbortError(`${input.label} aborted`))
  }
  if (input.signal?.aborted) {
    abortFromParent()
  } else {
    input.signal?.addEventListener('abort', abortFromParent, { once: true })
  }

  // The trace write must stay outside the timeout race below: writeEntry
  // serializes on a shared queue, so awaiting it here could push an otherwise
  // successful auxiliary call past its deadline.
  const traceCall = (response?: ChatResponse, error?: unknown): void => {
    logLlmCallTraceDetached({
      source: 'auxiliary',
      mode: 'auxiliary',
      sessionId: input.sessionId,
      provider: input.provider.name,
      model: input.request.model,
      iteration: 0,
      request: input.request,
      response,
      error: formatAuxiliaryTraceError(error),
      meta: { label: input.label, node: input.label },
    })
  }

  const chatPromise = executeAuxiliaryProviderRequest(input, controller.signal)

  if (timeoutMs === null) {
    return await chatPromise.then(
      (response) => {
        traceCall(response)
        return response
      },
      (error) => {
        traceCall(undefined, error)
        throw error
      },
    ).finally(() => {
      unregisterBudget()
      finishActiveRequest()
      input.signal?.removeEventListener('abort', abortFromParent)
    })
  }

  return await new Promise<ChatResponse>((resolve, reject) => {
    let settled = false
    const finish = (fn: () => void) => {
      if (settled) {
        return
      }
      settled = true
      clearTimeout(timer)
      unregisterBudget()
      finishActiveRequest()
      input.signal?.removeEventListener('abort', abortFromParent)
      fn()
    }
    const timer = setTimeout(() => {
      const error = new AuxiliaryLlmTimeoutError(input.label, timeoutMs)
      controller.abort(createAbortError(error.message))
      input.breaker?.noteExternalFailure(input.provider.id, input.request.model, error.message)
      input.budget?.exhaust(error)
      finish(() => reject(error))
    }, timeoutMs)
    timer.unref?.()

    chatPromise.then(
      (response) => {
        traceCall(response)
        finish(() => resolve(response))
      },
      (error) => {
        traceCall(undefined, error)
        finish(() => reject(
        input.budget?.isExhausted()
          ? new AuxiliaryLlmBudgetExhaustedError(input.budget.budgetMs)
          : error,
        ))
      },
    )
  })
}

async function executeAuxiliaryProviderRequest(
  input: {
    provider: ILLMProvider
    request: ChatRequest
    breaker?: ProviderCircuitBreaker
    maxRetries?: number
    transport?: AuxiliaryLlmTransport
  },
  signal: AbortSignal,
): Promise<ChatResponse> {
  const chat = () => guardedProviderChat({
    provider: input.provider,
    request: input.request,
    signal,
    breaker: input.breaker,
    maxRetries: input.maxRetries,
  })
  if ((input.transport ?? 'chat') === 'chat') return await chat()

  const streamed = await collectAuxiliaryStream(input, signal)
  if (
    input.transport === 'auto'
    && !hasUsefulAuxiliaryResponse(streamed, input.request)
  ) {
    return await chat()
  }
  return streamed
}

function hasUsefulAuxiliaryResponse(
  response: ChatResponse,
  request: ChatRequest,
): boolean {
  const satisfiesVisibleResponseContract =
    extractContent(response.message).trim().length > 0
    || (response.message.toolCalls?.length ?? 0) > 0
  // Hidden reasoning is useful for ordinary auxiliary prose, but it cannot
  // satisfy a request that structurally requires a tool selection. In auto
  // mode, retry that incomplete transport through chat so the provider gets
  // one chance to return the required visible/tool envelope.
  if (request.toolChoice === 'required') {
    return satisfiesVisibleResponseContract
  }
  return satisfiesVisibleResponseContract || Boolean(response.thinking?.trim())
}

async function collectAuxiliaryStream(
  input: {
    provider: ILLMProvider
    request: ChatRequest
    breaker?: ProviderCircuitBreaker
    maxRetries?: number
  },
  signal: AbortSignal,
): Promise<ChatResponse> {
  let text = ''
  let thinking = ''
  const toolCalls: ToolCall[] = []
  const toolCallArgs = new Map<string, string>()
  const usage: TokenUsage = { inputTokens: 0, outputTokens: 0 }
  let finishReason: ChatResponse['finishReason'] = 'stop'
  let sawDone = false
  let sawNonTerminalChunk = false

  for await (const chunk of guardedProviderStream({
    provider: input.provider,
    request: input.request,
    signal,
    breaker: input.breaker,
    maxRetries: input.maxRetries,
  })) {
    if (chunk.type !== 'done') sawNonTerminalChunk = true
    switch (chunk.type) {
      case 'text':
        text += chunk.text
        break
      case 'thinking':
        thinking += chunk.text
        break
      case 'tool_call_start': {
        const { id, name } = chunk.toolCall
        if (!id || !name) break
        toolCalls.push({ id, name, arguments: {} })
        toolCallArgs.set(id, '')
        break
      }
      case 'tool_call_delta':
        toolCallArgs.set(
          chunk.toolCallId,
          (toolCallArgs.get(chunk.toolCallId) ?? '') + chunk.delta,
        )
        break
      case 'tool_call_end':
        break
      case 'usage':
        usage.inputTokens += chunk.usage.inputTokens
        usage.outputTokens += chunk.usage.outputTokens
        usage.thinkingTokens = sumOptionalUsage(
          usage.thinkingTokens,
          chunk.usage.thinkingTokens,
        )
        usage.cacheReadTokens = sumOptionalUsage(
          usage.cacheReadTokens,
          chunk.usage.cacheReadTokens,
        )
        usage.cacheCreationTokens = sumOptionalUsage(
          usage.cacheCreationTokens,
          chunk.usage.cacheCreationTokens,
        )
        break
      case 'done':
        sawDone = true
        finishReason = chunk.finishReason
        break
      case 'error':
        throw new Error(chunk.error.message)
    }
  }

  if (!sawDone && sawNonTerminalChunk) finishReason = 'length'
  const usableToolCalls: ToolCall[] = []
  let truncatedToolCall = false
  for (const toolCall of toolCalls) {
    const parsed = parseToolCallArguments(toolCallArgs.get(toolCall.id))
    if (parsed.truncated) {
      truncatedToolCall = true
      continue
    }
    usableToolCalls.push({ ...toolCall, arguments: parsed.arguments })
  }
  if (truncatedToolCall && finishReason !== 'content_filter') {
    finishReason = 'length'
  }

  return {
    message: {
      role: 'assistant',
      content: text,
      toolCalls: usableToolCalls.length > 0 ? usableToolCalls : undefined,
    },
    thinking: thinking || undefined,
    usage,
    finishReason,
  }
}

function sumOptionalUsage(current: number | undefined, next: number | undefined): number | undefined {
  return next ? (current ?? 0) + next : current
}
