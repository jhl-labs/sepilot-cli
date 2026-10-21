import type {
  ChatRequest,
  ChatResponse,
  ILLMProvider,
  TokenUsage,
  ToolCall,
} from '@sepilotd/core'
import { Ajv } from 'ajv'
import { randomUUID } from 'node:crypto'
import { createAbortError, throwIfAborted } from '../abort.js'
import { parseToolCallArguments } from './tool-call-args.js'
import { extractContent } from '../providers/utils.js'
import {
  guardedProviderChat,
  guardedProviderStream,
  extractProviderErrorStatusCode,
  type ProviderCircuitBreaker,
} from '../providers/circuit-breaker.js'
import { logLlmCallTraceDetached } from '../observability/agent-trace.js'
import { resolveCallDeadline, resolveControlCallTimeoutMs } from './control-call-policy.js'
import { detectNativeToolTransportRejection } from './tool-transport-recovery.js'
import { closeTruncatedJson } from './json-repair.js'
import {
  detectProviderMaxOutputTokenLimit,
  recallObservedMaxOutputTokens,
  rememberObservedMaxOutputTokens,
} from './max-output-token-recovery.js'

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
    readonly kind: 'inactivity' | 'deadline' = 'deadline',
  ) {
    super(kind === 'inactivity'
      ? `${label} produced no output for ${timeoutMs}ms`
      : `${label} timed out after ${timeoutMs}ms`)
    this.name = 'AuxiliaryLlmTimeoutError'
  }
}

export class AuxiliaryLlmRepetitionError extends Error {
  constructor(readonly kind: 'exact-cycle' | 'stagnant-reasoning' = 'exact-cycle') {
    super(kind === 'exact-cycle'
      ? 'Required auxiliary control stream repeated an exact cycle without a visible response'
      : 'Required auxiliary control stream kept restating the same reasoning without a visible response')
    this.name = 'AuxiliaryLlmRepetitionError'
  }
}

const CONTROL_STAGNATION_WINDOW = 12_000
const CONTROL_STAGNATION_MIN_SENTENCE_CHARS = 24
const CONTROL_STAGNATION_MIN_SENTENCES = 40
const CONTROL_STAGNATION_MAX_DISTINCT_RATIO = 0.25

function hasStagnantControlReasoning(thinking: string): boolean {
  if (thinking.length < CONTROL_STAGNATION_WINDOW / 3) return false
  const window = thinking.slice(-CONTROL_STAGNATION_WINDOW)
  const sentences = window
    .split(/(?<=[.!?])\s+|\n+/u)
    .map((sentence) => sentence.trim().replace(/\s+/gu, ' ').toLowerCase())
    .filter((sentence) => sentence.length >= CONTROL_STAGNATION_MIN_SENTENCE_CHARS)
  if (sentences.length < CONTROL_STAGNATION_MIN_SENTENCES) return false
  return new Set(sentences).size / sentences.length <= CONTROL_STAGNATION_MAX_DISTINCT_RATIO
}

const CONTROL_REPETITION_WINDOW = 4_096
const CONTROL_REPETITION_MAX_PERIOD = 128
const CONTROL_REPETITION_CHECK_STEP = 256

function hasSustainedControlCycle(thinking: string): boolean {
  if (thinking.length < CONTROL_REPETITION_WINDOW) return false
  const suffix = thinking.slice(-CONTROL_REPETITION_WINDOW)
  const prefixes = new Uint16Array(suffix.length)
  for (let i = 1; i < suffix.length; i++) {
    let matched = prefixes[i - 1]!
    while (matched > 0 && suffix[i] !== suffix[matched]) matched = prefixes[matched - 1]!
    if (suffix[i] === suffix[matched]) matched++
    prefixes[i] = matched
  }
  return suffix.length - prefixes[suffix.length - 1]! <= CONTROL_REPETITION_MAX_PERIOD
}

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

  register(controller: AbortController, options: { admitWhenExhausted?: boolean } = {}): () => void {
    if (!options.admitWhenExhausted && this.isExhausted()) {
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
    Math.max(REASONING_RETRY_MAX_TOKENS, Math.ceil(current / 12)),
  )
  return next !== current ? next : null
}

function needsReasoningRetry(response: ChatResponse, request: ChatRequest): boolean {
  return response.finishReason === 'length'
    && response.finishDetail !== 'incomplete_stream'
    && extractContent(response.message).trim().length === 0
    && (response.message.toolCalls?.length ?? 0) === 0
    && (
      Boolean(response.thinking?.trim())
      || (Number.isFinite(request.maxTokens)
        && (request.maxTokens ?? 0) > 0
        && Number.isFinite(response.usage.outputTokens)
        && response.usage.outputTokens >= request.maxTokens!)
    )
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
      cacheReadTokens: sumOptionalUsage(first.usage.cacheReadTokens, retry.usage.cacheReadTokens),
      cacheCreationTokens: sumOptionalUsage(first.usage.cacheCreationTokens, retry.usage.cacheCreationTokens),
      estimatedCost: first.usage.estimatedCost !== undefined && retry.usage.estimatedCost !== undefined
        ? first.usage.estimatedCost + retry.usage.estimatedCost
        : undefined,
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
  controlResponseAccepted?: (response: ChatResponse) => boolean
}): Promise<ChatResponse> {
  const model = input.provider.models.find(model => model.id === input.request.model)
  const fallback = model?.capabilities?.thinking
    ? model.compatibility?.auxiliaryThinkingOffFallback
    : undefined
  const effectiveInput = input.request.thinkingLevel === 'off' && fallback
    ? {
        ...input,
        requestedThinkingLevel: input.request.thinkingLevel,
        request: { ...input.request, thinkingLevel: fallback as ChatRequest['thinkingLevel'] },
      }
    : input
  const first = await runSingleAuxiliaryLlmChat(effectiveInput)
  if (first.raw?.auxiliaryToolTransportRecovery || input.allowReasoningRetry === false) {
    return first
  }
  const exhausted = needsReasoningRetry(first, effectiveInput.request)
  const unaccepted = !exhausted && needsStructureRetry(first, input.controlResponseAccepted)
  if (!exhausted && !unaccepted) {
    return first
  }

  const maxTokens = reasoningRetryMaxTokens(effectiveInput.provider, effectiveInput.request)
  if (maxTokens === null) {
    return first
  }

  const retry = await runSingleAuxiliaryLlmChat({
    ...effectiveInput,
    admitWhenExhausted: true,
    request: {
      ...effectiveInput.request,
      maxTokens,
      messages: [
        ...effectiveInput.request.messages,
        {
          role: 'system',
          content: exhausted
            ? 'The previous attempt exhausted its output budget without returning the required visible response. Return the requested compact answer now, with no preamble or analysis.'
            : 'The previous reply did not contain the required structured response. Return only the requested compact answer now, starting immediately with it, with no preamble or analysis.',
        },
      ],
    },
  })
  return combineRetryUsage(first, retry)
}

function needsStructureRetry(
  response: ChatResponse,
  accepted: ((response: ChatResponse) => boolean) | undefined,
): boolean {
  if (!accepted || accepted(response)) return false
  return extractContent(response.message).trim().length > 0
    && (response.message.toolCalls?.length ?? 0) === 0
}

function hasExplicitControlDeadlineConfiguration(): boolean {
  return Boolean(process.env.SEPILOTD_AUX_LLM_TIMEOUT_MS?.trim())
    || Boolean(process.env.SEPILOTD_AUX_LLM_TURN_BUDGET_MS?.trim())
}

const CHAT_DEADLINE_FLOOR_TOKENS_PER_SECOND = 8

export function resolveChatAllowanceDeadlineMs(
  configuredLimitMs: number | null,
  maxTokens: number | undefined,
  controlCeilingMs: number | null = resolveControlCallTimeoutMs(),
): number | null {
  if (configuredLimitMs === null) return null
  const allowanceMs = Number.isSafeInteger(maxTokens) && maxTokens! > 0
    ? Math.ceil(maxTokens! / CHAT_DEADLINE_FLOOR_TOKENS_PER_SECOND) * 1000
    : 0
  const deadline = Math.max(configuredLimitMs, allowanceMs)
  return controlCeilingMs === null ? deadline : Math.min(deadline, Math.max(configuredLimitMs, controlCeilingMs))
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
  requestedThinkingLevel?: ChatRequest['thinkingLevel']
  admitWhenExhausted?: boolean
}): Promise<ChatResponse> {
  throwIfAborted(input.signal, `${input.label} aborted`)
  const remainingBudgetMs = input.budget?.remainingMs() ?? null
  if (remainingBudgetMs !== null && remainingBudgetMs <= 0 && !input.admitWhenExhausted) {
    input.budget?.exhaust()
    throw new AuxiliaryLlmBudgetExhaustedError(input.budget?.budgetMs ?? null)
  }
  const streaming = (input.transport ?? 'chat') !== 'chat'
  const configuredLimitMs = input.timeoutMs === undefined ? resolveAuxiliaryLlmTimeoutMs() : input.timeoutMs
  const timeoutMs = streaming
    ? configuredLimitMs
    : input.timeoutMs === undefined && !hasExplicitControlDeadlineConfiguration()
      ? resolveChatAllowanceDeadlineMs(configuredLimitMs, input.request.maxTokens)
      : resolveCallDeadline(input.timeoutMs, resolveAuxiliaryLlmTimeoutMs(), remainingBudgetMs)
  const controller = new AbortController()
  const unregisterBudget = input.budget?.register(controller, { admitWhenExhausted: input.admitWhenExhausted }) ?? (() => {})
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

  let traceRequest = input.request
  let recoveredTransport = false
  const traceCall = (response?: ChatResponse, error?: unknown): void => {
    logLlmCallTraceDetached({
      source: 'auxiliary',
      mode: 'auxiliary',
      sessionId: input.sessionId,
      provider: input.provider.name,
      model: input.request.model,
      iteration: 0,
      request: traceRequest,
      response,
      error: formatAuxiliaryTraceError(error ?? response?.raw?.auxiliaryToolTransportError),
      meta: {
        label: input.label,
        node: input.label,
        ...(recoveredTransport ? { recovery: 'required-tool-prompt' } : {}),
        ...(input.requestedThinkingLevel ? {
          requestedThinkingLevel: input.requestedThinkingLevel,
          effectiveThinkingLevel: input.request.thinkingLevel,
          compatibilityPolicy: 'auxiliary-thinking-off-fallback',
        } : {}),
      },
    })
  }

  let progressHook: (() => void) | null = null
  const chatPromise = executeAuxiliaryProviderRequest(
    { ...input, onProgress: () => progressHook?.() },
    controller,
    (error, request) => {
      traceCall(undefined, error)
      traceRequest = request
      recoveredTransport = true
    },
  )

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
    let timer: ReturnType<typeof setTimeout> | undefined
    const finish = (fn: () => void) => {
      if (settled) {
        return
      }
      settled = true
      progressHook = null
      clearTimeout(timer)
      unregisterBudget()
      finishActiveRequest()
      input.signal?.removeEventListener('abort', abortFromParent)
      fn()
    }
    const onTimeout = () => {
      const error = new AuxiliaryLlmTimeoutError(input.label, timeoutMs, streaming ? 'inactivity' : 'deadline')
      controller.abort(createAbortError(error.message))
      input.breaker?.noteExternalFailure(input.provider.id, input.request.model, error.message)
      input.budget?.exhaust(error)
      finish(() => reject(error))
    }
    const arm = () => {
      timer = setTimeout(onTimeout, timeoutMs)
      timer.unref?.()
    }
    arm()
    if (streaming) {
      progressHook = () => {
        if (settled) return
        clearTimeout(timer)
        arm()
      }
    }

    chatPromise.then(
      (response) => {
        traceCall(response)
        finish(() => resolve(response))
      },
      (error) => {
        traceCall(undefined, error)
        finish(() => reject(
          controller.signal.aborted && !input.signal?.aborted && input.budget?.isExhausted()
            ? new AuxiliaryLlmBudgetExhaustedError(input.budget.budgetMs)
            : error,
        ))
      },
    )
  })
}

const NATIVE_TOOL_SERIALIZATION_REJECTION_TTL_MS = 60 * 60_000
const MAX_NATIVE_TOOL_SERIALIZATION_REJECTIONS = 256
const nativeToolSerializationRejections = new Map<string, { at: number; reason: string }>()

function nativeToolSerializationKey(input: {
  provider: ILLMProvider
  request: ChatRequest
  sessionId?: string
}): string | null {
  if (!input.sessionId || input.request.toolChoice !== 'required' || !input.request.tools?.length) return null
  return JSON.stringify([
    input.sessionId,
    input.provider.id,
    input.request.model,
    input.request.tools.map((tool) => tool.name).sort(),
  ])
}

function recallNativeToolSerializationRejection(key: string, now = Date.now()): string | null {
  const entry = nativeToolSerializationRejections.get(key)
  if (!entry) return null
  if (now - entry.at > NATIVE_TOOL_SERIALIZATION_REJECTION_TTL_MS) {
    nativeToolSerializationRejections.delete(key)
    return null
  }
  return entry.reason
}

function rememberNativeToolSerializationRejection(key: string, reason: string, now = Date.now()): void {
  for (const [existingKey, entry] of nativeToolSerializationRejections) {
    if (now - entry.at > NATIVE_TOOL_SERIALIZATION_REJECTION_TTL_MS) nativeToolSerializationRejections.delete(existingKey)
  }
  while (nativeToolSerializationRejections.size >= MAX_NATIVE_TOOL_SERIALIZATION_REJECTIONS) {
    const oldest = nativeToolSerializationRejections.keys().next().value
    if (oldest === undefined) break
    nativeToolSerializationRejections.delete(oldest)
  }
  nativeToolSerializationRejections.set(key, { at: now, reason })
}

export function resetNativeToolSerializationRejections(): void {
  nativeToolSerializationRejections.clear()
}

function buildRequiredToolRecoveryRequest(request: ChatRequest): ChatRequest {
  return {
    ...request,
    tools: undefined,
    toolChoice: undefined,
    messages: [...request.messages, {
      role: 'user',
      content: [
        '[Internal required-tool transport recovery]',
        'The server rejected native tool serialization before producing a response. Continue the SAME assessment using the retained evidence; do not restart work.',
        'Return exactly one JSON object: {"name":"advertised tool name","arguments":{...}}. No markdown or other text.',
        'Select only an advertised tool. All original schema constraints, evidence requirements, and blocking outcomes still apply. Do not claim execution or approval merely because transport failed.',
        'The application validates this selection before its ordinary policy/approval/execution or decision handler. This response does not execute a tool.',
        JSON.stringify(request.tools),
      ].join('\n'),
    }],
  }
}

async function executeAuxiliaryProviderRequest(
  input: {
    provider: ILLMProvider
    request: ChatRequest
    breaker?: ProviderCircuitBreaker
    maxRetries?: number
    transport?: AuxiliaryLlmTransport
    sessionId?: string
    onProgress?: () => void
  },
  controller: AbortController,
  onRecovery: (error: unknown, request: ChatRequest) => void,
): Promise<ChatResponse> {
  const signal = controller.signal
  const remembered = recallObservedMaxOutputTokens(input.provider.id, input.request.model)
  if (remembered !== undefined && input.request.maxTokens !== undefined && input.request.maxTokens > remembered) {
    input = { ...input, request: { ...input.request, maxTokens: remembered } }
  }
  const chat = (request = input.request, maxRetries = input.maxRetries) => guardedProviderChat({
    provider: input.provider,
    request,
    signal,
    breaker: input.breaker,
    maxRetries,
  })
  const recoverWithEnvelope = async (
    error: unknown,
    source: 'rejected' | 'remembered',
  ): Promise<ChatResponse> => {
    const request = buildRequiredToolRecoveryRequest(input.request)
    onRecovery(error, request)
    const response = (input.transport ?? 'chat') === 'chat'
      ? await chat(request, 0)
      : await collectAuxiliaryStream({ ...input, request, maxRetries: 0, transport: 'stream', requiredControlResponse: true, onChunk: input.onProgress }, controller)
    throwIfAborted(signal)
    const tagged = { ...response, raw: {
      ...response.raw, auxiliaryToolTransportRecovery: true, auxiliaryToolTransportRecoverySource: source,
    } }
    if (response.finishReason !== 'stop') return {
      ...tagged, message: { role: 'assistant', content: '' },
    }
    try {
      const rawEnvelope = extractContent(response.message).trim()
      let envelope: unknown
      try {
        envelope = JSON.parse(rawEnvelope)
      } catch {
        envelope = JSON.parse(closeTruncatedJson(rawEnvelope))
      }
      if (!envelope || typeof envelope !== 'object' || Array.isArray(envelope)) throw new Error()
      const record = envelope as Record<string, unknown>
      if (Object.keys(record).length !== 2 || typeof record.name !== 'string'
        || !record.arguments || typeof record.arguments !== 'object' || Array.isArray(record.arguments)
        || response.message.toolCalls?.length) throw new Error()
      const tool = input.request.tools?.find(tool => tool.name === record.name)
      if (!tool) throw new Error()
      const validate = new Ajv({ strict: true }).compile(tool.inputSchema)
      if (('$async' in validate && validate.$async) || validate(record.arguments) !== true) throw new Error()
      return { ...tagged, finishReason: 'tool_use', message: {
        role: 'assistant', content: '', toolCalls: [{ id: randomUUID(), name: tool.name,
          arguments: record.arguments as Record<string, unknown> }],
      } }
    } catch {
      return { ...tagged, message: { role: 'assistant', content: '' }, raw: {
        ...tagged.raw, auxiliaryToolTransportError: 'Invalid required-tool recovery response',
      } }
    }
  }

  const memoryKey = nativeToolSerializationKey(input)
  const rememberedRejection = memoryKey ? recallNativeToolSerializationRejection(memoryKey) : null
  if (rememberedRejection !== null) {
    return await recoverWithEnvelope(
      new Error(`Native tool serialization was already rejected in this run: ${rememberedRejection}`),
      'remembered',
    )
  }
  let receivedChunk = false
  let streamed: ChatResponse
  try {
    if ((input.transport ?? 'chat') === 'chat') return await chat()
    streamed = await collectAuxiliaryStream({ ...input, onChunk: () => { receivedChunk = true; input.onProgress?.() } }, controller)
  } catch (error) {
    if (signal.aborted) throw error
    const observedLimit = receivedChunk
      ? null
      : detectProviderMaxOutputTokenLimit(error, input.request.maxTokens)
    if (observedLimit) {
      rememberObservedMaxOutputTokens(input.provider.id, input.request.model, observedLimit)
      const clamped = { ...input.request, maxTokens: observedLimit }
      return (input.transport ?? 'chat') === 'chat'
        ? await chat(clamped)
        : await collectAuxiliaryStream({ ...input, request: clamped, onChunk: input.onProgress }, controller)
    }
    const status = extractProviderErrorStatusCode(error)
    const grammarRejected = status === 400 && error instanceof Error
      && /(?:failed to parse|invalid|unsupported) grammar|grammar (?:compilation|initialization) failed/i.test(error.message)
    if (receivedChunk || input.request.toolChoice !== 'required' || !input.request.tools?.length
      || (status !== undefined && status !== 400 && status !== 422)
      || (!grammarRejected && !detectNativeToolTransportRejection(error))) throw error
    if (memoryKey) {
      rememberNativeToolSerializationRejection(
        memoryKey,
        error instanceof Error ? error.message.slice(0, 200) : 'native tool serialization rejected',
      )
    }
    return await recoverWithEnvelope(error, 'rejected')
  }
  if (
    input.transport === 'auto'
    && streamed.finishReason === 'stop'
    && !hasUsefulAuxiliaryResponse(streamed, input.request)
  ) {
    return combineRetryUsage(streamed, await chat())
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
    transport?: AuxiliaryLlmTransport
    onChunk?: () => void
    requiredControlResponse?: boolean
  },
  controller: AbortController,
): Promise<ChatResponse> {
  let text = ''
  let thinking = ''
  const toolCalls: ToolCall[] = []
  const toolCallArgs = new Map<string, string>()
  const usage: TokenUsage = { inputTokens: 0, outputTokens: 0 }
  let finishReason: ChatResponse['finishReason'] = 'stop'
  let sawDone = false
  let sawNonTerminalChunk = false
  let sawActionableChunk = false
  let lastRepetitionCheck = 0

  for await (const chunk of guardedProviderStream({
    provider: input.provider,
    request: input.request,
    signal: controller.signal,
    breaker: input.breaker,
    maxRetries: input.maxRetries,
    allowEmptyStreamFallback: input.transport === 'auto',
  })) {
    input.onChunk?.()
    if (chunk.type !== 'done') sawNonTerminalChunk = true
    switch (chunk.type) {
      case 'text':
        sawActionableChunk = true
        text += chunk.text
        break
      case 'thinking':
        thinking += chunk.text
        if ((input.requiredControlResponse || input.request.toolChoice === 'required') && !sawActionableChunk
          && thinking.length - lastRepetitionCheck >= CONTROL_REPETITION_CHECK_STEP) {
          lastRepetitionCheck = thinking.length
          const repetition = hasSustainedControlCycle(thinking)
            ? 'exact-cycle'
            : hasStagnantControlReasoning(thinking) ? 'stagnant-reasoning' : null
          if (repetition) {
            const error = new AuxiliaryLlmRepetitionError(repetition)
            controller.abort(error)
            throw error
          }
        }
        break
      case 'tool_call_start': {
        sawActionableChunk = true
        const { id, name } = chunk.toolCall
        if (!id || !name) break
        toolCalls.push({ id, name, arguments: {} })
        toolCallArgs.set(id, '')
        break
      }
      case 'tool_call_delta':
        sawActionableChunk = true
        toolCallArgs.set(
          chunk.toolCallId,
          (toolCallArgs.get(chunk.toolCallId) ?? '') + chunk.delta,
        )
        break
      case 'tool_call_end':
        sawActionableChunk = true
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
