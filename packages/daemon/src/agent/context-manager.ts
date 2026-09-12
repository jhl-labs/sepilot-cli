import type { ChatRequest, ContentPart, ILLMProvider, Message, TokenUsage } from '@sepilotd/core'
import { runAuxiliaryLlmChat } from './auxiliary-llm.js'
import type { ProviderCircuitBreaker } from '../providers/circuit-breaker.js'
import { CURRENT_AGENT_TURN_USER_METADATA_KEY } from './turn-context.js'
import { coalesceAdjacentAssistantToolCallMessages } from './tool-protocol-context.js'

/**
 * Context window assumed for a model that is not in the provider catalog.
 *
 * Every layer that sizes a prompt has to answer this question, and they were
 * answering it differently (8k in the run loops, 128k in the usage inspector,
 * 192k in the OpenAI-compatible provider defaults) — so the number the user
 * was shown was not the number the engine planned against. It is deliberately
 * conservative: overflowing an unknown window is a hard provider failure,
 * while compacting early only costs a summarization call. Operators fix the
 * real number by refreshing the provider's model list or declaring
 * `defaultContextWindow`.
 */
export const DEFAULT_UNKNOWN_MODEL_CONTEXT_WINDOW = 8_000

const DEFAULT_MINIMUM_OUTPUT_TOKENS = 512
const MAX_TOOL_PROTOCOL_UNIT_TOKENS = 24_000
const TOOL_PROTOCOL_CONTEXT_SHARE = 0.2
export const DEFAULT_MAX_CURRENT_TURN_TOOL_OBSERVATIONS = 16
// A large window can hold proportionally more distinct observations before the
// working-memory projection has to fold any of them into a digest. Keep the
// small-window default as a floor so narrow models are unaffected.
const CONTEXT_TOKENS_PER_OBSERVATION = 4_096
const MAX_CURRENT_TURN_TOOL_OBSERVATIONS_CEILING = 64
// Working-memory limits bound the *prompt*: `compactOversizedToolProtocolUnits`
// already folds anything above them into a rolling digest, so exceeding them is
// a compaction event, not a reason to stop gathering evidence. Termination is a
// separate, much larger bound: a turn that has accumulated several full
// working-memory loads is no longer converging, and only then is closing the
// tool surface the honest response.
const TOOL_OBSERVATION_SYNTHESIS_BUDGET_MULTIPLIER = 4
const TOOL_OBSERVATION_DIGEST_KIND = 'tool_observation_digest'

export interface ProviderContextFitOptions {
  contextWindowTokens: number
  requestedOutputTokens: number
  modelMaxOutputTokens?: number
  minimumOutputTokens?: number
  charsPerToken?: number
  safetyMarginTokens?: number
  tools?: ChatRequest['tools']
  renderMessages?: (messages: Message[]) => Message[]
}

export interface ProviderContextFitResult {
  baseMessages: Message[]
  requestMessages: Message[]
  maxOutputTokens: number
  estimatedInputTokens: number
  droppedMessageCount: number
  outputTokensReduced: boolean
}

export interface ToolObservationCompactionOptions {
  contextWindowTokens: number
  charsPerToken?: number
  maxProtocolUnitTokens?: number
  maxDigestChars?: number
  maxObservationCount?: number
}

export interface ToolObservationBudgetEvaluation {
  observationCount: number
  limitCount: number
  estimatedTokens: number
  limitTokens: number
  /**
   * The current turn's evidence no longer fits bounded working memory, so the
   * next provider request must be served from a compacted projection. This is
   * routine context management and never by itself a reason to stop gathering.
   */
  compactionRequired: boolean
  /** Evidence kept growing past several working-memory loads without converging. */
  exceeded: boolean
  synthesisLimitCount: number
  synthesisLimitTokens: number
}

interface ToolObservation {
  call: NonNullable<Message['toolCalls']>[number]
  result: Message
  resultIndex: number
  visualIndex?: number
  assistantIndex?: number
}

function stableSerialize(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stableSerialize).join(',')}]`
  if (value && typeof value === 'object') {
    return `{${Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, entry]) => `${JSON.stringify(key)}:${stableSerialize(entry)}`)
      .join(',')}}`
  }
  return JSON.stringify(value) ?? 'null'
}

function compactText(value: string, maxChars: number): string {
  const normalized = value.replace(/\s+/g, ' ').trim()
  if (normalized.length <= maxChars) return normalized
  const marker = ' … '
  const head = Math.max(0, Math.floor((maxChars - marker.length) * 0.65))
  const tail = Math.max(0, maxChars - marker.length - head)
  return `${normalized.slice(0, head)}${marker}${normalized.slice(-tail)}`
}

function visualToolCallId(message: Message): string | null {
  if (message.role !== 'user' || !Array.isArray(message.content)) return null
  const text = message.content.find(
    (part): part is Extract<ContentPart, { type: 'text' }> => part.type === 'text',
  )?.text
  if (!text?.startsWith('[Visual output from ')) return null
  return text.match(/Tool call id:\s*([^\s.]+)/)?.[1] ?? null
}

function selectRepresentativeObservations(
  observations: ToolObservation[],
  maximum: number,
): ToolObservation[] {
  if (observations.length <= maximum) return observations
  const selected = new Set<number>([0, observations.length - 1])
  const slots = Math.max(0, maximum - selected.size)
  for (let index = 1; index <= slots; index += 1) {
    selected.add(Math.floor(index * (observations.length - 1) / (slots + 1)))
  }
  return [...selected]
    .sort((left, right) => left - right)
    .map((index) => observations[index]!)
}

function buildToolObservationDigest(
  observations: ToolObservation[],
  previousDigest: string | undefined,
  maxChars: number,
): Message {
  const counts = new Map<string, number>()
  for (const observation of observations) {
    counts.set(observation.call.name, (counts.get(observation.call.name) ?? 0) + 1)
  }
  const countSummary = [...counts.entries()]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .map(([name, count]) => `${name} x${count}`)
    .join(', ')
  const header = [
    '[Compacted tool observations]',
    `${observations.length} earlier tool result(s) were reduced to a bounded working-memory digest; raw events remain in the session journal.`,
    `Counts: ${countSummary || 'none'}.`,
  ].join(' ')
  const prefix = previousDigest
    ? `${header}\nPrevious digest: ${compactText(previousDigest, Math.floor(maxChars * 0.2))}`
    : header
  const entryBudget = Math.max(1, maxChars - prefix.length - 80)
  const maximumEntries = Math.max(2, Math.min(24, Math.floor(entryBudget / 360)))
  const representatives = selectRepresentativeObservations(observations, maximumEntries)
  const entries: string[] = []
  let used = prefix.length
  for (const observation of representatives) {
    const args = compactText(stableSerialize(observation.call.arguments ?? {}), 160)
    const output = compactText(
      typeof observation.result.content === 'string'
        ? observation.result.content
        : observation.result.content
            .map((part) => part.type === 'text' ? part.text : `[${part.type}]`)
            .join(' '),
      240,
    )
    const entry = `- ${observation.call.name} ${args} -> ${output}`
    if (used + entry.length + 1 > maxChars) break
    entries.push(entry)
    used += entry.length + 1
  }
  const unlisted = Math.max(0, observations.length - entries.length)
  const suffix = unlisted > 0
    ? `\n- ${unlisted} additional observation(s) omitted from the digest; request only a narrow missing fact if it is essential.`
    : ''
  return {
    role: 'system',
    metadata: { contextCompactionKind: TOOL_OBSERVATION_DIGEST_KIND },
    content: compactText(`${prefix}\n${entries.join('\n')}${suffix}`, maxChars),
  }
}

function toolObservationTokens(
  observation: ToolObservation,
  messages: readonly Message[],
  charsPerToken: number,
): number {
  return estimateMessageTokens({
    role: 'assistant',
    content: '',
    toolCalls: [observation.call],
  }, charsPerToken)
    + estimateMessageTokens(observation.result, charsPerToken)
    + (observation.visualIndex === undefined
      ? 0
      : estimateMessageTokens(messages[observation.visualIndex]!, charsPerToken))
}

function resolveToolObservationTokenLimit(
  contextWindowTokens: number,
  configuredLimit?: number,
): number {
  return Math.max(
    256,
    Math.floor(configuredLimit ?? Math.min(
      MAX_TOOL_PROTOCOL_UNIT_TOKENS,
      Math.max(1, contextWindowTokens) * TOOL_PROTOCOL_CONTEXT_SHARE,
    )),
  )
}

/**
 * Resolve how many distinct observations one turn may hold in working memory
 * before the projection starts folding the oldest ones into a digest. An
 * explicit operator/caller limit always wins; otherwise the limit scales with
 * the model's context window and never drops below the small-window default.
 */
export function resolveToolObservationCountLimit(
  contextWindowTokens: number,
  configuredLimit?: number,
): number {
  if (
    typeof configuredLimit === 'number'
    && Number.isFinite(configuredLimit)
    && configuredLimit >= 1
  ) {
    return Math.floor(configuredLimit)
  }
  const windowScaled = Math.floor(
    Math.max(1, contextWindowTokens) / CONTEXT_TOKENS_PER_OBSERVATION,
  )
  return Math.min(
    MAX_CURRENT_TURN_TOOL_OBSERVATIONS_CEILING,
    Math.max(DEFAULT_MAX_CURRENT_TURN_TOOL_OBSERVATIONS, windowScaled),
  )
}

/**
 * Measure the current turn's accumulated external observations independently
 * of tool names.
 *
 * Two thresholds, deliberately distinct. `compactionRequired` says the prompt
 * must now be served from a compacted projection — routine, and already
 * handled by `compactOversizedToolProtocolUnits`. `exceeded` says the turn has
 * gathered several working-memory loads without converging and closing the
 * tool surface is the honest response. Treating the first as the second turns
 * ordinary long investigations (a repository review, a multi-file trace) into
 * premature, evidence-starved synthesis.
 */
export function evaluateCurrentTurnToolObservationBudget(
  messages: readonly Message[],
  options: ToolObservationCompactionOptions,
): ToolObservationBudgetEvaluation {
  const charsPerToken = Math.max(0.5, options.charsPerToken ?? 4)
  const limitTokens = resolveToolObservationTokenLimit(
    options.contextWindowTokens,
    options.maxProtocolUnitTokens,
  )
  const limitCount = Math.max(
    1,
    resolveToolObservationCountLimit(
      options.contextWindowTokens,
      options.maxObservationCount,
    ),
  )
  let currentTurnStartIndex = -1
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]
    if (
      message?.role === 'user'
      && message.metadata?.[CURRENT_AGENT_TURN_USER_METADATA_KEY] === true
    ) {
      currentTurnStartIndex = index
      break
    }
  }
  if (currentTurnStartIndex < 0) {
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      const message = messages[index]
      if (message?.role === 'user' && !isVisualToolObservation(message)) {
        currentTurnStartIndex = index
        break
      }
    }
  }

  let observationCount = 0
  let estimatedTokens = 0
  for (let index = currentTurnStartIndex + 1; index < messages.length; index += 1) {
    const message = messages[index]!
    if (message.role === 'assistant' && message.toolCalls?.length) {
      estimatedTokens += estimateMessageTokens({ ...message, content: '' }, charsPerToken)
    } else if (message.role === 'tool') {
      observationCount += 1
      estimatedTokens += estimateMessageTokens(message, charsPerToken)
    } else if (isVisualToolObservation(message)) {
      estimatedTokens += estimateMessageTokens(message, charsPerToken)
    }
  }
  const synthesisLimitCount = limitCount * TOOL_OBSERVATION_SYNTHESIS_BUDGET_MULTIPLIER
  const synthesisLimitTokens = limitTokens * TOOL_OBSERVATION_SYNTHESIS_BUDGET_MULTIPLIER
  return {
    observationCount,
    limitCount,
    estimatedTokens,
    limitTokens,
    compactionRequired: estimatedTokens > limitTokens || observationCount >= limitCount,
    synthesisLimitCount,
    synthesisLimitTokens,
    exceeded: estimatedTokens > synthesisLimitTokens
      || observationCount >= synthesisLimitCount,
  }
}

/**
 * Project oversized assistant→tool protocol batches into bounded working
 * memory. This is deliberately tool-agnostic: any model can overproduce calls
 * and any tool can return large observations. The raw journal remains intact;
 * only the next provider request receives this compact projection.
 */
export function compactOversizedToolProtocolUnits(
  messages: readonly Message[],
  options: ToolObservationCompactionOptions,
): Message[] {
  const charsPerToken = Math.max(0.5, options.charsPerToken ?? 4)
  const maxProtocolUnitTokens = resolveToolObservationTokenLimit(
    options.contextWindowTokens,
    options.maxProtocolUnitTokens,
  )
  const maxDigestChars = Math.max(
    800,
    Math.floor(options.maxDigestChars ?? Math.min(
      12_000,
      Math.max(1, options.contextWindowTokens) * charsPerToken * 0.08,
    )),
  )
  const previousDigests = messages.filter((message) =>
    message.role === 'system'
    && message.metadata?.contextCompactionKind === TOOL_OBSERVATION_DIGEST_KIND
    && typeof message.content === 'string'
  )
  const sourceMessages = messages.filter((message) =>
    message.metadata?.contextCompactionKind !== TOOL_OBSERVATION_DIGEST_KIND
  )
  const visualIndexByCallId = new Map<string, number>()
  const resultByCallId = new Map<string, { result: Message; resultIndex: number }>()
  sourceMessages.forEach((message, index) => {
    const callId = visualToolCallId(message)
    if (callId) visualIndexByCallId.set(callId, index)
    if (message.role === 'tool' && message.toolCallId) {
      resultByCallId.set(message.toolCallId, { result: message, resultIndex: index })
    }
  })

  const observationRecords: ToolObservation[] = []
  const observationsByAssistantIndex = new Map<number, ToolObservation[]>()
  sourceMessages.forEach((message, assistantIndex) => {
    if (message.role !== 'assistant' || !message.toolCalls?.length) return
    const observations = message.toolCalls.flatMap((call): ToolObservation[] => {
      const matched = resultByCallId.get(call.id)
      return matched
        ? [{
            call,
            result: matched.result,
            resultIndex: matched.resultIndex,
            visualIndex: visualIndexByCallId.get(call.id),
            assistantIndex,
          }]
        : []
    })
    observationsByAssistantIndex.set(assistantIndex, observations)
    observationRecords.push(...observations)
  })
  let currentTurnStartIndex = -1
  for (let index = sourceMessages.length - 1; index >= 0; index -= 1) {
    const message = sourceMessages[index]
    if (
      message?.role === 'user'
      && message.metadata?.[CURRENT_AGENT_TURN_USER_METADATA_KEY] === true
    ) {
      currentTurnStartIndex = index
      break
    }
  }
  if (currentTurnStartIndex < 0) {
    for (let index = sourceMessages.length - 1; index >= 0; index -= 1) {
      const message = sourceMessages[index]
      if (message?.role === 'user' && !isVisualToolObservation(message)) {
        currentTurnStartIndex = index
        break
      }
    }
  }
  const currentTurnObservations = observationRecords.filter((observation) =>
    (observation.assistantIndex ?? -1) > currentTurnStartIndex
  )
  const currentTurnObservationTokens = currentTurnObservations.reduce(
    (sum, observation) => sum
      + toolObservationTokens(observation, sourceMessages, charsPerToken),
    0,
  )
  const globallyRetainedCallIds = new Set<string>()
  if (currentTurnObservationTokens > maxProtocolUnitTokens) {
    let retainedTokens = 0
    for (let index = currentTurnObservations.length - 1; index >= 0; index -= 1) {
      const observation = currentTurnObservations[index]!
      const cost = toolObservationTokens(observation, sourceMessages, charsPerToken)
      if (retainedTokens + cost <= maxProtocolUnitTokens) {
        globallyRetainedCallIds.add(observation.call.id)
        retainedTokens += cost
      }
    }
  }

  const omittedCallIds = new Set<string>()
  const omittedResultIndexes = new Set<number>()
  const omittedVisualIndexes = new Set<number>()
  const retainedCallsByAssistantIndex = new Map<number, NonNullable<Message['toolCalls']>>()
  const omittedObservations: ToolObservation[] = []

  sourceMessages.forEach((message, assistantIndex) => {
    if (message.role !== 'assistant' || !message.toolCalls?.length) return
    const observations = observationsByAssistantIndex.get(assistantIndex) ?? []
    const unmatchedCalls = message.toolCalls.filter((call) =>
      !observations.some((observation) => observation.call.id === call.id)
    )
    const unitTokens = estimateMessageTokens(message, charsPerToken)
      + observations.reduce(
        (sum, observation) => sum
          + estimateMessageTokens(observation.result, charsPerToken)
          + (observation.visualIndex === undefined
            ? 0
            : estimateMessageTokens(sourceMessages[observation.visualIndex]!, charsPerToken)),
        0,
      )
    const globalPressure = currentTurnObservationTokens > maxProtocolUnitTokens
      && assistantIndex > currentTurnStartIndex
    if (unitTokens <= maxProtocolUnitTokens && !globalPressure) return

    let retainedTokens = estimateMessageTokens({
      ...message,
      toolCalls: unmatchedCalls.length > 0 ? unmatchedCalls : undefined,
    }, charsPerToken)
    const retainedIds = new Set(unmatchedCalls.map((call) => call.id))
    if (globalPressure) {
      for (const observation of observations) {
        if (globallyRetainedCallIds.has(observation.call.id)) {
          retainedIds.add(observation.call.id)
        }
      }
    } else {
      for (let index = observations.length - 1; index >= 0; index -= 1) {
        const observation = observations[index]!
        const cost = toolObservationTokens(observation, sourceMessages, charsPerToken)
        if (retainedTokens + cost <= maxProtocolUnitTokens) {
          retainedIds.add(observation.call.id)
          retainedTokens += cost
        }
      }
    }
    const retainedCalls = message.toolCalls.filter((call) => retainedIds.has(call.id))
    retainedCallsByAssistantIndex.set(assistantIndex, retainedCalls)
    for (const observation of observations) {
      if (retainedIds.has(observation.call.id)) continue
      omittedCallIds.add(observation.call.id)
      omittedResultIndexes.add(observation.resultIndex)
      if (observation.visualIndex !== undefined) {
        omittedVisualIndexes.add(observation.visualIndex)
      }
      omittedObservations.push(observation)
    }
  })

  if (omittedObservations.length === 0) {
    if (previousDigests.length <= 1) return messages as Message[]
    const latestDigest = previousDigests.at(-1)!
    return messages.filter((message) =>
      message.metadata?.contextCompactionKind !== TOOL_OBSERVATION_DIGEST_KIND
      || message === latestDigest
    ) as Message[]
  }

  const digest = buildToolObservationDigest(
    omittedObservations,
    previousDigests.at(-1)?.content as string | undefined,
    maxDigestChars,
  )
  const firstAffectedIndex = Math.min(
    ...[...retainedCallsByAssistantIndex.keys()],
  )
  const projected: Message[] = []
  sourceMessages.forEach((message, index) => {
    if (index === firstAffectedIndex) projected.push(digest)
    if (omittedResultIndexes.has(index) || omittedVisualIndexes.has(index)) return
    const retainedCalls = retainedCallsByAssistantIndex.get(index)
    if (!retainedCalls) {
      projected.push(message)
      return
    }
    const retainedToolCalls = retainedCalls.filter((call) => !omittedCallIds.has(call.id))
    if (retainedToolCalls.length === 0 && !String(message.content).trim()) return
    projected.push({
      ...message,
      toolCalls: retainedToolCalls.length > 0 ? retainedToolCalls : undefined,
    })
  })
  return projected
}

export class IrreducibleContextOverflowError extends Error {
  readonly code = 'IRREDUCIBLE_CONTEXT_OVERFLOW'

  constructor(
    readonly contextWindowTokens: number,
    readonly estimatedInputTokens: number,
    readonly minimumOutputTokens: number,
  ) {
    super(
      `Required prompt context (${estimatedInputTokens} estimated tokens) plus the minimum output budget (${minimumOutputTokens}) exceeds the ${contextWindowTokens}-token model context window.`,
    )
    this.name = 'IrreducibleContextOverflowError'
  }
}

interface ConversationUnit {
  messages: Message[]
  indexes: number[]
  hasToolProtocol: boolean
  providerSafe: boolean
}

function toolCallIds(message: Message): Set<string> {
  return new Set((message.toolCalls ?? []).map((toolCall) => toolCall.id).filter(Boolean))
}

function isVisualToolObservation(message: Message): boolean {
  if (message.role !== 'user' || !Array.isArray(message.content)) return false
  const firstText = message.content.find(
    (part): part is Extract<typeof part, { type: 'text' }> => part.type === 'text',
  )?.text
  return firstText?.startsWith('[Visual output from ') === true
}

function buildConversationUnits(messages: Message[]): ConversationUnit[] {
  const units: ConversationUnit[] = []

  for (let index = 0; index < messages.length; index += 1) {
    const message = messages[index]!
    if (message.role === 'system') {
      continue
    }
    const previous = units.at(-1)
    if (message.role === 'tool') {
      const callIds = previous?.messages[0]
        ? toolCallIds(previous.messages[0])
        : new Set<string>()
      if (previous?.hasToolProtocol && message.toolCallId && callIds.has(message.toolCallId)) {
        previous.messages.push(message)
        previous.indexes.push(index)
      } else {
        units.push({
          messages: [message],
          indexes: [index],
          hasToolProtocol: false,
          providerSafe: false,
        })
      }
      continue
    }

    if (previous?.hasToolProtocol && isVisualToolObservation(message)) {
      previous.messages.push(message)
      previous.indexes.push(index)
      continue
    }

    units.push({
      messages: [message],
      indexes: [index],
      hasToolProtocol: message.role === 'assistant' && (message.toolCalls?.length ?? 0) > 0,
      providerSafe: true,
    })
  }

  return units
}

function estimateToolDefinitionTokens(
  tools: ChatRequest['tools'] | undefined,
  charsPerToken: number,
): number {
  if (!tools?.length) return 0
  return Math.ceil(JSON.stringify(tools).length / charsPerToken) + tools.length * 4
}

function estimateProviderInputTokens(
  messages: Message[],
  tools: ChatRequest['tools'] | undefined,
  charsPerToken: number,
): number {
  const messageTokens = messages.reduce(
    (sum, message) => sum + estimateMessageTokens(message, charsPerToken) + 4,
    0,
  )
  return messageTokens + estimateToolDefinitionTokens(tools, charsPerToken)
}

/**
 * Fit one concrete provider request to a context window. Selection is done on
 * the native/persisted message representation so assistant tool calls and all
 * matching tool results remain atomic. `renderMessages` then accounts for the
 * actual provider representation (for example Prompt-ReAct's tool catalog).
 */
export function fitProviderContext(
  messages: readonly Message[],
  options: ProviderContextFitOptions,
): ProviderContextFitResult {
  const contextWindowTokens = Math.max(1, Math.floor(options.contextWindowTokens))
  const charsPerToken = Math.max(0.5, options.charsPerToken ?? 4)
  const requestedOutputTokens = Math.max(
    1,
    Math.floor(Math.min(
      options.requestedOutputTokens,
      options.modelMaxOutputTokens ?? Number.POSITIVE_INFINITY,
    )),
  )
  const minimumOutputTokens = Math.max(
    1,
    Math.min(
      Math.floor(options.minimumOutputTokens ?? DEFAULT_MINIMUM_OUTPUT_TOKENS),
      requestedOutputTokens,
      Math.max(1, Math.floor(contextWindowTokens * 0.25)),
    ),
  )
  const safetyMarginTokens = Math.max(
    0,
    Math.floor(options.safetyMarginTokens
      ?? Math.max(128, contextWindowTokens * 0.02)),
  )
  const renderMessages = options.renderMessages ?? ((selected: Message[]) => selected)
  const allMessages = compactOversizedToolProtocolUnits(
    coalesceAdjacentAssistantToolCallMessages(messages),
    { contextWindowTokens, charsPerToken },
  )
  const units = buildConversationUnits(allMessages)

  const evaluate = (selectedUnits: Set<number>) => {
    const selectedMessageIndexes = new Set(
      units.flatMap((unit, index) => selectedUnits.has(index) ? unit.indexes : []),
    )
    const baseMessages = allMessages.filter((message, index) =>
      message.role === 'system' || selectedMessageIndexes.has(index),
    )
    const requestMessages = renderMessages(baseMessages)
    const estimatedInputTokens = estimateProviderInputTokens(
      requestMessages,
      options.tools,
      charsPerToken,
    )
    return { baseMessages, requestMessages, estimatedInputTokens }
  }

  const allUnits = new Set(units.map((_, index) => index))
  const complete = evaluate(allUnits)
  if (
    complete.estimatedInputTokens
      + requestedOutputTokens
      + safetyMarginTokens
    <= contextWindowTokens
  ) {
    return {
      ...complete,
      maxOutputTokens: requestedOutputTokens,
      droppedMessageCount: 0,
      outputTokensReduced: false,
    }
  }

  let currentConversationIndex = -1
  for (let index = allMessages.length - 1; index >= 0; index -= 1) {
    const message = allMessages[index]
    if (
      message?.role === 'user'
      && message.metadata?.[CURRENT_AGENT_TURN_USER_METADATA_KEY] === true
    ) {
      currentConversationIndex = index
      break
    }
  }
  if (currentConversationIndex < 0) {
    for (let index = allMessages.length - 1; index >= 0; index -= 1) {
      if (allMessages[index]?.role === 'user') {
        currentConversationIndex = index
        break
      }
    }
  }

  const currentUnitIndex = units.findIndex(
    (unit) => unit.indexes.includes(currentConversationIndex),
  )
  let latestProtocolUnitIndex = -1
  for (let index = units.length - 1; index >= 0; index -= 1) {
    const unit = units[index]
    if (
      unit?.hasToolProtocol
      && (currentUnitIndex < 0 || index >= currentUnitIndex)
    ) {
      latestProtocolUnitIndex = index
      break
    }
  }
  let latestSafeUnitIndex = -1
  for (let index = units.length - 1; index >= 0; index -= 1) {
    if (units[index]?.providerSafe) {
      latestSafeUnitIndex = index
      break
    }
  }

  const selectedUnits = new Set<number>()
  for (const index of [currentUnitIndex, latestProtocolUnitIndex, latestSafeUnitIndex]) {
    if (index >= 0) selectedUnits.add(index)
  }

  const required = evaluate(selectedUnits)
  const maximumOutputTokens = Math.floor(
    contextWindowTokens - required.estimatedInputTokens - safetyMarginTokens,
  )
  if (maximumOutputTokens < minimumOutputTokens) {
    throw new IrreducibleContextOverflowError(
      contextWindowTokens,
      required.estimatedInputTokens + safetyMarginTokens,
      minimumOutputTokens,
    )
  }

  // Older units are optional. Add them newest-first only while the requested
  // output budget still fits; required current-turn units may instead consume
  // part of that output reserve, but never exceed the absolute context limit.
  for (let index = units.length - 1; index >= 0; index -= 1) {
    if (selectedUnits.has(index) || !units[index]?.providerSafe) continue
    const candidateUnits = new Set(selectedUnits)
    candidateUnits.add(index)
    const candidate = evaluate(candidateUnits)
    if (
      candidate.estimatedInputTokens
        + requestedOutputTokens
        + safetyMarginTokens
      <= contextWindowTokens
    ) {
      selectedUnits.add(index)
    }
  }

  const fitted = evaluate(selectedUnits)
  const maxOutputTokens = Math.min(
    requestedOutputTokens,
    Math.floor(contextWindowTokens - fitted.estimatedInputTokens - safetyMarginTokens),
  )
  if (maxOutputTokens < minimumOutputTokens) {
    throw new IrreducibleContextOverflowError(
      contextWindowTokens,
      fitted.estimatedInputTokens + safetyMarginTokens,
      minimumOutputTokens,
    )
  }

  return {
    ...fitted,
    maxOutputTokens,
    droppedMessageCount: allMessages.length - fitted.baseMessages.length,
    outputTokensReduced: maxOutputTokens < requestedOutputTokens,
  }
}

/**
 * Trim messages to fit within a model's context window.
 * Keeps system prompt + last N messages that fit.
 * Estimates 4 chars ≈ 1 token unless a calibrated ratio is supplied.
 */
export function trimToContextWindow(
  messages: Message[],
  maxTokens: number,
  reserveForOutput: number = 4096,
  charsPerToken: number = 4,
): Message[] {
  return fitProviderContext(messages, {
    contextWindowTokens: maxTokens,
    requestedOutputTokens: reserveForOutput,
    charsPerToken,
    safetyMarginTokens: 0,
  }).baseMessages
}

export function supersedeStaleReadResults(messages: Message[]): Message[] {
  interface ReadScope {
    path: string
    encoding: string
    lineNumbers: boolean
    start: number
    end: number
  }
  const positiveInt = (value: unknown, fallback: number): number => (
    typeof value === 'number' && Number.isSafeInteger(value) && value > 0
      ? value
      : fallback
  )
  const readScopeByToolCallId = new Map<string, ReadScope>()
  for (const msg of messages) {
    if (msg.role !== 'assistant') continue
    for (const toolCall of msg.toolCalls ?? []) {
      if (toolCall.name !== 'fs.read') continue
      const path = toolCall.arguments?.path
      if (typeof path === 'string' && path.trim()) {
        const start = positiveInt(toolCall.arguments?.offset, 1)
        const limit = positiveInt(toolCall.arguments?.limit, 200)
        readScopeByToolCallId.set(toolCall.id, {
          path,
          encoding: String(toolCall.arguments?.encoding ?? 'utf-8').toLowerCase(),
          lineNumbers: toolCall.arguments?.lineNumbers !== false,
          start,
          end: start + limit - 1,
        })
      }
    }
  }

  const toolResultScopes = messages.map((msg) =>
    msg.role === 'tool' && msg.toolCallId
      && msg.metadata?.toolResultStatus === 'success'
      && msg.metadata?.observationReuse !== true
      ? readScopeByToolCallId.get(msg.toolCallId) ?? null
      : null,
  )

  let changed = false
  const next = messages.map((msg, index) => {
    const scope = toolResultScopes[index]
    if (!scope) return msg
    const coveredByLaterRead = toolResultScopes.some((candidate, candidateIndex) => (
      candidateIndex > index
      && candidate !== null
      && candidate.path === scope.path
      && candidate.encoding === scope.encoding
      && candidate.lineNumbers === scope.lineNumbers
      && candidate.start <= scope.start
      && candidate.end >= scope.end
    ))
    if (!coveredByLaterRead) return msg
    changed = true
    return {
      ...msg,
      content: `[fs.read superseded: ${scope.path} lines ${scope.start}-${scope.end} were covered by a later read in this conversation]`,
    }
  })

  return changed ? next : messages
}

/**
 * Supersede stale recurring system reminders. Reminder push sites tag their
 * system message with `metadata.reminderKind` (a structural tag, e.g.
 * 'stuck' | 'backtrack' | 'completion_gate' | 'failed_attempt' | 'escalation').
 * These reminders are trim-exempt (role === 'system') so, over a long run,
 * every re-emission would accumulate unbounded. Here we keep only the most
 * recent message per kind and drop earlier duplicates. Untagged system
 * messages (static system prompt, board message, etc.) are left untouched.
 * Judgement is by tag + order only — no content matching. Only string-content
 * system messages can carry a reminderKind; non-string content is ignored.
 */
export function supersedeStaleSystemReminders<
  T extends { role: string; content: string | ContentPart[]; metadata?: Record<string, unknown> },
>(messages: T[]): T[] {
  const reminderKindOf = (m: T): string | undefined => {
    if (m.role !== 'system' || typeof m.content !== 'string') return undefined
    const kind = m.metadata?.reminderKind
    return typeof kind === 'string' && kind ? kind : undefined
  }

  const lastIndexByKind = new Map<string, number>()
  messages.forEach((m, i) => {
    const kind = reminderKindOf(m)
    if (kind) lastIndexByKind.set(kind, i)
  })

  return messages.filter((m, i) => {
    const kind = reminderKindOf(m)
    if (!kind) return true
    return lastIndexByKind.get(kind) === i
  })
}

export function estimateMessageTokens(msg: Message, charsPerToken: number = 4): number {
  const toolCallTokens = (msg.toolCalls ?? []).reduce(
    (sum, toolCall) => sum + Math.ceil(
      `${toolCall.name} ${JSON.stringify(toolCall.arguments ?? {})}`.length / charsPerToken,
    ),
    0,
  )
  if (typeof msg.content === 'string') {
    return Math.ceil(msg.content.length / charsPerToken) + toolCallTokens
  }
  if (Array.isArray(msg.content)) {
    return toolCallTokens + msg.content.reduce((sum, part) => {
      if (part.type === 'text') return sum + Math.ceil(part.text.length / charsPerToken)
      const minimumTokens = part.type === 'image' ? 1_024 : 2_048
      if (part.source.type !== 'base64') {
        return sum + minimumTokens
      }
      const padding = part.source.data.endsWith('==')
        ? 2
        : part.source.data.endsWith('=')
          ? 1
          : 0
      const estimatedBytes = Math.max(
        0,
        Math.floor(part.source.data.length * 3 / 4) - padding,
      )
      const sizeBasedTokens = Math.ceil(
        estimatedBytes / (part.type === 'image' ? 256 : 128),
      )
      const maximumTokens = part.type === 'image' ? 16_384 : 32_768
      return sum + Math.min(maximumTokens, Math.max(minimumTokens, sizeBasedTokens))
    }, 0)
  }
  return 10 + toolCallTokens
}

/**
 * Result of a semantic compression pass. `messages` is the new working
 * conversation: system messages + a single `[Compressed history]`
 * synthetic system message + the preserved tail. `summary` is the dense
 * paragraph the LLM produced; callers should stash it on AgentState so
 * future passes can extend rather than re-summarize. `upToIndex` is the
 * number of non-system conversation messages folded in so far.
 */
export interface SemanticCompressionResult {
  messages: Message[]
  summary: string
  upToIndex: number
  /**
   * Tokens consumed by the summarization round-trip itself. Callers
   * should fold this into the run's `totalUsage` so the cost gate sees
   * compression cost — otherwise frequent recompression on long-horizon
   * runs would be invisible to the budget.
   */
  usage?: TokenUsage
}

export interface SemanticCompressionOptions {
  /** Model id for the summarization LLM call. */
  model: string
  /** Tokens reserved for output during the *target* run. Default 4096. */
  reserveForOutput?: number
  /** How many recent messages to keep verbatim. Default 6. */
  preservedTailCount?: number
  /** Minimum messages worth summarizing on the first pass. Default 4. */
  minNewMessagesToCompress?: number
  /** Token budget for the summarization call itself. Default 400. */
  summarizationMaxTokens?: number
  /** Optional abort signal for the LLM call. */
  signal?: AbortSignal
  /** Provider circuit breaker so a provider outage fast-fails compression. */
  breaker?: ProviderCircuitBreaker
  /** Shared optional-LLM wall-clock budget for the current agent turn. */
  auxiliaryLlmBudget?: import('./auxiliary-llm.js').AuxiliaryLlmTurnBudget
  /** Calibrated chars-per-token ratio for estimation. Default 4. */
  charsPerToken?: number
  /**
   * Previously-stored compressed summary (rolling). When present the
   * pass extends this summary with whatever new messages have arrived
   * since `previousUpToIndex`.
   */
  previousSummary?: string
  /** Number of conversation messages already folded into previousSummary. */
  previousUpToIndex?: number
  /**
   * Maximum allowed length (in characters) for the rolling summary. When
   * `previousSummary` exceeds this on entry, the function runs a second
   * LLM pass that re-summarizes it down to a leaner paragraph before
   * folding new messages in. Prevents the summary from drifting toward
   * the model's full context window on long-horizon runs.
   *
   * Default 4000 chars (~1000 tokens). Set to 0 to disable.
   */
  maxPreviousSummaryChars?: number
}

/**
 * Semantic compression: when raw trimming would drop older messages,
 * call the LLM once to produce a dense paragraph that captures goal,
 * decisions, codebase map, files touched, validation, errors, and the
 * open question. The result
 * replaces the dropped tail; the most recent N messages are preserved
 * verbatim. Returns null when no compression was warranted (already
 * fits, or too few new messages to be worth a round-trip).
 */
export async function semanticCompress(
  messages: Message[],
  contextWindowTokens: number,
  provider: ILLMProvider,
  options: SemanticCompressionOptions,
): Promise<SemanticCompressionResult | null> {
  const reserve = options.reserveForOutput ?? 4096
  const budget = contextWindowTokens - reserve
  if (budget <= 0) return null

  const cpt = options.charsPerToken ?? 4
  const total = messages.reduce((sum, m) => sum + estimateMessageTokens(m, cpt), 0)
  if (total <= budget) return null

  const systemMsgs = messages.filter((m) => m.role === 'system')
  const convMsgs = messages.filter((m) => m.role !== 'system')

  const tailCount = options.preservedTailCount ?? 6
  if (convMsgs.length <= tailCount) return null

  const tailStartIndex = Math.max(0, convMsgs.length - tailCount)
  const currentTurnStartIndex = convMsgs.findIndex((message) =>
    message.role === 'user'
    && message.metadata?.[CURRENT_AGENT_TURN_USER_METADATA_KEY] === true,
  )
  const preservedStartIndex = currentTurnStartIndex >= 0
    ? Math.min(tailStartIndex, currentTurnStartIndex)
    : tailStartIndex
  const preservedTail = convMsgs.slice(preservedStartIndex)
  const toSummarize = convMsgs.slice(0, preservedStartIndex)

  let prevSummary = options.previousSummary ?? null
  const prevUpToIndex = options.previousUpToIndex ?? 0
  const newToSummarize = toSummarize.slice(prevUpToIndex)

  const minNew = options.minNewMessagesToCompress ?? 4
  if (newToSummarize.length === 0) return null
  if (!prevSummary && newToSummarize.length < minNew) return null

  // Rolling re-compression: when the previous summary itself has grown
  // past `maxPreviousSummaryChars`, fold it down before adding new
  // messages — otherwise it bloats every later compaction call and
  // eventually pushes the working window back over budget.
  let recompressionUsage: TokenUsage | undefined
  const maxSummaryChars = options.maxPreviousSummaryChars ?? 4000
  if (
    prevSummary
    && maxSummaryChars > 0
    && prevSummary.length > maxSummaryChars
  ) {
    const recompressed = await recompressSummary(prevSummary, provider, options)
    if (recompressed) {
      prevSummary = recompressed.summary
      recompressionUsage = recompressed.usage
    }
  }

  const sys = [
    'You are a strict conversation compactor.',
    'Output a single dense paragraph (under 250 words) that captures:',
    "the user's goal, key decisions made, files/symbols touched, errors encountered,",
    'codebase map/scout findings, validation commands/results, unresolved assumptions,',
    'and the current open question. Drop pleasantries. Do not invent.',
  ].join(' ')

  const userParts = [
    prevSummary ? `Earlier summary so far:\n${prevSummary}` : '',
    newToSummarize.length > 0
      ? `New messages to fold in:\n${formatMessagesForCompression(newToSummarize)}`
      : '',
  ].filter(Boolean)
  if (userParts.length === 0) return null

  const request: ChatRequest = {
    model: options.model,
    messages: [
      { role: 'system', content: sys },
      { role: 'user', content: userParts.join('\n\n') },
    ],
    temperature: 0.3,
    maxTokens: options.summarizationMaxTokens ?? 400,
  }

  let summary = ''
  let usage: TokenUsage | undefined
  try {
    const response = await runAuxiliaryLlmChat({
      provider,
      request,
      label: 'Semantic compression',
      signal: options.signal,
      breaker: options.breaker,
      budget: options.auxiliaryLlmBudget,
    })
    const content = response.message?.content
    if (typeof content === 'string') {
      summary = content.trim()
    } else if (Array.isArray(content)) {
      summary = content
        .map((part) => (part.type === 'text' ? part.text : ''))
        .join('')
        .trim()
    }
    if (response.usage) {
      usage = {
        inputTokens: response.usage.inputTokens ?? 0,
        outputTokens: response.usage.outputTokens ?? 0,
      }
    }
  } catch {
    return null
  }
  if (!summary) return null

  const summaryMsg: Message = {
    role: 'system',
    content: `[Compressed history] ${summary}`,
  }

  // Combine the recompression cost (if any) with the new fold pass so
  // the caller's totalUsage sees both round-trips.
  const combinedUsage =
    recompressionUsage || usage
      ? {
          inputTokens: (recompressionUsage?.inputTokens ?? 0) + (usage?.inputTokens ?? 0),
          outputTokens: (recompressionUsage?.outputTokens ?? 0) + (usage?.outputTokens ?? 0),
        }
      : undefined

  return {
    messages: [...systemMsgs, summaryMsg, ...preservedTail],
    summary,
    upToIndex: prevUpToIndex + newToSummarize.length,
    usage: combinedUsage,
  }
}

export interface EmergencyRecoveryOptions {
  model: string
  charsPerToken?: number
  signal?: AbortSignal
  previousSummary?: string
  previousUpToIndex?: number
  auxiliaryLlmBudget?: import('./auxiliary-llm.js').AuxiliaryLlmTurnBudget
}

/**
 * Last-resort recovery after a provider rejected a prompt as too large.
 * Uses a shrunken window because the declared provider context was optimistic.
 */
export async function emergencyContextRecovery(
  messages: Message[],
  contextWindowTokens: number,
  provider: ILLMProvider,
  options: EmergencyRecoveryOptions,
): Promise<{
  messages: Message[]
  effectiveContextWindowTokens: number
  summary?: string
  upToIndex?: number
}> {
  const shrunken = Math.max(
    1,
    Math.min(contextWindowTokens, Math.max(2000, Math.floor(contextWindowTokens * 0.6))),
  )
  try {
    const compressed = await semanticCompress(messages, shrunken, provider, {
      model: options.model,
      signal: options.signal,
      auxiliaryLlmBudget: options.auxiliaryLlmBudget,
      previousSummary: options.previousSummary,
      previousUpToIndex: options.previousUpToIndex,
      preservedTailCount: 4,
      charsPerToken: options.charsPerToken,
    })
    if (compressed) {
      return {
        messages: compressed.messages,
        effectiveContextWindowTokens: shrunken,
        summary: compressed.summary,
        upToIndex: compressed.upToIndex,
      }
    }
  } catch {
    // Keep provider context overflow from turning recovery into a hard failure.
  }
  return {
    messages: trimToContextWindow(
      supersedeStaleReadResults(messages),
      shrunken,
      4096,
      options.charsPerToken ?? 4,
    ),
    effectiveContextWindowTokens: shrunken,
  }
}

async function recompressSummary(
  previous: string,
  provider: ILLMProvider,
  options: SemanticCompressionOptions,
): Promise<{ summary: string; usage?: TokenUsage } | null> {
  const sys = [
    'You are a strict summary compactor.',
    'Re-summarize the input below into a single paragraph (under 200 words)',
    "that preserves the user's goal, key decisions, files/symbols touched,",
    'codebase map/scout findings, validation commands/results, unresolved assumptions,',
    'errors, and the open question. Drop redundant detail. Do not invent.',
  ].join(' ')

  const request: ChatRequest = {
    model: options.model,
    messages: [
      { role: 'system', content: sys },
      { role: 'user', content: previous },
    ],
    temperature: 0.3,
    maxTokens: Math.max(200, Math.floor((options.summarizationMaxTokens ?? 400) / 2)),
  }

  try {
    const response = await runAuxiliaryLlmChat({
      provider,
      request,
      label: 'Summary recompression',
      signal: options.signal,
      breaker: options.breaker,
      budget: options.auxiliaryLlmBudget,
    })
    const content = response.message?.content
    let summary = ''
    if (typeof content === 'string') {
      summary = content.trim()
    } else if (Array.isArray(content)) {
      summary = content
        .map((part) => (part.type === 'text' ? part.text : ''))
        .join('')
        .trim()
    }
    if (!summary) return null
    return {
      summary,
      usage: response.usage
        ? {
            inputTokens: response.usage.inputTokens ?? 0,
            outputTokens: response.usage.outputTokens ?? 0,
          }
        : undefined,
    }
  } catch {
    return null
  }
}

function formatMessagesForCompression(messages: Message[]): string {
  const out: string[] = []
  for (const m of messages) {
    let body = ''
    if (typeof m.content === 'string') {
      body = m.content
    } else if (Array.isArray(m.content)) {
      body = m.content
        .map((part) => (part.type === 'text' ? part.text : `[${part.type}]`))
        .join('')
    }
    body = body.replace(/\s+/g, ' ').trim()
    if (body.length > 600) body = body.slice(0, 600) + '…'
    out.push(`- ${m.role}: ${body}`)
  }
  return out.join('\n')
}
