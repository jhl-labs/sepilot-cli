import { randomUUID } from 'node:crypto'
import type {
  AgentRunContract,
  IHookRegistry,
  ILLMProvider,
  ISessionStore,
  Message,
} from '@sepilotd/core'
import { createLogger } from '../logger.js'
import { tokenCalibration } from '../providers/token-calibration.js'
import { loadSessionContext, trimMessagesToLimit } from './context-loader.js'

export { resolveSessionContextMaxMessages } from './context-loader.js'
import {
  compactSessionMessages,
  estimateMessagesTokens,
  getCompactContinuationMessage,
  type SessionCompactionConfig,
} from './session-compaction.js'

const log = createLogger('auto-compaction')

const DEFAULT_CONTEXT_WINDOW = 8000
const AUTO_COMPACT_CONTEXT_RATIO = 0.55
const AUTO_COMPACT_MIN_THRESHOLD = 3000
const DEFAULT_FIXED_OVERHEAD_TOKENS = 1500

function resolveFixedOverheadTokens(explicit?: number): number {
  if (typeof explicit === 'number' && Number.isFinite(explicit) && explicit >= 0) {
    return Math.floor(explicit)
  }
  const raw = Number(process.env.SEPILOTD_COMPACTION_FIXED_OVERHEAD_TOKENS)
  if (Number.isFinite(raw) && raw >= 0) return Math.floor(raw)
  return DEFAULT_FIXED_OVERHEAD_TOKENS
}

export async function loadObservedPromptOverheadTokens(
  sessionStore: ISessionStore,
  sessionId: string,
  charsPerToken: number,
): Promise<number | undefined> {
  const safeCharsPerToken = Math.max(0.5, charsPerToken)
  const events = await sessionStore.getEvents(sessionId)
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index]
    if (event?.type !== 'llm_request') continue
    const digest = event.requestDigest
    if (!digest || digest.auxiliary) continue
    const chars = (digest.systemPromptChars ?? 0) + (digest.toolSchemaChars ?? 0)
    if (chars > 0) return Math.ceil(chars / safeCharsPerToken)
  }
  return undefined
}

export interface AutoCompactionResult {
  messages: Message[]
  runContract?: AgentRunContract
  compacted: boolean
  beforeTokens: number
  afterTokens?: number
  removedMessageCount?: number
  wouldCompact?: boolean
}

function formatApproxTokens(tokens: number): string {
  return tokens >= 1000 ? `${(tokens / 1000).toFixed(1)}k` : String(tokens)
}

export function formatContextCompactionNotice(
  result: Pick<AutoCompactionResult, 'compacted' | 'beforeTokens' | 'afterTokens' | 'removedMessageCount'>,
): string | null {
  if (!result.compacted) return null
  const saved = result.afterTokens === undefined
    ? ''
    : ` (~${formatApproxTokens(result.beforeTokens)} → ~${formatApproxTokens(result.afterTokens)} tokens)`
  const removed = result.removedMessageCount ?? 0
  return `[context] Earlier conversation was summarized to free context space: `
    + `${removed} message${removed === 1 ? '' : 's'} folded into a summary${saved}.`
}

export interface AutoCompactionOptions {
  sessionStore: ISessionStore
  sessionId: string
  provider?: ILLMProvider
  model?: string
  hooks?: IHookRegistry
  maxMessages?: number
  tokenThreshold?: number
  config?: SessionCompactionConfig
  charsPerToken?: number
  fixedOverheadTokens?: number
  inspectOnly?: boolean
}

export async function loadAutoCompactedSessionContext(
  options: AutoCompactionOptions,
): Promise<AutoCompactionResult> {
  const runContract = await loadLatestRunContract(options.sessionStore, options.sessionId)
  const fullMessages = await loadSessionContext(
    options.sessionStore,
    options.sessionId,
    Number.MAX_SAFE_INTEGER,
  )
  const charsPerToken = options.charsPerToken
    ?? tokenCalibration.charsPerToken(options.provider?.id, options.model)
  const beforeTokens = estimateMessagesTokens(fullMessages, charsPerToken)
  const fixedOverheadTokens = options.fixedOverheadTokens
    ?? await loadObservedPromptOverheadTokens(
      options.sessionStore,
      options.sessionId,
      charsPerToken,
    )
  const tokenThreshold = options.tokenThreshold ?? resolveAutoCompactThreshold(
    options.provider,
    options.model,
    fixedOverheadTokens,
  )

  const exceedsMessageLimit = options.maxMessages !== undefined
    && fullMessages.length > options.maxMessages

  const wouldCompact = fullMessages.length >= 4
    && (beforeTokens >= tokenThreshold || exceedsMessageLimit)

  if (!wouldCompact) {
    return {
      messages: limitMessages(fullMessages, options.maxMessages),
      runContract,
      compacted: false,
      beforeTokens,
      ...(options.inspectOnly ? { wouldCompact: false } : {}),
    }
  }

  if (options.inspectOnly) {
    return {
      messages: limitMessages(fullMessages, options.maxMessages),
      runContract,
      compacted: false,
      beforeTokens,
      wouldCompact: true,
    }
  }

  const throughEventId = (await options.sessionStore.getEvents(options.sessionId)).at(-1)?.id
  const compaction = await compactSessionMessages({
    messages: fullMessages,
    provider: options.provider,
    model: options.model,
    hooks: options.hooks,
    sessionId: options.sessionId,
    config: options.config,
    charsPerToken,
  })

  if (
    !compaction
    || compaction.removedMessageCount === 0
    || (compaction.savedTokens <= 0 && !exceedsMessageLimit)
  ) {
    return {
      messages: limitMessages(fullMessages, options.maxMessages),
      runContract,
      compacted: false,
      beforeTokens,
    }
  }

  const continuationMessage: Message = {
    role: 'system',
    content: getCompactContinuationMessage(compaction.summary, {
      recentMessagesPreserved: (compaction.preservedMessages?.length ?? 0) > 0,
      suppressFollowUpQuestions: true,
    }),
  }
  const postCompactMessages = limitMessages(
    [continuationMessage, ...(compaction.preservedMessages ?? [])],
    options.maxMessages,
  )

  try {
    await options.sessionStore.appendEvent(options.sessionId, {
      type: 'context_compact',
      id: randomUUID(),
      timestamp: new Date().toISOString(),
      beforeTokens: compaction.originalTokens,
      afterTokens: compaction.compactedTokens,
      summary: compaction.summary,
      strategy: compaction.strategy,
      removedMessageCount: compaction.removedMessageCount,
      preservedMessageCount: compaction.preservedMessageCount,
      preservedMessages: compaction.preservedMessages,
      ...(throughEventId ? { throughEventId } : {}),
    })
  } catch (err) {
    log.error('failed to persist context_compact event; continuing with in-memory compaction', {
      sessionId: options.sessionId,
      error: err instanceof Error ? err.message : String(err),
    })
    return {
      messages: postCompactMessages,
      runContract,
      compacted: true,
      beforeTokens,
      afterTokens: compaction.compactedTokens,
      removedMessageCount: compaction.removedMessageCount,
    }
  }

  return {
    messages: postCompactMessages,
    runContract,
    compacted: true,
    beforeTokens,
    afterTokens: compaction.compactedTokens,
    removedMessageCount: compaction.removedMessageCount,
  }
}

export function resolveAutoCompactThreshold(
  provider: ILLMProvider | undefined,
  model: string | undefined,
  fixedOverheadTokens?: number,
): number {
  const modelInfo = model
    ? provider?.models.find((candidate) => candidate.id === model)
    : undefined
  const contextWindow = modelInfo?.contextWindow ?? DEFAULT_CONTEXT_WINDOW
  const overhead = resolveFixedOverheadTokens(fixedOverheadTokens)
  const effectiveWindow = Math.max(0, contextWindow - overhead)
  const byWindow = Math.floor(effectiveWindow * AUTO_COMPACT_CONTEXT_RATIO)
  return Math.max(AUTO_COMPACT_MIN_THRESHOLD, Math.min(byWindow, resolveAutoCompactMaxTokens()))
}

function resolveAutoCompactMaxTokens(): number {
  const raw = Number(process.env.SEPILOTD_AUTO_COMPACT_MAX_TOKENS)
  return Number.isFinite(raw) && raw > 0 ? Math.floor(raw) : Number.POSITIVE_INFINITY
}

function limitMessages(messages: Message[], maxMessages: number | undefined): Message[] {
  return trimMessagesToLimit(messages, maxMessages)
}

export async function loadLatestRunContract(
  sessionStore: ISessionStore,
  sessionId: string,
): Promise<AgentRunContract | undefined> {
  const events = await sessionStore.getEvents(sessionId)
  let latest: AgentRunContract | undefined
  let latestContractIndex = -1
  let latestUserMessageIndex = -1
  for (const [index, event] of events.entries()) {
    if (event.type === 'user_message') {
      latestUserMessageIndex = index
    }
    if (event.type === 'run_contract') {
      latest = event.contract
      latestContractIndex = index
    }
  }
  return latestContractIndex > latestUserMessageIndex ? latest : undefined
}

export async function loadLatestRoutedPersona(
  sessionStore: ISessionStore,
  sessionId: string,
): Promise<string | undefined> {
  const events = await sessionStore.getEvents(sessionId)
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index]!
    if (event.type !== 'router_decision') continue
    const decision = (event as { decision?: { persona?: string; fallback?: boolean } }).decision
    return decision?.persona && decision.fallback !== true ? decision.persona : undefined
  }
  return undefined
}

const DEFAULT_PREFLIGHT_COMPACTION_MS = 25_000
const compactionsInFlight = new Set<string>()

export function resolvePreflightCompactionBudgetMs(env: NodeJS.ProcessEnv = process.env): number {
  const raw = env.SEPILOTD_PREFLIGHT_COMPACTION_MS
  if (raw === undefined || raw.trim() === '') return DEFAULT_PREFLIGHT_COMPACTION_MS
  const parsed = Number.parseInt(raw, 10)
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : DEFAULT_PREFLIGHT_COMPACTION_MS
}

export async function loadAutoCompactedSessionContextWithinBudget(
  options: AutoCompactionOptions,
  budgetMs = resolvePreflightCompactionBudgetMs(),
): Promise<AutoCompactionResult & { deferred?: boolean }> {
  if (options.inspectOnly) return loadAutoCompactedSessionContext(options)
  if (compactionsInFlight.has(options.sessionId)) {
    return { ...(await loadAutoCompactedSessionContext({ ...options, inspectOnly: true })), deferred: true }
  }
  compactionsInFlight.add(options.sessionId)
  const compaction = loadAutoCompactedSessionContext(options)
    .finally(() => compactionsInFlight.delete(options.sessionId))
  if (budgetMs <= 0) return compaction
  let timer: ReturnType<typeof setTimeout> | undefined
  const timedOut = new Promise<'timeout'>((resolve) => { timer = setTimeout(() => resolve('timeout'), budgetMs) })
  try {
    const first = await Promise.race([compaction, timedOut])
    if (first !== 'timeout') return first
  } finally {
    if (timer) clearTimeout(timer)
  }
  compaction.catch((error) => log.warn('background compaction failed', {
    sessionId: options.sessionId,
    error: error instanceof Error ? error.message : String(error),
  }))
  log.info('compaction deferred past the preflight budget', { sessionId: options.sessionId, budgetMs })
  return { ...(await loadAutoCompactedSessionContext({ ...options, inspectOnly: true })), deferred: true }
}
