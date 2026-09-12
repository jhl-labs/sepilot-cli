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

// Re-exported so session entrypoints can size their context windows without
// importing the raw context loader (see session-context-guard test).
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
// The token estimate (`beforeTokens`) only counts conversation messages, but
// the request actually sent to the model also carries the system prompt, tool
// schemas and injected context (memory/state board). If the threshold ignores
// that fixed overhead, compaction fires too late and the real request overflows
// the window. Subtract a conservative overhead estimate from the effective
// window before applying the ratio. Operator-tunable; defaults to a safe
// (earlier-firing) value.
const DEFAULT_FIXED_OVERHEAD_TOKENS = 1500

function resolveFixedOverheadTokens(explicit?: number): number {
  if (typeof explicit === 'number' && Number.isFinite(explicit) && explicit >= 0) {
    return Math.floor(explicit)
  }
  const raw = Number(process.env.SEPILOTD_COMPACTION_FIXED_OVERHEAD_TOKENS)
  if (Number.isFinite(raw) && raw >= 0) return Math.floor(raw)
  return DEFAULT_FIXED_OVERHEAD_TOKENS
}

/**
 * Real prompt overhead observed on this session's latest non-auxiliary
 * provider request.
 *
 * The fixed default is a guess, and it is wrong by an order of magnitude on
 * the transports that matter: a prompt-react system prompt renders every tool
 * manual inline (tens of thousands of characters), and native transport moves
 * the same weight into tool schemas beside the messages. Both halves are
 * already measured on every `llm_request` event, so the threshold can use the
 * session's own observed size instead of a constant. Returns undefined for a
 * session that has not issued a main-loop request yet.
 */
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
    // Auxiliary calls (routers, judges, summarizers) carry a small bespoke
    // prompt that says nothing about the main loop's overhead.
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
  /** Messages folded into the summary. Present only when `compacted`. */
  removedMessageCount?: number
  /**
   * Inspect mode only: whether a real load would have compacted. Lets a
   * read-only inspector report the same decision without causing it.
   */
  wouldCompact?: boolean
}

function formatApproxTokens(tokens: number): string {
  return tokens >= 1000 ? `${(tokens / 1000).toFixed(1)}k` : String(tokens)
}

/**
 * User-facing notice for a compaction that already happened.
 *
 * Replacing the conversation with a summary changes what the assistant can
 * recall, so it is not an implementation detail the user should have to infer
 * from a degraded answer. Returns null when nothing was compacted.
 */
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
  /**
   * Estimated fixed prompt overhead (system prompt + tool schemas + injected
   * context) in tokens, subtracted from the effective window before the
   * compaction ratio is applied. Callers that know their real overhead should
   * pass it; otherwise a safe env-tunable default is used.
   */
  fixedOverheadTokens?: number
  /**
   * Report the compaction decision without performing it: no summarization
   * call, no `context_compact` event. Inspectors must not mutate the session
   * they are describing.
   */
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
  // Estimating with a flat 4 chars/token silently overstates the remaining
  // window for anything that does not tokenize like English prose, so the
  // threshold is reached long after the real request is already too big.
  // The calibrator learns the real ratio from provider usage; use it unless
  // the caller has a better number.
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

  // A message-count overflow is a compaction trigger too: silently slicing
  // the head off the context loses the user's goal and earlier decisions
  // without leaving a summary behind.
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

  const compaction = await compactSessionMessages({
    messages: fullMessages,
    provider: options.provider,
    model: options.model,
    hooks: options.hooks,
    sessionId: options.sessionId,
    config: options.config,
    charsPerToken,
  })

  // When triggered by token pressure the compaction must actually save
  // tokens. When triggered by a message-count overflow, shrinking the
  // message list with a summary is the goal even if the summary text costs
  // as many tokens as the short messages it replaced.
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

  // Build the post-compact in-memory message list FIRST. This
  // mirrors what loadSessionContext does when it encounters a
  // context_compact event:
  //   - drop everything before the compaction
  //   - push a system message with the continuation summary
  //   - push preservedMessages (the small recent tail)
  // Doing this in memory before appendEvent means: if the daemon
  // restarts or appendEvent fails partway, the agent loop returns
  // a coherent message list and the compaction simply replays on
  // the next call. The previous code appended first and re-loaded
  // from disk; an append-success-then-reload-fail would leave the
  // user's pending message floating with no in-memory context.
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
    })
  } catch (err) {
    // Persistence failed but the in-memory result is still
    // coherent — the agent loop can keep going on this turn and
    // the next turn will retry the compaction. Surface the
    // failure so an operator notices the disk problem rather than
    // letting the session quietly skip persistence forever.
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
  // Exact match only. `models[0]` is whatever order the provider returned —
  // alphabetical for Ollama, so an embedding model can sort first — and
  // borrowing an unrelated model's window sizes the threshold for a model that
  // is not running. An unknown model gets the conservative default, which
  // compacts early rather than overflowing.
  const contextWindow = modelInfo?.contextWindow ?? DEFAULT_CONTEXT_WINDOW
  const overhead = resolveFixedOverheadTokens(fixedOverheadTokens)
  const effectiveWindow = Math.max(0, contextWindow - overhead)
  return Math.max(
    AUTO_COMPACT_MIN_THRESHOLD,
    Math.floor(effectiveWindow * AUTO_COMPACT_CONTEXT_RATIO),
  )
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
  // A contract is renewed after the user message of every contracted turn.
  // If a newer user turn completed without emitting one (for example an
  // instant memory/scheduler action), the older task contract is no longer the
  // active task and must not leak into subsequent requests.
  return latestContractIndex > latestUserMessageIndex ? latest : undefined
}
