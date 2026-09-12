import { createLogger } from '../logger.js'

const log = createLogger('channel-intent')

export type ChannelFollowupIntent =
  | { category: 'status' }
  | { category: 'cancel' }
  | { category: 'duplicate' }
  | { category: 'correction'; note?: string }
  | { category: 'parallel_new' }

export type ChannelFollowupSource = 'command' | 'llm' | 'cache' | 'fallback'

export interface ChannelFollowupResolution {
  intent: ChannelFollowupIntent
  source: ChannelFollowupSource
  confidence?: number
  latencyMs?: number
}

export interface IntentClassifierActiveRunSnapshot {
  runId: string
  kind: 'main' | 'fork'
  userPrompt: string
  phase: 'processing' | 'waiting_approval' | 'waiting_question'
  elapsedMs: number
}

export interface IntentClassifierInput {
  chatKey: string
  newMessageText: string
  activeRuns: ReadonlyArray<IntentClassifierActiveRunSnapshot>
  recentTurns?: ReadonlyArray<string>
}

export interface IntentClassifierProviderModel {
  id: string
}

export interface IntentClassifierProvider {
  models: ReadonlyArray<IntentClassifierProviderModel>
  chat(
    request: {
      model: string
      temperature?: number
      maxTokens?: number
      timeoutMs?: number
      messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>
    },
    options?: { signal?: AbortSignal },
  ): Promise<{ message: { content: unknown } }>
}

export interface IntentClassifierConfig {
  enabled: boolean
  preferredModelId?: string
  timeoutMs: number
  rateLimitPerMin: number
  cacheTtlMs: number
  maxLlmInputChars: number
  maxOutputTokens: number
  defaultBias: 'parallel_new' | 'correction'
  // Fail-safe verdict for classifier failures (disabled/rate-limit/no-provider/
  // timeout/parse-fail). Must NOT be parallel_new: a failed classify must never
  // silently spawn a mutating second run against the same task. 'status' keeps
  // the active run and only acknowledges — the safe, no-new-work direction.
  failSafeBias: 'status' | 'correction'
}

export const DEFAULT_INTENT_CLASSIFIER_CONFIG: IntentClassifierConfig = {
  enabled: true,
  timeoutMs: 3000,
  rateLimitPerMin: 10,
  cacheTtlMs: 60_000,
  maxLlmInputChars: 1600,
  maxOutputTokens: 80,
  defaultBias: 'parallel_new',
  failSafeBias: 'status',
}

export interface IntentClassifierLogger {
  recordSkip?: (input: { chatKey: string; reason: 'rate_limited' | 'no_provider' | 'disabled' }) => void
  recordError?: (input: { chatKey: string; reason: 'timeout' | 'parse' | 'thrown'; errorName?: string }) => void
}

// Explicit slash command, not intent guessing: `/parallel ...`, `/new ...`,
// `/async ...`, `/fork ...` mean "run this as a separate parallel task". This
// is parsed (like `/approve`, `/session`), not heuristically inferred — the
// natural-language case ("아 그리고 미국 주식도 분석해줘") is the LLM classifier's
// job, never a regex.
const PARALLEL_COMMAND_PATTERN = /^(\/parallel|\/new|\/async|\/fork)\s+/i

export function explicitParallelCommand(text: string): { matched: boolean; remainder?: string } {
  const trimmed = text.trim()
  const match = trimmed.match(PARALLEL_COMMAND_PATTERN)
  if (!match) return { matched: false }
  return {
    matched: true,
    remainder: trimmed.slice(match[0].length).trim(),
  }
}

interface CacheEntry {
  intent: ChannelFollowupIntent
  expiresAt: number
}

interface RateLimitState {
  timestamps: number[]
}

export class ChannelIntentClassifier {
  private readonly cache = new Map<string, CacheEntry>()
  private readonly rateLimit = new Map<string, RateLimitState>()
  private readonly clock: () => number

  constructor(
    private readonly selectProvider: () => IntentClassifierProvider | undefined,
    private readonly config: IntentClassifierConfig = DEFAULT_INTENT_CLASSIFIER_CONFIG,
    private readonly observer: IntentClassifierLogger = {},
    clock: () => number = Date.now,
  ) {
    this.clock = clock
  }

  async classify(input: IntentClassifierInput): Promise<ChannelFollowupResolution> {
    if (!this.config.enabled) {
      this.observer.recordSkip?.({ chatKey: input.chatKey, reason: 'disabled' })
      return { intent: { category: this.config.failSafeBias }, source: 'fallback' }
    }

    const cacheKey = this.buildCacheKey(input)
    const cached = this.cache.get(cacheKey)
    if (cached && cached.expiresAt > this.clock()) {
      return { intent: cached.intent, source: 'cache' }
    }

    if (this.isRateLimited(input.chatKey)) {
      this.observer.recordSkip?.({ chatKey: input.chatKey, reason: 'rate_limited' })
      return { intent: { category: this.config.failSafeBias }, source: 'fallback' }
    }

    const provider = this.selectProvider()
    if (!provider) {
      this.observer.recordSkip?.({ chatKey: input.chatKey, reason: 'no_provider' })
      return { intent: { category: this.config.failSafeBias }, source: 'fallback' }
    }

    const start = this.clock()
    const llmIntent = await this.callLLM(provider, input)
    const latencyMs = this.clock() - start

    if (!llmIntent) {
      return { intent: { category: this.config.failSafeBias }, source: 'fallback', latencyMs }
    }

    this.cache.set(cacheKey, {
      intent: llmIntent,
      expiresAt: this.clock() + this.config.cacheTtlMs,
    })
    this.recordRateLimitHit(input.chatKey)

    return { intent: llmIntent, source: 'llm', latencyMs }
  }

  private async callLLM(
    provider: IntentClassifierProvider,
    input: IntentClassifierInput,
  ): Promise<ChannelFollowupIntent | null> {
    const modelId = this.config.preferredModelId ?? provider.models[0]?.id
    if (!modelId) {
      this.observer.recordSkip?.({ chatKey: input.chatKey, reason: 'no_provider' })
      return null
    }

    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), this.config.timeoutMs)

    try {
      const userContext = this.buildUserContext(input)
      const response = await provider.chat({
        model: modelId,
        temperature: 0,
        maxTokens: this.config.maxOutputTokens,
        timeoutMs: this.config.timeoutMs,
        messages: [
          { role: 'system', content: SYSTEM_PROMPT },
          { role: 'user', content: userContext },
        ],
      }, { signal: controller.signal })
      const parsed = parseClassifierJson(response.message.content)
      if (!parsed) {
        this.observer.recordError?.({ chatKey: input.chatKey, reason: 'parse' })
        return null
      }
      return parsed
    } catch (error) {
      const code = (error as { name?: string })?.name ?? 'Error'
      const reason = code === 'AbortError' ? 'timeout' : 'thrown'
      this.observer.recordError?.({ chatKey: input.chatKey, reason, errorName: code })
      log.debug('Intent classifier LLM call failed', { errorName: code })
      return null
    } finally {
      clearTimeout(timer)
    }
  }

  private buildUserContext(input: IntentClassifierInput): string {
    const lines: string[] = []
    lines.push(`Active runs (${input.activeRuns.length}):`)
    for (const [idx, run] of input.activeRuns.entries()) {
      const seconds = Math.floor(run.elapsedMs / 1000)
      const promptPreview = truncate(run.userPrompt, 200)
      lines.push(`  [${idx + 1}] kind=${run.kind} phase=${run.phase} elapsed=${seconds}s prompt="${promptPreview}"`)
    }
    if (input.recentTurns && input.recentTurns.length > 0) {
      lines.push('Recent turns:')
      for (const turn of input.recentTurns.slice(-3)) {
        lines.push(`  - ${truncate(turn, 200)}`)
      }
    }
    lines.push(`New user message: "${truncate(input.newMessageText, 600)}"`)
    return truncate(lines.join('\n'), this.config.maxLlmInputChars)
  }

  private buildCacheKey(input: IntentClassifierInput): string {
    const promptKey = input.activeRuns
      .map((run) => `${run.runId}|${truncate(run.userPrompt, 80)}`)
      .join('||')
    const turnsKey = (input.recentTurns ?? []).slice(-2).map((t) => truncate(t, 80)).join('||')
    return `${input.chatKey}::${promptKey}::${turnsKey}::${input.newMessageText.trim()}`
  }

  private isRateLimited(chatKey: string): boolean {
    const now = this.clock()
    const state = this.rateLimit.get(chatKey)
    if (!state) return false
    const oneMinAgo = now - 60_000
    state.timestamps = state.timestamps.filter((t) => t > oneMinAgo)
    return state.timestamps.length >= this.config.rateLimitPerMin
  }

  private recordRateLimitHit(chatKey: string): void {
    const state = this.rateLimit.get(chatKey) ?? { timestamps: [] }
    state.timestamps.push(this.clock())
    this.rateLimit.set(chatKey, state)
  }
}

const SYSTEM_PROMPT = [
  'You classify a user message that arrived while an agent run is still in progress.',
  'Output JSON only with this exact shape: {"category":"status|cancel|duplicate|correction|parallel_new"}.',
  'Definitions:',
  '- status: user is asking about progress or status of the running task ("결과 어떻게 됐어?", "still running?").',
  '- cancel: user wants to stop without a replacement task ("취소", "그만", "stop").',
  '- duplicate: the new message is essentially the SAME request an active run is already handling — a verbatim re-send, a near-identical rephrase, or "아 그거 다시 해줘 / can you do that again". The user is being impatient or unsure it was received, not asking for something different. Compare the new message against the listed active-run prompts; if it conveys the same goal as one of them, it is duplicate (NOT correction — nothing is being changed — and NOT parallel_new — it is not an additional task).',
  '- correction: user wants to fix or change the running task ("계산 이상해", "X 빼줘", "wait, that is wrong"). Implies the running task as-is is wrong; there is a delta to apply.',
  '- parallel_new: user wants to start an additional, genuinely different task while the current one keeps running.',
  'Decision order: if it matches an active-run prompt → duplicate. Else if it asks for a change to a running task → correction. Else → parallel_new. Be conservative with cancel: only choose it when there is no replacement task in the message.',
  'Output JSON only. No prose.',
].join(' ')

export function parseClassifierJson(content: unknown): ChannelFollowupIntent | null {
  const text = extractText(content)
  if (!text) return null
  const match = text.match(/\{[\s\S]*\}/)
  if (!match) return null
  try {
    const parsed = JSON.parse(match[0]) as Record<string, unknown>
    const raw = typeof parsed.category === 'string' ? parsed.category.toLowerCase() : ''
    switch (raw) {
      case 'status':
        return { category: 'status' }
      case 'cancel':
        return { category: 'cancel' }
      case 'duplicate':
      case 'duplicate_resend':
      case 'resend':
      case 'repeat':
        return { category: 'duplicate' }
      case 'correction': {
        const note = typeof parsed.note === 'string' && parsed.note.trim()
          ? parsed.note.trim()
          : undefined
        return { category: 'correction', note }
      }
      case 'parallel_new':
      case 'parallel':
      case 'new':
        return { category: 'parallel_new' }
      default:
        return null
    }
  } catch {
    return null
  }
}

function extractText(content: unknown): string {
  if (typeof content === 'string') return content
  if (Array.isArray(content)) {
    return content
      .map((part) =>
        part && typeof part === 'object' && 'text' in part
          ? String((part as { text: unknown }).text ?? '')
          : '',
      )
      .join('\n')
  }
  return ''
}

function truncate(text: string, max: number): string {
  if (text.length <= max) return text
  return `${text.slice(0, max - 3)}...`
}

export function loadIntentClassifierConfigFromEnv(env: NodeJS.ProcessEnv = process.env): IntentClassifierConfig {
  const enabled = (env.SEPILOTD_CHANNEL_INTENT_CLASSIFIER ?? 'on').toLowerCase() !== 'off'
  const timeoutMs = positiveInt(env.SEPILOTD_CHANNEL_INTENT_TIMEOUT_MS, DEFAULT_INTENT_CLASSIFIER_CONFIG.timeoutMs)
  const rateLimitPerMin = positiveInt(env.SEPILOTD_CHANNEL_INTENT_RATE_LIMIT_PER_MIN, DEFAULT_INTENT_CLASSIFIER_CONFIG.rateLimitPerMin)
  const preferredModelId = env.SEPILOTD_CHANNEL_INTENT_MODEL?.trim() || undefined
  return {
    ...DEFAULT_INTENT_CLASSIFIER_CONFIG,
    enabled,
    timeoutMs,
    rateLimitPerMin,
    preferredModelId,
  }
}

function positiveInt(raw: string | undefined, fallback: number): number {
  if (!raw) return fallback
  const parsed = Number(raw)
  if (!Number.isFinite(parsed) || parsed <= 0) return fallback
  return Math.floor(parsed)
}

export function maxParallelRunsFromEnv(env: NodeJS.ProcessEnv = process.env, fallback = 3): number {
  const raw = env.SEPILOTD_CHANNEL_MAX_PARALLEL_RUNS
  return positiveInt(raw, fallback)
}
