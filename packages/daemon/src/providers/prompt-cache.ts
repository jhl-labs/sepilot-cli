import { createHash } from 'node:crypto'
import type { ChatRequest, PromptCacheConfig, TokenUsage } from '@sepilotd/core'
import { partitionSystemMessages } from './system-messages.js'
import { RUNTIME_CONTROLS_METADATA_KEY } from '../agent/prompt-sections.js'

type WireObject = Record<string, unknown>

function record(value: unknown): WireObject | undefined {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? value as WireObject : undefined
}

function markMessage(message: WireObject | undefined, hint: WireObject): void {
  if (!message) return
  if (typeof message.content === 'string' && message.content.length > 0) {
    message.content = [{ type: 'text', text: message.content, ...hint }]
  } else if (Array.isArray(message.content)) {
    const index = message.content.findLastIndex(part => record(part)?.type === 'text')
    if (index >= 0) message.content[index] = { ...record(message.content[index]), ...hint }
  }
}

export function applyPromptCache(params: WireObject, request: ChatRequest, config?: PromptCacheConfig): void {
  if (!config) return
  const normalized = partitionSystemMessages(request.messages, request.systemPrompt)
  const keyMaterial = config.keyMode === 'session' ? request.promptCacheKey
    : config.keyMode === 'prefix' ? JSON.stringify([normalized.systemText, params.tools]) : undefined
  if (keyMaterial) params.prompt_cache_key = 'sepilot-' + createHash('sha256')
    .update(JSON.stringify([config.keyNamespace ?? '', request.model, keyMaterial])).digest('hex').slice(0, 48)
  if (config.retention) params.prompt_cache_retention = config.retention
  if (config.options) params.prompt_cache_options = { ...config.options }
  if (config.cachePrompt !== undefined) params.cache_prompt = config.cachePrompt
  if (!config.breakpoint) return
  const hint = config.breakpoint === 'cache-control'
    ? { cache_control: { type: 'ephemeral', ...(config.ttl ? { ttl: config.ttl } : {}) } }
    : { prompt_cache_breakpoint: { mode: 'explicit' } }
  const messages = params.messages as WireObject[]
  const offset = normalized.systemText ? 1 : 0
  if (offset) markMessage(messages[0], hint)
  const stableTail = normalized.messages.findLastIndex(message => message.metadata?.[RUNTIME_CONTROLS_METADATA_KEY] !== true)
  const boundaries = normalized.messages.flatMap((message, index) => (
    message.metadata?.[RUNTIME_CONTROLS_METADATA_KEY] !== true
    && normalized.messages[index + 1]?.metadata?.[RUNTIME_CONTROLS_METADATA_KEY] === true ? [index] : []
  ))
  if (stableTail >= 0) boundaries.push(stableTail)
  const hasToolBreakpoint = config.breakpoint === 'cache-control' && Array.isArray(params.tools) && params.tools.length > 0
  const slots = 4 - offset - (hasToolBreakpoint ? 1 : 0)
  for (const index of [...new Set(boundaries)].slice(-slots)) markMessage(messages[offset + index], hint)
  if (config.breakpoint === 'cache-control' && Array.isArray(params.tools) && params.tools.length) {
    const last = params.tools.length - 1
    params.tools[last] = { ...record(params.tools[last]), ...hint }
  }
}

function count(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 ? value : undefined
}

export function mapOpenAIUsage(value: unknown): TokenUsage {
  const usage = record(value) ?? {}
  const details = record(usage.prompt_tokens_details)
  const input = count(usage.prompt_tokens)
  const cachedCount = count(details?.cached_tokens) ?? count(usage.prompt_cache_hit_tokens)
  const createdCount = count(details?.cache_write_tokens) ?? count(usage.cache_write_tokens)
  const cached = input !== undefined && cachedCount !== undefined && cachedCount <= input ? cachedCount : undefined
  const created = input !== undefined && createdCount !== undefined && createdCount + (cached ?? 0) <= input ? createdCount : undefined
  return {
    inputTokens: input ?? 0,
    outputTokens: count(usage.completion_tokens) ?? 0,
    ...(cached !== undefined ? { cacheReadTokens: cached } : {}),
    ...(created !== undefined ? { cacheCreationTokens: created } : {}),
    ...(cached !== undefined || created !== undefined ? { inputTokensIncludeCache: true } : {}),
  }
}
