import type { AgentContextUsage, AgentEvent, Message, TokenUsage } from '@sepilotd/core'
import { estimateMessageTokens } from './context-manager.js'

export interface ContextUsageEventInput {
  inputTokens: number
  contextWindowTokens?: number
  reservedOutputTokens?: number
  iteration: number
  source: AgentContextUsage['source']
}

function finiteNonNegative(value: number | undefined): number {
  return Number.isFinite(value) ? Math.max(0, Math.floor(value ?? 0)) : 0
}

/**
 * Provider input usage can split uncached, cache-read, and cache-created
 * prompt tokens (Anthropic). All three occupy the request context even though
 * only the uncached portion is exposed as `inputTokens` there.
 */
export function providerContextInputTokens(usage: TokenUsage): number {
  return finiteNonNegative(usage.inputTokens)
    + finiteNonNegative(usage.cacheReadTokens)
    + finiteNonNegative(usage.cacheCreationTokens)
}

export function buildContextUsageEvent(
  input: ContextUsageEventInput,
): Extract<AgentEvent, { type: 'context_usage' }> {
  const context: AgentContextUsage = {
    inputTokens: finiteNonNegative(input.inputTokens),
    iteration: finiteNonNegative(input.iteration),
    source: input.source,
  }
  const contextWindowTokens = finiteNonNegative(input.contextWindowTokens)
  const reservedOutputTokens = finiteNonNegative(input.reservedOutputTokens)
  if (contextWindowTokens > 0) context.contextWindowTokens = contextWindowTokens
  if (reservedOutputTokens > 0) context.reservedOutputTokens = reservedOutputTokens
  return { type: 'context_usage', context }
}

export function buildProviderContextUsageEvent(input: {
  usage: TokenUsage
  contextWindowTokens?: number
  reservedOutputTokens?: number
  iteration: number
}): Extract<AgentEvent, { type: 'context_usage' }> | null {
  const inputTokens = providerContextInputTokens(input.usage)
  if (inputTokens <= 0) return null
  return buildContextUsageEvent({
    inputTokens,
    contextWindowTokens: input.contextWindowTokens,
    reservedOutputTokens: input.reservedOutputTokens,
    iteration: input.iteration,
    source: 'provider',
  })
}


export type ContextUsageCategoryId =
  | 'system'
  | 'tools'
  | 'memory'
  | 'skills'
  | 'workspace'
  | 'document'
  | 'conversation'

export interface ContextUsageCategory {
  id: ContextUsageCategoryId
  label: string
  estimatedTokens: number
  detail?: string
}

export interface ContextUsageEstimateInput {
  fullSystemPrompt: string
  systemPromptWithoutMemory: string
  systemPromptWithoutSkills: string
  systemPromptWithoutWorkspace: string
  systemPromptWithoutDocument: string
  messages: readonly Message[]
  tools: readonly unknown[]
  charsPerToken: number
  contextWindow: number
  reservedOutputTokens: number
  toolCount: number
}

export interface ContextUsageEstimate {
  categories: ContextUsageCategory[]
  usedTokens: number
  contextWindow: number
  reservedOutputTokens: number
  freeTokens: number
  percentage: number
}

const DEFAULT_CHARS_PER_TOKEN = 4

export function estimateContextTextTokens(
  text: string | null | undefined,
  charsPerToken = DEFAULT_CHARS_PER_TOKEN,
): number {
  if (!text) return 0
  const ratio = Number.isFinite(charsPerToken) && charsPerToken > 0
    ? charsPerToken
    : DEFAULT_CHARS_PER_TOKEN
  return Math.ceil(text.length / ratio)
}

export function estimateContextToolTokens(
  tools: readonly unknown[],
  charsPerToken = DEFAULT_CHARS_PER_TOKEN,
): number {
  if (tools.length === 0) return 0
  return estimateContextTextTokens(JSON.stringify(tools), charsPerToken) + tools.length * 4
}

/**
 * Builds the input-window estimate shown by context inspection surfaces.
 * Prompt variants are produced by the canonical system-prompt builder, so the
 * category deltas stay aligned when memory or skill formatting changes.
 */
export function estimateContextUsage(input: ContextUsageEstimateInput): ContextUsageEstimate {
  const ratio = Number.isFinite(input.charsPerToken) && input.charsPerToken > 0
    ? input.charsPerToken
    : DEFAULT_CHARS_PER_TOKEN
  const fullSystemTokens = estimateContextTextTokens(input.fullSystemPrompt, ratio)
  const memoryTokens = Math.max(
    0,
    fullSystemTokens - estimateContextTextTokens(input.systemPromptWithoutMemory, ratio),
  )
  const skillTokens = Math.max(
    0,
    fullSystemTokens - estimateContextTextTokens(input.systemPromptWithoutSkills, ratio),
  )
  const workspaceTokens = Math.max(
    0,
    fullSystemTokens - estimateContextTextTokens(input.systemPromptWithoutWorkspace, ratio),
  )
  const documentTokens = Math.max(
    0,
    fullSystemTokens - estimateContextTextTokens(input.systemPromptWithoutDocument, ratio),
  )
  const systemTokens = Math.max(
    0,
    fullSystemTokens - memoryTokens - skillTokens - workspaceTokens - documentTokens,
  )
  const conversationTokens = input.messages.reduce(
    (sum, message) => sum + estimateMessageTokens(message, ratio) + 4,
    0,
  )
  const toolTokens = estimateContextToolTokens(input.tools, ratio)
  const categories: ContextUsageCategory[] = [
    { id: 'system', label: '시스템 지침', estimatedTokens: systemTokens },
    {
      id: 'tools',
      label: '도구 정의',
      estimatedTokens: toolTokens,
      detail: `${input.toolCount}개 노출`,
    },
    { id: 'memory', label: '장기 메모리', estimatedTokens: memoryTokens },
    { id: 'skills', label: '스킬 카탈로그', estimatedTokens: skillTokens },
    { id: 'workspace', label: '작업공간 지침', estimatedTokens: workspaceTokens },
    { id: 'document', label: '활성 문서', estimatedTokens: documentTokens },
    {
      id: 'conversation',
      label: '대화',
      estimatedTokens: conversationTokens,
      detail: `${input.messages.length}개 메시지`,
    },
  ]
  const usedTokens = categories.reduce((sum, category) => sum + category.estimatedTokens, 0)
  const contextWindow = Math.max(1, Math.floor(input.contextWindow))
  const reservedOutputTokens = Math.max(0, Math.floor(input.reservedOutputTokens))

  return {
    categories,
    usedTokens,
    contextWindow,
    reservedOutputTokens,
    freeTokens: Math.max(0, contextWindow - usedTokens - reservedOutputTokens),
    percentage: Math.min(100, (usedTokens / contextWindow) * 100),
  }
}
