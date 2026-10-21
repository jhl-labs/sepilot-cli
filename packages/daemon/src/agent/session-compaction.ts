import type { ChatResponse, IHookRegistry, ILLMProvider, Message } from '@sepilotd/core'
import { runAuxiliaryLlmChat } from './auxiliary-llm.js'
import { canPublishUserFacingText } from './response-safety.js'
import type { ProviderCircuitBreaker } from '../providers/circuit-breaker.js'

const COMPACT_CONTINUATION_PREAMBLE =
  'Earlier conversation context was compacted to keep this session within the model context window.'
const COMPACT_RECENT_MESSAGES_NOTE = 'Recent messages are preserved below.'
const COMPACT_DIRECT_RESUME_INSTRUCTION =
  'Continue directly from the preserved recent messages. Do not recap the summary or ask the user to repeat prior context.'

const DEFAULT_MIN_MESSAGES_TO_COMPACT = 4

export type SessionCompactionStrategy = 'preserve_tail' | 'summary_only'

export interface SessionCompactionConfig {
  minMessagesToCompact?: number
  preserveRecentMessages?: number
}

export interface SessionCompactionResult {
  summary: string
  continuationMessage: string
  preservedMessages: Message[]
  strategy: SessionCompactionStrategy
  removedMessageCount: number
  preservedMessageCount: number
  originalTokens: number
  compactedTokens: number
  savedTokens: number
}

interface CompactionCandidate {
  summary: string
  continuationMessage: string
  preservedMessages: Message[]
  strategy: SessionCompactionStrategy
  removedMessageCount: number
  preservedMessageCount: number
  compactedTokens: number
}

export async function compactSessionMessages(options: {
  messages: Message[]
  provider?: ILLMProvider
  model?: string
  config?: SessionCompactionConfig
  hooks?: IHookRegistry
  sessionId?: string
  charsPerToken?: number
  signal?: AbortSignal
  breaker?: ProviderCircuitBreaker
}): Promise<SessionCompactionResult | null> {
  const messages = options.messages
  const charsPerToken = options.charsPerToken ?? 4
  const existingSummary = extractExistingCompactedSummary(messages[0])
  const compactedPrefixLen = existingSummary ? 1 : 0
  const compactableMessages = messages.slice(compactedPrefixLen)
  const minMessagesToCompact = options.config?.minMessagesToCompact ?? DEFAULT_MIN_MESSAGES_TO_COMPACT

  if (compactableMessages.length < minMessagesToCompact) {
    return null
  }

  if (options.hooks) {
    const preResult = await options.hooks.trigger({
      event: 'pre:context:compact',
      data: {
        sessionId: options.sessionId,
        eventCountBefore: messages.length,
      },
    })
    if (preResult.action === 'abort') {
      return null
    }
  }

  const originalTokens = estimateMessagesTokens(messages, charsPerToken)
  const preferredTailCount = resolvePreserveRecentMessages(
    compactableMessages.length,
    options.config?.preserveRecentMessages,
  )

  const preserveTailCandidate = await buildCompactionCandidate({
    existingSummary,
    messages,
    compactedPrefixLen,
    provider: options.provider,
    model: options.model,
    strategy: 'preserve_tail',
    preserveRecentMessages: preferredTailCount,
    charsPerToken,
    signal: options.signal,
    breaker: options.breaker,
  })

  const summaryOnlyCandidate = preserveTailCandidate.compactedTokens < originalTokens
    ? null
    : await buildCompactionCandidate({
      existingSummary,
      messages,
      compactedPrefixLen,
      provider: options.provider,
      model: options.model,
      strategy: 'summary_only',
      preserveRecentMessages: 0,
      charsPerToken,
      signal: options.signal,
      breaker: options.breaker,
    })

  const bestCandidate = pickBetterCandidate(
    preserveTailCandidate,
    summaryOnlyCandidate,
    originalTokens,
  )

  const result: SessionCompactionResult = {
    ...bestCandidate,
    originalTokens,
    savedTokens: Math.max(0, originalTokens - bestCandidate.compactedTokens),
  }

  if (options.hooks) {
    await options.hooks.trigger({
      event: 'post:context:compact',
      data: {
        sessionId: options.sessionId,
        eventCountBefore: messages.length,
        eventCountAfter: result.preservedMessageCount,
        summary: result.summary,
        strategy: result.strategy,
        savedTokens: result.savedTokens,
      },
    })
  }

  return result
}

export function formatCompactionSummary(summary: string): string {
  const withoutAnalysis = stripTagBlock(summary, 'analysis')
  const extracted = extractTagBlock(withoutAnalysis, 'summary') ?? withoutAnalysis
  const normalized = collapseBlankLines(extracted).trim()

  if (!normalized) {
    return 'Summary:\n- Earlier conversation context was compacted.'
  }

  if (/^summary:/i.test(normalized)) {
    return normalized
  }

  if (normalized.startsWith('- ')) {
    return `Summary:\n${normalized}`
  }

  return `Summary:\n- ${normalized}`
}

export function getCompactContinuationMessage(
  summary: string,
  options?: {
    recentMessagesPreserved?: boolean
    suppressFollowUpQuestions?: boolean
  },
): string {
  const sections = [
    COMPACT_CONTINUATION_PREAMBLE,
    formatCompactionSummary(summary),
  ]

  if (options?.recentMessagesPreserved) {
    sections.push(COMPACT_RECENT_MESSAGES_NOTE)
  }

  if (options?.suppressFollowUpQuestions !== false) {
    sections.push(COMPACT_DIRECT_RESUME_INSTRUCTION)
  }

  return sections.join('\n\n')
}

export function estimateMessagesTokens(messages: Message[], charsPerToken = 4): number {
  return messages.reduce(
    (sum, message) => sum + estimateMessageTokens(message, charsPerToken),
    0,
  )
}

function pickBetterCandidate(
  preserveTailCandidate: CompactionCandidate,
  summaryOnlyCandidate: CompactionCandidate | null,
  originalTokens: number,
): CompactionCandidate {
  if (!summaryOnlyCandidate) {
    return preserveTailCandidate
  }

  const preserveTailSavings = originalTokens - preserveTailCandidate.compactedTokens
  const summaryOnlySavings = originalTokens - summaryOnlyCandidate.compactedTokens

  return summaryOnlySavings > preserveTailSavings
    ? summaryOnlyCandidate
    : preserveTailCandidate
}

async function buildCompactionCandidate(options: {
  existingSummary?: string
  messages: Message[]
  compactedPrefixLen: number
  provider?: ILLMProvider
  model?: string
  strategy: SessionCompactionStrategy
  preserveRecentMessages: number
  charsPerToken: number
  signal?: AbortSignal
  breaker?: ProviderCircuitBreaker
}): Promise<CompactionCandidate> {
  const rawKeepFrom = options.strategy === 'summary_only'
    ? options.messages.length
    : Math.max(
      options.compactedPrefixLen,
      options.messages.length - options.preserveRecentMessages,
    )
  const keepFrom = options.strategy === 'summary_only'
    ? rawKeepFrom
    : adjustCompactionBoundary(options.messages, rawKeepFrom, options.compactedPrefixLen)

  const removedMessages = options.messages.slice(options.compactedPrefixLen, keepFrom)
  const preservedMessages = options.messages.slice(keepFrom)

  const removedSummary = await summarizeCompactedMessages({
    messages: removedMessages,
    provider: options.provider,
    model: options.model,
    signal: options.signal,
    breaker: options.breaker,
  })
  const merged = mergeCompactionSummaries(options.existingSummary, removedSummary)
  const summary = await condenseRollingSummary(merged, {
    provider: options.provider,
    model: options.model,
    signal: options.signal,
    breaker: options.breaker,
  })
  const continuationMessage = getCompactContinuationMessage(summary, {
    recentMessagesPreserved: preservedMessages.length > 0,
    suppressFollowUpQuestions: true,
  })
  const compactedTokens = estimateMessagesTokens(
    [
      { role: 'system', content: continuationMessage },
      ...preservedMessages,
    ],
    options.charsPerToken,
  )

  return {
    summary,
    continuationMessage,
    preservedMessages,
    strategy: options.strategy,
    removedMessageCount: removedMessages.length,
    preservedMessageCount: preservedMessages.length,
    compactedTokens,
  }
}

function resolvePreserveRecentMessages(
  compactableMessageCount: number,
  override?: number,
): number {
  if (override !== undefined) {
    return Math.max(0, Math.floor(override))
  }

  if (compactableMessageCount >= 20) {
    return 6
  }

  if (compactableMessageCount >= 10) {
    return 4
  }

  return 2
}

function adjustCompactionBoundary(
  messages: Message[],
  rawKeepFrom: number,
  compactedPrefixLen: number,
): number {
  let keepFrom = rawKeepFrom

  while (keepFrom > compactedPrefixLen && messages[keepFrom]?.role === 'tool') {
    keepFrom -= 1
  }

  return keepFrom
}

async function summarizeCompactedMessages(options: {
  messages: Message[]
  provider?: ILLMProvider
  model?: string
  signal?: AbortSignal
  breaker?: ProviderCircuitBreaker
}): Promise<string> {
  if (options.messages.length === 0) {
    return 'Summary:\n- No earlier messages were removed during compaction.'
  }

  if (!options.provider) {
    return buildFallbackSummary(options.messages)
  }

  try {
    const response = await runAuxiliaryLlmChat({
      provider: options.provider,
      request: {
        model: options.model ?? options.provider.models[0]?.id ?? 'default',
        messages: [
          {
            role: 'system',
            content: [
              'Summarize the earlier portion of an engineering assistant session for later continuation.',
              'Preserve the user goal, important constraints, decisions, pending work, files, tools, and notable failures.',
              'Return concise bullet points wrapped inside <summary>...</summary>.',
              'Do not include analysis or any text outside the summary block.',
            ].join(' '),
          },
          {
            role: 'user',
            content: options.messages.map(messageToSummaryLine).join('\n'),
          },
        ],
        maxTokens: 900,
      },
      label: 'Session compaction summarizer',
      signal: options.signal,
      breaker: options.breaker,
    })

    return completedCompactionSummary(response) ?? buildFallbackSummary(options.messages)
  } catch {
    return buildFallbackSummary(options.messages)
  }
}

function buildFallbackSummary(messages: Message[]): string {
  const roleCounts = {
    user: messages.filter((message) => message.role === 'user').length,
    assistant: messages.filter((message) => message.role === 'assistant').length,
    tool: messages.filter((message) => message.role === 'tool').length,
    system: messages.filter((message) => message.role === 'system').length,
  }
  const recentUserRequests = messages
    .filter((message) => message.role === 'user')
    .map((message) => summarizeInline(message))
    .filter(Boolean)
    .slice(-3)
  const recentAssistantReplies = messages
    .filter((message) => message.role === 'assistant')
    .map((message) => summarizeInline(message))
    .filter(Boolean)
    .slice(-2)
  const toolNames = collectToolNames(messages)
  const keyFiles = collectKeyFiles(messages)
  const currentWork = messages
    .map(summarizeInline)
    .filter(Boolean)
    .slice(-1)[0]

  const lines = [
    'Summary:',
    `- Earlier context compacted: ${messages.length} messages (user=${roleCounts.user}, assistant=${roleCounts.assistant}, tool=${roleCounts.tool}, system=${roleCounts.system}).`,
  ]

  if (recentUserRequests.length > 0) {
    lines.push('- Recent user requests:')
    for (const request of recentUserRequests) {
      lines.push(`  - ${request}`)
    }
  }

  if (recentAssistantReplies.length > 0) {
    lines.push('- Recent assistant replies:')
    for (const reply of recentAssistantReplies) {
      lines.push(`  - ${reply}`)
    }
  }

  if (toolNames.length > 0) {
    lines.push(`- Tools involved: ${toolNames.join(', ')}.`)
  }

  if (keyFiles.length > 0) {
    lines.push(`- Key files mentioned: ${keyFiles.join(', ')}.`)
  }

  if (currentWork) {
    lines.push(`- Current work at compaction time: ${currentWork}`)
  }

  return lines.join('\n')
}

function mergeCompactionSummaries(existingSummary: string | undefined, newSummary: string): string {
  if (!existingSummary) {
    return formatCompactionSummary(newSummary)
  }

  const previousLines = summaryBodyLines(existingSummary)
  const newLines = summaryBodyLines(newSummary)
  const lines = ['Summary:']

  if (previousLines.length > 0) {
    lines.push('- Previously compacted context:')
    for (const line of previousLines) {
      lines.push(`  ${line}`)
    }
  }

  if (newLines.length > 0) {
    lines.push('- Newly compacted context:')
    for (const line of newLines) {
      lines.push(`  ${line}`)
    }
  }

  return lines.join('\n')
}

function rollingSummaryCharCap(): number {
  const raw = Number(process.env.SEPILOTD_COMPACTION_SUMMARY_CHAR_CAP)
  if (Number.isFinite(raw) && raw >= 1000) return Math.floor(raw)
  return 8000
}

async function condenseRollingSummary(
  summary: string,
  options: {
    provider?: ILLMProvider
    model?: string
    signal?: AbortSignal
    breaker?: ProviderCircuitBreaker
  },
): Promise<string> {
  const cap = rollingSummaryCharCap()
  if (summary.length <= cap) return summary

  if (options.provider) {
    try {
      const response = await runAuxiliaryLlmChat({
        provider: options.provider,
        request: {
          model: options.model ?? options.provider.models[0]?.id ?? 'default',
          messages: [
            {
              role: 'system',
              content: [
                'You are compressing a running summary of an engineering assistant session.',
                'Merge duplicate points, keep the user goal, key decisions, files, pending work and notable failures.',
                'Return concise bullet points wrapped inside <summary>...</summary> and nothing else.',
              ].join(' '),
            },
            { role: 'user', content: summary },
          ],
          maxTokens: 900,
        },
        label: 'Session rolling-summary re-condenser',
        signal: options.signal,
        breaker: options.breaker,
      })
      const condensed = completedCompactionSummary(response)
      if (condensed && condensed.length <= cap) return condensed
    } catch {
    }
  }

  return formatCompactionSummary(headTailTruncate(summary.replace(/^Summary:\s*/i, ''), cap))
}

function completedCompactionSummary(response: ChatResponse): string | null {
  if (!canPublishUserFacingText(response.finishReason)) return null
  const content = typeof response.message.content === 'string'
    ? response.message.content
    : stringifyContent(response.message.content, Number.MAX_SAFE_INTEGER)
  for (const tag of ['analysis', 'summary']) {
    if (content.includes(`<${tag}>`) && extractTagBlock(content, tag) === null) return null
  }
  const withoutAnalysis = stripTagBlock(content, 'analysis')
  const body = extractTagBlock(withoutAnalysis, 'summary') ?? withoutAnalysis
  return body.trim() ? formatCompactionSummary(body) : null
}

function summaryBodyLines(summary: string): string[] {
  return formatCompactionSummary(summary)
    .replace(/^Summary:\s*/i, '')
    .split('\n')
    .map((line) => line.trimEnd())
    .filter((line) => line.trim().length > 0)
}

function extractExistingCompactedSummary(message: Message | undefined): string | undefined {
  if (!message || message.role !== 'system' || typeof message.content !== 'string') {
    return undefined
  }

  if (!message.content.startsWith(COMPACT_CONTINUATION_PREAMBLE)) {
    return undefined
  }

  let summary = message.content.slice(COMPACT_CONTINUATION_PREAMBLE.length).trim()

  const recentMessagesNoteIndex = summary.indexOf(`\n\n${COMPACT_RECENT_MESSAGES_NOTE}`)
  if (recentMessagesNoteIndex >= 0) {
    summary = summary.slice(0, recentMessagesNoteIndex)
  }

  const directResumeIndex = summary.indexOf(`\n\n${COMPACT_DIRECT_RESUME_INSTRUCTION}`)
  if (directResumeIndex >= 0) {
    summary = summary.slice(0, directResumeIndex)
  }

  return formatCompactionSummary(summary)
}

function compactionToolResultBudget(): number {
  const raw = Number(process.env.SEPILOTD_COMPACTION_TOOL_RESULT_BUDGET)
  if (Number.isFinite(raw) && raw >= 700) return Math.floor(raw)
  return 4000
}

function compactionLineBudget(): number {
  const raw = Number(process.env.SEPILOTD_COMPACTION_LINE_BUDGET)
  if (Number.isFinite(raw) && raw >= 300) return Math.floor(raw)
  return 1200
}

function headTailTruncate(content: string, maxChars: number): string {
  if (content.length <= maxChars) return content
  const elided = content.length - maxChars
  const marker = `\n...[${elided} chars elided]...\n`
  const budget = Math.max(0, maxChars - marker.length)
  const headLen = Math.ceil(budget * 0.6)
  const tailLen = budget - headLen
  return content.slice(0, headLen) + marker + content.slice(content.length - tailLen)
}

function compactionContent(message: Message, maxChars: number): string {
  if (typeof message.content === 'string') {
    return headTailTruncate(message.content.trim(), maxChars)
  }
  return stringifyContent(message.content, maxChars)
}

function messageToSummaryLine(message: Message): string {
  const budget = message.role === 'tool' ? compactionToolResultBudget() : compactionLineBudget()
  const parts = [`[${message.role}] ${compactionContent(message, budget)}`]

  if (message.toolCalls?.length) {
    parts.push(`tool_calls=${message.toolCalls.map((toolCall) => (
      `${toolCall.name}(${truncate(JSON.stringify(toolCall.arguments), 180)})`
    )).join(', ')}`)
  }

  if (message.role === 'tool' && message.toolCallId) {
    parts.push(`tool_call_id=${message.toolCallId}`)
  }

  return parts.join(' ')
}

function summarizeInline(message: Message, maxChars = 240): string {
  const segments: string[] = []
  const content = stringifyContent(message.content, maxChars)
  if (content) {
    segments.push(content)
  }

  if (message.toolCalls?.length) {
    segments.push(`tool calls: ${message.toolCalls.map((toolCall) => toolCall.name).join(', ')}`)
  }

  if (message.role === 'tool' && message.toolCallId) {
    segments.push(`tool result for ${message.toolCallId}`)
  }

  return truncate(segments.join(' | '), maxChars)
}

function stringifyContent(content: Message['content'], maxChars = 500): string {
  if (typeof content === 'string') {
    return truncate(content.trim(), maxChars)
  }

  const text = content
    .map((part) => {
      switch (part.type) {
        case 'text':
          return part.text
        case 'image':
          return `[image:${part.source.mediaType}]`
        case 'document':
          return `[document:${part.source.mediaType}]`
        default:
          return ''
      }
    })
    .join(' ')
    .trim()

  return truncate(text, maxChars)
}

function collectToolNames(messages: Message[]): string[] {
  const names = messages.flatMap((message) => message.toolCalls?.map((toolCall) => toolCall.name) ?? [])
  return Array.from(new Set(names)).slice(0, 8)
}

function collectKeyFiles(messages: Message[]): string[] {
  const candidates = messages.flatMap((message) => {
    const segments = [
      stringifyContent(message.content, 400),
      ...(message.toolCalls?.map((toolCall) => JSON.stringify(toolCall.arguments)) ?? []),
    ]
    return segments.flatMap(extractFileCandidates)
  })

  return Array.from(new Set(candidates)).slice(0, 8)
}

function extractFileCandidates(content: string): string[] {
  if (!content) {
    return []
  }

  return Array.from(content.matchAll(/\b(?:[\w.-]+\/)+[\w.-]+\.[A-Za-z0-9]+\b/g))
    .map((match) => match[0])
}

function estimateMessageTokens(message: Message, charsPerToken = 4): number {
  let chars = stringifyContent(message.content, Number.MAX_SAFE_INTEGER).length

  if (message.toolCalls?.length) {
    chars += JSON.stringify(message.toolCalls).length
  }

  if (message.toolCallId) {
    chars += message.toolCallId.length
  }

  return Math.max(1, Math.ceil(chars / charsPerToken))
}

function extractTagBlock(content: string, tag: string): string | null {
  const start = `<${tag}>`
  const end = `</${tag}>`
  const startIndex = content.indexOf(start)
  const endIndex = content.indexOf(end)

  if (startIndex < 0 || endIndex < 0 || endIndex <= startIndex) {
    return null
  }

  return content.slice(startIndex + start.length, endIndex)
}

function stripTagBlock(content: string, tag: string): string {
  const start = `<${tag}>`
  const end = `</${tag}>`
  const startIndex = content.indexOf(start)
  const endIndex = content.indexOf(end)

  if (startIndex < 0 || endIndex < 0 || endIndex <= startIndex) {
    return content
  }

  return `${content.slice(0, startIndex)}${content.slice(endIndex + end.length)}`
}

function collapseBlankLines(content: string): string {
  const lines = content.split('\n')
  const collapsed: string[] = []

  for (const line of lines) {
    const blank = line.trim().length === 0
    const lastBlank = collapsed.length > 0 && collapsed[collapsed.length - 1] === ''
    if (blank && lastBlank) {
      continue
    }
    collapsed.push(blank ? '' : line.trimEnd())
  }

  return collapsed.join('\n')
}

function truncate(content: string, maxChars: number): string {
  if (content.length <= maxChars) {
    return content
  }

  return `${content.slice(0, Math.max(0, maxChars - 3))}...`
}
