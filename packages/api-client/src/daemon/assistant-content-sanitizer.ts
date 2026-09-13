import type { RunStopReason } from '@sepilotd/core'

const INTERNAL_AGENT_FALLBACK_PATTERNS = [
  /^The run completed some tool work but the model returned an empty final reply\./i,
  /^The run ended with an empty final reply\./i,
  /^The run completed some tool work but ended with a progress-only reply instead of a finished answer\./i,
  /^The run ended with a progress-only reply instead of a finished answer\./i,
  /^Final reply missing after the run completed\./i,
]

const INTERNAL_AGENT_FALLBACK_PROGRESS_PATTERNS = [
  /Successful tool evidence:\s*([\s\S]+)/i,
  /Latest progress update:\s*([\s\S]+)/i,
  /\bLast progress update:\s*([\s\S]+)/i,
  /\bLast tool output:\s*([\s\S]+)/i,
]

const ANSWER_PROTOCOL_STEM_PATTERN = /^[ \t]*(ANSWER|INCOMPLETE):[ \t]*(.*)$/i
const MAX_ANSWER_PROTOCOL_STEM_SCAN_NONEMPTY_LINES = 4

type AnswerProtocolStem = 'ANSWER' | 'INCOMPLETE'

// Large enough to keep a gate-rejected "unverified draft" (often a full
// markdown report) readable instead of cutting it off mid-section.
const MAX_DETAIL_BLOCK_CHARS = 6000

function compactLine(value: string): string {
  const normalized = value.trim().replace(/\s+/g, ' ')
  if (normalized.length <= 220) return normalized
  return `${normalized.slice(0, 219).trimEnd()}...`
}

/**
 * Format a captured detail for display. Short single-line details stay inline
 * after the label; multi-line details (often a nearly-complete markdown
 * answer) keep their line breaks as an indented block instead of being
 * squashed into one truncated line.
 */
function formatDetail(label: string, value: string): string[] {
  const trimmed = value.trim()
  if (!trimmed) return []
  if (!trimmed.includes('\n') && trimmed.length <= 220) {
    return [`${label} ${compactLine(trimmed)}`]
  }
  const capped =
    trimmed.length <= MAX_DETAIL_BLOCK_CHARS
      ? trimmed
      : `${trimmed.slice(0, MAX_DETAIL_BLOCK_CHARS).trimEnd()}\n...`
  return [label, '', capped]
}

function stripThinkingArtifacts(text: string): string {
  let remaining = text.replace(/<(think|thinking)>[\s\S]*?<\/\1>/gi, '')
  const trimmedLeading = remaining.trimStart()
  if (/^<\/(think|thinking)>/i.test(trimmedLeading)) {
    remaining = trimmedLeading.replace(/^<\/(?:think|thinking)>/i, '')
  }
  return remaining.trim()
}

function findAnswerProtocolStem(content: string): {
  stem: AnswerProtocolStem
  body: string
  lineIndex: number
  lines: string[]
} | null {
  const lines = content.trim().split(/\r?\n/)
  const firstNonEmpty = lines.findIndex((line) => line.trim().length > 0)
  if (firstNonEmpty < 0) return null

  let nonEmptySeen = 0
  for (let i = firstNonEmpty; i < lines.length; i += 1) {
    const line = lines[i]!
    if (line.trim().length === 0) continue
    nonEmptySeen += 1
    if (nonEmptySeen > MAX_ANSWER_PROTOCOL_STEM_SCAN_NONEMPTY_LINES) break

    const match = line.match(ANSWER_PROTOCOL_STEM_PATTERN)
    if (!match) continue
    return {
      stem: match[1]!.toUpperCase() as AnswerProtocolStem,
      body: match[2] ?? '',
      lineIndex: i,
      lines,
    }
  }

  return null
}

function stripProtocolStem(content: string): string {
  const found = findAnswerProtocolStem(content)
  if (!found) return content.trim()

  let inFence = false
  const rest = found.lines.slice(found.lineIndex + 1).map((line) => {
    if (/^\s*(```|~~~)/.test(line)) {
      inFence = !inFence
      return line
    }
    if (inFence) return line

    const match = line.match(ANSWER_PROTOCOL_STEM_PATTERN)
    return match ? match[2] ?? '' : line
  })

  return [found.body, ...rest].join('\n').trim()
}

function extractInternalFallbackProgress(responseText: string): string | undefined {
  for (const pattern of INTERNAL_AGENT_FALLBACK_PROGRESS_PATTERNS) {
    const match = responseText.match(pattern)
    const progress = match?.[1]?.trim()
    if (progress) return progress
  }
  return undefined
}

function isInternalAgentFallbackResponse(responseText: string): boolean {
  const normalized = responseText.trim()
  return INTERNAL_AGENT_FALLBACK_PATTERNS.some((pattern) => pattern.test(normalized))
}

export interface SanitizeAssistantContentOptions {
  /**
   * Structured stop cause of the run that produced this content. When
   * present, it wins over the `INCOMPLETE:` prose protocol: the stem is
   * stripped and the blocker text is returned bare so the surface's stop
   * card carries the reason instead of a reconstructed sentence.
   */
  stopReason?: RunStopReason | null
}

export interface SanitizedAssistantContent {
  text: string
  /** True when the model reply was a protocol/fallback message, not an answer. */
  internalFallback: boolean
}

/**
 * Turn model-facing reply protocol (`ANSWER:`/`INCOMPLETE:` stems, thinking
 * tags, daemon fallback prose) into user-facing text. Pure and shared by
 * every chat surface so the transcript never shows raw protocol stems.
 */
export function sanitizeAssistantContent(
  responseText: string,
  options: SanitizeAssistantContentOptions = {},
): SanitizedAssistantContent {
  const visibleText = stripThinkingArtifacts(responseText)
  const protocolStem = findAnswerProtocolStem(visibleText)

  if (protocolStem?.stem === 'ANSWER') {
    return { text: stripProtocolStem(visibleText), internalFallback: false }
  }

  if (protocolStem?.stem === 'INCOMPLETE') {
    const blocker = stripProtocolStem(visibleText)
    const stopReason = options.stopReason
    if (stopReason && stopReason.code !== 'completed') {
      // The stop card renders the localized reason; keep only the blocker
      // detail (often a partial answer) as the message body.
      return { internalFallback: true, text: blocker }
    }
    return {
      internalFallback: true,
      text: [
        'Request incomplete.',
        ...formatDetail('Blocker:', blocker),
      ].join('\n'),
    }
  }

  if (!isInternalAgentFallbackResponse(visibleText)) {
    return { text: visibleText, internalFallback: false }
  }

  const progress = extractInternalFallbackProgress(visibleText)
  return {
    internalFallback: true,
    text: [
      'Run finished without a final answer.',
      'Some tool work may have completed, but the model did not produce a user-facing reply.',
      ...(progress ? formatDetail('Last detail:', progress) : []),
      '',
      'Try the request again to start a fresh run. If this repeats, check the selected model/provider and autonomy mode.',
    ].join('\n'),
  }
}

/**
 * Apply `sanitizeAssistantContent` to the assistant message a stream is
 * writing (by id) or, when no id is given, to the last assistant message.
 * Returns the same array when nothing changed.
 */
export function sanitizeAssistantMessages<T extends { id: string; role: string; content: string }>(
  messages: T[],
  options: SanitizeAssistantContentOptions & { assistantId?: string } = {},
): T[] {
  let targetIndex = -1
  if (options.assistantId) {
    targetIndex = messages.findIndex(
      (message) => message.id === options.assistantId && message.role === 'assistant',
    )
  } else {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      if (messages[i]!.role === 'assistant') {
        targetIndex = i
        break
      }
    }
  }
  if (targetIndex < 0) return messages
  const target = messages[targetIndex]!
  const next = sanitizeAssistantContent(target.content, options).text
  if (next === target.content) return messages
  const copy = messages.slice()
  copy[targetIndex] = { ...target, content: next }
  return copy
}
