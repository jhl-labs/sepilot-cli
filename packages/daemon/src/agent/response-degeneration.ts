import type { Message } from '@sepilotd/core'

const MIN_DETECTION_CHARS = 600
const MIN_REPEATED_UNIT_CHARS = 24
const MIN_REPEATED_UNIT_COUNT = 4
const SHINGLE_WORDS = 12
const MIN_REPEATED_SHINGLE_COUNT = 8

function isMarkdownTableRow(value: string): boolean {
  const trimmed = value.trim()
  return trimmed.startsWith('|') && trimmed.endsWith('|')
}

function normalize(value: string): string {
  return value.replace(/\s+/g, ' ').trim().toLocaleLowerCase()
}

/**
 * Detect provider output that has collapsed into a repeated phrase/reasoning
 * loop. This deliberately ignores language, model, tool names, and task type.
 * Exact sentence/tag units catch common thinking-tag loops; word shingles
 * cover repeated prose that lacks reliable punctuation.
 */
export function isDegenerateRepeatedResponse(value: string): boolean {
  const normalized = normalize(value)
  if (normalized.length < MIN_DETECTION_CHARS) return false

  const unitCounts = new Map<string, number>()
  const units = value
    .split(/<\/?(?:think|thinking)>|[\r\n]+|(?<=[.!?。！？])\s*/giu)
    .map(normalize)
    .filter((unit) => unit.length >= MIN_REPEATED_UNIT_CHARS)
  for (const unit of units) {
    const count = (unitCounts.get(unit) ?? 0) + 1
    unitCounts.set(unit, count)
    if (
      count >= MIN_REPEATED_UNIT_COUNT
      && unit.length * count >= normalized.length * 0.3
    ) {
      return true
    }
  }

  // Repeated cell values are normal in structured status tables (for example
  // several unavailable timestamps rendered as "unknown"). Feeding Markdown
  // rows into the prose-shingle detector makes those valid answers look like a
  // reasoning loop. Keep the exact-unit detector above on the full response so
  // genuinely duplicated rows still converge, but run overlapping word
  // shingles only over non-table prose.
  const shingleProse = value
    .split(/\r?\n/u)
    .filter((line) => !isMarkdownTableRow(line))
    .join('\n')
  const words = normalize(shingleProse).split(' ').filter(Boolean)
  if (words.length < SHINGLE_WORDS * MIN_REPEATED_SHINGLE_COUNT) return false
  const shingleCounts = new Map<string, number>()
  for (let index = 0; index <= words.length - SHINGLE_WORDS; index += 1) {
    const shingle = words.slice(index, index + SHINGLE_WORDS).join(' ')
    const count = (shingleCounts.get(shingle) ?? 0) + 1
    if (count >= MIN_REPEATED_SHINGLE_COUNT) return true
    shingleCounts.set(shingle, count)
  }
  return false
}

export function buildDegenerateResponseRecoveryMessage(): Message {
  return {
    role: 'system',
    metadata: { reminderKind: 'response_degeneration_recovery' },
    content: [
      '[Response degeneration recovery]',
      'The previous provider output collapsed into repeated text and was discarded.',
      'Do not continue or quote that output. Do not call tools.',
      'Use only the retained evidence to produce one concise final response now.',
      'Start with ANSWER: when the request is satisfied, otherwise start with INCOMPLETE: and name the smallest concrete gap.',
      'Stay below 1,200 words and do not include hidden reasoning or <think> tags.',
    ].join(' '),
  }
}

export function buildDegenerateResponseFallback(candidate?: string | null): string {
  const header =
    'INCOMPLETE: The model repeatedly hit an output-generation loop while repairing the final response.'
  if (!candidate?.trim()) {
    return `${header} Retry or resume the run; the collected tool evidence remains available.`
  }
  return [
    header,
    'The last usable draft is preserved below, but the requested follow-up verification may still be incomplete.',
    '',
    candidate.trim(),
  ].join('\n')
}
