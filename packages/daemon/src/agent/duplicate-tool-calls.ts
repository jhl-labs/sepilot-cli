import type { Message, ToolCall } from '@sepilotd/core'

export interface DuplicateToolCallGroup {
  signature: string
  name: string
  arguments: Record<string, unknown>
  count: number
  toolCalls: ToolCall[]
}

export interface DeduplicatedToolCalls {
  retained: ToolCall[]
  duplicates: ToolCall[]
  groups: DuplicateToolCallGroup[]
}

export const MAX_DUPLICATE_TOOL_CALL_REPAIRS = 2

function stableSerialize(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map((entry) => stableSerialize(entry)).join(',')}]`
  }

  if (value && typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
    return `{${entries.map(([key, entry]) => `${JSON.stringify(key)}:${stableSerialize(entry)}`).join(',')}}`
  }

  if (typeof value === 'number' && !Number.isFinite(value)) {
    return 'null'
  }

  return JSON.stringify(value) ?? 'null'
}

function summarizeArguments(argumentsObject: Record<string, unknown>): string {
  const serialized = stableSerialize(argumentsObject)
  return serialized.length <= 160
    ? serialized
    : `${serialized.slice(0, 157)}...`
}

function buildToolCallSignature(toolCall: ToolCall): string {
  return `${toolCall.name}:${stableSerialize(toolCall.arguments ?? {})}`
}

export function findDuplicateToolCallGroups(toolCalls: ToolCall[]): DuplicateToolCallGroup[] {
  const groups = new Map<string, DuplicateToolCallGroup>()

  for (const toolCall of toolCalls) {
    const signature = buildToolCallSignature(toolCall)
    const existing = groups.get(signature)
    if (existing) {
      existing.count += 1
      existing.toolCalls.push(toolCall)
      continue
    }

    groups.set(signature, {
      signature,
      name: toolCall.name,
      arguments: { ...(toolCall.arguments ?? {}) },
      count: 1,
      toolCalls: [toolCall],
    })
  }

  return [...groups.values()].filter((group) => group.count > 1)
}

/**
 * Calls in one assistant response are planned against the same pre-execution
 * state. An exact structural duplicate therefore cannot intentionally depend
 * on the first result; retain the first call and drop later copies before any
 * side effect occurs.
 */
export function deduplicateToolCalls(toolCalls: ToolCall[]): DeduplicatedToolCalls {
  const seen = new Set<string>()
  const retained: ToolCall[] = []
  const duplicates: ToolCall[] = []

  for (const toolCall of toolCalls) {
    const signature = buildToolCallSignature(toolCall)
    if (seen.has(signature)) {
      duplicates.push(toolCall)
      continue
    }
    seen.add(signature)
    retained.push(toolCall)
  }

  return {
    retained,
    duplicates,
    groups: findDuplicateToolCallGroups(toolCalls),
  }
}

export function shouldRepairDuplicateToolCalls(options: {
  duplicateGroups: DuplicateToolCallGroup[]
  repairedCount: number
  isLastIteration?: boolean
  maxRepairs?: number
}): boolean {
  const {
    duplicateGroups,
    repairedCount,
    isLastIteration = false,
    maxRepairs = MAX_DUPLICATE_TOOL_CALL_REPAIRS,
  } = options

  return !isLastIteration
    && repairedCount < maxRepairs
    && duplicateGroups.length > 0
}

export function buildDuplicateToolCallRepairMessage(
  duplicateGroups: DuplicateToolCallGroup[],
): Message {
  const repeatedCalls = duplicateGroups
    .slice(0, 3)
    .map((group) => `- ${group.name} ${summarizeArguments(group.arguments)} x${group.count}`)
    .join('\n')

  return {
    role: 'system',
    content: [
      'Your previous assistant reply repeated identical tool calls in the same turn.',
      'Do not call the same tool more than once with the same arguments unless the inputs have changed.',
      'Continue the task now by calling only the missing distinct tool(s), or provide the finished answer if you already have enough information.',
      repeatedCalls
        ? `Repeated identical calls detected:\n${repeatedCalls}`
        : '',
    ].filter(Boolean).join('\n'),
  }
}
