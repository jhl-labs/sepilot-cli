import type { ToolCall, ToolDefinition } from '@sepilotd/core'

export const ACTION_PROGRESS_TAG = 'sepilot_action_progress'
export const ACTION_PROGRESS_ARGUMENT = '__sepilot_action_progress'

const MAX_ACTION_PROGRESS_FIELD_CHARS = 180

export interface AgentActionProgress {
  /** Why the selected tool action is useful at this point in the run. */
  summary: string
  /** What the agent expects to decide or do after the tool result arrives. */
  nextStep: string
}

export const ACTION_PROGRESS_SYSTEM_PROMPT = [
  'Tool-use progress contract: when you make one or more tool calls, provide a concise user-visible action annotation in the same assistant response.',
  `Every exposed tool schema includes a required ${ACTION_PROGRESS_ARGUMENT} object. Fill its summary and nextStep fields as part of the tool arguments; the runtime removes this reserved object before executing the tool.`,
  'Write both fields in the language of the latest user message, keep each to one short line, and describe the purpose and expected follow-up rather than restating the tool name or arguments.',
  'This is a progress summary, not private chain-of-thought: do not reveal hidden reasoning, internal policies, or speculative conclusions.',
  `If a transport cannot carry the reserved argument but can carry assistant content beside a tool call, use <${ACTION_PROGRESS_TAG}>{"summary":"why this action is needed now","nextStep":"what follows after the result"}</${ACTION_PROGRESS_TAG}> as the fallback.`,
  'Do not emit this annotation for a tool-free final answer.',
].join(' ')

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

/**
 * Add a transport-neutral, model-authored progress field to each tool schema.
 * Keeping it in the ordinary argument object works for native function calls
 * and prompt-serialized tool calls without a second model request or a
 * special-purpose narration tool.
 */
export function withAgentActionProgressSchemas<T extends ToolDefinition>(
  tools: readonly T[],
): T[] {
  return tools.map((tool) => {
    const schema = tool.inputSchema
    const properties = isRecord(schema.properties) ? schema.properties : {}
    const required = Array.isArray(schema.required)
      ? schema.required.filter((item): item is string => typeof item === 'string')
      : []
    return {
      ...tool,
      inputSchema: {
        ...schema,
        type: schema.type ?? 'object',
        properties: {
          ...properties,
          [ACTION_PROGRESS_ARGUMENT]: {
            type: 'object',
            properties: {
              summary: { type: 'string' },
              nextStep: { type: 'string' },
            },
            required: ['summary', 'nextStep'],
            additionalProperties: false,
          },
        },
        required: [...new Set([...required, ACTION_PROGRESS_ARGUMENT])],
      },
    }
  })
}

function normalizeField(value: unknown): string | null {
  if (typeof value !== 'string') return null
  const normalized = value.replace(/\s+/gu, ' ').trim()
  if (!normalized) return null
  if (normalized.length <= MAX_ACTION_PROGRESS_FIELD_CHARS) return normalized
  return `${normalized.slice(0, MAX_ACTION_PROGRESS_FIELD_CHARS - 1)}…`
}

/**
 * Parse only the explicit model-authored action annotation. Untagged prose is
 * deliberately ignored: inferring intent from a tool name or arguments would
 * replace the model's judgment with a surface heuristic and could mislead the
 * operator about why the action was selected.
 */
export function extractAgentActionProgress(text: string): AgentActionProgress | null {
  const match = text.match(
    /<(?:[A-Za-z][\w.-]*:)?sepilot_action_progress\b[^>]*>\s*([\s\S]*?)\s*<\/(?:[A-Za-z][\w.-]*:)?sepilot_action_progress>/iu,
  )
  if (!match?.[1]) return null

  let parsed: unknown
  try {
    parsed = JSON.parse(match[1]) as unknown
  } catch {
    return null
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return null

  const value = parsed as Record<string, unknown>
  const summary = normalizeField(value.summary)
  const nextStep = normalizeField(value.nextStep)
  if (!summary || !nextStep) return null
  return { summary, nextStep }
}

/**
 * Consume progress metadata authored inside tool arguments. The reserved
 * field is always removed so validators, approval previews, signatures and
 * the actual tool implementation continue to see only the tool's real input.
 */
export function consumeToolCallActionProgress(
  toolCalls: ToolCall[],
  exposedTools: readonly ToolDefinition[] = [],
): AgentActionProgress | null {
  const exposedToolByName = new Map(exposedTools.map(tool => [tool.name, tool]))
  let firstProgress: AgentActionProgress | null = null
  for (const toolCall of toolCalls) {
    const rawProgress = toolCall.arguments[ACTION_PROGRESS_ARGUMENT]
    delete toolCall.arguments[ACTION_PROGRESS_ARGUMENT]
    if (!firstProgress && isRecord(rawProgress)) {
      const summary = normalizeField(rawProgress.summary)
      const nextStep = normalizeField(rawProgress.nextStep)
      if (summary && nextStep) firstProgress = { summary, nextStep }
    }

    // Some structured-output transports flatten the reserved object into
    // sibling `summary`/`nextStep` fields and may emit its property name as an
    // empty-key sentinel. Recover only when the exposed source schema proves
    // those sibling names are framework-only. Legitimate tool fields with the
    // same names remain untouched.
    const flattenedMarker = toolCall.arguments[''] === ACTION_PROGRESS_ARGUMENT
    if (flattenedMarker) delete toolCall.arguments['']
    const exposedSchema = exposedToolByName.get(toolCall.name)?.inputSchema
    const exposedProperties = isRecord(exposedSchema?.properties)
      ? exposedSchema.properties
      : null
    const summaryIsFrameworkOnly = exposedProperties !== null
      && !Object.hasOwn(exposedProperties, 'summary')
    const nextStepIsFrameworkOnly = exposedProperties !== null
      && !Object.hasOwn(exposedProperties, 'nextStep')
    if (!summaryIsFrameworkOnly || !nextStepIsFrameworkOnly || isRecord(rawProgress)) continue

    const summary = normalizeField(toolCall.arguments.summary)
    const nextStep = normalizeField(toolCall.arguments.nextStep)
    if (!summary || !nextStep) continue
    delete toolCall.arguments.summary
    delete toolCall.arguments.nextStep
    if (!firstProgress) firstProgress = { summary, nextStep }
  }
  return firstProgress
}
