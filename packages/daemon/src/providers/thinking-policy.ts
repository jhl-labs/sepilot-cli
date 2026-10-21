import { ThinkingLevel, type AgentRunContract, type ChatRequest, type ModelInfo, type ToolSecurityEffect } from '@sepilotd/core'

export function canAutomaticallyDisableThinking(model: ModelInfo | undefined): boolean {
  return Boolean(model?.capabilities.thinking && (
    model.capabilities.thinkingControl === 'chat-template-kwargs'
    || (model.capabilities.thinkingControl === 'reasoning-effort'
      && model.compatibility?.reasoningEffortOffValue !== undefined)
  ))
}

export const INVOCATION_FAILURE_CODES: ReadonlySet<string> = new Set([
  'INVALID_INPUT_PERMANENT',
  'ENOENT_PERMANENT',
  'EACCES_PERMANENT',
  'INVALID_URL_PERMANENT',
])

export function isInvocationFailure(context: {
  executionFailureCode?: string
  executionFailureBlocked?: boolean
}): boolean {
  return context.executionFailureBlocked === true
    || (context.executionFailureCode !== undefined && INVOCATION_FAILURE_CODES.has(context.executionFailureCode))
}

const STRUCTURED_TOOL_FAILURE_CODE_PATTERN = /^\[error:\s*([A-Z0-9_]+)\]/

export function structuredToolFailureCode(output: string | undefined): string | undefined {
  return output?.match(STRUCTURED_TOOL_FAILURE_CODE_PATTERN)?.[1]
}

export function consecutiveExecutionFailures(entries: readonly {
  status: 'success' | 'error'
  executionObserved?: boolean
  blocked?: boolean
  failureCode?: string
  output?: string
}[]): number {
  let count = 0
  for (let index = entries.length - 1; index >= 0; index -= 1) {
    const entry = entries[index]!
    if (entry.status === 'success') break
    if (entry.blocked === true || entry.executionObserved !== true) continue
    const code = entry.failureCode ?? structuredToolFailureCode(entry.output)
    if (code !== undefined && INVOCATION_FAILURE_CODES.has(code)) continue
    count += 1
  }
  return count
}

export function resolveThinkingLevel(
  requested: string | undefined,
  context: {
    phase?: string
    toolNames?: string[]
    readOnlyToolSurface?: boolean
    executionFailure?: boolean
    executionFailureEffect?: ToolSecurityEffect
    executionFailureCode?: string
    executionFailureBlocked?: boolean
    consecutiveExecutionFailures?: number
    taskComplexity?: AgentRunContract['reasoningComplexity']
    reasoningBudgetExhausted?: boolean
    observationBudgetPressure?: boolean
    observationBudgetExhausted?: boolean
    canDisableThinking?: boolean
  } = {},
): ChatRequest['thinkingLevel'] {
  if (requested !== 'auto') return requested as ChatRequest['thinkingLevel']
  const phase = context.phase ?? ''
  const readOnlyTools = context.readOnlyToolSurface ?? Boolean(
    context.toolNames?.length && context.toolNames.every((name) =>
      ['fs.read', 'fs.list', 'fs.glob', 'fs.search', 'git.status', 'git.diff'].includes(name)),
  )
  const failureStreak = context.executionFailure
    && context.executionFailureEffect !== 'observe'
    && !isInvocationFailure(context)
    ? Math.max(1, Math.floor(context.consecutiveExecutionFailures ?? 1))
    : 0
  let level = 'medium'
  if (/final|conclusion|report/.test(phase)) level = 'low'
  else if (failureStreak >= 3 || /review|diagnos/.test(phase)) level = 'high'
  else if (context.reasoningBudgetExhausted) level = 'low'
  else if (context.taskComplexity === 'complex') level = 'high'
  else if (failureStreak === 2) level = 'medium'
  else if (failureStreak >= 1 && context.observationBudgetExhausted) level = 'low'
  else if (context.observationBudgetExhausted && context.canDisableThinking) level = 'off'
  else if (context.observationBudgetPressure) level = 'low'
  else if (context.taskComplexity === 'routine' || readOnlyTools) level = 'low'
  else if (context.taskComplexity === undefined && context.toolNames?.length) level = 'low'
  return level as ChatRequest['thinkingLevel']
}

export function resolveControlThinkingLevel(
  requested: string | undefined,
  options: {
    binaryThinking?: boolean
    source?: ThinkingLevelSource
  } = {},
): ChatRequest['thinkingLevel'] {
  const level = options.source === 'default' && requested !== undefined && requested !== ThinkingLevel.Off
    ? 'auto'
    : requested
  if (level === undefined) return ThinkingLevel.Off
  if (level === 'auto') return options.binaryThinking ? ThinkingLevel.Off : ThinkingLevel.Low
  return level as ChatRequest['thinkingLevel']
}

export type ThinkingLevelSource = 'request' | 'default'

export function hasBinaryThinkingControl(model: ModelInfo | undefined): boolean {
  return Boolean(model?.capabilities.thinking && (
    (model.capabilities.thinkingEffort === 'binary' && model.capabilities.thinkingControl !== 'think-levels')
    || model.capabilities.thinkingControl === 'chat-template-kwargs'
  ))
}

const AUTOMATIC_THINKING_SATURATION_RATIO = 0.95
const AUTOMATIC_OBSERVATION_ALLOWANCE = 8192

export interface ObservationThinkingCost {
  finishReason: string
  finishDetail?: string
  outputTokens: number
  maxTokens?: number
  thinkingChars: number
  visibleChars: number
  observationOnlyTools: boolean
  taskComplexity?: AgentRunContract['reasoningComplexity']
}

function eligibleObservationCost(
  requested: string | undefined,
  effective: string | undefined,
  observation: ObservationThinkingCost,
): boolean {
  return requested === 'auto'
    && ['low', 'medium', 'high', 'max'].includes(effective ?? '')
    && observation.taskComplexity !== 'complex'
    && observation.observationOnlyTools
    && ['stop', 'tool_use'].includes(observation.finishReason)
    && observation.finishDetail !== 'incomplete_stream'
    && Number.isFinite(observation.outputTokens)
    && observation.outputTokens >= 0
    && Number.isFinite(observation.maxTokens)
    && (observation.maxTokens ?? 0) > 0
}

export function shouldReduceObservationThinking(
  requested: string | undefined,
  effective: string | undefined,
  observation: ObservationThinkingCost,
): boolean {
  return eligibleObservationCost(requested, effective, observation)
    && observation.outputTokens >= Math.ceil(
      (observation.maxTokens ?? 0) * AUTOMATIC_THINKING_SATURATION_RATIO,
    )
    && observation.thinkingChars > 0
    && observation.thinkingChars > observation.visibleChars
}

export function advanceAutomaticObservationBudget(
  requested: string | undefined,
  effective: string | undefined,
  previousFraction: number,
  observation: ObservationThinkingCost,
): number {
  const previous = Number.isFinite(previousFraction) ? Math.max(0, Math.min(2, previousFraction)) : 0
  if (!eligibleObservationCost(requested, effective, observation)) return previous
  const ceiling = effective === 'low' ? 2 : 1
  const allowance = Math.min(observation.maxTokens!, AUTOMATIC_OBSERVATION_ALLOWANCE)
  const cost = observation.outputTokens / allowance
  const charge = shouldReduceObservationThinking(requested, effective, observation) ? Math.max(1, cost) : cost
  return Math.max(previous, Math.min(ceiling, previous + charge))
}

export function shouldReduceAutomaticThinking(
  requested: string | undefined,
  effective: string | undefined,
  observation: {
    toolEnabled: boolean
    finishReason: string
    finishDetail?: string
    content: string
    thinking?: string
    hasToolOutput: boolean
    outputTokens?: number
    maxTokens?: number
  },
): boolean {
  return requested === 'auto'
    && ['medium', 'high', 'max'].includes(effective ?? '')
    && observation.toolEnabled
    && observation.finishReason === 'length'
    && observation.finishDetail !== 'incomplete_stream'
    && !observation.content.trim()
    && !observation.hasToolOutput
    && (
      Boolean(observation.thinking?.trim())
      || (Number.isFinite(observation.outputTokens)
        && Number.isFinite(observation.maxTokens)
        && (observation.maxTokens ?? 0) > 0
        && (observation.outputTokens ?? 0) >= observation.maxTokens!)
    )
}

export function resolveRequestThinking(request: ChatRequest): ChatRequest {
  return request.thinkingLevel === 'auto'
    ? { ...request, thinkingLevel: resolveThinkingLevel('auto', { toolNames: request.tools?.map((t) => t.name) }) }
    : request
}
