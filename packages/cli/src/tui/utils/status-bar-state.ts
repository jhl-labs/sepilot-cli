import type { DaemonAgentMode } from '@sepilotd/api-client'
import { colors } from '../theme.js'
import type { AutonomyLevel } from './autonomy.js'
import { formatAutonomyBadge, getAutonomyColor } from './autonomy.js'
import { contextFillPercent } from './context-token-estimate.js'

export interface StatusBarPresentationOptions {
  sessionId: string | null
  mode: DaemonAgentMode
  autonomy: AutonomyLevel
  primaryAgentId?: string | null
  usage: { input: number; output: number; cost: number }
  provider?: string | null
  model?: string | null
  contextInputTokens?: number | null
  contextEstimated?: boolean
  contextWindow?: number | null
  maxOutputTokens?: number | null
  isStreaming: boolean
  streamLabel?: string | null
  hasPendingApproval: boolean
  currentPhase?: string | null
  stateBoardCounts?: {
    criteriaTotal: number
    planTotal: number
    planDone: number
    todosTotal: number
    todosDone: number
  } | null
}

interface StatusBarStateLabel {
  text: string
  color: string
  bold?: boolean
}

export interface StatusBarPresentationState {
  sessionLabel: string
  modeLabel: string
  showPlanBadge: boolean
  autonomyLabel: string
  autonomyColor: string
  contextPercent: number | null
  contextEstimated: boolean
  contextColor: string
  costLabel: string
  costColor: string
  modelLabel: string | null
  maxOutputLabel: string | null
  rightStatus: StatusBarStateLabel | null
  phaseLabel: string | null
  criteriaLabel: string | null
  todosLabel: string | null
}

function formatCost(cost: number): string {
  if (!Number.isFinite(cost) || cost <= 0) return '$0'
  if (cost < 0.01) return `$${cost.toFixed(4)}`
  if (cost < 1) return `$${cost.toFixed(3)}`
  return `$${cost.toFixed(2)}`
}

function formatTokens(tokens: number): string {
  if (!Number.isFinite(tokens) || tokens <= 0) return '0'
  if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(1)}M`
  if (tokens >= 1_000) return `${(tokens / 1_000).toFixed(1)}k`
  return String(Math.round(tokens))
}

const MODEL_LABEL_DOT = '·'

function formatModelLabel(
  provider: string | null | undefined,
  model: string | null | undefined,
  contextWindow: number | null | undefined,
  maxOutputTokens: number | null | undefined,
): string | null {
  if (!model) return null

  const name = provider ? `${provider}/${model}` : model
  const hasContext = contextWindow != null && contextWindow > 0
  const hasMaxOutput = maxOutputTokens != null && maxOutputTokens > 0
  if (!hasContext && !hasMaxOutput) {
    return name
  }

  const ctxPart = hasContext ? formatTokens(contextWindow as number) : '?'
  const outPart = hasMaxOutput ? formatTokens(maxOutputTokens as number) : '?'
  return `${name} [${ctxPart}${MODEL_LABEL_DOT}${outPart}]`
}

function formatSessionId(sessionId: string | null): string {
  if (!sessionId) return '—'
  return sessionId.slice(0, 8)
}

function normalizeRunStatusLabel(
  streamLabel: string | null | undefined,
): string {
  const normalized = streamLabel
    ?.replace(/^\[[^\]]+\]\s*[•·]\s*in progress\s*[•·]\s*/i, '')
    .replace(/\s+/g, ' ')
    .trim()

  return normalized && normalized.length > 0
    ? normalized
    : 'running'
}

function deriveRunningStatus(
  isStreaming: boolean,
  streamLabel: string | null | undefined,
): StatusBarStateLabel | null {
  if (!isStreaming) {
    return null
  }

  const text = normalizeRunStatusLabel(streamLabel)
  const lowered = text.toLowerCase()

  return {
    text,
    color: lowered.includes('thinking')
      ? colors.warning
      : colors.info,
    bold: true,
  }
}

export function deriveStatusBarPresentationState(
  options: StatusBarPresentationOptions,
): StatusBarPresentationState {
  const contextTokens = options.contextInputTokens != null && options.contextInputTokens >= 0
    ? options.contextInputTokens
    : null
  const contextPercent = contextTokens == null
    ? null
    : contextFillPercent(contextTokens, options.contextWindow)
  const contextColor = contextPercent == null
    ? colors.dimText
    : contextPercent >= 80
      ? colors.warning
      : contextPercent >= 50
        ? colors.info
        : colors.text
  const maxOutputLabel = options.maxOutputTokens != null && options.maxOutputTokens > 0
    ? `maxOutput: ${formatTokens(options.maxOutputTokens)}`
    : null
  const modelLabel = formatModelLabel(
    options.provider,
    options.model,
    options.contextWindow,
    options.maxOutputTokens,
  )

  const rightStatus = options.hasPendingApproval
    ? {
        text: 'approval pending',
        color: colors.pending,
        bold: true,
      }
    : deriveRunningStatus(options.isStreaming, options.streamLabel)

  const phaseLabel = options.currentPhase
    ? `phase ${options.currentPhase}`
    : null
  const counts = options.stateBoardCounts ?? null
  const criteriaLabel = counts && counts.criteriaTotal > 0
    ? `criteria ${counts.criteriaTotal}`
    : null
  const todosLabel = counts && counts.todosTotal > 0
    ? `todos ${counts.todosDone}/${counts.todosTotal}`
    : null

  return {
    sessionLabel: formatSessionId(options.sessionId),
    modeLabel: options.mode.toUpperCase(),
    showPlanBadge: options.primaryAgentId === 'plan',
    autonomyLabel: formatAutonomyBadge(options.autonomy),
    autonomyColor: colors[getAutonomyColor(options.autonomy)],
    contextPercent,
    contextEstimated: options.contextEstimated === true,
    contextColor,
    costLabel: formatCost(options.usage.cost),
    costColor: options.usage.cost > 0 ? colors.text : colors.dimText,
    modelLabel,
    maxOutputLabel,
    rightStatus,
    phaseLabel,
    criteriaLabel,
    todosLabel,
  }
}
