import { Box, Text } from 'ink'
import type { CliLocale } from '../../../utils/locale.js'
import { colors } from '../../theme.js'
import { privilegeColor } from '../privilege.js'
import { fitStatusSegments, STATUS_SEPARATOR, type StatusSegment } from '../runtime/status-layout.js'

export interface StatusLineProps {
  model: string
  mode: string
  bot?: string
  engineOverride?: boolean
  tokensPerSecond?: number | null
  autonomy: string
  contextPercent: number | null
  contextTokens?: number | null
  contextWindow?: number | null
  contextEstimated?: boolean
  costLabel: string | null
  locale?: CliLocale
  width: number
}

interface ColoredSegment extends StatusSegment {
  color: string
  bold?: boolean
}

export function StatusLine({
  model, mode, bot, engineOverride, tokensPerSecond, autonomy,
  contextPercent, contextTokens = null, contextWindow = null,
  contextEstimated = false, costLabel, locale = 'en', width,
}: StatusLineProps) {
  const ko = locale === 'ko'
  const contextLabel = formatContextLabel(contextTokens, contextWindow, contextPercent, contextEstimated, false, locale)
  const compactContextLabel = contextPercent == null
    ? formatContextLabel(contextTokens, contextWindow, null, contextEstimated, true, locale)
    : `ctx ${contextEstimated ? '~' : ''}${contextPercent}%`
  const contextColor = contextPercent == null ? colors.dimText
    : contextPercent >= 85 ? colors.error : contextPercent >= 60 ? colors.warning : colors.success
  const segments: ColoredSegment[] = [
    { id: 'privilege', text: `${ko ? '권한' : 'privilege'} ${autonomy}`, compactText: autonomy, priority: 100, minWidth: 1, color: privilegeColor(autonomy), bold: autonomy === 'autonomous' },
    { id: 'model', text: model, compactText: model.slice(model.lastIndexOf('/') + 1), priority: 80, minWidth: 4, color: colors.primary },
    ...(contextLabel ? [{ id: 'context', text: contextLabel, compactText: compactContextLabel ?? contextLabel, priority: 90, color: contextColor }] : []),
    { id: 'bot', text: bot === undefined ? `mode ${mode}` : `${ko ? '봇' : 'bot'} ${bot}`, priority: 60, color: colors.info },
    ...(bot !== undefined && engineOverride ? [{ id: 'engine', text: `${ko ? '엔진' : 'engine'} ${mode}`, priority: 50, color: colors.warning }] : []),
    ...(costLabel ? [{ id: 'cost', text: costLabel, priority: 20, color: colors.dimText }] : []),
    ...(tokensPerSecond && tokensPerSecond > 0 ? [{ id: 'speed', text: `${tokensPerSecond} tps`, priority: 10, color: colors.success }] : []),
  ]
  const fitted = fitStatusSegments(segments, width)
  return (
    <Box width={width} height={1}>
      <Text wrap="truncate-end">
        {fitted.map((entry, index) => (
          <Text key={entry.id}>
            {index > 0 ? <Text color={colors.border}>{STATUS_SEPARATOR}</Text> : null}
            <Text color={entry.color} bold={entry.bold}>{entry.text}</Text>
          </Text>
        ))}
      </Text>
    </Box>
  )
}

function formatContextLabel(
  contextTokens: number | null,
  contextWindow: number | null,
  contextPercent: number | null,
  estimated: boolean,
  compact = false,
  locale: CliLocale = 'en',
): string | null {
  const prefix = compact ? 'ctx' : locale === 'ko' ? '컨텍스트' : 'context'
  if (contextTokens != null && contextWindow != null && contextWindow > 0) {
    const percent = contextPercent == null
      ? Math.round((contextTokens / contextWindow) * 100)
      : contextPercent
    return compact
      ? `${prefix} ${estimated ? '~' : ''}${formatTokens(contextTokens)}/${formatTokens(contextWindow)} ${percent}%`
      : `${prefix} ${estimated ? '~' : ''}${formatTokens(contextTokens)}/${formatTokens(contextWindow)} (${percent}%)`
  }
  if (contextPercent != null) return `${prefix} ${estimated ? '~' : ''}${contextPercent}%`
  if (contextTokens != null) return `${prefix} ${estimated ? '~' : ''}${formatTokens(contextTokens)}`
  if (contextWindow != null && contextWindow > 0) return `${prefix} ${compact ? '?' : locale === 'ko' ? '미측정' : 'unmeasured'}/${formatTokens(contextWindow)}`
  return null
}

function formatTokens(tokens: number): string {
  if (!Number.isFinite(tokens) || tokens < 0) return '0'
  if (tokens >= 1_000_000) {
    const value = tokens / 1_000_000
    return `${Number.isInteger(value) ? value : value.toFixed(1)}M`
  }
  if (tokens >= 1_000) {
    const value = tokens / 1_000
    return `${Number.isInteger(value) ? value : value.toFixed(1)}k`
  }
  return String(Math.round(tokens))
}
