import { describeStopReason } from '@sepilotd/api-client'
import { Box, Text } from 'ink'
import type { CliLocale } from '../../../utils/locale.js'
import { colors } from '../../theme.js'
import type { TurnOutcome } from '../../utils/turn-outcome.js'
import { formatRecapDuration } from '../../utils/turn-recap.js'
import { fitStatusSegments, STATUS_SEPARATOR, type StatusSegment } from '../runtime/status-layout.js'

export interface RunStatusLineProps {
  width: number
  locale?: CliLocale
  busy: boolean
  progressLabel?: string | null
  attention?: boolean
  attentionHint?: string
  dialogOpen?: boolean
  shellMode?: boolean
  hasDraft?: boolean
  leaderHintText?: string | null
  leaderKey?: string
  activityFrame?: string | null
  elapsedMs?: number | null
  error?: string | null
  outcome?: TurnOutcome | null
  recap?: { durationMs: number; toolCalls: number } | null
}

function outcomeTitle(outcome: TurnOutcome, locale: CliLocale): string {
  const ko = locale === 'ko'
  const errors: Record<string, string> = {
    RATE_LIMITED: ko ? '요청 한도 초과' : 'Rate limited',
    UNAUTHORIZED: ko ? '인증 실패' : 'Authentication failed',
    FORBIDDEN: ko ? '접근 거부' : 'Access denied',
    CONTEXT_LENGTH: ko ? '컨텍스트 한도 초과' : 'Context limit reached',
    PROVIDER_ERROR: ko ? '모델 제공자 오류' : 'Provider error',
    SERVICE_UNAVAILABLE: ko ? '서비스 사용 불가' : 'Service unavailable',
    SPEND_BUDGET_EXCEEDED: ko ? '지출 예산 소진' : 'Spend budget exhausted',
  }
  if (outcome.kind === 'error' && outcome.errorCode && errors[outcome.errorCode]) return errors[outcome.errorCode]
  if (outcome.reason && outcome.reason.code !== 'completed') return describeStopReason(outcome.reason, locale).title
  switch (outcome.kind) {
    case 'completed': return ko ? '턴 완료' : 'Turn completed'
    case 'cancelled': return ko ? '취소됨' : 'Cancelled'
    case 'timeout': return ko ? '응답 시간 초과' : 'Response timed out'
    case 'error': return ko ? '요청 실패' : 'Request failed'
    default: return ko ? '실행 종료' : 'Run stopped'
  }
}

export function RunStatusLine({
  width, locale = 'en', busy, progressLabel, attention = false, attentionHint,
  dialogOpen = false, shellMode = false, hasDraft = false, leaderHintText,
  leaderKey = 'Ctrl+X', activityFrame, elapsedMs, error, outcome, recap,
}: RunStatusLineProps) {
  const ko = locale === 'ko'
  const segments: StatusSegment[] = []
  let color: string = colors.dimText
  const add = (id: string, text: string, priority: number, minWidth?: number) => segments.push({ id, text, priority, minWidth })
  if (attention) {
    color = colors.warning
    add('state', `! ${progressLabel ?? (ko ? '입력 필요' : 'Input required')}`, 100, 1)
    add('action', attentionHint ?? (ko ? '요청 확인' : 'See request'), 90)
  } else if (error && !outcome) {
    color = colors.error
    add('state', `! ${error}`, 90, 4)
    add('action', ko ? '/error 상세' : '/error details', 100, 1)
    if (busy && !dialogOpen) add('stop', ko ? 'Esc 중지' : 'Esc stop', 95)
  } else if (busy) {
    color = colors.info
    add('state', `${activityFrame ?? '⠋'} ${progressLabel || (ko ? '작업 중' : 'Working…')}`, 90, 4)
    if (elapsedMs != null) add('elapsed', formatRecapDuration(elapsedMs), 95)
    add('action', dialogOpen ? (ko ? 'Esc 돌아가기' : 'Esc back') : (ko ? 'Esc 중지' : 'Esc stop'), 100, 1)
    if (!dialogOpen) {
      add('activity', ko ? 'Ctrl+O 활동' : 'Ctrl+O activity', 60)
      add('followup', ko ? 'Tab 후속 입력' : 'Tab follow-up', 50)
    }
  } else if (leaderHintText && !dialogOpen) {
    add('leader', leaderKey, 100)
    add('bindings', leaderHintText, 90, 1)
  } else if (shellMode) {
    color = colors.warning
    add('state', 'SHELL', 100)
    add('action', ko ? 'Enter 실행' : 'Enter run', 90)
    add('back', ko ? 'Esc 대화' : 'Esc chat', 80)
    add('rerun', ko ? '!! 재실행' : '!! rerun', 20)
  } else if (error || outcome || recap) {
    const kind = error && !outcome ? 'error' : outcome?.kind
    color = kind === 'completed' ? colors.success : kind === 'cancelled' || !kind ? colors.dimText : colors.warning
    const glyph = kind === 'completed' ? '✓' : kind === 'cancelled' ? '⊘' : !kind ? '·' : '!'
    add('state', `${glyph} ${outcome ? outcomeTitle(outcome, locale) : error ? (ko ? '오류' : 'Error') : (ko ? '실행 종료' : 'Run ended')}`, 95, 4)
    if (recap) add('duration', formatRecapDuration(recap.durationMs), 90)
    add('action', error ? (ko ? '/error 상세' : '/error details') : (ko ? '/recap 상세' : '/recap details'), 100, 1)
    if (recap?.toolCalls) add('tools', `${recap.toolCalls} ${ko ? '도구 호출' : recap.toolCalls === 1 ? 'tool call' : 'tool calls'}`, 40)
    if (kind === 'error' || kind === 'timeout') add('model', ko ? '/model 변경' : '/model change', 60)
    else if (!dialogOpen) add('continue', ko ? 'Enter 전송' : 'Enter send', 50)
  } else if (dialogOpen) {
    add('state', ko ? 'Esc 돌아가기' : 'Esc back', 100, 1)
  } else {
    add('state', ko ? '준비' : 'Ready', 100)
    add('send', ko ? 'Enter 전송' : 'Enter send', 95)
    add('help', '/help', 90)
    add('newline', ko ? 'Ctrl+J 줄바꿈' : 'Ctrl+J newline', 80)
    add('model', '/model', 70)
    add('privilege', ko ? 'Shift+Tab 권한' : 'Shift+Tab privilege', 60)
    add('bot', '/bot', 50)
    add('tasks', hasDraft ? (ko ? '←→ 커서' : '←→ cursor') : (ko ? '→ 작업' : '→ tasks'), 40)
    add('leader', `${leaderKey} ${ko ? '메뉴' : 'menu'}`, 75)
  }
  const text = fitStatusSegments(segments, width).map(({ text }) => text).join(STATUS_SEPARATOR)
  return <Box width={width} height={1}><Text color={color} bold={busy || attention || Boolean(error)} wrap="truncate-end">{text}</Text></Box>
}
