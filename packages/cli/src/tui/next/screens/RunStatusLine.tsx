import { describeStopReason } from '@sepilotd/api-client'
import { Box, Text } from 'ink'
import type { CliLocale } from '../../../utils/locale.js'
import { colors } from '../../theme.js'
import type { TurnOutcome } from '../../utils/turn-outcome.js'
import { formatRecapDuration } from '../../utils/turn-recap.js'
import { fitStatusSegments, STATUS_SEPARATOR, truncateStatusText, type StatusSegment } from '../runtime/status-layout.js'

export type FooterMetadataState = { status: 'loading' | 'ready' } | { status: 'error'; message: string }

export interface RunStatusLineProps {
  width: number
  metadata?: FooterMetadataState
  locale?: CliLocale
  busy: boolean
  progressLabel?: string | null
  attention?: boolean
  attentionHint?: string
  dialogOpen?: boolean
  shellMode?: boolean
  hasDraft?: boolean
  leaderHints?: string[] | null
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

export function deriveRunStatusPresentation({
  width, locale = 'en', busy, progressLabel, attention = false, attentionHint,
  dialogOpen = false, shellMode = false, hasDraft = false, leaderHints,
  leaderKey = 'Ctrl+X', activityFrame, elapsedMs, error, outcome, recap,
  metadata = { status: 'ready' },
}: RunStatusLineProps) {
  const ko = locale === 'ko'
  const owner = attention ? 'human' : dialogOpen ? 'dialog' : leaderHints?.length ? 'leader' : shellMode ? 'shell' : 'chat'
  const segments: StatusSegment[] = []
  let color: string = colors.dimText
  const add = (id: string, text: string, priority: number, minWidth?: number) => segments.push({ id, text, priority, minWidth })
  const metadataVisible = !busy && !error && !outcome && !recap
  if (attention) {
    color = colors.warning
    add('state', `! ${progressLabel ?? (ko ? '입력 필요' : 'Input required')}`, 100, 1)
  } else if (error && !outcome) {
    color = colors.error
    add('state', `! ${error}`, 90, 4)
  } else if (busy) {
    color = colors.info
    add('state', `${activityFrame ?? '⠋'} ${progressLabel || (ko ? '작업 중' : 'Working…')}`, 90, 4)
    if (elapsedMs != null) add('elapsed', formatRecapDuration(elapsedMs), 95)
  } else if (metadataVisible && metadata.status === 'error') {
    color = colors.error
    add('state', ko ? '! 모델 설정 조회 실패' : '! Model settings unavailable', 95, 4)
  } else if (metadataVisible && metadata.status === 'loading') {
    add('state', ko ? '모델 설정 조회 중…' : 'Loading model settings…', 95, 4)
  } else if (owner === 'leader' || owner === 'dialog') {
  } else if (owner === 'shell') {
    color = colors.warning
    add('state', 'SHELL', 100)
  } else if (error || outcome || recap) {
    const kind = error && !outcome ? 'error' : outcome?.kind
    color = kind === 'completed' ? colors.success : kind === 'cancelled' || !kind ? colors.dimText : colors.warning
    const glyph = kind === 'completed' ? '✓' : kind === 'cancelled' ? '⊘' : !kind ? '·' : '!'
    add('state', `${glyph} ${outcome ? outcomeTitle(outcome, locale) : error ? (ko ? '오류' : 'Error') : (ko ? '실행 종료' : 'Run ended')}`, 95, 4)
    if (recap) add('duration', formatRecapDuration(recap.durationMs), 90)
    if (recap?.toolCalls) add('tools', `${recap.toolCalls} ${ko ? '도구 호출' : recap.toolCalls === 1 ? 'tool call' : 'tool calls'}`, 40)
  } else {
    add('state', ko ? '준비' : 'Ready', 100)
  }

  if (owner === 'human') {
    add('action', attentionHint ?? (ko ? '요청 확인' : 'See request'), 90)
  } else if (owner === 'dialog') {
    add('action', ko ? 'Esc 돌아가기' : 'Esc back', 100, 1)
  } else if (owner === 'leader') {
    if (busy) add('stop', ko ? 'Esc 중지' : 'Esc stop', 100)
    add('leader', leaderKey, 99)
    leaderHints?.forEach((hint, index) => add(`binding-${index}`, hint, 97 - index))
    for (const segment of segments) {
      if (segment.id !== 'state') continue
      segment.text = truncateStatusText(segment.text, Math.max(4, Math.floor(width / 4)))
      segment.priority = 98
    }
  } else if (busy) {
    add('action', ko ? 'Esc 중지' : 'Esc stop', 100, 1)
    add('activity', ko ? 'Ctrl+O 활동' : 'Ctrl+O activity', 60)
    add('followup', ko ? 'Tab 후속 입력' : 'Tab follow-up', 50)
    if (error) add('details', ko ? '/error 상세' : '/error details', 75)
  } else if (owner === 'shell') {
    add('action', ko ? 'Enter 실행' : 'Enter run', 90)
    add('back', ko ? 'Esc 대화' : 'Esc chat', 80)
    add('rerun', ko ? '!! 재실행' : '!! rerun', 20)
  } else if (metadataVisible && metadata.status === 'error') {
    add('action', ko ? '/model 재시도' : '/model retry', 100)
    add('details', ko ? '/error 상세' : '/error details', 90)
  } else if (metadataVisible && metadata.status === 'loading') {
    add('help', '/help', 90)
  } else if (error || outcome || recap) {
    add('action', error ? (ko ? '/error 상세' : '/error details') : (ko ? '/recap 상세' : '/recap details'), 100, 1)
    if (error || outcome?.kind === 'error' || outcome?.kind === 'timeout') add('model', ko ? '/model 변경' : '/model change', 60)
    else add('continue', ko ? 'Enter 전송' : 'Enter send', 50)
  } else {
    add('send', ko ? 'Enter 전송' : 'Enter send', 95)
    add('help', '/help', 90)
    add('newline', ko ? 'Ctrl+J 줄바꿈' : 'Ctrl+J newline', 80)
    add('model', '/model', 70)
    add('privilege', ko ? 'Shift+Tab 권한' : 'Shift+Tab privilege', 60)
    add('bot', '/bot', 50)
    add('tasks', hasDraft ? (ko ? '←→ 커서' : '←→ cursor') : (ko ? '→ 작업' : '→ tasks'), 40)
    add('leader', `${leaderKey} ${ko ? '메뉴' : 'menu'}`, 75)
  }
  return { segments, color, bold: busy || attention || Boolean(error) }
}

export function RunStatusLine(props: RunStatusLineProps) {
  const { segments, color, bold } = deriveRunStatusPresentation(props)
  const text = fitStatusSegments(segments, props.width).map(({ text }) => text).join(STATUS_SEPARATOR)
  return <Box width={props.width} height={1}><Text color={color} bold={bold} wrap="truncate-end">{text}</Text></Box>
}
