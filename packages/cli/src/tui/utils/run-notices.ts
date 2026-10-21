import type { CliLocale } from '../../utils/locale.js'

const SUPERVISOR_PREFIX = '[supervisor]'

interface SupervisorNoticeRule {
  pattern: RegExp
  present: ((match: RegExpExecArray, locale: CliLocale) => string) | null
}

const SUPERVISOR_NOTICE_RULES: readonly SupervisorNoticeRule[] = [
  {
    pattern: /Reused (\d+) successful current-turn read observation/,
    present: (match, locale) => locale === 'ko'
      ? `이미 조회한 결과 ${match[1]}건은 다시 실행하지 않고 재사용합니다.`
      : `Reusing ${match[1]} earlier result(s) instead of running the same reads again.`,
  },
  {
    pattern: /exceeds bounded working memory/,
    present: (_match, locale) => locale === 'ko'
      ? '조사 결과가 많아 앞선 결과는 요약해 두고 계속 조사합니다.'
      : 'Many results so far: earlier ones are summarized and the investigation continues.',
  },
  {
    pattern: /Wall-clock limit is near/,
    present: (_match, locale) => locale === 'ko'
      ? '실행 시간 한도가 가까워, 지금까지 모은 결과로 답을 정리합니다.'
      : 'The time limit is near: writing the answer from what was gathered so far.',
  },
  {
    pattern: /switching to bounded synthesis|closing tool access|tools are closed/i,
    present: (_match, locale) => locale === 'ko'
      ? '추가 조회 없이 지금까지의 결과로 답을 정리합니다.'
      : 'No more tool calls: writing the answer from the results so far.',
  },
]

export function presentReasoningPreview(
  text: string,
  options: { details: boolean; locale: CliLocale },
): string | null {
  const trimmed = text.trim()
  if (!trimmed.startsWith(SUPERVISOR_PREFIX) || options.details) return text
  for (const rule of SUPERVISOR_NOTICE_RULES) {
    const match = rule.pattern.exec(trimmed)
    if (match) return rule.present ? rule.present(match, options.locale) : null
  }
  return null
}

export interface TurnToolCounts {
  done: number
  failed: number
}

export function currentTurnToolCounts(
  messages: ReadonlyArray<{ role: string; toolCall?: { status: string } }>,
): TurnToolCounts {
  let done = 0
  let failed = 0
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]!
    if (message.role === 'user') break
    if (message.role !== 'tool' || !message.toolCall) continue
    if (message.toolCall.status === 'success') done += 1
    else if (message.toolCall.status === 'error') { done += 1; failed += 1 }
  }
  return { done, failed }
}

export function formatElapsed(ms: number): string {
  const seconds = Math.max(0, Math.floor(ms / 1000))
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  return minutes < 60 ? `${minutes}m${String(seconds % 60).padStart(2, '0')}s` : `${Math.floor(minutes / 60)}h${String(minutes % 60).padStart(2, '0')}m`
}

export const SLOW_MODEL_HINT_AFTER_MS = 60_000

export function formatRunWatchLabel(input: {
  elapsedMs: number
  tools: TurnToolCounts
  modelWaitMs?: number | null
  locale: CliLocale
  showElapsed?: boolean
}): string | null {
  const ko = input.locale === 'ko'
  const parts = input.showElapsed === false ? [] : [formatElapsed(input.elapsedMs)]
  if (input.tools.done > 0) {
    const failed = input.tools.failed > 0
      ? (ko ? `, 실패 ${input.tools.failed}` : `, ${input.tools.failed} failed`)
      : ''
    parts.push(ko ? `도구 ${input.tools.done}회${failed}` : `${input.tools.done} tool call${input.tools.done === 1 ? '' : 's'}${failed}`)
  }
  if ((input.modelWaitMs ?? 0) >= SLOW_MODEL_HINT_AFTER_MS) {
    parts.push(ko ? '모델 응답이 느립니다 · Esc 중단' : 'slow model response · Esc to stop')
  }
  return parts.length > 0 ? parts.join(' · ') : null
}

export function formatTurnSummary(input: { elapsedMs: number; tools: TurnToolCounts; locale: CliLocale }): string | null {
  if (input.tools.done === 0) return null
  const ko = input.locale === 'ko'
  const failed = input.tools.failed > 0
    ? (ko ? ` (실패 ${input.tools.failed})` : ` (${input.tools.failed} failed)`)
    : ''
  return ko
    ? `완료 · ${formatElapsed(input.elapsedMs)} · 도구 ${input.tools.done}회${failed}`
    : `Done · ${formatElapsed(input.elapsedMs)} · ${input.tools.done} tool call${input.tools.done === 1 ? '' : 's'}${failed}`
}
