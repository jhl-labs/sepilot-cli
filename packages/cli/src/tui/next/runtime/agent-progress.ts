import type { CliLocale } from '../../../utils/locale.js'
import { formatRecapDuration } from '../../utils/turn-recap.js'

export interface InlineAgentProgressInput {
  phase?: string | null
  streamStatus?: string | null
  providerWait?: { startedAt: number; auxiliary: boolean } | null
  now?: number
  locale?: CliLocale
  counts?: {
    criteriaTotal: number
    planTotal: number
    planDone: number
    todosTotal: number
    todosDone: number
  } | null
}

export function formatInlineAgentProgress(input: InlineAgentProgressInput): string | null {
  const parts: string[] = []
  const ko = input.locale === 'ko'
  const activity = input.providerWait
    ? `${input.providerWait.auxiliary ? (ko ? '검토 중' : 'Reviewing') : (ko ? '모델 응답 대기' : 'Waiting for model')} ${formatRecapDuration(Math.max(0, (input.now ?? Date.now()) - input.providerWait.startedAt))}`
    : input.phase?.trim() || input.streamStatus?.trim()
  if (activity) parts.push(activity)
  if (input.counts?.planTotal) parts.push(`${ko ? '계획' : 'plan'} ${input.counts.planDone}/${input.counts.planTotal}`)
  if (input.counts?.todosTotal) parts.push(`${ko ? '작업' : 'tasks'} ${input.counts.todosDone}/${input.counts.todosTotal}`)
  if (input.counts?.criteriaTotal) parts.push(`${ko ? '검사' : 'checks'} ${input.counts.criteriaTotal}`)
  return parts.length > 0 ? parts.join(' · ') : null
}
