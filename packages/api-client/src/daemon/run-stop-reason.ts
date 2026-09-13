import type { RunStopCode, RunStopNextAction, RunStopReason } from '@sepilotd/core'

export type { RunStopCode, RunStopNextAction }

export type RunStopLocale = 'en' | 'ko'

export interface RunStopCardAction {
  kind: RunStopNextAction
  label: string
}

export interface RunStopCardCopy {
  title: string
  body: string
  actions: RunStopCardAction[]
}

interface StopCopyEntry {
  title: string
  body: string
  /**
   * Body used when the daemon reported `detail.used`/`detail.budget`. Keeps
   * the numbers in the sentence instead of a bare "budget exhausted".
   */
  bodyWithBudget?: string
}

type StopCopyTable = Record<Exclude<RunStopCode, 'completed'>, StopCopyEntry>

const EN_COPY: StopCopyTable = {
  iteration_budget: {
    title: 'Iteration budget exhausted',
    body: 'The run reached its iteration limit before finishing.',
    bodyWithBudget: 'The run used {used} of {budget} iterations before finishing.',
  },
  node_budget: {
    title: 'Step budget exhausted',
    body: 'The run reached its graph step limit before finishing.',
    bodyWithBudget: 'The run used {used} of {budget} graph steps before finishing.',
  },
  observation_budget: {
    title: 'Observation budget exhausted',
    body: 'The run reached its tool-observation limit before finishing.',
    bodyWithBudget: 'The run used {used} of {budget} tool observations before finishing.',
  },
  no_progress: {
    title: 'No progress',
    body: 'The run stopped because recent steps produced no new progress.',
  },
  stuck_repeat: {
    title: 'Repeating the same action',
    body: 'The run stopped after repeating the same failed action.',
  },
  completion_gate: {
    title: 'Completion gate blocked',
    body: 'The reply did not satisfy every acceptance criterion with verified evidence.',
  },
  cost_gate: {
    title: 'Cost gate reached',
    body: 'The run stopped at its cost limit before finishing.',
  },
  spend_budget: {
    title: 'Spend budget exhausted',
    body: 'The run stopped because the spend budget is used up.',
  },
  approval_denied: {
    title: 'Approval denied',
    body: 'A required tool approval was denied, so the run stopped.',
  },
  user_action_required: {
    title: 'Action needed from you',
    body: 'The run stopped because a prerequisite only you can satisfy is missing. Complete it, then resume.',
  },
  approval_timeout: {
    title: 'Waiting for your decision',
    body: 'The run paused because a pending approval or question received no answer in time. Answer it to continue.',
  },
  policy_blocked: {
    title: 'Blocked by policy',
    body: 'The current autonomy or tool policy blocked a required action.',
  },
  wall_clock: {
    title: 'Time limit reached',
    body: 'The run stopped at its wall-clock limit.',
  },
  inactivity: {
    title: 'Run stalled',
    body: 'The agent produced no output for too long and was aborted to release the run slot.',
  },
  user_abort: {
    title: 'Run cancelled',
    body: 'The run was cancelled.',
  },
  provider_error: {
    title: 'Provider error',
    body: 'The model provider failed, so the run stopped.',
  },
}

const KO_COPY: StopCopyTable = {
  iteration_budget: {
    title: '반복 예산 소진',
    body: '완료 전에 반복 횟수 한도에 도달했습니다.',
    bodyWithBudget: '완료 전에 반복 {budget}회 중 {used}회를 사용했습니다.',
  },
  node_budget: {
    title: '단계 예산 소진',
    body: '완료 전에 그래프 단계 한도에 도달했습니다.',
    bodyWithBudget: '완료 전에 그래프 단계 {budget}회 중 {used}회를 사용했습니다.',
  },
  observation_budget: {
    title: '관찰 예산 소진',
    body: '완료 전에 도구 관찰 한도에 도달했습니다.',
    bodyWithBudget: '완료 전에 도구 관찰 {budget}회 중 {used}회를 사용했습니다.',
  },
  no_progress: {
    title: '진행 없음',
    body: '최근 단계에서 새로운 진전이 없어 실행을 멈췄습니다.',
  },
  stuck_repeat: {
    title: '같은 동작 반복',
    body: '실패한 동작을 반복해서 실행을 멈췄습니다.',
  },
  completion_gate: {
    title: '완료 게이트 차단',
    body: '답변이 모든 수용 기준을 검증된 증거로 충족하지 못했습니다.',
  },
  cost_gate: {
    title: '비용 한도 도달',
    body: '완료 전에 비용 한도에 도달해 실행을 멈췄습니다.',
  },
  spend_budget: {
    title: '지출 예산 소진',
    body: '지출 예산을 모두 사용해 실행을 멈췄습니다.',
  },
  approval_denied: {
    title: '승인 거부됨',
    body: '필요한 도구 승인이 거부되어 실행을 멈췄습니다.',
  },
  user_action_required: {
    title: '사용자 조치 필요',
    body: '사용자만 해결할 수 있는 선행 조건이 없어 실행을 멈췄습니다. 조치 후 이어서 실행하세요.',
  },
  approval_timeout: {
    title: '결정을 기다리는 중',
    body: '대기 중인 승인 또는 질문에 제때 응답이 없어 실행을 일시정지했습니다. 응답하면 이어집니다.',
  },
  policy_blocked: {
    title: '정책으로 차단됨',
    body: '현재 자율성 또는 도구 정책이 필요한 동작을 차단했습니다.',
  },
  wall_clock: {
    title: '시간 제한 도달',
    body: '실행 시간 한도에 도달해 멈췄습니다.',
  },
  inactivity: {
    title: '실행 정지',
    body: '에이전트가 오랫동안 출력을 내지 않아 실행 슬롯을 회수했습니다.',
  },
  user_abort: {
    title: '실행 취소됨',
    body: '실행이 취소되었습니다.',
  },
  provider_error: {
    title: '프로바이더 오류',
    body: '모델 프로바이더가 실패해 실행을 멈췄습니다.',
  },
}

const ACTION_LABELS: Record<RunStopLocale, Record<RunStopNextAction, string>> = {
  en: {
    resume: 'Resume run',
    switch_autonomy: 'Change autonomy',
    approve_pending: 'Answer pending approval',
    raise_budget: 'Retry with a larger budget',
    retry: 'Retry',
  },
  ko: {
    resume: '이어서 실행',
    switch_autonomy: '자율성 변경',
    approve_pending: '대기 중 승인 처리',
    raise_budget: '예산 늘려 재시도',
    retry: '다시 시도',
  },
}

/**
 * Fallback next actions when the daemon sent an empty `nextActions` list.
 * Older daemons may omit the list; newer ones own the decision.
 */
const DEFAULT_ACTIONS: Record<Exclude<RunStopCode, 'completed'>, RunStopNextAction[]> = {
  iteration_budget: ['resume', 'raise_budget'],
  node_budget: ['resume', 'raise_budget'],
  observation_budget: ['resume', 'raise_budget'],
  no_progress: ['retry'],
  stuck_repeat: ['retry'],
  completion_gate: ['resume'],
  cost_gate: ['raise_budget'],
  spend_budget: ['raise_budget'],
  approval_denied: ['switch_autonomy', 'retry'],
  user_action_required: ['resume'],
  approval_timeout: ['approve_pending', 'resume'],
  policy_blocked: ['switch_autonomy'],
  wall_clock: ['resume'],
  inactivity: ['retry'],
  user_abort: ['resume'],
  provider_error: ['retry'],
}

const UNKNOWN_COPY: Record<RunStopLocale, StopCopyEntry> = {
  en: { title: 'Run stopped', body: 'The run stopped before finishing.' },
  ko: { title: '실행 중단', body: '완료 전에 실행이 멈췄습니다.' },
}

function interpolate(template: string, values: Record<string, string | number | undefined>): string {
  return template.replace(/\{(\w+)\}/g, (match, name: string) => {
    const value = values[name]
    return value === undefined || value === null ? match : String(value)
  })
}

/** True when a surface should render a stop card for this reason. */
export function shouldRenderStopCard(
  reason: RunStopReason | null | undefined,
): reason is RunStopReason {
  return Boolean(reason && reason.code !== 'completed')
}

/**
 * Localized copy for a run-stop card. Pure: the same reason always produces
 * the same title/body/actions, so surfaces can render it without parsing the
 * daemon's prose. `completed` yields an empty card (no title, no actions) so
 * callers can guard with `shouldRenderStopCard` and never show it.
 */
export function describeStopReason(
  reason: RunStopReason,
  locale: RunStopLocale = 'en',
): RunStopCardCopy {
  if (reason.code === 'completed') {
    return { title: '', body: '', actions: [] }
  }
  const table = locale === 'ko' ? KO_COPY : EN_COPY
  const entry = table[reason.code as Exclude<RunStopCode, 'completed'>] ?? UNKNOWN_COPY[locale]
  const detail = reason.detail ?? {}
  const hasBudget =
    typeof detail.budget === 'number' && typeof detail.used === 'number' && entry.bodyWithBudget
  let body = hasBudget
    ? interpolate(entry.bodyWithBudget!, { used: detail.used, budget: detail.budget })
    : entry.body
  if (detail.tool && (reason.code === 'approval_denied' || reason.code === 'policy_blocked')) {
    body = locale === 'ko' ? `${body} (도구: ${detail.tool})` : `${body} (tool: ${detail.tool})`
  }
  if (reason.code === 'completion_gate' && detail.criteria && detail.criteria.length > 0) {
    body =
      locale === 'ko'
        ? `${body} 미충족: ${detail.criteria.join(', ')}`
        : `${body} Unmet: ${detail.criteria.join(', ')}`
  }
  const kinds =
    reason.nextActions && reason.nextActions.length > 0
      ? reason.nextActions
      : DEFAULT_ACTIONS[reason.code as Exclude<RunStopCode, 'completed'>] ?? []
  const labels = ACTION_LABELS[locale]
  const seen = new Set<RunStopNextAction>()
  const actions: RunStopCardAction[] = []
  for (const kind of kinds) {
    if (seen.has(kind) || !labels[kind]) continue
    seen.add(kind)
    actions.push({ kind, label: labels[kind] })
  }
  return { title: entry.title, body, actions }
}

/** Every code the copy tables cover; used by parity tests and pickers. */
export const RUN_STOP_CODES: RunStopCode[] = [
  'completed',
  ...(Object.keys(EN_COPY) as Array<Exclude<RunStopCode, 'completed'>>),
]

/**
 * Map a daemon stream error code to the stop code it implies when the frame
 * did not carry a structured `stopReason` (older daemons). Returns undefined
 * for codes that are not run stops.
 */
export function stopCodeFromErrorCode(code: string | undefined): RunStopCode | undefined {
  switch (code) {
    case 'APPROVAL_TIMEOUT':
    case 'QUESTION_TIMEOUT':
      return 'approval_timeout'
    case 'AGENT_INACTIVITY':
      return 'inactivity'
    default:
      return undefined
  }
}
