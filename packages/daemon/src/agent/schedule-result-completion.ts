import type { Message, ToolCall } from '@sepilotd/core'
import { redactSensitive } from '../memory/sensitive.js'
import { TOOL_RESULT_STATUS_METADATA_KEY } from './memory-write-completion.js'
import {
  isExplicitScheduleCreateRequest,
  isExplicitScheduleListRequest,
  resolveExplicitScheduleInspectionIntent,
  resolveExplicitScheduleManagementIntent,
  scheduleListStatusForRequest,
  SCHEDULE_CANCEL_TOOL_NAME,
  SCHEDULE_CREATE_TOOL_NAME,
  SCHEDULE_GET_TOOL_NAME,
  SCHEDULE_LIST_TOOL_NAME,
  SCHEDULE_PAUSE_TOOL_NAME,
  SCHEDULE_RESUME_TOOL_NAME,
  SCHEDULE_RUN_NOW_TOOL_NAME,
  SCHEDULE_RUNS_TOOL_NAME,
  SCHEDULE_UPDATE_TOOL_NAME,
  type ScheduleInspectionIntent,
  type ScheduleManagementIntent,
} from './schedule-intent.js'
import { CURRENT_AGENT_TURN_USER_METADATA_KEY } from './turn-context.js'

interface ScheduleListRow {
  id: string
  name: string
  instruction: string
  kind: 'oneshot' | 'recurring'
  when_human: string
  timezone: string | null
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  enabled: boolean
  nextRunAt: number | null
  lastRunAt: number | null
  lastError: string | null
  createdBy: 'agent' | 'rest' | 'cli' | 'channel-cmd' | 'internal' | null
}

interface ScheduleCreateResult {
  id: string
  name: string
  kind: 'oneshot' | 'recurring'
  when_human: string
  timezone: string | null
  done: boolean
}

interface ScheduleJobDetail {
  id: string
  name: string
  instruction: string
  kind: 'oneshot' | 'recurring'
  when_human: string
  timezone: string | null
  status: ScheduleListRow['status']
  enabled: boolean
  nextRunAt: number | null
  lastRunAt: number | null
  attempt: number
  maxAttempts: number
  retryBackoffMs: number
  lastError: string | null
  channelType: string | null
  channelTarget: string | null
  replyToMessageId: string | null
  unattended: boolean
  notificationPriority: 'normal' | 'high' | 'critical'
}

interface ScheduleRunRow {
  id: string
  jobId: string
  startedAt: number
  finishedAt: number | null
  status: 'running' | 'success' | 'failed'
  attempt: number
  durationMs: number | null
  error: string | null
  output: string | null
}

export function normalizeExplicitScheduleListToolCalls(
  toolCalls: readonly ToolCall[],
  userInput: string,
): ToolCall[] {
  const listRequest = isExplicitScheduleListRequest(userInput)
  const inspectionIntent = resolveExplicitScheduleInspectionIntent(userInput)
  const managementIntent = resolveExplicitScheduleManagementIntent(userInput)
  if (!listRequest && !inspectionIntent && !managementIntent) return [...toolCalls]
  const status = inspectionIntent
    ? 'all'
    : managementIntent
      ? 'pending'
      : scheduleListStatusForRequest(userInput)
  return toolCalls.map((toolCall) => toolCall.name === SCHEDULE_LIST_TOOL_NAME
    ? {
        ...toolCall,
        arguments: {
          ...toolCall.arguments,
          status,
        },
      }
    : toolCall)
}

/**
 * Turn a successful scheduler result into the final answer without asking an
 * LLM to restate structured rows. This prevents evidence truncation and model
 * priors from inventing or dropping scheduled jobs.
 */
export function deterministicScheduleResultFromMessages(
  messages: readonly Message[],
  userInput: string,
): string | null {
  const wantsCreate = isExplicitScheduleCreateRequest(userInput)
  const wantsList = isExplicitScheduleListRequest(userInput)
  const inspectionIntent = resolveExplicitScheduleInspectionIntent(userInput)
  const managementIntent = resolveExplicitScheduleManagementIntent(userInput)
  if (!wantsCreate && !wantsList && !inspectionIntent && !managementIntent) return null

  const currentTurnMessages = messagesAfterLatestUser(messages)
  const callsById = new Map<string, ToolCall>()
  for (const message of currentTurnMessages) {
    for (const toolCall of message.toolCalls ?? []) {
      callsById.set(toolCall.id, toolCall)
    }
  }

  if (inspectionIntent) {
    return deterministicScheduleInspectionResult(
      currentTurnMessages,
      callsById,
      inspectionIntent,
      userInput,
    )
  }

  if (managementIntent) {
    const mutationResult = deterministicScheduleMutationResult(
      currentTurnMessages,
      callsById,
      managementIntent,
      userInput,
    )
    if (mutationResult) return mutationResult
  }

  for (const message of [...currentTurnMessages].reverse()) {
    if (
      message.role !== 'tool'
      || message.metadata?.[TOOL_RESULT_STATUS_METADATA_KEY] !== 'success'
      || !message.toolCallId
    ) {
      continue
    }
    const toolCall = callsById.get(message.toolCallId)
    if (!toolCall) continue
    const output = messageText(message)

    if (wantsCreate && toolCall.name === SCHEDULE_CREATE_TOOL_NAME) {
      const created = parseScheduleCreateResult(output)
      return created ? renderScheduleCreated(created, userInput) : null
    }
    if (wantsList && toolCall.name === SCHEDULE_LIST_TOOL_NAME) {
      const rows = parseScheduleListRows(output)
      return rows ? renderScheduleList(rows, userInput) : null
    }
  }
  return null
}

function deterministicScheduleInspectionResult(
  messages: readonly Message[],
  callsById: ReadonlyMap<string, ToolCall>,
  intent: ScheduleInspectionIntent,
  userInput: string,
): string | null {
  const expectedTool = intent === 'detail' ? SCHEDULE_GET_TOOL_NAME : SCHEDULE_RUNS_TOOL_NAME
  let listedRows: ScheduleListRow[] | null = null
  let result: string | null = null

  for (const message of messages) {
    if (
      message.role !== 'tool'
      || message.metadata?.[TOOL_RESULT_STATUS_METADATA_KEY] !== 'success'
      || !message.toolCallId
    ) {
      continue
    }
    const toolCall = callsById.get(message.toolCallId)
    if (!toolCall) continue
    const output = messageText(message)
    if (toolCall.name === SCHEDULE_LIST_TOOL_NAME) {
      listedRows = parseScheduleListRows(output) ?? listedRows
      continue
    }
    if (toolCall.name !== expectedTool || !listedRows) continue

    const canonicalJobId = resolveListedJobId(toolCall.arguments.id, listedRows)
    if (!canonicalJobId) continue
    const listedJob = listedRows.find((row) => row.id === canonicalJobId)
    if (!listedJob) continue

    if (intent === 'detail') {
      const detail = parseScheduleGetResult(output)
      if (!detail || detail.id !== canonicalJobId) continue
      result = renderScheduleDetail(detail, userInput)
      continue
    }

    const runs = parseScheduleRuns(output)
    if (!runs || runs.some((run) => run.jobId !== canonicalJobId)) continue
    result = renderScheduleRuns(listedJob, runs, userInput)
  }

  if (result) return result
  if (listedRows?.length === 0) return renderNoScheduleInspectionTarget(intent, userInput)
  return null
}

function resolveListedJobId(
  rawId: unknown,
  rows: readonly ScheduleListRow[],
): string | null {
  if (typeof rawId !== 'string') return null
  const id = rawId.trim()
  if (!id) return null
  if (rows.some((row) => row.id === id)) return id
  if (id.length < 4) return null
  const matches = rows.filter((row) => row.id.startsWith(id))
  return matches.length === 1 ? matches[0]!.id : null
}

function deterministicScheduleMutationResult(
  messages: readonly Message[],
  callsById: ReadonlyMap<string, ToolCall>,
  intent: ScheduleManagementIntent,
  userInput: string,
): string | null {
  const expectedTool = managementToolName(intent)
  let listedRows: ScheduleListRow[] | null = null
  const mutatedIds: string[] = []
  const mutationNamesById = new Map<string, string>()

  for (const message of messages) {
    if (
      message.role !== 'tool'
      || message.metadata?.[TOOL_RESULT_STATUS_METADATA_KEY] !== 'success'
      || !message.toolCallId
    ) {
      continue
    }
    const toolCall = callsById.get(message.toolCallId)
    if (!toolCall) continue
    const output = messageText(message)
    if (toolCall.name === SCHEDULE_LIST_TOOL_NAME) {
      listedRows = parseScheduleListRows(output) ?? listedRows
      continue
    }
    if (toolCall.name !== expectedTool) continue
    const mutation = parseScheduleMutation(output, intent)
    if (mutation && !mutatedIds.includes(mutation.id)) mutatedIds.push(mutation.id)
    if (mutation?.name) mutationNamesById.set(mutation.id, mutation.name)
  }

  if (mutatedIds.length > 0) {
    const namesById = new Map((listedRows ?? []).map((row) => [row.id, row.name]))
    for (const [id, name] of mutationNamesById) namesById.set(id, name)
    return renderScheduleMutation(intent, mutatedIds, namesById, userInput)
  }
  if (listedRows?.length === 0) {
    return renderNoScheduleMutationTarget(intent, userInput)
  }
  return null
}

function managementToolName(intent: ScheduleManagementIntent): string {
  if (intent === 'cancel') return SCHEDULE_CANCEL_TOOL_NAME
  if (intent === 'pause') return SCHEDULE_PAUSE_TOOL_NAME
  if (intent === 'resume') return SCHEDULE_RESUME_TOOL_NAME
  if (intent === 'run_now') return SCHEDULE_RUN_NOW_TOOL_NAME
  return SCHEDULE_UPDATE_TOOL_NAME
}

function parseScheduleMutation(
  output: string,
  intent: ScheduleManagementIntent,
): { id: string; name: string | null } | null {
  try {
    const value = JSON.parse(output) as Record<string, unknown>
    if (value.ok !== true) return null
    if (intent === 'update') {
      if (!value.job || typeof value.job !== 'object' || Array.isArray(value.job)) return null
      const job = value.job as Record<string, unknown>
      if (typeof job.id !== 'string') return null
      return {
        id: job.id,
        name: typeof job.name === 'string' && job.name.trim() ? job.name : null,
      }
    }
    if (intent === 'run_now') {
      if (
        value.started !== true
        || typeof value.jobId !== 'string'
        || typeof value.runId !== 'string'
        || !value.jobId.trim()
        || !value.runId.trim()
        || (value.status !== 'success' && value.status !== 'running')
      ) {
        return null
      }
      return {
        id: value.jobId,
        name: typeof value.name === 'string' && value.name.trim() ? value.name : null,
      }
    }
    if (typeof value.id !== 'string') return null
    if (intent === 'pause' && value.enabled !== false) return null
    if (intent === 'resume' && value.enabled !== true) return null
    return {
      id: value.id,
      name: typeof value.name === 'string' && value.name.trim() ? value.name : null,
    }
  } catch {
    return null
  }
}

function parseScheduleCreateResult(output: string): ScheduleCreateResult | null {
  try {
    const value = JSON.parse(output) as Record<string, unknown>
    if (
      typeof value.id !== 'string'
      || typeof value.name !== 'string'
      || (value.kind !== 'oneshot' && value.kind !== 'recurring')
      || typeof value.when_human !== 'string'
      || value.done !== true
    ) {
      return null
    }
    return {
      id: value.id,
      name: value.name,
      kind: value.kind,
      when_human: value.when_human,
      timezone: typeof value.timezone === 'string' ? value.timezone : null,
      done: true,
    }
  } catch {
    return null
  }
}

function parseScheduleListRows(output: string): ScheduleListRow[] | null {
  try {
    const value = JSON.parse(output)
    if (!Array.isArray(value)) return null
    const rows: ScheduleListRow[] = []
    for (const item of value) {
      if (!item || typeof item !== 'object' || Array.isArray(item)) return null
      const row = item as Record<string, unknown>
      if (
        typeof row.id !== 'string'
        || typeof row.name !== 'string'
        || typeof row.instruction !== 'string'
        || (row.kind !== 'oneshot' && row.kind !== 'recurring')
        || typeof row.when_human !== 'string'
        || !isScheduleStatus(row.status)
        || typeof row.enabled !== 'boolean'
      ) {
        return null
      }
      rows.push({
        id: row.id,
        name: row.name,
        instruction: row.instruction,
        kind: row.kind,
        when_human: row.when_human,
        timezone: typeof row.timezone === 'string' ? row.timezone : null,
        status: row.status,
        enabled: row.enabled,
        nextRunAt: typeof row.nextRunAt === 'number' ? row.nextRunAt : null,
        lastRunAt: typeof row.lastRunAt === 'number' ? row.lastRunAt : null,
        lastError: typeof row.lastError === 'string' ? row.lastError : null,
        createdBy: isCreatedBy(row.createdBy) ? row.createdBy : null,
      })
    }
    return rows
  } catch {
    return null
  }
}

function parseScheduleGetResult(output: string): ScheduleJobDetail | null {
  try {
    const value = JSON.parse(output) as Record<string, unknown>
    if (
      (value.scope !== 'current_channel' && value.scope !== 'all_visible')
      || !value.job
      || typeof value.job !== 'object'
      || Array.isArray(value.job)
    ) return null
    return parseScheduleJobDetail(value.job as Record<string, unknown>)
  } catch {
    return null
  }
}

function parseScheduleJobDetail(job: Record<string, unknown>): ScheduleJobDetail | null {
  if (
    typeof job.id !== 'string'
    || !job.id.trim()
    || typeof job.name !== 'string'
    || typeof job.instruction !== 'string'
    || (job.kind !== 'oneshot' && job.kind !== 'recurring')
    || typeof job.when_human !== 'string'
    || !isScheduleStatus(job.status)
    || typeof job.enabled !== 'boolean'
    || !isNonNegativeInteger(job.attempt)
    || !isPositiveInteger(job.maxAttempts)
    || !isNonNegativeInteger(job.retryBackoffMs)
    || typeof job.unattended !== 'boolean'
    || !isNotificationPriority(job.notificationPriority)
    || !isNullableString(job.timezone)
    || !isNullableFiniteNumber(job.nextRunAt)
    || !isNullableFiniteNumber(job.lastRunAt)
    || !isNullableString(job.lastError)
    || !isNullableString(job.channelType)
    || !isNullableString(job.channelTarget)
    || !isNullableString(job.replyToMessageId)
  ) return null
  return {
    id: job.id,
    name: job.name,
    instruction: job.instruction,
    kind: job.kind,
    when_human: job.when_human,
    timezone: nullableString(job.timezone),
    status: job.status,
    enabled: job.enabled,
    nextRunAt: nullableFiniteNumber(job.nextRunAt),
    lastRunAt: nullableFiniteNumber(job.lastRunAt),
    attempt: job.attempt,
    maxAttempts: job.maxAttempts,
    retryBackoffMs: job.retryBackoffMs,
    lastError: nullableString(job.lastError),
    channelType: nullableString(job.channelType),
    channelTarget: nullableString(job.channelTarget),
    replyToMessageId: nullableString(job.replyToMessageId),
    unattended: job.unattended,
    notificationPriority: job.notificationPriority,
  }
}

function parseScheduleRuns(output: string): ScheduleRunRow[] | null {
  try {
    const value = JSON.parse(output)
    if (!Array.isArray(value)) return null
    const runs: ScheduleRunRow[] = []
    for (const item of value) {
      if (!item || typeof item !== 'object' || Array.isArray(item)) return null
      const run = item as Record<string, unknown>
      if (
        typeof run.id !== 'string'
        || !run.id.trim()
        || typeof run.jobId !== 'string'
        || !run.jobId.trim()
        || !Number.isFinite(run.startedAt)
        || !isJobRunStatus(run.status)
        || !isNonNegativeInteger(run.attempt)
        || !isNullableFiniteNumber(run.finishedAt)
        || !isNullableFiniteNumber(run.durationMs)
        || !isNullableString(run.error)
        || !isNullableString(run.output)
      ) return null
      runs.push({
        id: run.id,
        jobId: run.jobId,
        startedAt: run.startedAt as number,
        finishedAt: nullableFiniteNumber(run.finishedAt),
        status: run.status,
        attempt: run.attempt,
        durationMs: nullableFiniteNumber(run.durationMs),
        error: nullableString(run.error),
        output: nullableString(run.output),
      })
    }
    return runs
  } catch {
    return null
  }
}

function nullableString(value: unknown): string | null {
  return typeof value === 'string' ? value : null
}

function isNullableString(value: unknown): value is string | null {
  return value === null || typeof value === 'string'
}

function nullableFiniteNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function isNullableFiniteNumber(value: unknown): value is number | null {
  return value === null || (typeof value === 'number' && Number.isFinite(value))
}

function isNonNegativeInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0
}

function isPositiveInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isInteger(value) && value > 0
}

function isNotificationPriority(value: unknown): value is ScheduleJobDetail['notificationPriority'] {
  return value === 'normal' || value === 'high' || value === 'critical'
}

function isJobRunStatus(value: unknown): value is ScheduleRunRow['status'] {
  return value === 'running' || value === 'success' || value === 'failed'
}

function renderScheduleCreated(result: ScheduleCreateResult, userInput: string): string {
  const name = safeInline(result.name, 100)
  const when = safeInline(result.when_human, 100)
  const timezone = result.timezone ? ` · ${safeInline(result.timezone, 60)}` : ''
  if (isKorean(userInput)) {
    return `예약했습니다 — ${name} · ${when}${timezone}`
  }
  return `Scheduled — ${name} · ${when}${timezone}`
}

function renderScheduleDetail(job: ScheduleJobDetail, userInput: string): string {
  const korean = isKorean(userInput)
  const status = scheduleStatusLabel({ ...job, createdBy: null }, korean)
  const kind = korean
    ? job.kind === 'recurring' ? '반복' : '일회'
    : job.kind === 'recurring' ? 'recurring' : 'one-shot'
  const lines = korean
    ? [
        `예약 상세 — ${safeInline(job.name, 100)}`,
        `- 작업 ID: ${safeInline(job.id, 100)}`,
        `- 일정: ${safeInline(job.when_human, 120)} · ${kind} · ${status}`,
        `- 활성: ${job.enabled ? '예' : '아니오'} · 실행 권한: ${job.unattended ? '무인 승인' : '승인 대기'}`,
        `- 지시: ${safeInline(job.instruction, 240)}`,
        `- 재시도: 최대 ${job.maxAttempts}회 · 현재 시도 ${job.attempt} · backoff ${job.retryBackoffMs}ms`,
        `- 다음 실행: ${formatTimestamp(job.nextRunAt, '없음')}`,
        `- 최근 실행: ${formatTimestamp(job.lastRunAt, '없음')}`,
        `- 전송: ${renderScheduleRoute(job, true)}`,
        `- 알림 우선순위: ${job.notificationPriority}`,
      ]
    : [
        `Schedule details — ${safeInline(job.name, 100)}`,
        `- Job ID: ${safeInline(job.id, 100)}`,
        `- Schedule: ${safeInline(job.when_human, 120)} · ${kind} · ${status}`,
        `- Enabled: ${job.enabled ? 'yes' : 'no'} · authority: ${job.unattended ? 'unattended' : 'attended'}`,
        `- Instruction: ${safeInline(job.instruction, 240)}`,
        `- Retries: max ${job.maxAttempts} · current attempt ${job.attempt} · backoff ${job.retryBackoffMs}ms`,
        `- Next run: ${formatTimestamp(job.nextRunAt, 'none')}`,
        `- Last run: ${formatTimestamp(job.lastRunAt, 'none')}`,
        `- Delivery: ${renderScheduleRoute(job, false)}`,
        `- Notification priority: ${job.notificationPriority}`,
      ]
  if (job.timezone) {
    lines.splice(3, 0, `- ${korean ? '시간대' : 'Timezone'}: ${safeInline(job.timezone, 60)}`)
  }
  if (job.lastError) {
    lines.push(`- ${korean ? '최근 저장 오류' : 'Last stored error'}: ${safeInline(job.lastError, 240)}`)
  }
  return lines.join('\n')
}

function renderScheduleRoute(job: ScheduleJobDetail, korean: boolean): string {
  if (!job.channelType || !job.channelTarget) return korean ? '로컬' : 'local'
  const thread = job.replyToMessageId
    ? ` · ${korean ? '스레드' : 'thread'} ${safeInline(job.replyToMessageId, 80)}`
    : ''
  return `${safeInline(job.channelType, 60)} → ${safeInline(job.channelTarget, 120)}${thread}`
}

function renderScheduleRuns(
  job: ScheduleListRow,
  runs: readonly ScheduleRunRow[],
  userInput: string,
): string {
  const korean = isKorean(userInput)
  const diagnosis = isFailureDiagnosisRequest(userInput)
  const name = safeInline(job.name, 100)
  if (runs.length === 0) {
    if (korean) {
      return diagnosis
        ? `기록된 실행이 없습니다 — ${name}. 실패 원인을 확인할 영속 실행 증거가 없습니다.`
        : `기록된 실행이 없습니다 — ${name}.`
    }
    return diagnosis
      ? `There are no persisted runs for ${name}, so there is no run evidence to diagnose.`
      : `There are no persisted runs for ${name}.`
  }

  const lines = [
    korean
      ? `최근 실행 기록 ${runs.length}개 — ${name} · 작업 ID ${safeInline(job.id, 100)}`
      : `${runs.length} recent run${runs.length === 1 ? '' : 's'} — ${name} · job ID ${safeInline(job.id, 100)}`,
  ]
  for (const run of runs) {
    const status = jobRunStatusLabel(run.status, korean)
    const finished = run.finishedAt === null
      ? ''
      : ` · ${korean ? '종료' : 'finished'} ${formatTimestamp(run.finishedAt, '-')}`
    const duration = run.durationMs === null ? '' : ` · ${run.durationMs}ms`
    const error = run.error
      ? ` · ${korean ? '저장된 오류' : 'stored error'}: ${safeInline(run.error, 200)}`
      : ''
    const output = run.output
      ? ` · ${korean ? '결과' : 'output'}: ${safeInline(run.output, 200)}`
      : ''
    lines.push(
      `- run ${safeInline(run.id, 100)} · ${status} · ${korean ? '시작' : 'started'} ${formatTimestamp(run.startedAt, '-')}${finished} · ${korean ? '시도' : 'attempt'} ${run.attempt}${duration}${error}${output}`,
    )
  }

  if (diagnosis) {
    const failed = runs.filter((run) => run.status === 'failed')
    const hasStoredError = failed.some((run) => Boolean(run.error))
    if (failed.length === 0) {
      lines.push('', korean
        ? '조회된 실행 기록에는 실패가 없어 실패 원인을 확인할 수 없습니다.'
        : 'The returned run history contains no failed run, so it does not establish a failure cause.')
    } else if (hasStoredError) {
      lines.push('', korean
        ? '위 오류는 실행 당시 저장된 증거입니다. 추가 로그나 대상 점검 없이 더 구체적인 근본 원인을 단정하지 않습니다.'
        : 'The errors above are persisted run evidence. A more specific root cause requires additional logs or target inspection.')
    } else {
      lines.push('', korean
        ? '실패 상태는 기록됐지만 저장된 오류가 없어 근본 원인을 확인할 수 없습니다.'
        : 'A failed run is persisted, but no error was stored, so the root cause is not established.')
    }
  }
  return lines.join('\n')
}

function renderNoScheduleInspectionTarget(
  intent: ScheduleInspectionIntent,
  userInput: string,
): string {
  if (isKorean(userInput)) {
    return intent === 'detail'
      ? '상세를 조회할 예약 작업이 없습니다.'
      : '실행 기록을 조회할 예약 작업이 없습니다.'
  }
  return intent === 'detail'
    ? 'There is no scheduled task to inspect.'
    : 'There is no scheduled task whose runs can be inspected.'
}

function formatTimestamp(value: number | null, empty: string): string {
  if (value === null || !Number.isFinite(value)) return empty
  try {
    return new Date(value).toISOString()
  } catch {
    return empty
  }
}

function jobRunStatusLabel(status: ScheduleRunRow['status'], korean: boolean): string {
  if (korean) {
    return status === 'success' ? '성공' : status === 'failed' ? '실패' : '실행 중'
  }
  return status
}

function isFailureDiagnosisRequest(input: string): boolean {
  const normalized = input.normalize('NFKC')
  return /(?:왜|실패\s*(?:원인|이유)|안\s*(?:됐|되었|돌았|실행))/u.test(normalized)
    || /\b(?:why|fail(?:ed|ure)?|did(?:n['’]t|\s+not)\s+(?:run|fire|execute))\b/iu.test(normalized)
}

function renderScheduleMutation(
  intent: ScheduleManagementIntent,
  ids: readonly string[],
  namesById: ReadonlyMap<string, string>,
  userInput: string,
): string {
  const korean = isKorean(userInput)
  const labels = ids.map((id) => safeInline(namesById.get(id) ?? id, 100))
  if (ids.length === 1) {
    const prefix = korean
      ? {
          cancel: '예약을 취소했습니다',
          pause: '예약을 일시정지했습니다',
          resume: '예약을 재개했습니다',
          run_now: '예약 작업의 즉시 실행을 시작했습니다',
          update: '예약을 수정했습니다',
        }[intent]
      : {
          cancel: 'Cancelled the schedule',
          pause: 'Paused the schedule',
          resume: 'Resumed the schedule',
          run_now: 'Started the scheduled task now',
          update: 'Updated the schedule',
        }[intent]
    return `${prefix} — ${labels[0]}`
  }

  const summary = korean
    ? {
        cancel: `예약 ${ids.length}개를 취소했습니다.`,
        pause: `예약 ${ids.length}개를 일시정지했습니다.`,
        resume: `예약 ${ids.length}개를 재개했습니다.`,
        run_now: `예약 작업 ${ids.length}개의 즉시 실행을 시작했습니다.`,
        update: `예약 ${ids.length}개를 수정했습니다.`,
      }[intent]
    : {
        cancel: `Cancelled ${ids.length} schedules.`,
        pause: `Paused ${ids.length} schedules.`,
        resume: `Resumed ${ids.length} schedules.`,
        run_now: `Started ${ids.length} scheduled tasks now.`,
        update: `Updated ${ids.length} schedules.`,
      }[intent]
  return [summary, ...labels.map((label) => `- ${label}`)].join('\n')
}

function renderNoScheduleMutationTarget(
  intent: ScheduleManagementIntent,
  userInput: string,
): string {
  if (isKorean(userInput)) {
    const action = {
      cancel: '취소할',
      pause: '일시정지할',
      resume: '재개할',
      run_now: '지금 실행할',
      update: '수정할',
    }[intent]
    return `${action} 현재 예약 작업이 없습니다.`
  }
  const action = {
    cancel: 'cancel',
    pause: 'pause',
    resume: 'resume',
    run_now: 'run now',
    update: 'update',
  }[intent]
  return `There are no current scheduled tasks to ${action}.`
}

function renderScheduleList(rows: ScheduleListRow[], userInput: string): string {
  const korean = isKorean(userInput)
  const historical = scheduleListStatusForRequest(userInput) === 'all'
  if (rows.length === 0) {
    return korean
      ? historical
        ? '등록된 예약 작업 이력이 없습니다.'
        : '현재 등록된 예약 작업이 없습니다.'
      : historical
        ? 'There is no scheduled-task history.'
        : 'There are no currently registered scheduled tasks.'
  }

  const userRows = rows.filter((row) => row.createdBy !== 'internal')
  const internalRows = rows.filter((row) => row.createdBy === 'internal')
  const lines = [
    korean
      ? historical
        ? `예약 작업은 총 ${rows.length}개입니다.`
        : `현재 등록된 예약 작업은 ${rows.length}개입니다.`
      : historical
        ? `There are ${rows.length} scheduled tasks in total.`
        : `There are ${rows.length} currently registered scheduled tasks.`,
  ]
  appendScheduleGroup(lines, userRows, korean ? '사용자 예약' : 'User schedules', korean)
  appendScheduleGroup(lines, internalRows, korean ? '시스템 예약' : 'System schedules', korean)
  return lines.join('\n')
}

function appendScheduleGroup(
  lines: string[],
  rows: ScheduleListRow[],
  title: string,
  korean: boolean,
): void {
  if (rows.length === 0) return
  lines.push('', `${title} (${rows.length})`)
  for (const row of rows) {
    const name = safeInline(row.name, 100)
    const when = safeInline(row.when_human, 100)
    const status = scheduleStatusLabel(row, korean)
    const timezone = row.timezone ? ` · ${safeInline(row.timezone, 60)}` : ''
    const error = row.lastError ? ` · ${korean ? '오류' : 'error'}: ${safeInline(row.lastError, 120)}` : ''
    lines.push(`- ${name} — ${when} · ${status}${timezone}${error}`)
  }
}

function scheduleStatusLabel(row: ScheduleListRow, korean: boolean): string {
  if (row.status === 'pending' && !row.enabled) return korean ? '일시정지' : 'paused'
  const labels = korean
    ? {
        pending: '대기 중',
        running: '실행 중',
        completed: '완료',
        failed: '실패',
        cancelled: '취소',
      }
    : {
        pending: 'pending',
        running: 'running',
        completed: 'completed',
        failed: 'failed',
        cancelled: 'cancelled',
      }
  return labels[row.status]
}

function safeInline(value: string, maxLength: number): string {
  const compact = redactSensitive(value).redacted
    .replace(/[\r\n\t]+/g, ' ')
    .replace(/\s{2,}/g, ' ')
    .trim()
  const truncated = compact.length > maxLength
    ? `${compact.slice(0, maxLength - 1)}…`
    : compact
  return truncated.replace(/([\\`*_[\]<>])/g, '\\$1')
}

function isScheduleStatus(value: unknown): value is ScheduleListRow['status'] {
  return value === 'pending'
    || value === 'running'
    || value === 'completed'
    || value === 'failed'
    || value === 'cancelled'
}

function isCreatedBy(value: unknown): value is NonNullable<ScheduleListRow['createdBy']> {
  return value === 'agent'
    || value === 'rest'
    || value === 'cli'
    || value === 'channel-cmd'
    || value === 'internal'
}

function messageText(message: Message): string {
  if (typeof message.content === 'string') return message.content
  return message.content
    .filter((part): part is { type: 'text'; text: string } => part.type === 'text')
    .map((part) => part.text)
    .join('\n')
}

function messagesAfterLatestUser(messages: readonly Message[]): readonly Message[] {
  const index = currentTurnUserMessageIndex(messages)
  return index >= 0 ? messages.slice(index + 1) : messages
}

function currentTurnUserMessageIndex(messages: readonly Message[]): number {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]
    if (
      message?.role === 'user'
      && message.metadata?.[CURRENT_AGENT_TURN_USER_METADATA_KEY] === true
    ) return index
  }
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === 'user') return index
  }
  return -1
}

function isKorean(input: string): boolean {
  return /[가-힣]/u.test(input)
}
