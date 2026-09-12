import chalk from 'chalk'
import {
  daemonSchedulerRunAgentSessionId,
  normalizeScheduledAgentSkillRefs,
  scheduledAgentSkillRefsFromMetadata,
  type DaemonSchedulerJob,
  type DaemonSchedulerJobRun,
  type DaemonSchedulerManualRunResult,
} from '@sepilotd/api-client'
import { DaemonClient } from '../client/http.js'
import { output, outputError } from '../output/formatter.js'
import { friendlyErrorMessage } from '../utils/error-message.js'
import { detectCliLocale } from '../utils/locale.js'
import { hasPendingScheduledRun } from '../utils/scheduler-state.js'

type UrlOpt = { url?: string }
type ScheduleRouteOptions = {
  channelType?: string
  channelTarget?: string
  replyTo?: string
}
type DeliveryRouteInput = {
  channelType: string
  channelTarget: string
  replyToMessageId?: string
}
const SCHEDULE_RUN_POLL_INTERVAL_MS = 1_000

async function waitForScheduledRun(
  client: DaemonClient,
  initial: Extract<DaemonSchedulerManualRunResult, { started: true }>,
): Promise<DaemonSchedulerManualRunResult> {
  if (initial.status !== 'running') return initial
  while (true) {
    const run = await client.getScheduledTaskRun(initial.jobId, initial.runId)
    if (run.status !== 'running') {
      return {
        started: true,
        jobId: initial.jobId,
        runId: initial.runId,
        status: run.status,
        ...(initial.deliverySuppressed ? { deliverySuppressed: true } : {}),
      }
    }
    await new Promise<void>((resolve) => setTimeout(resolve, SCHEDULE_RUN_POLL_INTERVAL_MS))
  }
}

const SCHEDULE_COPY = {
  en: {
    failedList: 'Failed to list scheduled tasks',
    failedLoad: (id: string) => `Failed to load scheduled task ${id}`,
    failedCreate: 'Failed to create scheduled task',
    failedUpdate: (id: string) => `Failed to update scheduled task ${id}`,
    failedRemove: (id: string) => `Failed to remove scheduled task ${id}`,
    failedTrigger: (id: string) => `Failed to trigger scheduled task ${id}`,
    failedPause: (id: string) => `Failed to pause scheduled task ${id}`,
    failedResume: (id: string) => `Failed to resume scheduled task ${id}`,
    failedUnattended: (id: string) => `Failed to change unattended execution for scheduled task ${id}`,
    failedRoute: (id: string) => `Failed to change the delivery route for scheduled task ${id}`,
    failedRuns: (id: string) => `Failed to load run history for ${id}`,
    failedRun: (id: string, runId: string) => `Failed to load run ${runId} for ${id}`,
    invalidRunsOptions: 'Choose either --run or --limit, not both',
    invalidRunLimit: 'Run history limit must be a whole number',
    pauseResumeConflict: 'Choose either --pause or --resume, not both',
    invalidUnattendedState: 'Unattended state must be "on" or "off".',
    invalidDeliveryRoute:
      'Delivery routing requires both --channel-type and --channel-target; --reply-to also requires that pair.',
    invalidRouteCommand:
      'Use a channel type plus target, or use "clear" without a target or --reply-to.',
    invalidSkillSelection: 'Scheduled skills must be valid stable skill ids (maximum 8 unique ids).',
    skillEditConflict: 'Choose --skill or --clear-skills, not both.',
    now: 'now',
    inMinutes: (minutes: number) => `in ${minutes}m`,
    minutesAgo: (minutes: number) => `${minutes}m ago`,
    inHours: (hours: number) => `in ${hours}h`,
    hoursAgo: (hours: number) => `${hours}h ago`,
    inDays: (days: number) => `in ${days}d`,
    daysAgo: (days: number) => `${days}d ago`,
    once: 'once',
    noTasks: 'No scheduled tasks.',
    next: 'next',
    started: 'started',
    last: 'last',
    noActiveRun: 'no active run',
    enabled: 'enabled',
    paused: 'paused',
    nameLabel: 'name:',
    scheduleLabel: 'schedule:',
    nextRunLabel: 'next run:',
    lastRunLabel: 'last run:',
    retriesLabel: 'retries:',
    executionLabel: 'execution:',
    deliveryLabel: 'delivery:',
    skillsLabel: 'skills:',
    invalidSkills: 'invalid persisted selection',
    unattended: 'unattended',
    attended: 'attended',
    localNotification: 'local notification',
    retriesSummary: (maxAttempts: number, backoffMs: number, attempt: number) =>
      `up to ${maxAttempts} (backoff ${backoffMs}ms, attempt ${attempt})`,
    createdByLabel: 'created by:',
    instructionLabel: 'instruction:',
    lastErrorLabel: 'last error:',
    scheduled: 'scheduled',
    updated: 'updated',
    removed: 'removed',
    triggered: 'triggered',
    deliverySuppressed:
      'scheduler-owned delivery suppressed; agent tools and job state still executed',
    seeRuns: (id: string) => `see 'sepilot schedule runs ${id}'`,
    resumed: 'resumed',
    unattendedChanged: (unattended: boolean) =>
      `unattended execution ${unattended ? 'enabled' : 'disabled'}`,
    routeChanged: 'delivery route updated',
    routeCleared: 'delivery route cleared',
    noRuns: 'No runs recorded.',
    ok: 'ok',
    failed: 'failed',
    incomplete: 'incomplete',
    legacyIncompleteConflict: '(stored as success by an older runtime)',
    running: 'running',
    manual: 'manual',
    attempt: (attempt: number) => `attempt ${attempt}`,
  },
  ko: {
    failedList: '예약 작업 목록을 불러오지 못했습니다',
    failedLoad: (id: string) => `예약 작업 ${id}을(를) 불러오지 못했습니다`,
    failedCreate: '예약 작업을 만들지 못했습니다',
    failedUpdate: (id: string) => `예약 작업 ${id}을(를) 수정하지 못했습니다`,
    failedRemove: (id: string) => `예약 작업 ${id}을(를) 삭제하지 못했습니다`,
    failedTrigger: (id: string) => `예약 작업 ${id}을(를) 실행하지 못했습니다`,
    failedPause: (id: string) => `예약 작업 ${id}을(를) 일시정지하지 못했습니다`,
    failedResume: (id: string) => `예약 작업 ${id}을(를) 재개하지 못했습니다`,
    failedUnattended: (id: string) => `예약 작업 ${id}의 무인 실행 설정을 변경하지 못했습니다`,
    failedRoute: (id: string) => `예약 작업 ${id}의 전달 경로를 변경하지 못했습니다`,
    failedRuns: (id: string) => `${id}의 실행 기록을 불러오지 못했습니다`,
    failedRun: (id: string, runId: string) => `${id}의 실행 ${runId}을(를) 불러오지 못했습니다`,
    invalidRunsOptions: '--run과 --limit은 동시에 사용할 수 없습니다',
    invalidRunLimit: '실행 기록 limit은 정수여야 합니다',
    pauseResumeConflict: '--pause와 --resume은 동시에 사용할 수 없습니다',
    invalidUnattendedState: '무인 실행 상태는 "on" 또는 "off"여야 합니다.',
    invalidDeliveryRoute:
      '전달 경로에는 --channel-type과 --channel-target이 모두 필요하며 --reply-to도 이 쌍이 있어야 합니다.',
    invalidRouteCommand:
      '채널 유형과 대상을 함께 지정하거나, 대상과 --reply-to 없이 "clear"를 사용하세요.',
    invalidSkillSelection: '예약 스킬은 유효한 고정 스킬 ID여야 하며 고유 ID는 최대 8개입니다.',
    skillEditConflict: '--skill과 --clear-skills는 동시에 사용할 수 없습니다.',
    now: '지금',
    inMinutes: (minutes: number) => `${minutes}분 후`,
    minutesAgo: (minutes: number) => `${minutes}분 전`,
    inHours: (hours: number) => `${hours}시간 후`,
    hoursAgo: (hours: number) => `${hours}시간 전`,
    inDays: (days: number) => `${days}일 후`,
    daysAgo: (days: number) => `${days}일 전`,
    once: '1회',
    noTasks: '예약된 작업이 없습니다.',
    next: '다음',
    started: '시작',
    last: '최근',
    noActiveRun: '활성 실행 없음',
    enabled: '활성',
    paused: '일시정지',
    nameLabel: '이름:',
    scheduleLabel: '스케줄:',
    nextRunLabel: '다음 실행:',
    lastRunLabel: '마지막 실행:',
    retriesLabel: '재시도:',
    executionLabel: '실행 방식:',
    deliveryLabel: '전달 경로:',
    skillsLabel: '스킬:',
    invalidSkills: '저장된 선택이 손상됨',
    unattended: '무인 실행',
    attended: '승인 대기',
    localNotification: '로컬 알림',
    retriesSummary: (maxAttempts: number, backoffMs: number, attempt: number) =>
      `최대 ${maxAttempts}회 (backoff ${backoffMs}ms, 현재 시도 ${attempt})`,
    createdByLabel: '생성자:',
    instructionLabel: '지시문:',
    lastErrorLabel: '마지막 오류:',
    scheduled: '예약됨',
    updated: '수정됨',
    removed: '삭제됨',
    triggered: '실행됨',
    deliverySuppressed:
      'scheduler 소유 전달만 억제됨; agent tool과 job 상태 전이는 정상 실행됨',
    seeRuns: (id: string) => `'sepilot schedule runs ${id}'에서 확인`,
    resumed: '재개됨',
    unattendedChanged: (unattended: boolean) =>
      `무인 실행 ${unattended ? '활성화됨' : '비활성화됨'}`,
    routeChanged: '전달 경로 수정됨',
    routeCleared: '전달 경로 제거됨',
    noRuns: '기록된 실행이 없습니다.',
    ok: '성공',
    failed: '실패',
    incomplete: '미완료',
    legacyIncompleteConflict: '(이전 runtime이 success로 저장한 기록)',
    running: '실행 중',
    manual: '수동',
    attempt: (attempt: number) => `${attempt}번째 시도`,
  },
} as const

type ScheduleCopy = (typeof SCHEDULE_COPY)[keyof typeof SCHEDULE_COPY]

function scheduleCopy(): ScheduleCopy {
  return SCHEDULE_COPY[detectCliLocale()] ?? SCHEDULE_COPY.en
}

function fail(err: unknown, msg: string): never {
  outputError({ ok: false, error: friendlyErrorMessage(err) }, () =>
    chalk.red(`${msg}: ${friendlyErrorMessage(err)}`),
  )
  process.exit(1)
}

function failPlain(msg: string): never {
  outputError({ ok: false, error: msg }, () => chalk.red(msg))
  process.exit(1)
}

function relWhen(ts: number, copy: ScheduleCopy): string {
  const d = ts - Date.now()
  const m = Math.round(d / 60_000)
  if (Math.abs(m) < 1) return copy.now
  if (Math.abs(m) < 60) return d > 0 ? copy.inMinutes(m) : copy.minutesAgo(-m)
  const h = Math.round(m / 60)
  if (Math.abs(h) < 24) return d > 0 ? copy.inHours(h) : copy.hoursAgo(-h)
  const days = Math.round(h / 24)
  return d > 0 ? copy.inDays(days) : copy.daysAgo(-days)
}

function scheduleLabel(job: DaemonSchedulerJob, copy: ScheduleCopy): string {
  if (job.kind === 'recurring') return job.cron ?? '?'
  return job.runAt ? new Date(job.runAt).toISOString() : copy.once
}

function scheduleTimingLabel(job: DaemonSchedulerJob, copy: ScheduleCopy): string {
  if (hasPendingScheduledRun(job)) {
    return `${copy.next} ${relWhen(job.nextRunAt, copy)}`
  }
  if (job.status === 'running') {
    return `${copy.started} ${relWhen(job.lastRunAt ?? job.nextRunAt, copy)}`
  }
  if (job.lastRunAt != null && Number.isFinite(job.lastRunAt)) {
    return `${copy.last} ${relWhen(job.lastRunAt, copy)}`
  }
  return copy.noActiveRun
}

function clip(text: string, max = 80): string {
  const flat = text.replace(/\s+/g, ' ').trim()
  return flat.length > max ? `${flat.slice(0, max - 1)}…` : flat
}

function parseUnattendedState(state: string, copy: ScheduleCopy): boolean {
  if (state === 'on') return true
  if (state === 'off') return false
  return failPlain(copy.invalidUnattendedState)
}

function deliveryRouteInput(
  options: ScheduleRouteOptions,
  copy: ScheduleCopy,
): DeliveryRouteInput | null {
  const channelType = options.channelType?.trim() || null
  const channelTarget = options.channelTarget?.trim() || null
  const replyToMessageId = options.replyTo?.trim() || null
  if (Boolean(channelType) !== Boolean(channelTarget) || (replyToMessageId && !channelType)) {
    return failPlain(copy.invalidDeliveryRoute)
  }
  return channelType && channelTarget
    ? {
        channelType,
        channelTarget,
        ...(replyToMessageId ? { replyToMessageId } : {}),
      }
    : null
}

function deliveryLabel(job: DaemonSchedulerJob, copy: ScheduleCopy): string {
  if (!job.channelType || !job.channelTarget) return copy.localNotification
  const reply = job.replyToMessageId ? ` reply=${clip(job.replyToMessageId, 24)}` : ''
  return `${job.channelType} -> ${clip(job.channelTarget, 40)}${reply}`
}

function scheduledSkillSummary(
  job: DaemonSchedulerJob,
  copy: ScheduleCopy,
): { refs: Array<{ name: string }>; label: string } {
  try {
    const refs = scheduledAgentSkillRefsFromMetadata(job.metadata)
    return {
      refs,
      label: refs.length > 0 ? refs.map((ref) => ref.name).join(', ') : '—',
    }
  } catch {
    return { refs: [], label: copy.invalidSkills }
  }
}

function normalizedSkillRefs(ids: readonly string[] | undefined, copy: ScheduleCopy) {
  try {
    return normalizeScheduledAgentSkillRefs((ids ?? []).map((name) => ({ name })))
  } catch {
    return failPlain(copy.invalidSkillSelection)
  }
}

function safeScheduledJob(job: DaemonSchedulerJob, copy: ScheduleCopy) {
  const { metadata: _metadata, ...safe } = job
  const skills = scheduledSkillSummary(job, copy)
  return { ...safe, skillRefs: skills.refs, skillProfileStatus: skills.label === copy.invalidSkills ? 'invalid' : 'valid' }
}

export async function scheduleListCommand(options: UrlOpt & { all?: boolean }) {
  const copy = scheduleCopy()
  const client = new DaemonClient(options.url)
  let jobs: DaemonSchedulerJob[]
  try {
    jobs = await client.listScheduledTasks({ all: options.all })
  } catch (err) {
    fail(err, copy.failedList)
  }
  output({ ok: true, jobs: jobs.map((job) => safeScheduledJob(job, copy)) }, () => {
    if (jobs.length === 0) return chalk.dim(copy.noTasks)
    return jobs
      .map((j) => {
        const flag = j.enabled ? chalk.green('●') : chalk.dim('○')
        const st =
          j.status === 'failed'
            ? chalk.red(j.status)
            : j.lastError
              ? chalk.yellow(j.status)
              : chalk.dim(j.status)
        const tz = j.timezone ? chalk.dim(` [${j.timezone}]`) : ''
        const warn = j.lastError ? chalk.red(`  ⚠ ${clip(j.lastError, 60)}`) : ''
        const execution = j.unattended ? chalk.magenta(copy.unattended) : chalk.dim(copy.attended)
        const delivery = chalk.dim(deliveryLabel(j, copy))
        const skills = scheduledSkillSummary(j, copy)
        const skillLabel = skills.refs.length > 0 || skills.label === copy.invalidSkills
          ? chalk.dim(`  skills=${clip(skills.label, 48)}`)
          : ''
        return `${flag} ${chalk.bold(j.id.slice(0, 8))}  ${chalk.cyan(scheduleLabel(j, copy))}${tz}  ${scheduleTimingLabel(j, copy)}  ${st}  ${execution}  ${delivery}${skillLabel}  ${j.name}${warn}`
      })
      .join('\n')
  })
}

export async function scheduleShowCommand(id: string, options: UrlOpt) {
  const copy = scheduleCopy()
  const client = new DaemonClient(options.url)
  let job: DaemonSchedulerJob
  try {
    job = await client.getScheduledTask(id)
  } catch (err) {
    fail(err, copy.failedLoad(id))
  }
  output({ ok: true, job: safeScheduledJob(job, copy) }, () => {
    const skills = scheduledSkillSummary(job, copy)
    const lines = [
      `${chalk.bold(job.id)}  ${job.enabled ? chalk.green(copy.enabled) : chalk.dim(copy.paused)}  ${job.status}`,
      `${copy.nameLabel.padEnd(12)}${job.name}`,
      `${copy.scheduleLabel.padEnd(12)}${scheduleLabel(job, copy)}${job.timezone ? `  (${job.timezone})` : ''}`,
      `${copy.nextRunLabel.padEnd(12)}${hasPendingScheduledRun(job) ? `${new Date(job.nextRunAt).toISOString()} (${relWhen(job.nextRunAt, copy)})` : '—'}`,
      `${copy.lastRunLabel.padEnd(12)}${job.lastRunAt ? new Date(job.lastRunAt).toISOString() : '—'}`,
      `${copy.retriesLabel.padEnd(12)}${copy.retriesSummary(job.maxAttempts, job.retryBackoffMs, job.attempt)}`,
      `${copy.executionLabel.padEnd(12)}${job.unattended ? copy.unattended : copy.attended}`,
      `${copy.deliveryLabel.padEnd(12)}${deliveryLabel(job, copy)}`,
      `${copy.skillsLabel.padEnd(12)}${skills.label}`,
      `${copy.createdByLabel.padEnd(12)}${job.createdBy}`,
      `${copy.instructionLabel.padEnd(12)}${clip(job.instruction, 200)}`,
    ]
    if (job.lastError) {
      lines.push(chalk.red(`${copy.lastErrorLabel.padEnd(12)}${clip(job.lastError, 200)}`))
    }
    return lines.join('\n')
  })
}

export async function scheduleAddCommand(
  when: string,
  instruction: string,
  options: UrlOpt & ScheduleRouteOptions & {
    name?: string
    timezone?: string
    retries?: string
    backoff?: string
    unattended?: boolean
    skill?: string[]
  },
) {
  const copy = scheduleCopy()
  const client = new DaemonClient(options.url)
  const deliveryRoute = deliveryRouteInput(options, copy)
  const maxAttempts = options.retries
    ? Math.max(1, Number.parseInt(options.retries, 10))
    : undefined
  const retryBackoffMs = options.backoff
    ? Math.max(1000, Number.parseInt(options.backoff, 10))
    : undefined
  const skillRefs = normalizedSkillRefs(options.skill, copy)
  let job: DaemonSchedulerJob
  try {
    job = await client.createScheduledTask({
      when,
      instruction,
      name: options.name,
      timezone: options.timezone,
      maxAttempts,
      retryBackoffMs,
      ...(options.unattended === true ? { unattended: true } : {}),
      ...(skillRefs.length > 0 ? { skillRefs } : {}),
      ...(deliveryRoute ?? {}),
    })
  } catch (err) {
    fail(err, copy.failedCreate)
  }
  output(
    { ok: true, job },
    () =>
      `${chalk.green('✓')} ${copy.scheduled} ${chalk.bold(job.id)} — ${chalk.cyan(scheduleLabel(job, copy))}, ${copy.next} ${relWhen(job.nextRunAt, copy)}`,
  )
}

export async function scheduleEditCommand(
  id: string,
  when: string,
  instruction: string | undefined,
  options: UrlOpt & {
    name?: string
    timezone?: string
    retries?: string
    backoff?: string
    pause?: boolean
    resume?: boolean
    skill?: string[]
    clearSkills?: boolean
  },
) {
  const copy = scheduleCopy()
  if (options.pause && options.resume) {
    failPlain(copy.pauseResumeConflict)
  }
  if (options.clearSkills && options.skill?.length) {
    failPlain(copy.skillEditConflict)
  }
  const client = new DaemonClient(options.url)
  const maxAttempts = options.retries
    ? Math.max(1, Number.parseInt(options.retries, 10))
    : undefined
  const retryBackoffMs = options.backoff
    ? Math.max(1000, Number.parseInt(options.backoff, 10))
    : undefined
  const skillRefs = options.clearSkills
    ? []
    : options.skill
      ? normalizedSkillRefs(options.skill, copy)
      : undefined
  let job: DaemonSchedulerJob
  try {
    job = await client.updateScheduledTask(id, {
      when,
      instruction,
      name: options.name,
      timezone: options.timezone,
      maxAttempts,
      retryBackoffMs,
      ...(skillRefs !== undefined ? { skillRefs } : {}),
      enabled: options.pause ? false : options.resume ? true : undefined,
    })
  } catch (err) {
    fail(err, copy.failedUpdate(id))
  }
  output(
    { ok: true, job },
    () =>
      `${chalk.green('✓')} ${copy.updated} ${chalk.bold(job.id)} — ${chalk.cyan(scheduleLabel(job, copy))}, ${copy.next} ${relWhen(job.nextRunAt, copy)}`,
  )
}

export async function scheduleRescheduleCommand(
  id: string,
  when: string,
  options: UrlOpt & { timezone?: string },
) {
  const copy = scheduleCopy()
  const client = new DaemonClient(options.url)
  let job: DaemonSchedulerJob
  try {
    job = await client.updateScheduledTask(id, {
      when,
      timezone: options.timezone,
    })
  } catch (err) {
    fail(err, copy.failedUpdate(id))
  }
  output(
    { ok: true, job },
    () =>
      `${chalk.green('✓')} ${copy.updated} ${chalk.bold(job.id)} — ${chalk.cyan(scheduleLabel(job, copy))}, ${copy.next} ${relWhen(job.nextRunAt, copy)}`,
  )
}

export async function scheduleRemoveCommand(id: string, options: UrlOpt) {
  const copy = scheduleCopy()
  const client = new DaemonClient(options.url)
  try {
    await client.deleteScheduledTask(id)
  } catch (err) {
    fail(err, copy.failedRemove(id))
  }
  output(
    { ok: true, id, removed: true },
    () => `${chalk.green('✓')} ${copy.removed} ${chalk.bold(id)}`,
  )
}

export async function scheduleRunCommand(
  id: string,
  options: UrlOpt & { delivery?: boolean },
) {
  const copy = scheduleCopy()
  const client = new DaemonClient(options.url)
  let result: DaemonSchedulerManualRunResult
  try {
    const started = options.delivery === false
      ? await client.runScheduledTask(id, {
          suppressDelivery: true,
          waitForCompletion: false,
        })
      : await client.runScheduledTask(id, { waitForCompletion: false })
    result = started.started ? await waitForScheduledRun(client, started) : started
  } catch (err) {
    fail(err, copy.failedTrigger(id))
  }
  if (!result.started) {
    failPlain(`${copy.failedTrigger(result.jobId)}: manual run was not started`)
  }
  const deliveryNote = result.deliverySuppressed
    ? `\n${chalk.yellow(copy.deliverySuppressed)}`
    : ''
  if (result.status === 'failed') {
    outputError(
      { ok: false, error: 'scheduled task execution failed', ...result },
      () =>
        chalk.red(
          `${copy.failedTrigger(result.jobId)} (run: ${result.runId}) — ${copy.seeRuns(result.jobId)}${deliveryNote}`,
        ),
    )
    process.exit(1)
  }
  output(
    { ok: true, triggered: true, ...result },
    () =>
      `${chalk.green('✓')} ${copy.triggered} ${chalk.bold(result.jobId)} (run: ${result.runId}) — ${copy.seeRuns(result.jobId)}${deliveryNote}`,
  )
}

export async function schedulePauseCommand(id: string, options: UrlOpt) {
  const copy = scheduleCopy()
  const client = new DaemonClient(options.url)
  let job: DaemonSchedulerJob
  try {
    job = await client.pauseScheduledTask(id)
  } catch (err) {
    fail(err, copy.failedPause(id))
  }
  output({ ok: true, job }, () => `${chalk.yellow('⏸')} ${copy.paused} ${chalk.bold(id)}`)
}

export async function scheduleResumeCommand(id: string, options: UrlOpt) {
  const copy = scheduleCopy()
  const client = new DaemonClient(options.url)
  let job: DaemonSchedulerJob
  try {
    job = await client.resumeScheduledTask(id)
  } catch (err) {
    fail(err, copy.failedResume(id))
  }
  output({ ok: true, job }, () => `${chalk.green('▶')} ${copy.resumed} ${chalk.bold(id)}`)
}

export async function scheduleUnattendedCommand(
  id: string,
  state: string,
  options: UrlOpt,
) {
  const copy = scheduleCopy()
  const unattended = parseUnattendedState(state, copy)
  const client = new DaemonClient(options.url)
  let job: DaemonSchedulerJob
  try {
    job = await client.updateScheduledTask(id, { unattended })
  } catch (err) {
    fail(err, copy.failedUnattended(id))
  }
  output(
    { ok: true, job },
    () => `${chalk.green('✓')} ${copy.unattendedChanged(unattended)} ${chalk.bold(job.id)}`,
  )
}

export async function scheduleRouteCommand(
  id: string,
  channelType: string,
  channelTarget: string | undefined,
  options: UrlOpt & { replyTo?: string },
) {
  const copy = scheduleCopy()
  const clear = channelType === 'clear'
  if (clear && (channelTarget || options.replyTo)) return failPlain(copy.invalidRouteCommand)
  if (!clear && !channelTarget?.trim()) return failPlain(copy.invalidRouteCommand)

  const route = clear
    ? { channelType: null, channelTarget: null, replyToMessageId: null }
    : {
        ...deliveryRouteInput({ channelType, channelTarget, replyTo: options.replyTo }, copy)!,
        replyToMessageId: options.replyTo?.trim() || null,
      }
  const client = new DaemonClient(options.url)
  let job: DaemonSchedulerJob
  try {
    job = await client.updateScheduledTask(id, route)
  } catch (err) {
    fail(err, copy.failedRoute(id))
  }
  output(
    { ok: true, job },
    () => `${chalk.green('✓')} ${clear ? copy.routeCleared : copy.routeChanged} ${chalk.bold(job.id)}`,
  )
}

export async function scheduleRunsCommand(
  id: string,
  options: UrlOpt & { limit?: string; run?: string },
) {
  const copy = scheduleCopy()
  if (options.run && options.limit !== undefined) failPlain(copy.invalidRunsOptions)
  if (options.limit !== undefined && !/^\d+$/u.test(options.limit)) {
    failPlain(copy.invalidRunLimit)
  }
  const client = new DaemonClient(options.url)
  const limit = options.limit ? Math.max(1, Math.min(Number.parseInt(options.limit, 10), 200)) : 20
  let runs: DaemonSchedulerJobRun[]
  try {
    runs = options.run
      ? [await client.getScheduledTaskRun(id, options.run)]
      : await client.listScheduledTaskRuns(id, limit)
  } catch (err) {
    fail(err, options.run ? copy.failedRun(id, options.run) : copy.failedRuns(id))
  }
  output({ ok: true, id, runs }, () => {
    if (runs.length === 0) return chalk.dim(copy.noRuns)
    return runs
      .map((r) => {
        const taskOutcome = r.taskOutcome
          ?? (r.status === 'success' ? 'complete' : r.status)
        const st =
          taskOutcome === 'complete'
            ? chalk.green(copy.ok)
            : taskOutcome === 'incomplete'
              ? chalk.yellow(copy.incomplete)
              : taskOutcome === 'failed'
              ? chalk.red(copy.failed)
              : chalk.yellow(copy.running)
        const integrity = r.statusIntegrity === 'legacy-incomplete-conflict'
          ? chalk.yellow(` ${copy.legacyIncompleteConflict}`)
          : ''
        const dur = r.durationMs != null ? `${r.durationMs}ms` : '—'
        const att = r.attempt === 0 ? copy.manual : copy.attempt(r.attempt)
        const tail = r.error
          ? chalk.red(`  ⚠ ${clip(r.error)}`)
          : r.outputExcerpt
            ? chalk.dim(`  ${clip(r.outputExcerpt)}`)
            : ''
        const agentSessionId = daemonSchedulerRunAgentSessionId(r)
        const agentSession = agentSessionId
          ? ` session=${agentSessionId}`
          : ''
        const inspectSession = agentSessionId
          ? `\n${chalk.dim(`  inspect: sepilot sessions show ${agentSessionId}`)}`
          : ''
        return `${st}${integrity}\t${new Date(r.startedAt).toISOString()}  ${dur}  ${chalk.dim(att)}  job=${r.jobId} run=${r.id}${agentSession}${tail}${inspectSession}`
      })
      .join('\n')
  })
}
