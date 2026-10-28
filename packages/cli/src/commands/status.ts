import chalk from 'chalk'
import { writeFile } from 'node:fs/promises'
import type {
  DaemonChatBackgroundListItem,
  DaemonHealthComponent,
  DaemonProviderInfo,
} from '@sepilotd/api-client'
import { DaemonClient } from '../client/http.js'
import { ensureDaemon } from '../client/ensure-daemon.js'
import { output, getOutputFormat } from '../output/formatter.js'
import { formatHealthComponentIcon } from './health-display.js'
import { detectCliLocale } from '../utils/locale.js'

const STATUS_COPY = {
  en: {
    exportFailed: 'Failed to export health report',
    reportWrittenTo: (path: string) => `Health report written to ${path}`,
    uptimePrefix: 'Uptime:',
    memoryPrefix: (rss: number, heap: number) => `Memory: ${rss}MB RSS / ${heap}MB Heap`,
    providersPrefix: (list: string) => `\nProviders: ${list}`,
    usagePrefix: (requests: number, tokens: number, cost: number) =>
      `Usage: ${requests} requests, ${tokens} tokens, $${cost.toFixed(4)}`,
    backgroundJobsHeader: (count: number) =>
      `\nBackground jobs: ${count} active`,
    backgroundRunning: (jobId: string, sessionId: string, label: string) =>
      `  - ${jobId} running session=${sessionId}${label ? ` (${label})` : ''}`,
    backgroundWaitingApproval: (jobId: string, sessionId: string, toolName: string) =>
      `  - ${jobId} waiting approval session=${sessionId} tool=${toolName}`,
    backgroundWaitingQuestion: (jobId: string, sessionId: string) =>
      `  - ${jobId} waiting answer session=${sessionId}`,
    backgroundRequest: (preview: string) => `    request: ${preview}`,
    backgroundQuestion: (prompt: string) => `    question: ${prompt}`,
    backgroundChoices: (choices: string[]) => `    choices: ${choices.join(' · ')}`,
    backgroundWait: (jobId: string) => `    wait: sepilot ask --background-status ${jobId} --wait`,
    backgroundApprove: (requestId: string, sessionId: string) =>
      `    approve: sepilot approve ${requestId} --session ${sessionId} --scope run`,
    backgroundDeny: (requestId: string, sessionId: string) =>
      `    deny: sepilot deny ${requestId} --session ${sessionId}`,
    backgroundAnswer: (sessionId: string, questionId: string) =>
      `    answer: sepilot answer ${sessionId} ${questionId} <reply>`,
    backgroundCancel: (jobId: string) => `    cancel: sepilot ask --background-cancel ${jobId}`,
    cannotConnect: 'Cannot connect to sepilotd.',
    httpFailure: (status: number) => `Daemon responded with HTTP ${status}. Check credentials and daemon diagnostics.`,
    probeTimeout: 'Daemon liveness check timed out. Check server load and connection settings.',
    diagnosticsFailed: 'Daemon is reachable, but detailed status collection failed.',
    startHint: 'Is the daemon running? Start with: sepilot start',
    unknown: 'unknown',
  },
  ko: {
    exportFailed: '상태 리포트 내보내기 실패',
    reportWrittenTo: (path: string) => `상태 리포트가 ${path}에 작성되었습니다`,
    uptimePrefix: '가동 시간:',
    memoryPrefix: (rss: number, heap: number) => `메모리: ${rss}MB RSS / ${heap}MB Heap`,
    providersPrefix: (list: string) => `\nProvider: ${list}`,
    usagePrefix: (requests: number, tokens: number, cost: number) =>
      `사용량: 요청 ${requests}회, 토큰 ${tokens}개, $${cost.toFixed(4)}`,
    backgroundJobsHeader: (count: number) =>
      `\n백그라운드 작업: 활성 ${count}개`,
    backgroundRunning: (jobId: string, sessionId: string, label: string) =>
      `  - ${jobId} 실행 중 session=${sessionId}${label ? ` (${label})` : ''}`,
    backgroundWaitingApproval: (jobId: string, sessionId: string, toolName: string) =>
      `  - ${jobId} 승인 대기 session=${sessionId} tool=${toolName}`,
    backgroundWaitingQuestion: (jobId: string, sessionId: string) =>
      `  - ${jobId} 답변 대기 session=${sessionId}`,
    backgroundRequest: (preview: string) => `    요청: ${preview}`,
    backgroundQuestion: (prompt: string) => `    질문: ${prompt}`,
    backgroundChoices: (choices: string[]) => `    선택지: ${choices.join(' · ')}`,
    backgroundWait: (jobId: string) => `    대기: sepilot ask --background-status ${jobId} --wait`,
    backgroundApprove: (requestId: string, sessionId: string) =>
      `    승인: sepilot approve ${requestId} --session ${sessionId} --scope run`,
    backgroundDeny: (requestId: string, sessionId: string) =>
      `    거부: sepilot deny ${requestId} --session ${sessionId}`,
    backgroundAnswer: (sessionId: string, questionId: string) =>
      `    답변: sepilot answer ${sessionId} ${questionId} <reply>`,
    backgroundCancel: (jobId: string) => `    취소: sepilot ask --background-cancel ${jobId}`,
    cannotConnect: 'sepilotd에 연결할 수 없습니다.',
    httpFailure: (status: number) => `daemon이 HTTP ${status}로 응답했습니다. 인증 정보와 서버 진단을 확인하세요.`,
    probeTimeout: 'daemon 연결 확인 시간이 초과됐습니다. 서버 부하와 연결 설정을 확인하세요.',
    diagnosticsFailed: 'daemon에 연결됐지만 상세 상태를 수집하지 못했습니다.',
    startHint: 'daemon이 실행 중인가요? 시작: sepilot start',
    unknown: '알 수 없음',
  },
} as const

type StatusCopy = (typeof STATUS_COPY)[keyof typeof STATUS_COPY]

function truncateStatusDetail(text: string | undefined, limit = 160): string | undefined {
  if (!text) return undefined
  const normalized = text.replace(/\s+/g, ' ').trim()
  return normalized.length > limit
    ? `${normalized.slice(0, limit - 3).trimEnd()}...`
    : normalized
}

function formatBackgroundJobLines(
  job: DaemonChatBackgroundListItem,
  copy: StatusCopy,
): string[] {
  const action = job.progress?.action
  if (action?.type === 'approval') {
    const preview = truncateStatusDetail(action.preview ?? job.progress?.detail)
    return [
      copy.backgroundWaitingApproval(job.jobId, job.sessionId, action.toolName),
      preview ? copy.backgroundRequest(preview) : '',
      copy.backgroundApprove(action.requestId, job.sessionId),
      copy.backgroundDeny(action.requestId, job.sessionId),
      copy.backgroundCancel(job.jobId),
    ].filter(Boolean)
  }

  if (action?.type === 'question') {
    const prompt = truncateStatusDetail(job.progress?.detail)
    return [
      copy.backgroundWaitingQuestion(job.jobId, job.sessionId),
      prompt ? copy.backgroundQuestion(prompt) : '',
      action.choices?.length ? copy.backgroundChoices(action.choices.slice(0, 8)) : '',
      copy.backgroundAnswer(job.sessionId, action.questionId),
      copy.backgroundCancel(job.jobId),
    ].filter(Boolean)
  }

  return [
    copy.backgroundRunning(job.jobId, job.sessionId, truncateStatusDetail(job.progress?.label, 80) ?? ''),
    copy.backgroundWait(job.jobId),
    copy.backgroundCancel(job.jobId),
  ]
}

function reportStatusFailure(error: unknown, copy: StatusCopy, connected: boolean): never {
  const status = (error as { status?: number } | null)?.status
  const timeout = (error as { code?: string } | null)?.code === 'DAEMON_PROBE_TIMEOUT'
    || ['TimeoutError', 'ApiTimeoutError'].includes((error as { name?: string } | null)?.name ?? '')
  const code = typeof status === 'number' ? 'daemon-http-error' : timeout ? 'daemon-probe-timeout' : connected ? 'daemon-diagnostics-failed' : 'daemon-unreachable'
  const message = typeof status === 'number' ? copy.httpFailure(status) : timeout ? copy.probeTimeout : connected ? copy.diagnosticsFailed : copy.cannotConnect
  if (getOutputFormat() === 'json') console.log(JSON.stringify({ ok: false, error: code, message, ...(typeof status === 'number' ? { statusCode: status } : {}) }, null, 2))
  else { console.error(chalk.red(message)); if (code === 'daemon-unreachable') console.error(chalk.gray(copy.startHint)) }
  process.exit(1)
}

export async function statusCommand(options: {
  url?: string
  report?: boolean
  output?: string
}) {
  const copy = STATUS_COPY[detectCliLocale()] ?? STATUS_COPY.en
  const client = new DaemonClient(options.url)

  try {
    await ensureDaemon(client, { url: options.url, quiet: true })
  } catch (error) { reportStatusFailure(error, copy, false) }

  if (options.report) {
    const reportFormat = getOutputFormat() === 'json' ? 'json' : 'markdown'
    try {
      if (reportFormat === 'json') {
        const payload = await client.healthReport('json')
        if (options.output) {
          await writeFile(options.output, JSON.stringify(payload, null, 2), 'utf-8')
          console.log(chalk.green(copy.reportWrittenTo(options.output)))
          return
        }
        output(payload)
        return
      }

      const markdown = await client.healthReport('markdown')
      if (options.output) {
        await writeFile(options.output, markdown, 'utf-8')
        console.log(chalk.green(copy.reportWrittenTo(options.output)))
        return
      }
      console.log(markdown)
      return
    } catch (error) {
      const details = error instanceof Error ? error.message : String(error)
      if (getOutputFormat() === 'json') {
        output({ ok: false, error: 'health-report-export-failed', details })
      } else {
        console.error(chalk.red(`${copy.exportFailed}: ${details}`))
      }
      process.exit(1)
    }
  }

  try {
    const health = await client.health()

    if (getOutputFormat() === 'json') {
      output(health)
      return
    }

    console.log(chalk.green(`sepilotd ${health.version} — ${health.status}`))
    console.log(chalk.gray(`${copy.uptimePrefix} ${formatUptime(health.uptime, copy.unknown)}`))

    if (health.memory) {
      console.log(chalk.gray(copy.memoryPrefix(health.memory.rss, health.memory.heap)))
    }

    if (health.components) {
      console.log()
      for (const [name, comp] of Object.entries(health.components) as [string, DaemonHealthComponent][]) {
        const icon = formatHealthComponentIcon(comp.status)
        const optional = comp.optional === true ? chalk.gray(' optional') : ''
        const details = comp.details ? chalk.gray(` (${comp.details})`) : ''
        console.log(`  ${icon} ${name.padEnd(15)}${optional}${details}`)
      }
    }

    try {
      const providers = await client.providers()
      if (providers?.length) {
        console.log(chalk.gray(copy.providersPrefix(providers.map((p: DaemonProviderInfo) => p.id).join(', '))))
      }
    } catch {}

    try {
      const usage = await client.usage()
      if (usage.requestCount > 0) {
        console.log(chalk.gray(copy.usagePrefix(usage.requestCount, usage.inputTokens + usage.outputTokens, usage.costUsd)))
      }
    } catch {}

    try {
      const backgroundJobs = await client.backgroundChatJobs()
      const activeJobs = backgroundJobs.jobs.filter((job) => job.status === 'running')
      if (activeJobs.length > 0) {
        console.log(chalk.gray(copy.backgroundJobsHeader(activeJobs.length)))
        for (const job of activeJobs.slice(0, 5)) {
          for (const line of formatBackgroundJobLines(job, copy)) {
            console.log(chalk.gray(line))
          }
        }
        if (activeJobs.length > 5) {
          console.log(chalk.gray(`  ... ${activeJobs.length - 5} more`))
        }
      }
    } catch {}

  } catch (error) { reportStatusFailure(error, copy, true) }
}

function formatUptime(seconds: number | undefined, unknownLabel = 'unknown'): string {
  if (!seconds) return unknownLabel
  if (seconds < 60) return `${Math.round(seconds)}s`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  return `${h}h ${m}m`
}

export const __testables = {
  formatUptime,
}
