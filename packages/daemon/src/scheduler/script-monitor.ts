import { createHash } from 'node:crypto'
import { spawn } from 'node:child_process'
import { isAbsolute } from 'node:path'
import { z } from 'zod'
import { redactSensitive } from '../memory/sensitive.js'
import {
  createMonitorStateRepo,
  type MonitorEvaluationResult,
  type MonitorObservedStatus,
  type MonitorSeverity,
} from '../monitoring/state-repo.js'
import type { ScheduledJob } from './job-store.js'

const OUTPUT_MAX_BYTES = 64 * 1024
const SUMMARY_MAX_CHARS = 1_200
const ARG_MAX_CHARS = 2_048
export const quoteRemoteArgument = (value: string) => `'${value.replaceAll("'", "'\\''")}'`

const portableId = z.string().trim().min(1).max(120)
  .regex(/^[a-z0-9](?:[a-z0-9._:-]{0,118}[a-z0-9])?$/i)
const contractId = z.string().trim().min(1).max(80)
  .regex(/^[a-z0-9](?:[a-z0-9._:-]{0,78}[a-z0-9])?$/i)
const safeText = z.string().max(ARG_MAX_CHARS).refine((value) => !/[\u0000-\u001f\u007f]/.test(value))
const absolutePath = z.string().trim().min(1).max(4_096)
  .refine((value) => isAbsolute(value) && !/[\u0000-\u001f\u007f]/.test(value), 'must be an absolute path')

const targetSchema = z.discriminatedUnion('kind', [
  z.object({ kind: z.literal('local') }).strict(),
  z.object({
    kind: z.literal('ssh'),
    host: z.string().trim().min(1).max(253).regex(/^[a-zA-Z0-9:][a-zA-Z0-9._:-]*$/),
    user: z.string().trim().min(1).max(128).regex(/^[a-zA-Z0-9_][a-zA-Z0-9_.-]*$/).optional(),
    port: z.number().int().min(1).max(65_535).optional(),
  }).strict(),
])

export const scriptMonitorConfigSchema = z.object({
  version: z.literal(1),
  monitorId: portableId,
  contractVersion: contractId,
  target: targetSchema,
  scriptPath: absolutePath,
  args: z.array(safeText).max(32).default([]),
  cwd: absolutePath.optional(),
  timeoutMs: z.number().int().min(1_000).max(10 * 60_000).default(30_000),
  anomalyConsecutiveSamples: z.number().int().min(1).max(100).default(2),
  recoveryConsecutiveSamples: z.number().int().min(1).max(100).default(2),
  errorConsecutiveSamples: z.number().int().min(1).max(100).default(2),
  /** Optional periodic status report. Transitions are always delivered immediately. */
  reportEveryMs: z.number().int().min(5 * 60_000).max(30 * 24 * 60 * 60_000).nullable().default(null),
}).strict()

export type ScriptMonitorConfig = z.infer<typeof scriptMonitorConfigSchema>

const observationSchema = z.object({
  version: z.literal(1),
  status: z.enum(['healthy', 'anomaly', 'unknown']),
  severity: z.enum(['warning', 'high', 'critical']).optional(),
  summary: z.string().trim().min(1).max(SUMMARY_MAX_CHARS),
  metrics: z.record(z.number().finite()).refine((value) => Object.keys(value).length <= 32).default({}),
  observedAt: z.string().datetime({ offset: true }).optional(),
}).strict().superRefine((value, ctx) => {
  if (value.status === 'anomaly' && !value.severity) {
    ctx.addIssue({ code: z.ZodIssueCode.custom, path: ['severity'], message: 'severity is required for anomaly' })
  }
  if (value.status !== 'anomaly' && value.severity) {
    ctx.addIssue({ code: z.ZodIssueCode.custom, path: ['severity'], message: 'severity is only valid for anomaly' })
  }
})

interface ScriptMonitorRuntimeState {
  lastReportAt?: number
}

export interface ScriptMonitorRunResult {
  output: string
  evaluation: MonitorEvaluationResult
  notified: boolean
  notificationKind: 'transition' | 'periodic' | null
}

export interface ScriptMonitorRunnerDeps {
  updateMetadata(jobId: string, metadata: Record<string, unknown>): void
  notify(input: {
    id: string
    title: string
    body: string
    priority: 'normal' | 'high' | 'critical'
    jobId: string
    runId?: string
  }): void
  now?: () => number
}

export function scriptMonitorConfigFromMetadata(
  metadata: Record<string, unknown> | null | undefined,
): ScriptMonitorConfig | null {
  if (!metadata || !Object.prototype.hasOwnProperty.call(metadata, 'scriptMonitor')) return null
  return scriptMonitorConfigSchema.parse(metadata.scriptMonitor)
}

function hash(value: string): string {
  return createHash('sha256').update(value).digest('hex')
}

function monitorEnvironment(): NodeJS.ProcessEnv {
  const allowed = ['PATH', 'HOME', 'LANG', 'LC_ALL', 'TMPDIR', 'USER', 'LOGNAME', 'SHELL', 'SSH_AUTH_SOCK'] as const
  return Object.fromEntries(
    allowed.flatMap((name) => process.env[name] === undefined ? [] : [[name, process.env[name]]]),
  )
}

function runtimeState(metadata: Record<string, unknown> | null): ScriptMonitorRuntimeState {
  const value = metadata?.scriptMonitorState
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {}
  const lastReportAt = (value as Record<string, unknown>).lastReportAt
  return Number.isFinite(lastReportAt) ? { lastReportAt: Number(lastReportAt) } : {}
}

export function scriptMonitorLaunch(config: ScriptMonitorConfig): { command: string; args: string[]; cwd: string } {
  if (config.target.kind !== 'ssh') {
    return { command: config.scriptPath, args: config.args, cwd: config.cwd ?? process.cwd() }
  }
  const remote = [config.scriptPath, ...config.args].map(quoteRemoteArgument).join(' ')
  const command = config.cwd ? `cd ${quoteRemoteArgument(config.cwd)} && exec ${remote}` : `exec ${remote}`
  const args = [
    '-o', 'BatchMode=yes',
    '-o', 'StrictHostKeyChecking=yes',
    '-o', 'ForwardAgent=no',
    '-o', 'ClearAllForwardings=yes',
    '-o', 'ConnectTimeout=10',
    '-o', 'ServerAliveInterval=15',
    '-o', 'ServerAliveCountMax=2',
  ]
  if (config.target.port) args.push('-p', String(config.target.port))
  if (config.target.user) args.push('-l', config.target.user)
  args.push(config.target.host, `exec "\${SHELL:-/bin/sh}" -lc ${quoteRemoteArgument(command)}`)
  return { command: 'ssh', args, cwd: process.cwd() }
}

async function executeProbe(
  config: ScriptMonitorConfig,
  signal?: AbortSignal,
): Promise<string> {
  const launch = scriptMonitorLaunch(config)
  return await new Promise<string>((resolve, reject) => {
    const child = spawn(launch.command, launch.args, {
      cwd: launch.cwd,
      // Scheduled probes do not inherit provider tokens or daemon credentials.
      env: monitorEnvironment(),
      shell: false,
      stdio: ['ignore', 'pipe', 'pipe'],
    })
    let stdout = ''
    let stderr = ''
    let outputBytes = 0
    let settled = false
    const finish = (error?: Error) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      signal?.removeEventListener('abort', abort)
      if (error) reject(error)
      else resolve(stdout)
    }
    const append = (current: string, chunk: Buffer): string => {
      outputBytes += chunk.length
      if (outputBytes > OUTPUT_MAX_BYTES) {
        child.kill('SIGKILL')
        finish(new Error(`script output exceeded ${OUTPUT_MAX_BYTES} bytes`))
      }
      return current + chunk.toString('utf8')
    }
    child.stdout.on('data', (chunk: Buffer) => { stdout = append(stdout, chunk) })
    child.stderr.on('data', (chunk: Buffer) => { stderr = append(stderr, chunk) })
    child.once('error', (error) => finish(error))
    child.once('close', (code, terminatedBy) => {
      if (code === 0) return finish()
      const detail = redactSensitive(stderr.trim().slice(0, 500)).redacted
      finish(new Error(`script exited ${code ?? terminatedBy ?? 'unknown'}${detail ? `: ${detail}` : ''}`))
    })
    const abort = () => {
      child.kill('SIGTERM')
      finish(new Error('script monitor aborted'))
    }
    signal?.addEventListener('abort', abort, { once: true })
    const timer = setTimeout(() => {
      child.kill('SIGKILL')
      finish(new Error(`script timed out after ${config.timeoutMs}ms`))
    }, config.timeoutMs)
    timer.unref?.()
    if (signal?.aborted) abort()
  })
}

function priorityFor(evaluation: MonitorEvaluationResult): 'normal' | 'high' | 'critical' {
  if (evaluation.stableSeverity === 'critical') return 'critical'
  if (evaluation.stableSeverity === 'high' || evaluation.transitions.includes('observation_error')) return 'high'
  return 'normal'
}

function notificationTitle(job: ScheduledJob, evaluation: MonitorEvaluationResult, periodic: boolean): string {
  if (periodic) return `Monitor report: ${job.name}`
  if (evaluation.transitions.includes('anomaly_recovered')) return `Monitor recovered: ${job.name}`
  if (evaluation.transitions.includes('observation_error')) return `Monitor probe failed: ${job.name}`
  if (evaluation.transitions.includes('severity_changed')) return `Monitor severity changed: ${job.name}`
  return `Monitor alert: ${job.name}`
}

export function createScriptMonitorRunner(deps: ScriptMonitorRunnerDeps) {
  return async (
    job: ScheduledJob,
    context?: { runId?: string; signal?: AbortSignal; suppressDelivery?: boolean },
  ): Promise<ScriptMonitorRunResult> => {
    const config = scriptMonitorConfigFromMetadata(job.metadata)
    if (!config) throw new Error('script monitor configuration is missing')
    const now = deps.now?.() ?? Date.now()
    let status: MonitorObservedStatus
    let severity: MonitorSeverity | null
    let summary: string
    let metrics: Record<string, number>
    let observedAt = now
    let observedAtForHash: string | null = null
    let evidenceRedacted = false
    try {
      const parsed = observationSchema.parse(JSON.parse((await executeProbe(config, context?.signal)).trim()))
      status = parsed.status
      severity = parsed.status === 'anomaly' ? parsed.severity! : null
      const redacted = redactSensitive(parsed.summary)
      summary = redacted.redacted
      evidenceRedacted = redacted.found
      metrics = parsed.metrics
      observedAt = parsed.observedAt ? Date.parse(parsed.observedAt) : now
      observedAtForHash = parsed.observedAt ?? null
      if (observedAt > now + 5 * 60_000) throw new Error('observedAt is more than 5 minutes in the future')
    } catch (error) {
      if (context?.signal?.aborted) throw error
      status = 'unknown'
      severity = null
      const redacted = redactSensitive(error instanceof Error ? error.message : String(error))
      summary = `Probe error: ${redacted.redacted}`.slice(0, SUMMARY_MAX_CHARS)
      evidenceRedacted = redacted.found
      metrics = {}
    }
    const thresholds = {
      anomaly: config.anomalyConsecutiveSamples,
      recovery: config.recoveryConsecutiveSamples,
      error: config.errorConsecutiveSamples,
    }
    const scopeKey = `scheduler-script-monitor:${job.id}`
    const sampleId = context?.runId ?? `${job.id}:${job.nextRunAt}:${job.attempt}`
    const contractHash = hash(JSON.stringify({
      target: config.target,
      scriptPath: config.scriptPath,
      args: config.args,
      cwd: config.cwd ?? null,
      thresholds,
    }))
    const observation = { status, severity, summary, metrics, observedAt: observedAtForHash }
    const evaluation = createMonitorStateRepo().evaluate({
      scopeKey,
      monitorId: config.monitorId,
      contractVersion: config.contractVersion,
      contractHash,
      sampleHash: hash(`${scopeKey}\0${config.monitorId}\0${config.contractVersion}\0${sampleId}`),
      observationHash: hash(JSON.stringify(observation)),
      observedStatus: status,
      severity,
      evidenceSummary: summary,
      evidenceRedacted,
      metrics,
      thresholds,
      observedAt,
      now,
    })

    const state = runtimeState(job.metadata)
    const periodic = config.reportEveryMs !== null
      && now - (state.lastReportAt ?? 0) >= config.reportEveryMs
    const notificationKind = evaluation.shouldNotify ? 'transition' : periodic ? 'periodic' : null
    const notified = Boolean(notificationKind && !context?.suppressDelivery)
    if (notified) {
      const deliveryIdentity = evaluation.notificationDedupKey
        ?? `${job.id}:periodic:${Math.floor(now / config.reportEveryMs!)}`
      const id = `script-monitor:${hash(`${job.id}\0${deliveryIdentity}`).slice(0, 32)}`
      deps.notify({
        id,
        title: notificationTitle(job, evaluation, periodic && !evaluation.shouldNotify),
        body: summary,
        priority: priorityFor(evaluation),
        jobId: job.id,
        ...(context?.runId ? { runId: context.runId } : {}),
      })
      deps.updateMetadata(job.id, {
        ...(job.metadata ?? {}),
        scriptMonitorState: { lastReportAt: now },
      })
    }
    const metricText = Object.entries(metrics).map(([name, value]) => `${name}=${value}`).join(', ')
    return {
      output: `${status}${severity ? `/${severity}` : ''}: ${summary}${metricText ? ` (${metricText})` : ''}`.slice(0, 2_000),
      evaluation,
      notified,
      notificationKind,
    }
  }
}
