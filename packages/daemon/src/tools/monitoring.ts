import { createHash } from 'node:crypto'
import { SessionInbox } from '../server/runtime/session-inbox.js'
import { redactSensitive } from '../memory/sensitive.js'
import {
  createMonitorStateRepo,
  type MonitorObservedStatus,
  type MonitorSeverity,
  type MonitorThresholds,
} from '../monitoring/state-repo.js'
import type {
  ToolDefinitionRuntime,
  ToolExecutionContext,
  ToolResult,
} from './registry.js'

const MONITOR_ID_PATTERN = /^[a-z0-9](?:[a-z0-9._:-]{0,118}[a-z0-9])?$/i
const CONTRACT_VERSION_PATTERN = /^[a-z0-9](?:[a-z0-9._-]{0,78}[a-z0-9])?$/i
const METRIC_NAME_PATTERN = /^[a-z][a-z0-9._:-]{0,62}[a-z0-9]$/i
const MAX_EVIDENCE_LENGTH = 1_200
const MAX_METRICS = 32
const MAX_CONSECUTIVE_SAMPLES = 100
const MAX_REPORT_HOURS = 24 * 90
const MAX_FUTURE_OBSERVATION_SKEW_MS = 5 * 60 * 1_000

class MonitorInputError extends Error {
  constructor(
    message: string,
    readonly code = 'INVALID_INPUT_PERMANENT',
  ) {
    super(message)
  }
}

function failure(error: unknown, startedAt: number): ToolResult {
  const message = error instanceof Error ? error.message : String(error)
  const explicitCode = error instanceof MonitorInputError
    ? error.code
    : /^MONITOR_[A-Z_]+(?:_PERMANENT|_USER)$/.test(message)
      ? message
      : null
  return {
    status: 'error',
    code: explicitCode ?? 'MONITOR_STATE_TRANSIENT',
    output: explicitCode
      ? explicitCode === 'MONITOR_CONTRACT_MISMATCH_USER'
        ? 'The thresholds changed without a new contractVersion. Keep the accepted thresholds or increment contractVersion after the user accepts the new contract.'
        : explicitCode === 'MONITOR_SAMPLE_CONFLICT_PERMANENT'
          ? 'sampleId was already used for a different observation. Reuse it only for an exact retry, or submit the new observation with a new sampleId.'
          : message
      : 'The monitoring state could not be updated. Retry this exact sampleId once; idempotency prevents a duplicate transition.',
    durationMs: Date.now() - startedAt,
  }
}

function requiredString(
  value: unknown,
  name: string,
  pattern: RegExp,
  maxLength: number,
): string {
  const text = typeof value === 'string' ? value.trim() : ''
  if (!text || text.length > maxLength || !pattern.test(text)) {
    throw new MonitorInputError(
      `${name} must be a portable ${maxLength}-character identifier using letters, numbers, dot, underscore, colon, or hyphen.`,
    )
  }
  return text
}

function boundedInteger(value: unknown, name: string, max: number): number {
  if (!Number.isInteger(value) || Number(value) < 1 || Number(value) > max) {
    throw new MonitorInputError(`${name} must be an integer from 1 to ${max}.`)
  }
  return Number(value)
}

function observedStatus(value: unknown): MonitorObservedStatus {
  if (value === 'healthy' || value === 'anomaly' || value === 'unknown') return value
  throw new MonitorInputError('observedStatus must be healthy, anomaly, or unknown.')
}

function severityFor(status: MonitorObservedStatus, value: unknown): MonitorSeverity | null {
  if (status !== 'anomaly') {
    if (value !== undefined && value !== null) {
      throw new MonitorInputError('severity must be omitted unless observedStatus is anomaly.')
    }
    return null
  }
  if (value === 'warning' || value === 'high' || value === 'critical') return value
  throw new MonitorInputError('severity must be warning, high, or critical for an anomaly.')
}

function evidenceSummary(value: unknown): {
  original: string
  redacted: string
  foundSensitive: boolean
} {
  const original = typeof value === 'string' ? value.trim().replace(/\s+/g, ' ') : ''
  if (!original) throw new MonitorInputError('evidenceSummary is required.')
  if (original.length > MAX_EVIDENCE_LENGTH) {
    throw new MonitorInputError(`evidenceSummary must be at most ${MAX_EVIDENCE_LENGTH} characters.`)
  }
  const sanitized = redactSensitive(original)
  return { original, redacted: sanitized.redacted, foundSensitive: sanitized.found }
}

function metrics(value: unknown): Record<string, number> {
  if (value === undefined) return {}
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new MonitorInputError('metrics must be an object of bounded numeric observations.')
  }
  const entries = Object.entries(value)
  if (entries.length > MAX_METRICS) {
    throw new MonitorInputError(`metrics may contain at most ${MAX_METRICS} values.`)
  }
  const normalized: Array<[string, number]> = []
  for (const [rawName, rawValue] of entries) {
    const name = rawName.trim()
    if (!METRIC_NAME_PATTERN.test(name)) {
      throw new MonitorInputError(
        'metric names must use 2-64 letters, numbers, dot, underscore, colon, or hyphen and begin with a letter.',
      )
    }
    if (typeof rawValue !== 'number' || !Number.isFinite(rawValue)) {
      throw new MonitorInputError(`metric ${name} must be a finite number.`)
    }
    normalized.push([name, rawValue])
  }
  return Object.fromEntries(normalized.sort(([left], [right]) => left.localeCompare(right)))
}

function timestamp(value: unknown, name: string, fallback: number): number {
  if (value === undefined) return fallback
  if (typeof value !== 'string' || value.length > 80) {
    throw new MonitorInputError(`${name} must be an ISO-8601 timestamp.`)
  }
  const parsed = Date.parse(value)
  if (!Number.isFinite(parsed)) throw new MonitorInputError(`${name} must be an ISO-8601 timestamp.`)
  return parsed
}

function hash(value: string): string {
  return createHash('sha256').update(value).digest('hex')
}

function scopeKey(context: ToolExecutionContext | undefined): string {
  const channel = context?.channelContext?.channel?.trim()
  const chatKey = context?.channelContext?.chatKey?.trim()
  if (channel && chatKey) return `channel:${hash(`${channel}\0${chatKey}`)}`
  const scopeTags = [...new Set(context?.scopeTags ?? [])]
    .map((tag) => tag.trim())
    .filter(Boolean)
    .sort()
  return scopeTags.length > 0 ? `scope:${hash(scopeTags.join('\n'))}` : 'global'
}

function stableObservationHash(input: {
  observedStatus: MonitorObservedStatus
  severity: MonitorSeverity | null
  evidenceSummary: string
  metrics: Record<string, number>
  thresholds: MonitorThresholds
  observedAt: string | null
}): string {
  return hash(JSON.stringify(input))
}

function reportInput(input: Record<string, unknown>, context?: ToolExecutionContext) {
  const monitorId = requiredString(input.monitorId, 'monitorId', MONITOR_ID_PATTERN, 120)
  const lookbackHours = typeof input.lookbackHours === 'number'
    && Number.isFinite(input.lookbackHours)
    && input.lookbackHours >= 1
    && input.lookbackHours <= MAX_REPORT_HOURS
    ? input.lookbackHours
    : null
  if (lookbackHours === null) {
    throw new MonitorInputError(`lookbackHours must be a number from 1 to ${MAX_REPORT_HOURS}.`)
  }
  const until = timestamp(input.until, 'until', Date.now())
  return {
    scopeKey: scopeKey(context),
    monitorId,
    since: until - lookbackHours * 60 * 60 * 1_000,
    until,
  }
}

export function createMonitorEvaluateTool(): ToolDefinitionRuntime {
  return {
    name: 'monitor.evaluate',
    description:
      'Atomically evaluate one structured infrastructure observation against an accepted monitor contract. The tool persists transition state, applies consecutive anomaly/recovery/error thresholds, deduplicates exact sampleId retries, and returns shouldNotify plus a stable non-secret notificationDedupKey. Call once after each read-only probe. `unknown` is never healthy. Do not call notification.publish for a scheduled result; scheduled output already uses the originating channel and relay. When shouldNotify=false, a transition-check schedule must return no visible final text.',
    resumeSafety: 'replay-safe',
    inputSchema: {
      type: 'object',
      additionalProperties: false,
      properties: {
        monitorId: {
          type: 'string',
          minLength: 1,
          maxLength: 120,
          description: 'Stable non-secret monitor identifier.',
        },
        contractVersion: {
          type: 'string',
          minLength: 1,
          maxLength: 80,
          description: 'Portable version for the user-accepted thresholds; increment after an accepted contract change.',
        },
        sampleId: {
          type: 'string',
          minLength: 1,
          maxLength: 200,
          description: 'Stable non-secret identity for this probe sample or scheduler run. Stored only as a hash.',
        },
        observedStatus: { type: 'string', enum: ['healthy', 'anomaly', 'unknown'] },
        severity: { type: 'string', enum: ['warning', 'high', 'critical'] },
        evidenceSummary: {
          type: 'string',
          minLength: 1,
          maxLength: MAX_EVIDENCE_LENGTH,
          description: 'Bounded evidence summary without credentials or secret values. Sensitive-looking values are redacted before persistence and output.',
        },
        metrics: {
          type: 'object',
          maxProperties: MAX_METRICS,
          additionalProperties: { type: 'number' },
          description: 'Optional bounded numeric measurements for later resource reports.',
        },
        anomalyConsecutiveSamples: { type: 'integer', minimum: 1, maximum: MAX_CONSECUTIVE_SAMPLES },
        recoveryConsecutiveSamples: { type: 'integer', minimum: 1, maximum: MAX_CONSECUTIVE_SAMPLES },
        errorConsecutiveSamples: { type: 'integer', minimum: 1, maximum: MAX_CONSECUTIVE_SAMPLES },
        observedAt: {
          type: 'string',
          description: 'Optional ISO-8601 probe timestamp; defaults to tool execution time.',
        },
      },
      required: [
        'monitorId',
        'contractVersion',
        'sampleId',
        'observedStatus',
        'evidenceSummary',
        'anomalyConsecutiveSamples',
        'recoveryConsecutiveSamples',
        'errorConsecutiveSamples',
      ],
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      try {
        const monitorId = requiredString(input.monitorId, 'monitorId', MONITOR_ID_PATTERN, 120)
        const contractVersion = requiredString(
          input.contractVersion,
          'contractVersion',
          CONTRACT_VERSION_PATTERN,
          80,
        )
        const sampleId = typeof input.sampleId === 'string' ? input.sampleId.trim() : ''
        if (!sampleId || sampleId.length > 200 || /[\u0000-\u001f\u007f]/.test(sampleId)) {
          throw new MonitorInputError('sampleId must be 1-200 characters without control characters.')
        }
        const status = observedStatus(input.observedStatus)
        const severity = severityFor(status, input.severity)
        const evidence = evidenceSummary(input.evidenceSummary)
        const normalizedMetrics = metrics(input.metrics)
        const thresholds: MonitorThresholds = {
          anomaly: boundedInteger(
            input.anomalyConsecutiveSamples,
            'anomalyConsecutiveSamples',
            MAX_CONSECUTIVE_SAMPLES,
          ),
          recovery: boundedInteger(
            input.recoveryConsecutiveSamples,
            'recoveryConsecutiveSamples',
            MAX_CONSECUTIVE_SAMPLES,
          ),
          error: boundedInteger(
            input.errorConsecutiveSamples,
            'errorConsecutiveSamples',
            MAX_CONSECUTIVE_SAMPLES,
          ),
        }
        const observedAt = timestamp(input.observedAt, 'observedAt', startedAt)
        if (observedAt > startedAt + MAX_FUTURE_OBSERVATION_SKEW_MS) {
          throw new MonitorInputError('observedAt must not be more than 5 minutes in the future.')
        }
        const scoped = scopeKey(context)
        const result = createMonitorStateRepo().evaluate({
          scopeKey: scoped,
          monitorId,
          contractVersion,
          contractHash: hash(JSON.stringify(thresholds)),
          sampleHash: hash(`${scoped}\0${monitorId}\0${contractVersion}\0${sampleId}`),
          observationHash: stableObservationHash({
            observedStatus: status,
            severity,
            evidenceSummary: evidence.original,
            metrics: normalizedMetrics,
            thresholds,
            observedAt: typeof input.observedAt === 'string' ? input.observedAt : null,
          }),
          observedStatus: status,
          severity,
          evidenceSummary: evidence.redacted,
          evidenceRedacted: evidence.foundSensitive,
          metrics: normalizedMetrics,
          thresholds,
          observedAt,
          now: startedAt,
        })
        if (result.shouldNotify && context?.sessionId && !context.channelContext) {
          new SessionInbox().tryPublish({ sessionId: context.sessionId, source: 'monitor',
            sourceId: monitorId, eventKey: `${scoped}:${monitorId}:${result.stateVersion}`,
            title: `Monitor ${monitorId}: ${status}`, body: evidence.redacted })
        }
        return {
          status: 'success',
          output: JSON.stringify(result),
          metadata: {
            schedulerDelivery: {
              version: 1,
              disposition: result.shouldNotify ? 'deliver' : 'suppress',
              source: 'monitor.evaluate',
              stateVersion: result.stateVersion,
            },
          },
          durationMs: Date.now() - startedAt,
        }
      } catch (error) {
        return failure(error, startedAt)
      }
    },
  }
}

export function createMonitorReportTool(): ToolDefinitionRuntime {
  return {
    name: 'monitor.report',
    description:
      'Read a bounded aggregate report for a monitor over a requested lookback: sample availability, unknown intervals, current/min/max/average numeric metrics, stable state, and transition counts. This tool does not probe targets, change state, or send notifications. Use it for a scheduled digest after monitor.evaluate has recorded samples.',
    resumeSafety: 'replay-safe',
    inputSchema: {
      type: 'object',
      additionalProperties: false,
      properties: {
        monitorId: { type: 'string', minLength: 1, maxLength: 120 },
        lookbackHours: { type: 'number', minimum: 1, maximum: MAX_REPORT_HOURS },
        until: {
          type: 'string',
          description: 'Optional ISO-8601 report end; defaults to the current time.',
        },
      },
      required: ['monitorId', 'lookbackHours'],
    },
    async execute(input, context): Promise<ToolResult> {
      const startedAt = Date.now()
      try {
        const normalized = reportInput(input, context)
        const report = createMonitorStateRepo().report(normalized)
        if (!report) {
          throw new MonitorInputError(
            'No state exists for this monitor in the current user scope.',
            'MONITOR_NOT_FOUND_USER',
          )
        }
        return {
          status: 'success',
          output: JSON.stringify(report),
          durationMs: Date.now() - startedAt,
        }
      } catch (error) {
        return failure(error, startedAt)
      }
    },
  }
}
