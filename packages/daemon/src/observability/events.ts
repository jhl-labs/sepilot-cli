import { randomUUID } from 'node:crypto'
import { mkdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import type { SqliteDatabase } from '../db/sqlite.js'
import { openDomainDb } from '../storage/domain-db.js'
import { sepilotdHome } from '../storage/home.js'
import {
  createTraceRedactionContext,
  redactSecretKeys,
  redactSensitiveText,
} from './trace-redaction.js'

export type ObservabilityRange = '24h' | '7d' | '30d'
export type ObservabilitySource =
  | 'daemon'
  | 'cli'
  | 'tui'
  | 'desktop-main'
  | 'desktop-renderer'
  | 'web'
  | 'channel'
export type ObservabilitySeverity =
  | 'debug'
  | 'info'
  | 'warning'
  | 'error'
  | 'fatal'
export type ObservabilityPrivacy =
  | 'operational'
  | 'diagnostic'
  | 'sensitive'

export interface ObservabilityEventInput {
  id?: string
  schemaVersion?: number
  timestamp?: string
  source: ObservabilitySource
  surface?: string
  eventType: string
  severity?: ObservabilitySeverity
  privacy?: ObservabilityPrivacy
  sessionId?: string
  runId?: string
  messageId?: string
  taskId?: string
  channelIdHash?: string
  userIdHash?: string
  provider?: string
  model?: string
  attributes?: Record<string, unknown>
}

export interface ObservabilityEvent extends Required<
  Pick<
    ObservabilityEventInput,
    'id' | 'source' | 'eventType' | 'severity' | 'privacy'
  >
> {
  schemaVersion: number
  timestamp: string
  surface?: string
  sessionId?: string
  runId?: string
  messageId?: string
  taskId?: string
  channelIdHash?: string
  userIdHash?: string
  provider?: string
  model?: string
  attributes: Record<string, unknown>
}

export type FeedbackRating = 'positive' | 'neutral' | 'negative'

export interface FeedbackInput {
  id?: string
  timestamp?: string
  sessionId?: string
  messageId?: string
  runId?: string
  rating: FeedbackRating
  reason?: string
  note?: string
  source?: ObservabilitySource
  surface?: string
}

export interface FeedbackRecord {
  id: string
  timestamp: string
  sessionId?: string
  messageId?: string
  runId?: string
  rating: FeedbackRating
  reason?: string
  noteRedacted?: string
  source: ObservabilitySource
  surface?: string
}

export interface ObservabilityPrivacySettings {
  localCollectionEnabled: boolean
  diagnosticCollectionEnabled: boolean
  sensitiveCollectionEnabled: boolean
  feedbackCollectionEnabled: boolean
  retentionDays: number
  feedbackPromptCooldownHours: number
  updatedAt: string
}

export interface ObservabilityPrivacySettingsInput {
  localCollectionEnabled?: boolean
  diagnosticCollectionEnabled?: boolean
  sensitiveCollectionEnabled?: boolean
  feedbackCollectionEnabled?: boolean
  retentionDays?: number
  feedbackPromptCooldownHours?: number
}

export interface FeedbackPromptStateInput {
  surface?: string
  sessionId?: string
  messageId?: string
  userIdHash?: string
}

export interface FeedbackPromptState {
  shouldPrompt: boolean
  reason: 'eligible' | 'disabled' | 'cooldown' | 'recent-feedback'
  cooldownHours: number
  promptCount24h: number
  feedbackCount24h: number
  lastPromptAt?: string
  nextPromptAfter?: string
}

export interface ObservabilityTrendPoint {
  label: string
  from: string
  to: string
  totalEvents: number
  errorEvents: number
  crashReports: number
  feedbackPositive: number
  feedbackNeutral: number
  feedbackNegative: number
  explicitSatisfaction: number | null
  tasksStarted: number
  tasksCompleted: number
  completionRate: number | null
  assistedTaskThroughputPerDay: number
  channelTasksStarted: number
  channelTasksCompleted: number
  channelResolutionRate: number | null
  toolSuccessRate: number | null
}

export interface ObservabilityComparison {
  previousFrom: string
  previousTo: string
  errorEventsDelta: number
  crashReportsDelta: number
  satisfactionDelta: number | null
  completionRateDelta: number | null
  throughputPerDayDelta: number
  channelResolutionRateDelta: number | null
  toolSuccessRateDelta: number | null
}

export interface ObservabilitySegment {
  key: string
  label: string
  source: ObservabilitySource
  surface?: string
  totalEvents: number
  errorEvents: number
  crashReports: number
  feedbackPositive: number
  feedbackNeutral: number
  feedbackNegative: number
  explicitSatisfaction: number | null
  tasksStarted: number
  tasksCompleted: number
  completionRate: number | null
  channelTasksStarted: number
  channelTasksCompleted: number
  channelResolutionRate: number | null
  toolSuccessRate: number | null
}

export interface ObservabilityHotspot {
  key: string
  label: string
  count: number
  source?: ObservabilitySource
  surface?: string
  eventType?: string
  severity?: ObservabilitySeverity
}

export interface ObservabilityAlert {
  id: string
  level: 'info' | 'warning' | 'critical'
  title: string
  detail: string
  metric: string
  value: number | null
  threshold?: number
}

export interface ObservabilitySnapshot {
  generatedAt: string
  range: {
    key: ObservabilityRange
    from: string
    to: string
  }
  reliability: {
    totalEvents: number
    errorEvents: number
    fatalEvents: number
    crashReports: number
    crashFreeSessions: number | null
    providerFailureRate: number | null
    channelDeliverySuccessRate: number | null
    routeErrorRate: number | null
  }
  quality: {
    feedbackPositive: number
    feedbackNeutral: number
    feedbackNegative: number
    explicitSatisfaction: number | null
    promptResponseRate: number | null
    implicitAcceptance: number | null
    regenerationRate: number | null
    stopRate: number | null
  }
  productivity: {
    tasksStarted: number
    tasksCompleted: number
    completionRate: number | null
    assistedTaskThroughputPerDay: number
    autonomousCompletionRate: number | null
    approvalFrictionMs: number | null
    toolSuccessRate: number | null
    costPerResolvedTask: number | null
    tokensPerResolvedTask: number | null
    recoverySuccessRate: number | null
    channelResolutionRate: number | null
  }
  comparison: ObservabilityComparison
  trend: ObservabilityTrendPoint[]
  segments: ObservabilitySegment[]
  hotspots: {
    errorEvents: ObservabilityHotspot[]
    feedbackReasons: ObservabilityHotspot[]
  }
  alerts: ObservabilityAlert[]
  recent: ObservabilityEvent[]
}

export interface ObservabilityExportInput {
  range?: ObservabilityRange
  limit?: number
  includeEvents?: boolean
  includeCrashes?: boolean
  includeFeedback?: boolean
  persist?: boolean
}

export interface ObservabilityExportBundle {
  generatedAt: string
  privacy: ObservabilityPrivacySettings
  snapshot: ObservabilitySnapshot
  events: ObservabilityEvent[]
  crashes: ObservabilityEvent[]
  feedback: FeedbackRecord[]
  redaction: {
    attributes: 'sanitized-at-ingest'
    notes: 'redacted-or-truncated'
  }
}

export interface ObservabilityExportResult {
  path?: string
  eventCount: number
  crashCount: number
  feedbackCount: number
  bundle: ObservabilityExportBundle
}

export interface ObservabilityPruneResult {
  cutoff: string
  deletedEvents: number
  deletedFeedback: number
}

interface EventRow {
  id: string
  schema_version: number
  timestamp_ms: number
  timestamp_iso: string
  source: ObservabilitySource
  surface: string | null
  event_type: string
  severity: ObservabilitySeverity
  privacy: ObservabilityPrivacy
  session_id: string | null
  run_id: string | null
  message_id: string | null
  task_id: string | null
  channel_id_hash: string | null
  user_id_hash: string | null
  provider: string | null
  model: string | null
  attributes_json: string
}

interface FeedbackRow {
  id: string
  timestamp_ms: number
  timestamp_iso: string
  session_id: string | null
  message_id: string | null
  run_id: string | null
  rating: FeedbackRating
  reason: string | null
  note_redacted: string | null
  source: ObservabilitySource
  surface: string | null
}

const RANGE_MS: Record<ObservabilityRange, number> = {
  '24h': 24 * 60 * 60 * 1000,
  '7d': 7 * 24 * 60 * 60 * 1000,
  '30d': 30 * 24 * 60 * 60 * 1000,
}

const DEFAULT_PRIVACY_SETTINGS: ObservabilityPrivacySettings = {
  localCollectionEnabled: true,
  diagnosticCollectionEnabled: true,
  sensitiveCollectionEnabled: false,
  feedbackCollectionEnabled: true,
  retentionDays: 30,
  feedbackPromptCooldownHours: 24,
  updatedAt: '1970-01-01T00:00:00.000Z',
}

const SETTINGS_PRIVACY_KEY = 'privacy'
const DAY_MS = 24 * 60 * 60 * 1000
const REDACT_KEY =
  /(authorization|api[_-]?key|password|secret|token|prompt|response|content|input|output|stack|trace|path)/i
const MAX_STRING_LENGTH = 1_000
const MAX_ARRAY_ITEMS = 32
const MAX_OBJECT_KEYS = 48
const MAX_DEPTH = 5

function ensureSchema(db: SqliteDatabase): void {
  db.prepare(
    `CREATE TABLE IF NOT EXISTS events (
      id TEXT PRIMARY KEY,
      schema_version INTEGER NOT NULL,
      timestamp_ms INTEGER NOT NULL,
      timestamp_iso TEXT NOT NULL,
      source TEXT NOT NULL,
      surface TEXT,
      event_type TEXT NOT NULL,
      severity TEXT NOT NULL,
      privacy TEXT NOT NULL,
      session_id TEXT,
      run_id TEXT,
      message_id TEXT,
      task_id TEXT,
      channel_id_hash TEXT,
      user_id_hash TEXT,
      provider TEXT,
      model TEXT,
      attributes_json TEXT NOT NULL
    )`,
  ).run()
  db.prepare(
    `CREATE INDEX IF NOT EXISTS events_timestamp
      ON events(timestamp_ms DESC)`,
  ).run()
  db.prepare(
    `CREATE INDEX IF NOT EXISTS events_type_timestamp
      ON events(event_type, timestamp_ms DESC)`,
  ).run()
  db.prepare(
    `CREATE INDEX IF NOT EXISTS events_severity_timestamp
      ON events(severity, timestamp_ms DESC)`,
  ).run()
  db.prepare(
    `CREATE INDEX IF NOT EXISTS events_session_timestamp
      ON events(session_id, timestamp_ms DESC)`,
  ).run()
  db.prepare(
    `CREATE TABLE IF NOT EXISTS feedback (
      id TEXT PRIMARY KEY,
      timestamp_ms INTEGER NOT NULL,
      timestamp_iso TEXT NOT NULL,
      session_id TEXT,
      message_id TEXT,
      run_id TEXT,
      rating TEXT NOT NULL,
      reason TEXT,
      note_redacted TEXT,
      source TEXT NOT NULL,
      surface TEXT
    )`,
  ).run()
  db.prepare(
    `CREATE INDEX IF NOT EXISTS feedback_timestamp
      ON feedback(timestamp_ms DESC)`,
  ).run()
  db.prepare(
    `CREATE INDEX IF NOT EXISTS feedback_session_timestamp
      ON feedback(session_id, timestamp_ms DESC)`,
  ).run()
  db.prepare(
    `CREATE TABLE IF NOT EXISTS settings (
      key TEXT PRIMARY KEY,
      value_json TEXT NOT NULL,
      updated_at_iso TEXT NOT NULL
    )`,
  ).run()
}

function normalizeTimestamp(input?: string): { iso: string; ms: number } {
  const parsed = input ? Date.parse(input) : Number.NaN
  const ms = Number.isFinite(parsed) ? parsed : Date.now()
  return { iso: new Date(ms).toISOString(), ms }
}

function truncate(value: string, limit = MAX_STRING_LENGTH): string {
  return value.length > limit ? `${value.slice(0, limit - 1)}...` : value
}

function sanitizeValue(value: unknown, depth = 0): unknown {
  if (depth >= MAX_DEPTH) return '[max-depth]'
  if (value === null || typeof value === 'number' || typeof value === 'boolean') {
    return value
  }
  if (typeof value === 'string') return truncate(value)
  if (typeof value === 'bigint' || typeof value === 'symbol') return String(value)
  if (Array.isArray(value)) {
    const items = value
      .slice(0, MAX_ARRAY_ITEMS)
      .map((item) => sanitizeValue(item, depth + 1))
    if (value.length > MAX_ARRAY_ITEMS) {
      items.push(`[+${value.length - MAX_ARRAY_ITEMS} more items]`)
    }
    return items
  }
  if (typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>)
    const output: Record<string, unknown> = {}
    for (const [key, item] of entries.slice(0, MAX_OBJECT_KEYS)) {
      output[key] = REDACT_KEY.test(key)
        ? '[redacted]'
        : sanitizeValue(item, depth + 1)
    }
    if (entries.length > MAX_OBJECT_KEYS) {
      output.__truncated_keys__ = entries.length - MAX_OBJECT_KEYS
    }
    return output
  }
  if (typeof value === 'undefined') return null
  return String(value)
}

function sanitizeAttributes(
  attributes?: Record<string, unknown>,
): Record<string, unknown> {
  return sanitizeValue(attributes ?? {}) as Record<string, unknown>
}

function normalizeRetentionDays(value: unknown, fallback: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) return fallback
  return Math.min(Math.max(Math.round(value), 1), 365)
}

function normalizeCooldownHours(value: unknown, fallback: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) return fallback
  return Math.min(Math.max(Math.round(value), 1), 720)
}

function mergePrivacySettings(
  current: ObservabilityPrivacySettings,
  input: ObservabilityPrivacySettingsInput,
  updatedAt: string,
): ObservabilityPrivacySettings {
  return {
    localCollectionEnabled:
      input.localCollectionEnabled ?? current.localCollectionEnabled,
    diagnosticCollectionEnabled:
      input.diagnosticCollectionEnabled ?? current.diagnosticCollectionEnabled,
    sensitiveCollectionEnabled:
      input.sensitiveCollectionEnabled ?? current.sensitiveCollectionEnabled,
    feedbackCollectionEnabled:
      input.feedbackCollectionEnabled ?? current.feedbackCollectionEnabled,
    retentionDays: normalizeRetentionDays(input.retentionDays, current.retentionDays),
    feedbackPromptCooldownHours: normalizeCooldownHours(
      input.feedbackPromptCooldownHours,
      current.feedbackPromptCooldownHours,
    ),
    updatedAt,
  }
}

function toEvent(row: EventRow): ObservabilityEvent {
  let attributes: Record<string, unknown> = {}
  try {
    attributes = JSON.parse(row.attributes_json) as Record<string, unknown>
  } catch {
    attributes = {}
  }
  return {
    id: row.id,
    schemaVersion: row.schema_version,
    timestamp: row.timestamp_iso,
    source: row.source,
    surface: row.surface ?? undefined,
    eventType: row.event_type,
    severity: row.severity,
    privacy: row.privacy,
    sessionId: row.session_id ?? undefined,
    runId: row.run_id ?? undefined,
    messageId: row.message_id ?? undefined,
    taskId: row.task_id ?? undefined,
    channelIdHash: row.channel_id_hash ?? undefined,
    userIdHash: row.user_id_hash ?? undefined,
    provider: row.provider ?? undefined,
    model: row.model ?? undefined,
    attributes,
  }
}

function toFeedback(row: FeedbackRow): FeedbackRecord {
  return {
    id: row.id,
    timestamp: row.timestamp_iso,
    sessionId: row.session_id ?? undefined,
    messageId: row.message_id ?? undefined,
    runId: row.run_id ?? undefined,
    rating: row.rating,
    reason: row.reason ?? undefined,
    noteRedacted: row.note_redacted ?? undefined,
    source: row.source,
    surface: row.surface ?? undefined,
  }
}

function shouldRecordEvent(
  input: ObservabilityEventInput,
  privacy: ObservabilityPrivacySettings,
): boolean {
  if (!privacy.localCollectionEnabled) return false
  const classification = input.privacy ?? 'operational'
  if (classification === 'diagnostic' && !privacy.diagnosticCollectionEnabled) {
    return false
  }
  if (classification === 'sensitive') {
    return privacy.sensitiveCollectionEnabled
  }
  return true
}

function ratio(numerator: number, denominator: number): number | null {
  if (denominator <= 0) return null
  return numerator / denominator
}

function countEvents(
  db: SqliteDatabase,
  fromMs: number,
  toMs: number,
  eventTypes: string[],
): number {
  if (eventTypes.length === 0) return 0
  const placeholders = eventTypes.map(() => '?').join(',')
  const row = db.prepare(
    `SELECT COUNT(*) AS count FROM events
     WHERE timestamp_ms >= ? AND timestamp_ms <= ?
       AND event_type IN (${placeholders})`,
  ).get(fromMs, toMs, ...eventTypes) as { count: number }
  return row.count
}

function countSeverity(
  db: SqliteDatabase,
  fromMs: number,
  toMs: number,
  severities: ObservabilitySeverity[],
): number {
  const placeholders = severities.map(() => '?').join(',')
  const row = db.prepare(
    `SELECT COUNT(*) AS count FROM events
     WHERE timestamp_ms >= ? AND timestamp_ms <= ?
       AND severity IN (${placeholders})`,
  ).get(fromMs, toMs, ...severities) as { count: number }
  return row.count
}

// Count crash-class events in a SINGLE OR pass so an event that is both a
// crash-typed event AND severity=fatal is counted once. Summing three separate
// counts (LIKE '%.crash' + LIKE '%.renderer_crash' + severity fatal) double-
// counted the overlap and inflated the CRITICAL crash alert. Mirrors the
// per-session crash predicate used elsewhere in this module.
function countCrashReports(db: SqliteDatabase, fromMs: number, toMs: number): number {
  const row = db.prepare(
    `SELECT COUNT(*) AS count FROM events
     WHERE timestamp_ms >= ? AND timestamp_ms <= ?
       AND (severity = 'fatal'
         OR event_type LIKE '%.crash'
         OR event_type LIKE '%.renderer_crash')`,
  ).get(fromMs, toMs) as { count: number }
  return row.count
}

function averageAttribute(
  rows: ObservabilityEvent[],
  keys: string[],
): number | null {
  const values: number[] = []
  for (const row of rows) {
    for (const key of keys) {
      const value = row.attributes[key]
      if (typeof value === 'number' && Number.isFinite(value)) {
        values.push(value)
        break
      }
    }
  }
  if (values.length === 0) return null
  return values.reduce((sum, value) => sum + value, 0) / values.length
}

function sumAttribute(rows: ObservabilityEvent[], keys: string[]): number {
  let sum = 0
  for (const row of rows) {
    for (const key of keys) {
      const value = row.attributes[key]
      if (typeof value === 'number' && Number.isFinite(value)) {
        sum += value
        break
      }
    }
  }
  return sum
}

interface WindowMetrics {
  totalEvents: number
  errorEvents: number
  crashReports: number
  feedbackPositive: number
  feedbackNeutral: number
  feedbackNegative: number
  explicitSatisfaction: number | null
  tasksStarted: number
  tasksCompleted: number
  completionRate: number | null
  assistedTaskThroughputPerDay: number
  channelTasksStarted: number
  channelTasksCompleted: number
  channelResolutionRate: number | null
  toolSucceeded: number
  toolFailed: number
  toolSuccessRate: number | null
}

function countAllEvents(
  db: SqliteDatabase,
  fromMs: number,
  toMs: number,
): number {
  const row = db.prepare(
    `SELECT COUNT(*) AS count FROM events
     WHERE timestamp_ms >= ? AND timestamp_ms <= ?`,
  ).get(fromMs, toMs) as { count: number }
  return row.count
}

function countFeedbackByRating(
  db: SqliteDatabase,
  fromMs: number,
  toMs: number,
  rating: FeedbackRating,
): number {
  const row = db.prepare(
    `SELECT COUNT(*) AS count FROM feedback
     WHERE timestamp_ms >= ? AND timestamp_ms <= ? AND rating = ?`,
  ).get(fromMs, toMs, rating) as { count: number }
  return row.count
}

function buildWindowMetrics(
  db: SqliteDatabase,
  fromMs: number,
  toMs: number,
): WindowMetrics {
  const feedbackPositive = countFeedbackByRating(db, fromMs, toMs, 'positive')
  const feedbackNeutral = countFeedbackByRating(db, fromMs, toMs, 'neutral')
  const feedbackNegative = countFeedbackByRating(db, fromMs, toMs, 'negative')
  const tasksStarted = countEvents(db, fromMs, toMs, [
    'task.started',
    'channel.task_started',
  ])
  const tasksCompleted = countEvents(db, fromMs, toMs, [
    'task.completed',
    'channel.task_completed',
  ])
  const channelTasksStarted = countEvents(db, fromMs, toMs, [
    'channel.task_started',
  ])
  const channelTasksCompleted = countEvents(db, fromMs, toMs, [
    'channel.task_completed',
  ])
  const toolSucceeded = countEvents(db, fromMs, toMs, [
    'tool.succeeded',
    'mcp.tool_succeeded',
  ])
  const toolFailed = countEvents(db, fromMs, toMs, [
    'tool.failed',
    'mcp.tool_failed',
  ])
  const days = Math.max((toMs - fromMs) / DAY_MS, 1 / 24)

  return {
    totalEvents: countAllEvents(db, fromMs, toMs),
    errorEvents: countSeverity(db, fromMs, toMs, ['error', 'fatal']),
    crashReports: countCrashReports(db, fromMs, toMs),
    feedbackPositive,
    feedbackNeutral,
    feedbackNegative,
    explicitSatisfaction: ratio(
      feedbackPositive,
      feedbackPositive + feedbackNegative,
    ),
    tasksStarted,
    tasksCompleted,
    completionRate: ratio(tasksCompleted, tasksStarted),
    assistedTaskThroughputPerDay: tasksCompleted / days,
    channelTasksStarted,
    channelTasksCompleted,
    channelResolutionRate: ratio(channelTasksCompleted, channelTasksStarted),
    toolSucceeded,
    toolFailed,
    toolSuccessRate: ratio(toolSucceeded, toolSucceeded + toolFailed),
  }
}

function deltaNullable(
  current: number | null,
  previous: number | null,
): number | null {
  if (current === null || previous === null) return null
  return current - previous
}

function buildTrend(
  db: SqliteDatabase,
  range: ObservabilityRange,
  fromMs: number,
  toMs: number,
): ObservabilityTrendPoint[] {
  const buckets = range === '24h' ? 24 : range === '7d' ? 7 : 30
  const bucketMs = (toMs - fromMs) / buckets
  const points: ObservabilityTrendPoint[] = []
  for (let index = 0; index < buckets; index += 1) {
    const startMs = Math.round(fromMs + bucketMs * index)
    const endMs = index === buckets - 1
      ? toMs
      : Math.round(fromMs + bucketMs * (index + 1)) - 1
    const metrics = buildWindowMetrics(db, startMs, endMs)
    const start = new Date(startMs)
    points.push({
      label: range === '24h'
        ? start.toISOString().slice(11, 16)
        : start.toISOString().slice(0, 10),
      from: start.toISOString(),
      to: new Date(endMs).toISOString(),
      totalEvents: metrics.totalEvents,
      errorEvents: metrics.errorEvents,
      crashReports: metrics.crashReports,
      feedbackPositive: metrics.feedbackPositive,
      feedbackNeutral: metrics.feedbackNeutral,
      feedbackNegative: metrics.feedbackNegative,
      explicitSatisfaction: metrics.explicitSatisfaction,
      tasksStarted: metrics.tasksStarted,
      tasksCompleted: metrics.tasksCompleted,
      completionRate: metrics.completionRate,
      assistedTaskThroughputPerDay: metrics.assistedTaskThroughputPerDay,
      channelTasksStarted: metrics.channelTasksStarted,
      channelTasksCompleted: metrics.channelTasksCompleted,
      channelResolutionRate: metrics.channelResolutionRate,
      toolSuccessRate: metrics.toolSuccessRate,
    })
  }
  return points
}

interface SegmentRow {
  source: ObservabilitySource
  surface: string | null
  total_events: number
  error_events: number
  crash_reports: number
  tasks_started: number
  tasks_completed: number
  channel_tasks_started: number
  channel_tasks_completed: number
  tool_succeeded: number
  tool_failed: number
}

interface FeedbackSegmentRow {
  source: ObservabilitySource
  surface: string | null
  feedback_positive: number
  feedback_neutral: number
  feedback_negative: number
}

function segmentKey(source: ObservabilitySource, surface?: string | null): string {
  return `${source}:${surface ?? ''}`
}

function segmentLabel(source: ObservabilitySource, surface?: string | null): string {
  return surface ? `${surface} (${source})` : source
}

function buildSegments(
  db: SqliteDatabase,
  fromMs: number,
  toMs: number,
): ObservabilitySegment[] {
  const eventRows = db.prepare(
    `SELECT
       source,
       surface,
       COUNT(*) AS total_events,
       SUM(CASE WHEN severity IN ('error', 'fatal') THEN 1 ELSE 0 END) AS error_events,
       SUM(CASE
         WHEN severity = 'fatal'
           OR event_type LIKE '%.crash'
           OR event_type LIKE '%.renderer_crash'
         THEN 1 ELSE 0 END) AS crash_reports,
       SUM(CASE WHEN event_type IN ('task.started', 'channel.task_started') THEN 1 ELSE 0 END) AS tasks_started,
       SUM(CASE WHEN event_type IN ('task.completed', 'channel.task_completed') THEN 1 ELSE 0 END) AS tasks_completed,
       SUM(CASE WHEN event_type = 'channel.task_started' THEN 1 ELSE 0 END) AS channel_tasks_started,
       SUM(CASE WHEN event_type = 'channel.task_completed' THEN 1 ELSE 0 END) AS channel_tasks_completed,
       SUM(CASE WHEN event_type IN ('tool.succeeded', 'mcp.tool_succeeded') THEN 1 ELSE 0 END) AS tool_succeeded,
       SUM(CASE WHEN event_type IN ('tool.failed', 'mcp.tool_failed') THEN 1 ELSE 0 END) AS tool_failed
     FROM events
     WHERE timestamp_ms >= ? AND timestamp_ms <= ?
     GROUP BY source, surface`,
  ).all(fromMs, toMs) as SegmentRow[]
  const feedbackRows = db.prepare(
    `SELECT
       source,
       surface,
       SUM(CASE WHEN rating = 'positive' THEN 1 ELSE 0 END) AS feedback_positive,
       SUM(CASE WHEN rating = 'neutral' THEN 1 ELSE 0 END) AS feedback_neutral,
       SUM(CASE WHEN rating = 'negative' THEN 1 ELSE 0 END) AS feedback_negative
     FROM feedback
     WHERE timestamp_ms >= ? AND timestamp_ms <= ?
     GROUP BY source, surface`,
  ).all(fromMs, toMs) as FeedbackSegmentRow[]
  const feedbackByKey = new Map<string, FeedbackSegmentRow>()
  for (const row of feedbackRows) {
    feedbackByKey.set(segmentKey(row.source, row.surface), row)
  }

  const segments = new Map<string, ObservabilitySegment>()
  for (const row of eventRows) {
    const key = segmentKey(row.source, row.surface)
    const feedback = feedbackByKey.get(key)
    const feedbackPositive = feedback?.feedback_positive ?? 0
    const feedbackNeutral = feedback?.feedback_neutral ?? 0
    const feedbackNegative = feedback?.feedback_negative ?? 0
    segments.set(key, {
      key,
      label: segmentLabel(row.source, row.surface),
      source: row.source,
      surface: row.surface ?? undefined,
      totalEvents: row.total_events,
      errorEvents: row.error_events,
      crashReports: row.crash_reports,
      feedbackPositive,
      feedbackNeutral,
      feedbackNegative,
      explicitSatisfaction: ratio(
        feedbackPositive,
        feedbackPositive + feedbackNegative,
      ),
      tasksStarted: row.tasks_started,
      tasksCompleted: row.tasks_completed,
      completionRate: ratio(row.tasks_completed, row.tasks_started),
      channelTasksStarted: row.channel_tasks_started,
      channelTasksCompleted: row.channel_tasks_completed,
      channelResolutionRate: ratio(
        row.channel_tasks_completed,
        row.channel_tasks_started,
      ),
      toolSuccessRate: ratio(
        row.tool_succeeded,
        row.tool_succeeded + row.tool_failed,
      ),
    })
  }
  for (const feedback of feedbackRows) {
    const key = segmentKey(feedback.source, feedback.surface)
    if (segments.has(key)) continue
    segments.set(key, {
      key,
      label: segmentLabel(feedback.source, feedback.surface),
      source: feedback.source,
      surface: feedback.surface ?? undefined,
      totalEvents: 0,
      errorEvents: 0,
      crashReports: 0,
      feedbackPositive: feedback.feedback_positive,
      feedbackNeutral: feedback.feedback_neutral,
      feedbackNegative: feedback.feedback_negative,
      explicitSatisfaction: ratio(
        feedback.feedback_positive,
        feedback.feedback_positive + feedback.feedback_negative,
      ),
      tasksStarted: 0,
      tasksCompleted: 0,
      completionRate: null,
      channelTasksStarted: 0,
      channelTasksCompleted: 0,
      channelResolutionRate: null,
      toolSuccessRate: null,
    })
  }

  return [...segments.values()]
    .sort((a, b) =>
      (b.errorEvents - a.errorEvents)
      || (b.tasksCompleted - a.tasksCompleted)
      || (b.feedbackNegative - a.feedbackNegative)
      || (b.totalEvents - a.totalEvents)
    )
    .slice(0, 12)
}

interface ErrorHotspotRow {
  event_type: string
  source: ObservabilitySource
  surface: string | null
  severity: ObservabilitySeverity
  count: number
}

interface FeedbackHotspotRow {
  reason: string | null
  source: ObservabilitySource
  surface: string | null
  count: number
}

function buildHotspots(
  db: SqliteDatabase,
  fromMs: number,
  toMs: number,
): ObservabilitySnapshot['hotspots'] {
  const errorEvents = (
    db.prepare(
      `SELECT event_type, source, surface, severity, COUNT(*) AS count
       FROM events
       WHERE timestamp_ms >= ? AND timestamp_ms <= ?
         AND severity IN ('error', 'fatal')
       GROUP BY event_type, source, surface, severity
       ORDER BY count DESC
       LIMIT 5`,
    ).all(fromMs, toMs) as ErrorHotspotRow[]
  ).map((row) => ({
    key: `${row.source}:${row.surface ?? ''}:${row.event_type}:${row.severity}`,
    label: `${row.event_type}${row.surface ? ` @ ${row.surface}` : ''}`,
    count: row.count,
    source: row.source,
    surface: row.surface ?? undefined,
    eventType: row.event_type,
    severity: row.severity,
  }))
  const feedbackReasons = (
    db.prepare(
      `SELECT reason, source, surface, COUNT(*) AS count
       FROM feedback
       WHERE timestamp_ms >= ? AND timestamp_ms <= ?
         AND rating IN ('negative', 'neutral')
       GROUP BY reason, source, surface
       ORDER BY count DESC
       LIMIT 5`,
    ).all(fromMs, toMs) as FeedbackHotspotRow[]
  ).map((row) => ({
    key: `${row.source}:${row.surface ?? ''}:${row.reason ?? 'unspecified'}`,
    label: row.reason ?? 'unspecified feedback',
    count: row.count,
    source: row.source,
    surface: row.surface ?? undefined,
  }))
  return { errorEvents, feedbackReasons }
}

function buildAlerts(
  current: WindowMetrics,
  comparison: ObservabilityComparison,
): ObservabilityAlert[] {
  const alerts: ObservabilityAlert[] = []
  if (current.totalEvents === 0) {
    alerts.push({
      id: 'no-observability-events',
      level: 'info',
      title: 'No observability events',
      detail: 'No events were collected in this range.',
      metric: 'reliability.totalEvents',
      value: 0,
    })
  }
  if (current.crashReports > 0) {
    alerts.push({
      id: 'crash-reports-present',
      level: 'critical',
      title: 'Crash reports detected',
      detail: `${current.crashReports} crash-class events were recorded.`,
      metric: 'reliability.crashReports',
      value: current.crashReports,
      threshold: 0,
    })
  }
  if (comparison.errorEventsDelta >= 3 && current.errorEvents >= 5) {
    alerts.push({
      id: 'error-events-rising',
      level: 'warning',
      title: 'Error events increased',
      detail: `${comparison.errorEventsDelta} more error events than the previous range.`,
      metric: 'comparison.errorEventsDelta',
      value: comparison.errorEventsDelta,
      threshold: 3,
    })
  }
  if (
    current.completionRate !== null
    && current.tasksStarted >= 3
    && current.completionRate < 0.75
  ) {
    alerts.push({
      id: 'completion-rate-low',
      level: 'warning',
      title: 'Task completion is low',
      detail: 'Completed assisted tasks are below the operating threshold.',
      metric: 'productivity.completionRate',
      value: current.completionRate,
      threshold: 0.75,
    })
  }
  if (
    current.channelResolutionRate !== null
    && current.channelTasksStarted >= 3
    && current.channelResolutionRate < 0.8
  ) {
    alerts.push({
      id: 'channel-resolution-low',
      level: 'warning',
      title: 'Channel resolution is low',
      detail: 'External channel tasks are not being resolved reliably.',
      metric: 'productivity.channelResolutionRate',
      value: current.channelResolutionRate,
      threshold: 0.8,
    })
  }
  if (
    current.explicitSatisfaction !== null
    && current.feedbackPositive + current.feedbackNegative >= 3
    && current.explicitSatisfaction < 0.7
  ) {
    alerts.push({
      id: 'satisfaction-low',
      level: 'warning',
      title: 'Satisfaction is low',
      detail: 'Explicit feedback is below the target threshold.',
      metric: 'quality.explicitSatisfaction',
      value: current.explicitSatisfaction,
      threshold: 0.7,
    })
  }
  if (
    current.toolSuccessRate !== null
    && current.toolSucceeded + current.toolFailed >= 5
    && current.toolSuccessRate < 0.9
  ) {
    alerts.push({
      id: 'tool-success-low',
      level: 'warning',
      title: 'Tool success is low',
      detail: 'Tool failures are high enough to affect AX reliability.',
      metric: 'productivity.toolSuccessRate',
      value: current.toolSuccessRate,
      threshold: 0.9,
    })
  }
  return alerts
}

export interface ObservabilityRepo {
  recordEvents(input: ObservabilityEventInput[]): { accepted: number }
  recordFeedback(input: FeedbackInput): FeedbackRecord
  getPrivacySettings(): ObservabilityPrivacySettings
  updatePrivacySettings(
    input: ObservabilityPrivacySettingsInput,
  ): ObservabilityPrivacySettings
  feedbackPromptState(input?: FeedbackPromptStateInput): FeedbackPromptState
  exportBundle(input?: ObservabilityExportInput): ObservabilityExportResult
  prune(): ObservabilityPruneResult
  snapshot(range?: ObservabilityRange): ObservabilitySnapshot
  listEvents(input?: {
    limit?: number
    severity?: ObservabilitySeverity
    eventType?: string
  }): ObservabilityEvent[]
  listCrashes(input?: { limit?: number }): ObservabilityEvent[]
}

export function createObservabilityRepo(): ObservabilityRepo {
  const db = openDomainDb({ name: 'observability', filename: 'events.db' })
  ensureSchema(db)
  // INSERT OR IGNORE (not REPLACE): the id is client-supplied, so REPLACE let a
  // caller overwrite/clobber an already-stored event or feedback row by reusing
  // its id. IGNORE makes re-delivery idempotent and keeps the first-written row
  // authoritative.
  const insertEvent = db.prepare(
    `INSERT OR IGNORE INTO events (
      id, schema_version, timestamp_ms, timestamp_iso, source, surface,
      event_type, severity, privacy, session_id, run_id, message_id, task_id,
      channel_id_hash, user_id_hash, provider, model, attributes_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  )
  const insertFeedback = db.prepare(
    `INSERT OR IGNORE INTO feedback (
      id, timestamp_ms, timestamp_iso, session_id, message_id, run_id,
      rating, reason, note_redacted, source, surface
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  )
  const selectSetting = db.prepare(
    'SELECT value_json FROM settings WHERE key=?',
  )
  const upsertSetting = db.prepare(
    `INSERT INTO settings (key, value_json, updated_at_iso)
     VALUES (?, ?, ?)
     ON CONFLICT(key) DO UPDATE SET
       value_json=excluded.value_json,
       updated_at_iso=excluded.updated_at_iso`,
  )

  function getPrivacySettings(): ObservabilityPrivacySettings {
    const row = selectSetting.get(SETTINGS_PRIVACY_KEY) as
      | { value_json: string }
      | undefined
    if (!row) return { ...DEFAULT_PRIVACY_SETTINGS }
    try {
      const parsed = JSON.parse(row.value_json) as Partial<ObservabilityPrivacySettings>
      return {
        ...DEFAULT_PRIVACY_SETTINGS,
        ...parsed,
        retentionDays: normalizeRetentionDays(
          parsed.retentionDays,
          DEFAULT_PRIVACY_SETTINGS.retentionDays,
        ),
        feedbackPromptCooldownHours: normalizeCooldownHours(
          parsed.feedbackPromptCooldownHours,
          DEFAULT_PRIVACY_SETTINGS.feedbackPromptCooldownHours,
        ),
        updatedAt: parsed.updatedAt ?? DEFAULT_PRIVACY_SETTINGS.updatedAt,
      }
    } catch {
      return { ...DEFAULT_PRIVACY_SETTINGS }
    }
  }

  function savePrivacySettings(
    settings: ObservabilityPrivacySettings,
  ): ObservabilityPrivacySettings {
    upsertSetting.run(
      SETTINGS_PRIVACY_KEY,
      JSON.stringify(settings),
      settings.updatedAt,
    )
    return settings
  }

  function insertEventInput(
    input: ObservabilityEventInput,
    privacy: ObservabilityPrivacySettings,
  ): boolean {
    if (!shouldRecordEvent(input, privacy)) return false
    const timestamp = normalizeTimestamp(input.timestamp)
    const attributes = sanitizeAttributes(input.attributes)
    insertEvent.run(
      input.id ?? randomUUID(),
      input.schemaVersion ?? 1,
      timestamp.ms,
      timestamp.iso,
      input.source,
      input.surface ?? null,
      input.eventType,
      input.severity ?? 'info',
      input.privacy ?? 'operational',
      input.sessionId ?? null,
      input.runId ?? null,
      input.messageId ?? null,
      input.taskId ?? null,
      input.channelIdHash ?? null,
      input.userIdHash ?? null,
      input.provider ?? null,
      input.model ?? null,
      JSON.stringify(attributes),
    )
    return true
  }

  function scopedWhere(
    input: FeedbackPromptStateInput,
    includeUserHash: boolean,
  ): { sql: string; params: unknown[] } {
    const where: string[] = []
    const params: unknown[] = []
    if (input.surface) {
      where.push('surface = ?')
      params.push(input.surface)
    }
    if (input.sessionId) {
      where.push('session_id = ?')
      params.push(input.sessionId)
    }
    if (input.messageId) {
      where.push('message_id = ?')
      params.push(input.messageId)
    }
    if (includeUserHash && input.userIdHash) {
      where.push('user_id_hash = ?')
      params.push(input.userIdHash)
    }
    return {
      sql: where.length ? ` AND ${where.join(' AND ')}` : '',
      params,
    }
  }

  function feedbackRowsInRange(
    fromMs: number,
    toMs: number,
    limit: number,
  ): FeedbackRecord[] {
    return (
      db.prepare(
        `SELECT * FROM feedback
         WHERE timestamp_ms >= ? AND timestamp_ms <= ?
         ORDER BY timestamp_ms DESC
         LIMIT ?`,
      ).all(fromMs, toMs, limit) as FeedbackRow[]
    ).map(toFeedback)
  }

  function pruneExpired(nowMs = Date.now()): ObservabilityPruneResult {
    const settings = getPrivacySettings()
    const cutoffMs = nowMs - settings.retentionDays * DAY_MS
    const cutoff = new Date(cutoffMs).toISOString()
    const eventResult = db.prepare(
      'DELETE FROM events WHERE timestamp_ms < ?',
    ).run(cutoffMs)
    const feedbackResult = db.prepare(
      'DELETE FROM feedback WHERE timestamp_ms < ?',
    ).run(cutoffMs)
    return {
      cutoff,
      deletedEvents: eventResult.changes,
      deletedFeedback: feedbackResult.changes,
    }
  }

  return {
    recordEvents(input) {
      const privacy = getPrivacySettings()
      let accepted = 0
      const tx = db.transaction((events: ObservabilityEventInput[]) => {
        for (const event of events) {
          if (insertEventInput(event, privacy)) {
            accepted += 1
          }
        }
      })
      tx(input)
      return { accepted }
    },

    recordFeedback(input) {
      const timestamp = normalizeTimestamp(input.timestamp)
      const id = input.id ?? randomUUID()
      const source = input.source ?? 'daemon'
      const privacy = getPrivacySettings()
      // note_redacted must actually be redacted, not merely truncated: run the
      // note through the shared secret redactor before persisting/exporting.
      const noteRedacted =
        input.note && privacy.diagnosticCollectionEnabled
          ? truncate(redactSensitiveText(input.note), 2_000)
          : null
      const record: FeedbackRecord = {
        id,
        timestamp: timestamp.iso,
        sessionId: input.sessionId,
        messageId: input.messageId,
        runId: input.runId,
        rating: input.rating,
        reason: input.reason,
        noteRedacted: noteRedacted ?? undefined,
        source,
        surface: input.surface,
      }
      if (!privacy.localCollectionEnabled || !privacy.feedbackCollectionEnabled) {
        return record
      }
      const tx = db.transaction(() => {
        insertFeedback.run(
          id,
          timestamp.ms,
          timestamp.iso,
          input.sessionId ?? null,
          input.messageId ?? null,
          input.runId ?? null,
          input.rating,
          input.reason ?? null,
          noteRedacted,
          source,
          input.surface ?? null,
        )
        insertEventInput({
          id: `feedback-event-${id}`,
          timestamp: timestamp.iso,
          source,
          surface: input.surface,
          eventType: 'feedback.submitted',
          severity: 'info',
          privacy: input.note ? 'diagnostic' : 'operational',
          sessionId: input.sessionId,
          runId: input.runId,
          messageId: input.messageId,
          attributes: {
            rating: input.rating,
            reason: input.reason,
            hasNote: Boolean(input.note),
          },
        }, privacy)
      })
      tx()
      const row = db.prepare('SELECT * FROM feedback WHERE id=?').get(id) as FeedbackRow
      return toFeedback(row)
    },

    getPrivacySettings,

    updatePrivacySettings(input) {
      const next = mergePrivacySettings(
        getPrivacySettings(),
        input,
        new Date().toISOString(),
      )
      savePrivacySettings(next)
      pruneExpired()
      return next
    },

    feedbackPromptState(input = {}) {
      const privacy = getPrivacySettings()
      const nowMs = Date.now()
      const from24h = nowMs - DAY_MS
      const cooldownMs = privacy.feedbackPromptCooldownHours * 60 * 60 * 1000
      const eventScope = scopedWhere(input, true)
      const feedbackScope = scopedWhere(input, false)
      const promptCount24h = (db.prepare(
        `SELECT COUNT(*) AS count FROM events
         WHERE event_type='feedback.prompt_shown' AND timestamp_ms >= ?
         ${eventScope.sql}`,
      ).get(from24h, ...eventScope.params) as { count: number }).count
      const feedbackCount24h = (db.prepare(
        `SELECT COUNT(*) AS count FROM feedback
         WHERE timestamp_ms >= ?
         ${feedbackScope.sql}`,
      ).get(from24h, ...feedbackScope.params) as { count: number }).count
      const lastPrompt = db.prepare(
        `SELECT timestamp_ms, timestamp_iso FROM events
         WHERE event_type='feedback.prompt_shown'
         ${eventScope.sql}
         ORDER BY timestamp_ms DESC
         LIMIT 1`,
      ).get(...eventScope.params) as
        | { timestamp_ms: number; timestamp_iso: string }
        | undefined

      if (!privacy.localCollectionEnabled || !privacy.feedbackCollectionEnabled) {
        return {
          shouldPrompt: false,
          reason: 'disabled',
          cooldownHours: privacy.feedbackPromptCooldownHours,
          promptCount24h,
          feedbackCount24h,
          lastPromptAt: lastPrompt?.timestamp_iso,
        }
      }
      if (feedbackCount24h > 0) {
        return {
          shouldPrompt: false,
          reason: 'recent-feedback',
          cooldownHours: privacy.feedbackPromptCooldownHours,
          promptCount24h,
          feedbackCount24h,
          lastPromptAt: lastPrompt?.timestamp_iso,
        }
      }
      if (lastPrompt && nowMs - lastPrompt.timestamp_ms < cooldownMs) {
        return {
          shouldPrompt: false,
          reason: 'cooldown',
          cooldownHours: privacy.feedbackPromptCooldownHours,
          promptCount24h,
          feedbackCount24h,
          lastPromptAt: lastPrompt.timestamp_iso,
          nextPromptAfter: new Date(lastPrompt.timestamp_ms + cooldownMs).toISOString(),
        }
      }
      return {
        shouldPrompt: true,
        reason: 'eligible',
        cooldownHours: privacy.feedbackPromptCooldownHours,
        promptCount24h,
        feedbackCount24h,
        lastPromptAt: lastPrompt?.timestamp_iso,
      }
    },

    exportBundle(input = {}) {
      const range = input.range ?? '7d'
      const limit = Math.min(Math.max(input.limit ?? 250, 1), 1_000)
      const toMs = Date.now()
      const fromMs = toMs - RANGE_MS[range]
      const privacy = getPrivacySettings()
      const events = input.includeEvents === false
        ? []
        : (
            db.prepare(
              `SELECT * FROM events
               WHERE timestamp_ms >= ? AND timestamp_ms <= ?
               ORDER BY timestamp_ms DESC
               LIMIT ?`,
            ).all(fromMs, toMs, limit) as EventRow[]
          ).map(toEvent)
      const crashes = input.includeCrashes === false
        ? []
        : (
            db.prepare(
              `SELECT * FROM events
               WHERE timestamp_ms >= ? AND timestamp_ms <= ?
                 AND (
                   severity='fatal'
                   OR event_type LIKE '%.crash'
                   OR event_type LIKE '%.renderer_crash'
                 )
               ORDER BY timestamp_ms DESC
               LIMIT ?`,
            ).all(fromMs, toMs, limit) as EventRow[]
          ).map(toEvent)
      const feedback = input.includeFeedback === false
        ? []
        : feedbackRowsInRange(fromMs, toMs, limit)
      // Direct /export previously emitted event messages, attributes, snapshot
      // recents, and feedback notes verbatim to the caller AND to a file on
      // disk. Redact the whole bundle with the same secret-key redactor the
      // support bundle uses (idempotent, so the support-bundle path that
      // re-redacts is unaffected) so credentials cannot leak through a direct
      // export or its persisted artifact.
      const bundle = redactSecretKeys(
        {
          generatedAt: new Date(toMs).toISOString(),
          privacy,
          snapshot: this.snapshot(range),
          events,
          crashes,
          feedback,
          redaction: {
            attributes: 'sanitized-at-ingest',
            notes: 'redacted-or-truncated',
          },
        },
        createTraceRedactionContext(),
      ) as ObservabilityExportBundle
      let path: string | undefined
      if (input.persist !== false) {
        const dir = join(sepilotdHome(), 'observability', 'exports')
        mkdirSync(dir, { recursive: true })
        const stamp = bundle.generatedAt.replace(/[:.]/g, '-')
        path = join(dir, `observability-${stamp}.json`)
        writeFileSync(path, `${JSON.stringify(bundle, null, 2)}\n`, 'utf8')
      }
      return {
        path,
        eventCount: events.length,
        crashCount: crashes.length,
        feedbackCount: feedback.length,
        bundle,
      }
    },

    prune() {
      return pruneExpired()
    },

    snapshot(range = '7d') {
      const toMs = Date.now()
      const fromMs = toMs - RANGE_MS[range]
      const fromIso = new Date(fromMs).toISOString()
      const toIso = new Date(toMs).toISOString()
      const totalEvents = countAllEvents(db, fromMs, toMs)
      const errorEvents = countSeverity(db, fromMs, toMs, ['error', 'fatal'])
      const fatalEvents = countSeverity(db, fromMs, toMs, ['fatal'])
      const crashReports = countCrashReports(db, fromMs, toMs)
      const sessionTotals = db.prepare(
        `SELECT
          COUNT(DISTINCT session_id) AS total,
          COUNT(DISTINCT CASE
            WHEN severity='fatal' OR event_type LIKE '%.crash'
              OR event_type LIKE '%.renderer_crash'
            THEN session_id END) AS crashed
         FROM events
         WHERE timestamp_ms >= ? AND timestamp_ms <= ? AND session_id IS NOT NULL`,
      ).get(fromMs, toMs) as { total: number; crashed: number }

      const providerFailures = countEvents(db, fromMs, toMs, ['provider.error'])
      const providerAttempts = countEvents(db, fromMs, toMs, [
        'provider.call',
        'provider.error',
        'assistant.completed',
      ])
      const channelSucceeded = countEvents(db, fromMs, toMs, [
        'channel.delivery_succeeded',
      ])
      const channelFailed = countEvents(db, fromMs, toMs, [
        'channel.delivery_failed',
      ])
      const routeErrors = countEvents(db, fromMs, toMs, ['route.error'])
      const routeRequests = countEvents(db, fromMs, toMs, [
        'route.request',
        'route.error',
      ])

      const feedbackRows = db.prepare(
        `SELECT * FROM feedback
         WHERE timestamp_ms >= ? AND timestamp_ms <= ?`,
      ).all(fromMs, toMs) as FeedbackRow[]
      const feedback = feedbackRows.map(toFeedback)
      const feedbackPositive = feedback.filter((f) => f.rating === 'positive').length
      const feedbackNeutral = feedback.filter((f) => f.rating === 'neutral').length
      const feedbackNegative = feedback.filter((f) => f.rating === 'negative').length
      const feedbackPromptShown = countEvents(db, fromMs, toMs, [
        'feedback.prompt_shown',
      ])

      const assistantCompleted = countEvents(db, fromMs, toMs, [
        'assistant.completed',
      ])
      const assistantStarted = countEvents(db, fromMs, toMs, ['assistant.started'])
      const passivePositive = countEvents(db, fromMs, toMs, [
        'message.copied',
        'message.starred',
        'snippet.saved',
      ])
      const regenerated = countEvents(db, fromMs, toMs, ['response.regenerated'])
      const stopped = countEvents(db, fromMs, toMs, ['stream.stopped'])
      const correctionFollowups = countEvents(db, fromMs, toMs, [
        'assistant.followup_within_window',
      ])

      const tasksStarted = countEvents(db, fromMs, toMs, [
        'task.started',
        'channel.task_started',
      ])
      const tasksCompleted = countEvents(db, fromMs, toMs, [
        'task.completed',
        'channel.task_completed',
      ])
      const toolSucceeded = countEvents(db, fromMs, toMs, [
        'tool.succeeded',
        'mcp.tool_succeeded',
      ])
      const toolFailed = countEvents(db, fromMs, toMs, [
        'tool.failed',
        'mcp.tool_failed',
      ])
      const recoverySucceeded = countEvents(db, fromMs, toMs, [
        'run.recovered',
        'session.recovered',
      ])
      const recoveryFailed = countEvents(db, fromMs, toMs, [
        'run.recovery_failed',
        'session.recovery_failed',
      ])
      const channelTasksStarted = countEvents(db, fromMs, toMs, [
        'channel.task_started',
      ])
      const channelTasksCompleted = countEvents(db, fromMs, toMs, [
        'channel.task_completed',
      ])

      const productivityRows = (
        db.prepare(
          `SELECT * FROM events
           WHERE timestamp_ms >= ? AND timestamp_ms <= ?
             AND event_type IN (
               'task.completed',
               'channel.task_completed',
               'approval.responded',
               'usage.recorded',
               'cost.recorded'
             )`,
        ).all(fromMs, toMs) as EventRow[]
      ).map(toEvent)
      const completedTaskRows = productivityRows.filter(
        (event) => event.eventType === 'task.completed'
          || event.eventType === 'channel.task_completed',
      )
      const autonomousCompletionRows = completedTaskRows.filter(
        (event) => typeof event.attributes.manualInterventions === 'number',
      )
      const autonomousCompleted = autonomousCompletionRows.filter((event) => {
        const manual = event.attributes.manualInterventions
        return typeof manual === 'number' && manual <= 0
      }).length
      const days = Math.max(RANGE_MS[range] / (24 * 60 * 60 * 1000), 1)
      const costUsd = sumAttribute(productivityRows, ['costUsd', 'totalCostUsd'])
      const tokens = sumAttribute(productivityRows, ['tokens', 'totalTokens'])
      const completionRate = ratio(tasksCompleted, tasksStarted)
      const assistedTaskThroughputPerDay = tasksCompleted / days
      const toolSuccessRate = ratio(toolSucceeded, toolSucceeded + toolFailed)
      const channelResolutionRate = ratio(
        channelTasksCompleted,
        channelTasksStarted,
      )
      const previousFromMs = fromMs - RANGE_MS[range]
      const previousToMs = fromMs - 1
      const previousMetrics = buildWindowMetrics(db, previousFromMs, previousToMs)
      const currentMetrics: WindowMetrics = {
        totalEvents,
        errorEvents,
        crashReports,
        feedbackPositive,
        feedbackNeutral,
        feedbackNegative,
        explicitSatisfaction: ratio(
          feedbackPositive,
          feedbackPositive + feedbackNegative,
        ),
        tasksStarted,
        tasksCompleted,
        completionRate,
        assistedTaskThroughputPerDay,
        channelTasksStarted,
        channelTasksCompleted,
        channelResolutionRate,
        toolSucceeded,
        toolFailed,
        toolSuccessRate,
      }
      const comparison: ObservabilityComparison = {
        previousFrom: new Date(previousFromMs).toISOString(),
        previousTo: new Date(previousToMs).toISOString(),
        errorEventsDelta: currentMetrics.errorEvents - previousMetrics.errorEvents,
        crashReportsDelta:
          currentMetrics.crashReports - previousMetrics.crashReports,
        satisfactionDelta: deltaNullable(
          currentMetrics.explicitSatisfaction,
          previousMetrics.explicitSatisfaction,
        ),
        completionRateDelta: deltaNullable(
          currentMetrics.completionRate,
          previousMetrics.completionRate,
        ),
        throughputPerDayDelta:
          currentMetrics.assistedTaskThroughputPerDay
          - previousMetrics.assistedTaskThroughputPerDay,
        channelResolutionRateDelta: deltaNullable(
          currentMetrics.channelResolutionRate,
          previousMetrics.channelResolutionRate,
        ),
        toolSuccessRateDelta: deltaNullable(
          currentMetrics.toolSuccessRate,
          previousMetrics.toolSuccessRate,
        ),
      }
      const trend = buildTrend(db, range, fromMs, toMs)
      const segments = buildSegments(db, fromMs, toMs)
      const hotspots = buildHotspots(db, fromMs, toMs)
      const alerts = buildAlerts(currentMetrics, comparison)

      const recent = (
        db.prepare(
          `SELECT * FROM events
           WHERE timestamp_ms >= ? AND timestamp_ms <= ?
           ORDER BY timestamp_ms DESC
           LIMIT 10`,
        ).all(fromMs, toMs) as EventRow[]
      ).map(toEvent)

      return {
        generatedAt: new Date(toMs).toISOString(),
        range: { key: range, from: fromIso, to: toIso },
        reliability: {
          totalEvents,
          errorEvents,
          fatalEvents,
          crashReports,
          crashFreeSessions: sessionTotals.total > 0
            ? 1 - (sessionTotals.crashed / sessionTotals.total)
            : null,
          providerFailureRate: ratio(providerFailures, providerAttempts),
          channelDeliverySuccessRate: ratio(
            channelSucceeded,
            channelSucceeded + channelFailed,
          ),
          routeErrorRate: ratio(routeErrors, routeRequests),
        },
        quality: {
          feedbackPositive,
          feedbackNeutral,
          feedbackNegative,
          explicitSatisfaction: currentMetrics.explicitSatisfaction,
          promptResponseRate: ratio(feedback.length, feedbackPromptShown),
          implicitAcceptance: ratio(
            passivePositive,
            passivePositive + regenerated + stopped + correctionFollowups,
          ),
          regenerationRate: ratio(regenerated, assistantCompleted),
          stopRate: ratio(stopped, assistantStarted),
        },
        productivity: {
          tasksStarted,
          tasksCompleted,
          completionRate,
          assistedTaskThroughputPerDay,
          autonomousCompletionRate: autonomousCompletionRows.length > 0
            ? autonomousCompleted / autonomousCompletionRows.length
            : null,
          approvalFrictionMs: averageAttribute(productivityRows, [
            'approvalLatencyMs',
            'durationMs',
          ]),
          toolSuccessRate,
          costPerResolvedTask: ratio(costUsd, tasksCompleted),
          tokensPerResolvedTask: ratio(tokens, tasksCompleted),
          recoverySuccessRate: ratio(
            recoverySucceeded,
            recoverySucceeded + recoveryFailed,
          ),
          channelResolutionRate,
        },
        comparison,
        trend,
        segments,
        hotspots,
        alerts,
        recent,
      }
    },

    listEvents(input = {}) {
      const limit = Math.min(Math.max(input.limit ?? 50, 1), 200)
      const where: string[] = []
      const params: unknown[] = []
      if (input.severity) {
        where.push('severity = ?')
        params.push(input.severity)
      }
      if (input.eventType) {
        where.push('event_type = ?')
        params.push(input.eventType)
      }
      const sql = [
        'SELECT * FROM events',
        where.length ? `WHERE ${where.join(' AND ')}` : '',
        'ORDER BY timestamp_ms DESC',
        'LIMIT ?',
      ].filter(Boolean).join(' ')
      return (db.prepare(sql).all(...params, limit) as EventRow[]).map(toEvent)
    },

    listCrashes(input = {}) {
      const limit = Math.min(Math.max(input.limit ?? 50, 1), 200)
      return (
        db.prepare(
          `SELECT * FROM events
           WHERE severity='fatal'
             OR event_type LIKE '%.crash'
             OR event_type LIKE '%.renderer_crash'
           ORDER BY timestamp_ms DESC
           LIMIT ?`,
        ).all(limit) as EventRow[]
      ).map(toEvent)
    },
  }
}
