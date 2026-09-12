import type { SqliteDatabase } from '../db/sqlite.js'
import { openDomainDb } from '../storage/domain-db.js'

export const MONITOR_SAMPLE_RETENTION = 4_096

export type MonitorObservedStatus = 'healthy' | 'anomaly' | 'unknown'
export type MonitorSeverity = 'warning' | 'high' | 'critical'
export type MonitorTransitionType =
  | 'anomaly_opened'
  | 'severity_changed'
  | 'anomaly_recovered'
  | 'observation_error'
  | 'observation_recovered'

export interface MonitorThresholds {
  anomaly: number
  recovery: number
  error: number
}

export interface MonitorEvaluationInput {
  scopeKey: string
  monitorId: string
  contractVersion: string
  contractHash: string
  sampleHash: string
  observationHash: string
  observedStatus: MonitorObservedStatus
  severity: MonitorSeverity | null
  evidenceSummary: string
  evidenceRedacted: boolean
  metrics: Record<string, number>
  thresholds: MonitorThresholds
  observedAt: number
  now: number
}

export interface MonitorEvaluationResult {
  outcome: 'quiet' | 'pending' | 'transition' | 'duplicate' | 'stale'
  originalOutcome?: 'quiet' | 'pending' | 'transition'
  monitorId: string
  contractVersion: string
  observedStatus: MonitorObservedStatus
  stableStatus: 'healthy' | 'anomaly' | null
  stableSeverity: MonitorSeverity | null
  candidate: {
    status: 'healthy' | 'anomaly' | null
    severity: MonitorSeverity | null
    count: number
    required: number | null
  }
  observationErrors: {
    open: boolean
    consecutive: number
    required: number
  }
  transitions: MonitorTransitionType[]
  stateVersion: number
  shouldNotify: boolean
  notificationDedupKey: string | null
  evidenceSummary: string
  evidenceRedacted: boolean
  observedAt: string
}

export interface MonitorReportInput {
  scopeKey: string
  monitorId: string
  since: number
  until: number
}

export interface MonitorReportResult {
  monitorId: string
  contractVersion: string
  stateVersion: number
  stableStatus: 'healthy' | 'anomaly' | null
  stableSeverity: MonitorSeverity | null
  period: { since: string; until: string }
  samples: {
    count: number
    healthy: number
    anomaly: number
    unknown: number
    unknownIntervals: number
    firstObservedAt: string | null
    lastObservedAt: string | null
    retentionLimit: number
    retentionLimited: boolean
  }
  transitions: Record<MonitorTransitionType, number>
  metrics: Record<string, {
    current: number
    min: number
    max: number
    average: number
    samples: number
  }>
}

interface MonitorStateRow {
  scope_key: string
  monitor_id: string
  contract_version: string
  contract_hash: string
  stable_status: 'healthy' | 'anomaly' | null
  stable_severity: MonitorSeverity | null
  candidate_status: 'healthy' | 'anomaly' | null
  candidate_severity: MonitorSeverity | null
  candidate_count: number
  error_open: number
  error_count: number
  state_version: number
  pruned_samples: number
  last_observed_at: number | null
  updated_at: number
}

interface MonitorSampleRow {
  observation_hash: string
  contract_version: string
  observed_status: MonitorObservedStatus
  severity: MonitorSeverity | null
  evidence_summary: string
  evidence_redacted: number
  metrics_json: string
  outcome: 'quiet' | 'pending' | 'transition'
  transitions_json: string
  state_version: number
  should_notify: number
  notification_dedup_key: string | null
  observed_at: number
}

interface MonitorReportSampleRow {
  observed_status: MonitorObservedStatus
  metrics_json: string
  transitions_json: string
  observed_at: number
}

function ensureSchema(db: SqliteDatabase): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS monitor_states (
      scope_key TEXT NOT NULL,
      monitor_id TEXT NOT NULL,
      contract_version TEXT NOT NULL,
      contract_hash TEXT NOT NULL,
      stable_status TEXT,
      stable_severity TEXT,
      candidate_status TEXT,
      candidate_severity TEXT,
      candidate_count INTEGER NOT NULL DEFAULT 0,
      error_open INTEGER NOT NULL DEFAULT 0,
      error_count INTEGER NOT NULL DEFAULT 0,
      state_version INTEGER NOT NULL DEFAULT 0,
      pruned_samples INTEGER NOT NULL DEFAULT 0,
      last_observed_at INTEGER,
      updated_at INTEGER NOT NULL,
      PRIMARY KEY (scope_key, monitor_id)
    );
    CREATE TABLE IF NOT EXISTS monitor_samples (
      scope_key TEXT NOT NULL,
      monitor_id TEXT NOT NULL,
      sample_hash TEXT NOT NULL,
      observation_hash TEXT NOT NULL,
      contract_version TEXT NOT NULL,
      observed_status TEXT NOT NULL,
      severity TEXT,
      evidence_summary TEXT NOT NULL,
      evidence_redacted INTEGER NOT NULL DEFAULT 0,
      metrics_json TEXT NOT NULL DEFAULT '{}',
      outcome TEXT NOT NULL,
      transitions_json TEXT NOT NULL DEFAULT '[]',
      state_version INTEGER NOT NULL,
      should_notify INTEGER NOT NULL DEFAULT 0,
      notification_dedup_key TEXT,
      observed_at INTEGER NOT NULL,
      created_at INTEGER NOT NULL,
      PRIMARY KEY (scope_key, monitor_id, sample_hash)
    );
    CREATE INDEX IF NOT EXISTS idx_monitor_samples_period
      ON monitor_samples(scope_key, monitor_id, observed_at DESC);
  `)
  const stateColumns = new Set(
    (db.prepare('PRAGMA table_info(monitor_states)').all() as Array<{ name: string }>)
      .map((column) => column.name),
  )
  if (!stateColumns.has('last_observed_at')) {
    db.prepare('ALTER TABLE monitor_states ADD COLUMN last_observed_at INTEGER').run()
  }
}

function parseTransitions(raw: string): MonitorTransitionType[] {
  try {
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed as MonitorTransitionType[] : []
  } catch {
    return []
  }
}

function parseMetrics(raw: string): Record<string, number> {
  try {
    const parsed = JSON.parse(raw) as Record<string, unknown>
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return {}
    return Object.fromEntries(
      Object.entries(parsed).filter((entry): entry is [string, number] => (
        typeof entry[1] === 'number' && Number.isFinite(entry[1])
      )),
    )
  } catch {
    return {}
  }
}

function requiredForCandidate(
  candidateStatus: 'healthy' | 'anomaly' | null,
  thresholds: MonitorThresholds,
): number | null {
  if (candidateStatus === 'anomaly') return thresholds.anomaly
  if (candidateStatus === 'healthy') return thresholds.recovery
  return null
}

function shouldNotify(transitions: readonly MonitorTransitionType[]): boolean {
  return transitions.some((transition) => transition !== 'observation_recovered')
}

function duplicateResult(
  input: MonitorEvaluationInput,
  sample: MonitorSampleRow,
  state: MonitorStateRow,
): MonitorEvaluationResult {
  return {
    outcome: 'duplicate',
    originalOutcome: sample.outcome,
    monitorId: input.monitorId,
    contractVersion: sample.contract_version,
    observedStatus: sample.observed_status,
    stableStatus: state.stable_status,
    stableSeverity: state.stable_severity,
    candidate: {
      status: state.candidate_status,
      severity: state.candidate_severity,
      count: state.candidate_count,
      required: requiredForCandidate(state.candidate_status, input.thresholds),
    },
    observationErrors: {
      open: state.error_open === 1,
      consecutive: state.error_count,
      required: input.thresholds.error,
    },
    transitions: parseTransitions(sample.transitions_json),
    stateVersion: state.state_version,
    shouldNotify: false,
    notificationDedupKey: null,
    evidenceSummary: sample.evidence_summary,
    evidenceRedacted: sample.evidence_redacted === 1,
    observedAt: new Date(sample.observed_at).toISOString(),
  }
}

function staleResult(
  input: MonitorEvaluationInput,
  state: MonitorStateRow,
): MonitorEvaluationResult {
  return {
    outcome: 'stale',
    monitorId: input.monitorId,
    contractVersion: state.contract_version,
    observedStatus: input.observedStatus,
    stableStatus: state.stable_status,
    stableSeverity: state.stable_severity,
    candidate: {
      status: state.candidate_status,
      severity: state.candidate_severity,
      count: state.candidate_count,
      required: requiredForCandidate(state.candidate_status, input.thresholds),
    },
    observationErrors: {
      open: state.error_open === 1,
      consecutive: state.error_count,
      required: input.thresholds.error,
    },
    transitions: [],
    stateVersion: state.state_version,
    shouldNotify: false,
    notificationDedupKey: null,
    evidenceSummary: input.evidenceSummary,
    evidenceRedacted: input.evidenceRedacted,
    observedAt: new Date(input.observedAt).toISOString(),
  }
}

function notificationKey(
  input: MonitorEvaluationInput,
  version: number,
  transitions: readonly MonitorTransitionType[],
): string {
  return `monitor:${input.monitorId}:${version}:${transitions.join('+')}`
}

export interface MonitorStateRepo {
  evaluate(input: MonitorEvaluationInput): MonitorEvaluationResult
  report(input: MonitorReportInput): MonitorReportResult | null
}

export function createMonitorStateRepo(): MonitorStateRepo {
  const db = openDomainDb({ name: 'monitoring' })
  ensureSchema(db)

  const evaluateTransaction = db.transaction((input: MonitorEvaluationInput) => {
    const stateQuery = db.prepare(
      'SELECT * FROM monitor_states WHERE scope_key = ? AND monitor_id = ?',
    )
    const existingState = stateQuery.get(input.scopeKey, input.monitorId) as MonitorStateRow | undefined
    const existingSample = db.prepare(
      `SELECT observation_hash, contract_version, observed_status, severity,
        evidence_summary, evidence_redacted, metrics_json, outcome,
        transitions_json, state_version, should_notify,
        notification_dedup_key, observed_at
       FROM monitor_samples
       WHERE scope_key = ? AND monitor_id = ? AND sample_hash = ?`,
    ).get(input.scopeKey, input.monitorId, input.sampleHash) as MonitorSampleRow | undefined

    if (existingSample) {
      if (existingSample.observation_hash !== input.observationHash) {
        throw new Error('MONITOR_SAMPLE_CONFLICT_PERMANENT')
      }
      if (!existingState) throw new Error('MONITOR_STATE_INCONSISTENT_PERMANENT')
      return duplicateResult(input, existingSample, existingState)
    }

    if (
      existingState
      && existingState.contract_version === input.contractVersion
      && existingState.contract_hash !== input.contractHash
    ) {
      throw new Error('MONITOR_CONTRACT_MISMATCH_USER')
    }
    if (
      existingState?.last_observed_at != null
      && input.observedAt < existingState.last_observed_at
    ) {
      return staleResult(input, existingState)
    }

    const contractChanged = existingState?.contract_version !== input.contractVersion
    const state: MonitorStateRow = existingState
      ? {
          ...existingState,
          contract_version: input.contractVersion,
          contract_hash: input.contractHash,
          candidate_status: contractChanged ? null : existingState.candidate_status,
          candidate_severity: contractChanged ? null : existingState.candidate_severity,
          candidate_count: contractChanged ? 0 : existingState.candidate_count,
          error_count: contractChanged ? 0 : existingState.error_count,
        }
      : {
          scope_key: input.scopeKey,
          monitor_id: input.monitorId,
          contract_version: input.contractVersion,
          contract_hash: input.contractHash,
          stable_status: null,
          stable_severity: null,
          candidate_status: null,
          candidate_severity: null,
          candidate_count: 0,
          error_open: 0,
          error_count: 0,
          state_version: 0,
          pruned_samples: 0,
          last_observed_at: null,
          updated_at: input.now,
        }

    const transitions: MonitorTransitionType[] = []
    if (input.observedStatus === 'unknown') {
      state.candidate_status = null
      state.candidate_severity = null
      state.candidate_count = 0
      if (state.error_open === 0) {
        state.error_count += 1
        if (state.error_count >= input.thresholds.error) {
          state.error_open = 1
          transitions.push('observation_error')
        }
      }
    } else {
      if (state.error_open === 1) {
        state.error_open = 0
        transitions.push('observation_recovered')
      }
      state.error_count = 0

      const desiredStatus = input.observedStatus
      const desiredSeverity = desiredStatus === 'anomaly' ? input.severity : null
      const stableMatches = state.stable_status === desiredStatus
        && (desiredStatus !== 'anomaly' || state.stable_severity === desiredSeverity)

      if (state.stable_status === null && desiredStatus === 'healthy') {
        state.stable_status = 'healthy'
        state.stable_severity = null
        state.candidate_status = null
        state.candidate_severity = null
        state.candidate_count = 0
      } else if (stableMatches) {
        state.candidate_status = null
        state.candidate_severity = null
        state.candidate_count = 0
      } else {
        if (
          state.candidate_status === desiredStatus
          && state.candidate_severity === desiredSeverity
        ) {
          state.candidate_count += 1
        } else {
          state.candidate_status = desiredStatus
          state.candidate_severity = desiredSeverity
          state.candidate_count = 1
        }

        const required = desiredStatus === 'anomaly'
          ? input.thresholds.anomaly
          : input.thresholds.recovery
        if (state.candidate_count >= required) {
          const previousStatus = state.stable_status
          const previousSeverity = state.stable_severity
          state.stable_status = desiredStatus
          state.stable_severity = desiredSeverity
          state.candidate_status = null
          state.candidate_severity = null
          state.candidate_count = 0
          if (desiredStatus === 'healthy') {
            transitions.push('anomaly_recovered')
          } else if (previousStatus !== 'anomaly') {
            transitions.push('anomaly_opened')
          } else if (previousSeverity !== desiredSeverity) {
            transitions.push('severity_changed')
          }
        }
      }
    }

    if (transitions.length > 0) state.state_version += 1
    state.last_observed_at = input.observedAt
    state.updated_at = input.now
    const notificationRequired = shouldNotify(transitions)
    const dedupKey = notificationRequired
      ? notificationKey(input, state.state_version, transitions)
      : null
    const pending = transitions.length === 0 && (
      state.candidate_count > 0 || (state.error_open === 0 && state.error_count > 0)
    )
    const outcome: 'quiet' | 'pending' | 'transition' = transitions.length > 0
      ? 'transition'
      : pending
        ? 'pending'
        : 'quiet'

    db.prepare(
      `INSERT INTO monitor_states (
        scope_key, monitor_id, contract_version, contract_hash,
        stable_status, stable_severity, candidate_status, candidate_severity,
        candidate_count, error_open, error_count, state_version, pruned_samples,
        last_observed_at, updated_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(scope_key, monitor_id) DO UPDATE SET
        contract_version=excluded.contract_version,
        contract_hash=excluded.contract_hash,
        stable_status=excluded.stable_status,
        stable_severity=excluded.stable_severity,
        candidate_status=excluded.candidate_status,
        candidate_severity=excluded.candidate_severity,
        candidate_count=excluded.candidate_count,
        error_open=excluded.error_open,
        error_count=excluded.error_count,
        state_version=excluded.state_version,
        last_observed_at=excluded.last_observed_at,
        updated_at=excluded.updated_at`,
    ).run(
      state.scope_key,
      state.monitor_id,
      state.contract_version,
      state.contract_hash,
      state.stable_status,
      state.stable_severity,
      state.candidate_status,
      state.candidate_severity,
      state.candidate_count,
      state.error_open,
      state.error_count,
      state.state_version,
      state.pruned_samples,
      state.last_observed_at,
      state.updated_at,
    )

    db.prepare(
      `INSERT INTO monitor_samples (
        scope_key, monitor_id, sample_hash, observation_hash, contract_version,
        observed_status, severity, evidence_summary, evidence_redacted, metrics_json,
        outcome, transitions_json, state_version, should_notify,
        notification_dedup_key, observed_at, created_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    ).run(
      input.scopeKey,
      input.monitorId,
      input.sampleHash,
      input.observationHash,
      input.contractVersion,
      input.observedStatus,
      input.severity,
      input.evidenceSummary,
      input.evidenceRedacted ? 1 : 0,
      JSON.stringify(input.metrics),
      outcome,
      JSON.stringify(transitions),
      state.state_version,
      notificationRequired ? 1 : 0,
      dedupKey,
      input.observedAt,
      input.now,
    )

    const deleted = db.prepare(
      `DELETE FROM monitor_samples
       WHERE scope_key = ? AND monitor_id = ? AND sample_hash IN (
         SELECT sample_hash FROM monitor_samples
         WHERE scope_key = ? AND monitor_id = ?
         ORDER BY observed_at DESC, created_at DESC
         LIMIT -1 OFFSET ?
       )`,
    ).run(
      input.scopeKey,
      input.monitorId,
      input.scopeKey,
      input.monitorId,
      MONITOR_SAMPLE_RETENTION,
    ).changes
    if (deleted > 0) {
      state.pruned_samples += deleted
      db.prepare(
        `UPDATE monitor_states SET pruned_samples = ?
         WHERE scope_key = ? AND monitor_id = ?`,
      ).run(state.pruned_samples, input.scopeKey, input.monitorId)
    }

    return {
      outcome,
      monitorId: input.monitorId,
      contractVersion: input.contractVersion,
      observedStatus: input.observedStatus,
      stableStatus: state.stable_status,
      stableSeverity: state.stable_severity,
      candidate: {
        status: state.candidate_status,
        severity: state.candidate_severity,
        count: state.candidate_count,
        required: requiredForCandidate(state.candidate_status, input.thresholds),
      },
      observationErrors: {
        open: state.error_open === 1,
        consecutive: state.error_count,
        required: input.thresholds.error,
      },
      transitions,
      stateVersion: state.state_version,
      shouldNotify: notificationRequired,
      notificationDedupKey: dedupKey,
      evidenceSummary: input.evidenceSummary,
      evidenceRedacted: input.evidenceRedacted,
      observedAt: new Date(input.observedAt).toISOString(),
    } satisfies MonitorEvaluationResult
  })

  return {
    evaluate(input) {
      return evaluateTransaction(input)
    },
    report(input) {
      const state = db.prepare(
        'SELECT * FROM monitor_states WHERE scope_key = ? AND monitor_id = ?',
      ).get(input.scopeKey, input.monitorId) as MonitorStateRow | undefined
      if (!state) return null

      const rows = db.prepare(
        `SELECT observed_status, metrics_json, transitions_json, observed_at
         FROM monitor_samples
         WHERE scope_key = ? AND monitor_id = ? AND observed_at >= ? AND observed_at <= ?
         ORDER BY observed_at ASC`,
      ).all(input.scopeKey, input.monitorId, input.since, input.until) as MonitorReportSampleRow[]
      const earliestRetained = db.prepare(
        `SELECT MIN(observed_at) AS observed_at FROM monitor_samples
         WHERE scope_key = ? AND monitor_id = ?`,
      ).get(input.scopeKey, input.monitorId) as { observed_at: number | null }

      const counts = { healthy: 0, anomaly: 0, unknown: 0 }
      const transitionCounts: Record<MonitorTransitionType, number> = {
        anomaly_opened: 0,
        severity_changed: 0,
        anomaly_recovered: 0,
        observation_error: 0,
        observation_recovered: 0,
      }
      let unknownIntervals = 0
      let previousUnknown = false
      const metricStats = new Map<string, {
        current: number
        min: number
        max: number
        total: number
        samples: number
      }>()

      for (const row of rows) {
        counts[row.observed_status] += 1
        const unknown = row.observed_status === 'unknown'
        if (unknown && !previousUnknown) unknownIntervals += 1
        previousUnknown = unknown
        for (const transition of parseTransitions(row.transitions_json)) {
          transitionCounts[transition] += 1
        }
        for (const [name, value] of Object.entries(parseMetrics(row.metrics_json))) {
          const current = metricStats.get(name)
          metricStats.set(name, current
            ? {
                current: value,
                min: Math.min(current.min, value),
                max: Math.max(current.max, value),
                total: current.total + value,
                samples: current.samples + 1,
              }
            : { current: value, min: value, max: value, total: value, samples: 1 })
        }
      }

      const metrics = Object.fromEntries(
        [...metricStats.entries()].map(([name, stat]) => [name, {
          current: stat.current,
          min: stat.min,
          max: stat.max,
          average: stat.total / stat.samples,
          samples: stat.samples,
        }]),
      )
      return {
        monitorId: input.monitorId,
        contractVersion: state.contract_version,
        stateVersion: state.state_version,
        stableStatus: state.stable_status,
        stableSeverity: state.stable_severity,
        period: {
          since: new Date(input.since).toISOString(),
          until: new Date(input.until).toISOString(),
        },
        samples: {
          count: rows.length,
          ...counts,
          unknownIntervals,
          firstObservedAt: rows[0] ? new Date(rows[0].observed_at).toISOString() : null,
          lastObservedAt: rows.at(-1) ? new Date(rows.at(-1)!.observed_at).toISOString() : null,
          retentionLimit: MONITOR_SAMPLE_RETENTION,
          retentionLimited: state.pruned_samples > 0
            && earliestRetained.observed_at !== null
            && earliestRetained.observed_at > input.since,
        },
        transitions: transitionCounts,
        metrics,
      }
    },
  }
}
