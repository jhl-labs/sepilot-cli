import { enforceOutputLimit } from './output-limit.js'
import { redactSensitive } from '../memory/sensitive.js'
import type {
  TerminalRunner,
  TerminalRunnerResult,
} from './terminal.js'
import type {
  ToolDefinitionRuntime,
  ToolExecutionContext,
  ToolResult,
} from './registry.js'

const DEFAULT_LIMIT = 20
const MAX_LIMIT = 50
const MAX_RUN_DETAILS = 5
const MAX_JOBS_PER_DETAIL = 10
const COMMAND_TIMEOUT_MS = 30_000

export interface GiteaActionsRunsToolOptions {
  runner?: TerminalRunner
}

function finiteLimit(value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) return DEFAULT_LIMIT
  return Math.max(1, Math.min(MAX_LIMIT, Math.trunc(value)))
}

function giteaActionsObservationCovers(
  observed: Record<string, unknown>,
  requested: Record<string, unknown>,
): boolean {
  const observedRemote = remoteAlias(observed.remote)
  const requestedRemote = remoteAlias(requested.remote)
  if (!observedRemote || observedRemote !== requestedRemote) return false
  const observedLimit = finiteLimit(observed.limit)
  const requestedLimit = finiteLimit(requested.limit)
  if (observedLimit < requestedLimit) return false
  const observedIncludesRunDetails = observed.includeRunDetails !== false
  const requestedIncludesRunDetails = requested.includeRunDetails !== false
  if (!observedIncludesRunDetails && requestedIncludesRunDetails) return false
  const observedIncludesDetails = observed.includeInProgressDetails !== false
  const requestedIncludesDetails = requested.includeInProgressDetails !== false
  if (observedLimit !== requestedLimit
    && (requestedIncludesRunDetails || requestedIncludesDetails)) return false
  return observedIncludesDetails || !requestedIncludesDetails
}

function remoteAlias(value: unknown): string | null {
  if (typeof value !== 'string') return null
  const normalized = value.trim()
  return /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/u.test(normalized)
    ? normalized
    : null
}

interface GiteaRunProjection {
  id: string
  status: string
  conclusion: string | null
  workflow?: string
  branch?: string
  event?: string
  headSha?: string
  runNumber?: string
  path?: string
  createdAt: string | null
  startedAt: string | null
  completedAt: string | null
  duration: string | null
  url: string | null
}

interface GiteaJobProjection {
  id?: string
  name?: string
  status: string
  conclusion?: string
  startedAt: string | null
  duration: string | null
}

interface GiteaRunDetailProjection extends GiteaRunProjection {
  jobsObserved: number
  jobsTruncated: boolean
  jobs: GiteaJobProjection[]
}

const CONTROL_CHARACTER_PATTERN = /[\u0000-\u001f\u007f]+/gu
const IN_PROGRESS_STATUSES = new Set(['in_progress', 'running'])

function boundedText(value: unknown, maxLength: number): string | null {
  if (typeof value !== 'string') return null
  const normalized = value
    .replace(CONTROL_CHARACTER_PATTERN, ' ')
    .replace(/\s+/gu, ' ')
    .trim()
  if (!normalized) return null
  return normalized.slice(0, maxLength)
}

function firstValue(record: Record<string, unknown>, keys: readonly string[]): unknown {
  for (const key of keys) {
    if (record[key] !== undefined && record[key] !== null) return record[key]
  }
  return undefined
}

function positiveId(value: unknown): string | null {
  const candidate = typeof value === 'number' && Number.isSafeInteger(value)
    ? String(value)
    : typeof value === 'string'
      ? value.trim()
      : ''
  if (!/^\d{1,19}$/u.test(candidate)) return null
  return BigInt(candidate) > 0n ? candidate : null
}

function normalizedStatus(value: unknown): string {
  return boundedText(value, 32)
    ?.toLowerCase()
    .replace(/[\s-]+/gu, '_') ?? 'unknown'
}

function normalizedTimestamp(value: unknown): string | null {
  const candidate = boundedText(value, 64)
  if (!candidate || /^1970-01-01(?:[T ]|$)/u.test(candidate)) return null
  if (!/(?:Z|[+-]\d{2}:?\d{2})$/iu.test(candidate)) return null
  return Number.isFinite(Date.parse(candidate)) ? candidate : null
}

function normalizedTiming(
  startedAt: string | null,
  completedAt: string | null,
  durationValue: unknown,
): { completedAt: string | null; duration: string | null } {
  const inverted = Boolean(
    startedAt
    && completedAt
    && Date.parse(completedAt) < Date.parse(startedAt),
  )
  const duration = boundedText(durationValue, 24)
  return {
    completedAt: inverted ? null : completedAt,
    duration: startedAt && !inverted && duration && !duration.startsWith('-')
      ? duration
      : null,
  }
}

function normalizedTeaUtcHeaderTimestamp(value: string | undefined): string | undefined {
  if (!value) return value
  const match = /^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2})(?::(\d{2}))?$/u.exec(value.trim())
  if (!match?.[1] || !match[2]) return value
  return `${match[1]}T${match[2]}:${match[3] ?? '00'}Z`
}

function normalizedHeadSha(value: unknown): string | null {
  const candidate = boundedText(value, 64)
  return candidate && /^[a-f0-9]{7,64}$/iu.test(candidate) ? candidate : null
}

function normalizedRunUrl(value: unknown, runId: string): string | null {
  const candidate = boundedText(value, 500)
  if (!candidate) return null
  try {
    const url = new URL(candidate)
    if (!['http:', 'https:'].includes(url.protocol)) return null
    if (url.username || url.password || url.search || url.hash) return null
    const match = /\/actions\/runs\/(\d+)\/?$/u.exec(url.pathname)
    if (match?.[1] !== runId) return null
    return url.toString()
  } catch {
    return null
  }
}

function projectRun(record: Record<string, unknown>): GiteaRunProjection | null {
  const id = positiveId(firstValue(record, ['id', 'runId', 'run_id', 'index', 'runID']))
  if (!id) return null
  const createdAt = normalizedTimestamp(firstValue(record, [
    'createdAt', 'created_at', 'created',
  ]))
  const startedAt = normalizedTimestamp(firstValue(record, [
    'startedAt', 'started_at', 'runStartedAt', 'run_started_at', 'started',
  ]))
  const completedRaw = firstValue(record, ['completedAt', 'completed_at', 'completed'])
  const conclusion = boundedText(record.conclusion, 32)
  const workflow = boundedText(firstValue(record, ['workflow', 'name', 'title']), 120)
  const branch = boundedText(firstValue(record, ['branch', 'headBranch', 'head_branch']), 80)
  const event = boundedText(record.event, 32)
  const headSha = normalizedHeadSha(firstValue(record, ['headSha', 'head_sha', 'sha']))
  const runNumber = positiveId(firstValue(record, ['runNumber', 'run_number', 'number']))
  const path = boundedText(record.path, 200)
  const timing = normalizedTiming(
    startedAt,
    normalizedTimestamp(completedRaw),
    firstValue(record, ['duration', 'elapsed']),
  )
  return {
    id,
    status: normalizedStatus(record.status),
    conclusion: conclusion?.toLowerCase() ?? null,
    ...(workflow ? { workflow } : {}),
    ...(branch ? { branch } : {}),
    ...(event ? { event } : {}),
    ...(headSha ? { headSha } : {}),
    ...(runNumber ? { runNumber } : {}),
    ...(path ? { path } : {}),
    createdAt,
    startedAt,
    completedAt: timing.completedAt,
    duration: timing.duration,
    url: normalizedRunUrl(firstValue(record, ['htmlUrl', 'html_url', 'url']), id),
  }
}

function extractRunRecords(value: unknown): Array<Record<string, unknown>> | null {
  if (Array.isArray(value)) {
    if (value.some((item) => item === null || typeof item !== 'object' || Array.isArray(item))) {
      return null
    }
    return value as Array<Record<string, unknown>>
  }
  if (!value || typeof value !== 'object') return null
  const record = value as Record<string, unknown>
  for (const key of ['data', 'runs', 'items']) {
    if (Array.isArray(record[key])) return extractRunRecords(record[key])
    if (record[key] && typeof record[key] === 'object') {
      const nested = extractRunRecords(record[key])
      if (nested !== null) return nested
    }
  }
  return null
}

function projectJob(record: Record<string, unknown>): GiteaJobProjection {
  const id = positiveId(firstValue(record, ['id', 'jobId', 'job_id']))
  const name = boundedText(record.name, 120)
  const conclusion = boundedText(record.conclusion, 32)
  const startedAt = normalizedTimestamp(firstValue(record, [
    'startedAt', 'started_at', 'started', 'createdAt', 'created_at',
  ]))
  return {
    ...(id ? { id } : {}),
    ...(name ? { name } : {}),
    status: normalizedStatus(record.status),
    ...(conclusion ? { conclusion: conclusion.toLowerCase() } : {}),
    startedAt,
    duration: normalizedTiming(
      startedAt,
      null,
      firstValue(record, ['duration', 'elapsed']),
    ).duration,
  }
}

function projectRunDetail(
  record: Record<string, unknown>,
  jobsValue: unknown,
): GiteaRunDetailProjection | null {
  const run = projectRun(record)
  if (!run) return null
  const jobRecords = Array.isArray(jobsValue)
    ? jobsValue.filter((item): item is Record<string, unknown> => (
      item !== null && typeof item === 'object' && !Array.isArray(item)
    ))
    : []
  return {
    ...run,
    jobsObserved: jobRecords.length,
    jobsTruncated: jobRecords.length > MAX_JOBS_PER_DETAIL,
    jobs: jobRecords.slice(0, MAX_JOBS_PER_DETAIL).map(projectJob),
  }
}

function parseJsonRunDetail(value: unknown): GiteaRunDetailProjection | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  const outer = value as Record<string, unknown>
  const nestedRun = outer.run
  if (nestedRun && typeof nestedRun === 'object' && !Array.isArray(nestedRun)) {
    const run = nestedRun as Record<string, unknown>
    return projectRunDetail(run, outer.jobs ?? run.jobs)
  }
  return projectRunDetail(outer, outer.jobs)
}

function parseMixedTeaRunDetail(output: string): GiteaRunDetailProjection | null {
  const lines = output.replace(/\r/gu, '').split('\n')
  const jobsIndex = lines.findIndex((line) => /^Jobs:\s*$/iu.test(line.trim()))
  if (jobsIndex < 0) return null

  const headers = new Map<string, string>()
  for (const line of lines.slice(0, jobsIndex)) {
    const match = /^([A-Za-z][A-Za-z ]{0,39}):\s*(.*)$/u.exec(line.trim())
    if (match?.[1] && match[2] !== undefined) {
      headers.set(match[1].toLowerCase().replace(/\s+/gu, ' '), match[2])
    }
  }
  const jobsText = lines.slice(jobsIndex + 1).join('\n').trim()
  let jobs: unknown
  try {
    jobs = JSON.parse(jobsText)
  } catch {
    return null
  }
  if (!Array.isArray(jobs)) return null
  return projectRunDetail({
    id: headers.get('run id'),
    runNumber: headers.get('run number'),
    status: headers.get('status'),
    conclusion: headers.get('conclusion'),
    workflow: headers.get('workflow'),
    path: headers.get('path'),
    branch: headers.get('branch'),
    event: headers.get('event'),
    headSha: headers.get('head sha'),
    createdAt: normalizedTeaUtcHeaderTimestamp(headers.get('created')),
    startedAt: normalizedTeaUtcHeaderTimestamp(headers.get('started')),
    completedAt: normalizedTeaUtcHeaderTimestamp(headers.get('completed')),
    duration: headers.get('duration'),
    url: headers.get('url'),
  }, jobs)
}

function parseTeaRunDetail(output: string): GiteaRunDetailProjection | null {
  try {
    return parseJsonRunDetail(JSON.parse(output))
  } catch {
    return parseMixedTeaRunDetail(output)
  }
}

function runnerOutput(result: TerminalRunnerResult): string {
  const raw = [result.stdout.trim(), result.stderr.trim()].filter(Boolean).join('\n')
  return boundedText(redactSensitive(raw).redacted, 2_000) ?? ''
}

function trustedReadOnlyPosture(result: TerminalRunnerResult): boolean {
  return result.executionPosture.sandbox.active
    && result.executionPosture.filesystem.isolated
    && result.executionPosture.filesystem.readOnly === true
    && result.executionPosture.network.mode === 'host'
}

function inProgressRunIds(runs: readonly GiteaRunProjection[]): string[] {
  return [...new Set(runs
    .filter((run) => IN_PROGRESS_STATUSES.has(run.status))
    .map((run) => run.id))]
}

function mergeRunDetail(
  run: GiteaRunProjection,
  detail: GiteaRunDetailProjection,
): GiteaRunProjection {
  const startedAt = detail.startedAt ?? run.startedAt
  const timing = normalizedTiming(
    startedAt,
    detail.completedAt ?? run.completedAt,
    detail.duration ?? run.duration,
  )
  return {
    ...run,
    status: detail.status === 'unknown' ? run.status : detail.status,
    conclusion: detail.conclusion ?? run.conclusion,
    ...(detail.workflow ? { workflow: detail.workflow } : {}),
    ...(detail.branch ? { branch: detail.branch } : {}),
    ...(detail.event ? { event: detail.event } : {}),
    ...(detail.headSha ? { headSha: detail.headSha } : {}),
    ...(detail.runNumber ? { runNumber: detail.runNumber } : {}),
    ...(detail.path ? { path: detail.path } : {}),
    createdAt: detail.createdAt ?? run.createdAt,
    startedAt,
    completedAt: timing.completedAt,
    duration: timing.duration,
    url: detail.url ?? run.url,
  }
}

async function runTea(
  runner: TerminalRunner,
  args: string[],
  context: ToolExecutionContext,
): Promise<TerminalRunnerResult> {
  return runner.run({
    executable: 'tea',
    args,
    cwd: context.cwd ?? context.workspaceRoot,
    workspaceRoot: context.workspaceRoot,
    timeoutMs: COMMAND_TIMEOUT_MS,
    signal: context.signal,
    cwdBoundary: 'strict_workspace',
    capability: 'gitea-actions-readonly',
  })
}

export function createGiteaActionsRunsTool(
  options: GiteaActionsRunsToolOptions = {},
): ToolDefinitionRuntime {
  return {
    name: 'gitea.actions.runs.inspect',
    description:
      'Inspect a bounded, daemon-normalized Gitea Actions run projection for one existing local workspace remote alias. The tool lists recent runs and, by default, enriches up to five priority runs with conclusion, timestamps, and a validated run URL; actual in_progress/running runs are prioritized and may also include bounded job metadata. Every internal detail read reuses the alias byte-for-byte. Zero/uninitialized timestamps, negative durations, chronologically inverted completion times, and their bogus durations become null instead of runtime evidence. Preserve exact status/conclusion categories: only failure means failed; cancelled, skipped, queued, and in_progress remain distinct. Use this canonical read instead of separate tea list/view terminal calls. Discover remote names with one read-only `git remote` call when needed; pass a literal alias such as origin, never a URL. It exposes no action logs, secrets, variables, repository/login override, workflow mutation, cancellation, deletion, or rerun capability.',
    resumeSafety: 'replay-safe',
    scheduling: { mode: 'parallel-safe', resource: 'gitea-actions-runs' },
    observationCoverage: {
      covers: (observed, requested) => giteaActionsObservationCovers(observed, requested),
    },
    inputSchema: {
      type: 'object',
      properties: {
        remote: {
          type: 'string',
          minLength: 1,
          maxLength: 64,
          pattern: '^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$',
          description: 'Exact existing local workspace remote alias, for example origin. Never pass a URL.',
        },
        limit: {
          type: 'integer',
          minimum: 1,
          maximum: MAX_LIMIT,
          default: DEFAULT_LIMIT,
          description: 'Maximum recent runs to list. Defaults to 20.',
        },
        includeRunDetails: {
          type: 'boolean',
          default: true,
          description: 'Enrich up to five priority runs with bounded summary details. Defaults to true.',
        },
        includeInProgressDetails: {
          type: 'boolean',
          default: true,
          description: 'Prioritize and include bounded job details for up to five runs whose list status is in_progress or running.',
        },
      },
      required: ['remote'],
      additionalProperties: false,
    },
    async execute(
      input: Record<string, unknown>,
      context?: ToolExecutionContext,
    ): Promise<ToolResult> {
      const startedAt = Date.now()
      const remote = remoteAlias(input.remote)
      if (!remote) {
        return {
          output: 'A Gitea Actions inspection requires one literal local remote alias, never a URL.',
          status: 'error',
          code: 'GITEA_REMOTE_ALIAS_INVALID_PERMANENT',
          durationMs: Date.now() - startedAt,
        }
      }
      const runner = options.runner
      if (!context?.workspaceRoot || !runner) {
        return {
          output: 'Gitea Actions inspection requires an active strict workspace and its credential-safe read-only runner.',
          status: 'error',
          code: 'SANDBOX_UNAVAILABLE',
          durationMs: Date.now() - startedAt,
        }
      }

      const limit = finiteLimit(input.limit)
      const listResult = await runTea(runner, [
        'actions',
        'runs',
        'list',
        '--remote',
        remote,
        '--limit',
        String(limit),
        '--output',
        'json',
      ], context)
      if (!trustedReadOnlyPosture(listResult)) {
        return {
          output: 'The credential-safe read-only Gitea Actions sandbox did not become active.',
          status: 'error',
          code: 'SANDBOX_UNAVAILABLE',
          durationMs: Date.now() - startedAt,
          executionPosture: listResult.executionPosture,
        }
      }
      if (listResult.status !== 'success') {
        return {
          output: runnerOutput(listResult) || 'Gitea Actions run listing failed.',
          status: 'error',
          code: listResult.code ?? 'GITEA_ACTIONS_LIST_FAILED_TRANSIENT',
          durationMs: Date.now() - startedAt,
          executionPosture: listResult.executionPosture,
        }
      }

      let parsedList: unknown
      try {
        parsedList = JSON.parse(listResult.stdout)
      } catch {
        return {
          output: 'Gitea Actions run listing returned malformed JSON.',
          status: 'error',
          code: 'GITEA_ACTIONS_INVALID_JSON_PERMANENT',
          durationMs: Date.now() - startedAt,
          executionPosture: listResult.executionPosture,
        }
      }
      const sourceRuns = extractRunRecords(parsedList)
      if (sourceRuns === null) {
        return {
          output: 'Gitea Actions run listing returned an unsupported JSON schema.',
          status: 'error',
          code: 'GITEA_ACTIONS_INVALID_SCHEMA_PERMANENT',
          durationMs: Date.now() - startedAt,
          executionPosture: listResult.executionPosture,
        }
      }
      const projectedRuns = sourceRuns
        .map(projectRun)
        .filter((run): run is GiteaRunProjection => run !== null)
      let runs = projectedRuns.slice(0, limit)
      if (sourceRuns.length > 0 && projectedRuns.length === 0) {
        return {
          output: 'Gitea Actions run listing did not contain any valid positive run identities.',
          status: 'error',
          code: 'GITEA_ACTIONS_INVALID_SCHEMA_PERMANENT',
          durationMs: Date.now() - startedAt,
          executionPosture: listResult.executionPosture,
        }
      }

      const inProgressIds = inProgressRunIds(runs)
      const detailExpansionRequested = input.includeInProgressDetails !== false
      const runDetailExpansionRequested = input.includeRunDetails !== false
      const detailCandidateIds = [...new Set([
        ...(detailExpansionRequested ? inProgressIds : []),
        ...(runDetailExpansionRequested ? runs.map((run) => run.id) : []),
      ])]
      const detailIds = detailCandidateIds.slice(0, MAX_RUN_DETAILS)
      const detailResults = await Promise.all(detailIds.map(async (runId) => {
        const detail = await runTea(runner, [
          'actions',
          'runs',
          'view',
          runId,
          '--remote',
          remote,
          '--output',
          'json',
        ], context)
        if (!trustedReadOnlyPosture(detail)) {
          return { runId, status: 'error' as const, error: 'read-only sandbox unavailable' }
        }
        if (detail.status !== 'success') {
          return {
            runId,
            status: 'error' as const,
            error: runnerOutput(detail) || detail.code || 'detail lookup failed',
          }
        }
        const projectedDetail = parseTeaRunDetail(detail.stdout)
        if (!projectedDetail) {
          return {
            runId,
            status: 'error' as const,
            error: 'detail lookup returned an unsupported output schema',
          }
        }
        if (projectedDetail.id !== runId) {
          return {
            runId,
            status: 'error' as const,
            error: `detail lookup returned mismatched run id ${projectedDetail.id}`,
          }
        }
        return { runId, status: 'success' as const, data: projectedDetail }
      }))
      const successfulDetails = new Map(detailResults.flatMap((detail) => (
        detail.status === 'success' ? [[detail.runId, detail.data] as const] : []
      )))
      runs = runs.map((run) => {
        const detail = successfulDetails.get(run.id)
        return detail ? mergeRunDetail(run, detail) : run
      })

      const inProgressDetailIds = detailExpansionRequested
        ? inProgressIds.slice(0, MAX_RUN_DETAILS)
        : []
      const details: Array<{
        runId: string
        status: 'success' | 'error'
        data?: GiteaRunDetailProjection
        error?: string
      }> = inProgressDetailIds.map((runId) => detailResults.find((detail) => (
        detail.runId === runId
      )) ?? {
        runId,
        status: 'error',
        error: 'detail lookup exceeded the bounded priority limit',
      })

      const detailLimitExceeded = detailExpansionRequested
        && inProgressIds.length > MAX_RUN_DETAILS
      const complete = !detailLimitExceeded
        && details.every((detail) => detail.status === 'success')
      const runDetailLimitExceeded = detailCandidateIds.length > MAX_RUN_DETAILS
      const runDetailsComplete = !runDetailLimitExceeded
        && detailResults.every((detail) => detail.status === 'success')
      const projection = {
        schemaVersion: 1,
        remote,
        limit,
        inventory: {
          observed: sourceRuns.length,
          returned: runs.length,
          omittedInvalid: Math.max(0, sourceRuns.length - projectedRuns.length),
          omittedByLimit: Math.max(0, projectedRuns.length - runs.length),
        },
        runs,
        runDetails: {
          expansionRequested: runDetailExpansionRequested,
          candidates: detailCandidateIds.length,
          attempted: detailIds.length,
          detailLimit: MAX_RUN_DETAILS,
          truncated: runDetailLimitExceeded,
          complete: runDetailsComplete,
          failures: detailResults.flatMap((detail) => detail.status === 'error'
            ? [{ runId: detail.runId, error: detail.error }]
            : []),
        },
        inProgress: {
          observed: inProgressIds.length,
          detailExpansionRequested,
          expanded: inProgressDetailIds.length,
          detailLimit: MAX_RUN_DETAILS,
          truncated: detailLimitExceeded,
          complete,
          details,
        },
      }
      let limited = enforceOutputLimit(JSON.stringify(projection), {
        toolName: 'gitea.actions.runs.inspect',
        maxBytes: 40_000,
        resumeHint: 'reduce limit; run detail expansion is already bounded to five exact run ids',
      })
      if (limited.truncated) {
        const detailsWithoutJobs = details.map((detail) => detail.data
          ? {
              ...detail,
              data: {
                ...detail.data,
                jobs: [],
                jobsTruncated: detail.data.jobsObserved > 0,
              },
            }
          : detail)
        limited = enforceOutputLimit(JSON.stringify({
          ...projection,
          outputCompacted: true,
          inProgress: {
            ...projection.inProgress,
            complete: false,
            details: detailsWithoutJobs,
          },
        }), {
          toolName: 'gitea.actions.runs.inspect',
          maxBytes: 40_000,
          resumeHint: 'reduce limit',
        })
      }
      if (limited.truncated) {
        return {
          output: 'The bounded Gitea Actions projection exceeded its safe output limit; retry with a smaller limit.',
          status: 'error',
          code: 'GITEA_ACTIONS_OUTPUT_TOO_LARGE_PERMANENT',
          durationMs: Date.now() - startedAt,
          executionPosture: listResult.executionPosture,
        }
      }
      return {
        output: limited.output,
        status: 'success',
        durationMs: Date.now() - startedAt,
        executionPosture: listResult.executionPosture,
      }
    },
  }
}
