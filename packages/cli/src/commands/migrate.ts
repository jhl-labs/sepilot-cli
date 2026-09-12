import { writeFileSync } from 'node:fs'
import chalk from 'chalk'
import type { MigrationClient } from '@sepilotd/api-client'
import { DaemonClient } from '../client/http.js'
import { ensureDaemon } from '../client/ensure-daemon.js'
import { detectSource } from '../migrate/source-detect.js'
import { errorMessage } from '../utils/error-message.js'

interface MigrateFacadeStepEvent {
  name: string
  status: string
  copied: number
  skipped: number
  errors: number
}

export interface MigrateFacadeInput {
  sourcePath: string
  steps?: string[]
  exclude?: string[]
  dryRun?: boolean
  conflict?: 'skip' | 'overwrite'
  migration: MigrationClient
  pollMs?: number
  /**
   * Total elapsed-time cap for the polling loop, in milliseconds. When
   * the daemon never transitions to a terminal state (network blip,
   * crash without status update) the CLI would otherwise hang. Default:
   * 30 minutes — overridable via `SEPILOTD_MIGRATE_TIMEOUT_MS`. Tests
   * pass an explicit small value via this field instead of relying on
   * the env var. `0` disables the watchdog (used by tests with stubbed
   * pollers).
   */
  timeoutMs?: number
  onStep?: (event: MigrateFacadeStepEvent) => void
}

export interface MigrateFacadeResult {
  migrationId: string
  status: string
  summary: { copied: number; skipped: number; errors: number }
}

const TERMINAL_STATES = new Set(['completed', 'failed', 'canceled'])

const DEFAULT_MIGRATE_TIMEOUT_MS = 30 * 60 * 1000

function resolveMigrateTimeoutMs(explicit?: number): number {
  if (typeof explicit === 'number') return explicit
  const envRaw = process.env.SEPILOTD_MIGRATE_TIMEOUT_MS
  if (envRaw && envRaw.length > 0) {
    const parsed = Number.parseInt(envRaw, 10)
    if (Number.isFinite(parsed) && parsed >= 0) return parsed
  }
  return DEFAULT_MIGRATE_TIMEOUT_MS
}

/**
 * Thrown by `runMigrateFacade` when the polling loop exceeds its
 * elapsed-time cap. The verb wrapper catches this and prints a
 * resume-with-`migrate:status` hint, then exits 2. Tests assert the
 * `migrationId` so they can confirm the watchdog tripped on the right
 * run.
 */
export class MigratePollingTimeoutError extends Error {
  readonly migrationId: string
  readonly elapsedMs: number
  constructor(migrationId: string, elapsedMs: number) {
    super(
      `migration polling timed out after ${Math.round(elapsedMs / 1000)}s (jobId ${migrationId})`,
    )
    this.name = 'MigratePollingTimeoutError'
    this.migrationId = migrationId
    this.elapsedMs = elapsedMs
  }
}

/**
 * Pure facade over the daemon's `/api/v1/migration/*` routes. The CLI
 * verb is now a thin client: POST /run, poll /:id until the snapshot
 * status is terminal, then GET /report for the summary tally. The CLI
 * no longer owns the migration steps — the daemon's MigrationRunner
 * does — so this surface only translates the daemon's snapshot stream
 * into operator-visible progress rows.
 */
export async function runMigrateFacade(
  input: MigrateFacadeInput,
): Promise<MigrateFacadeResult> {
  const { migration, pollMs = 500, onStep } = input
  const timeoutMs = resolveMigrateTimeoutMs(input.timeoutMs)

  const submitted = await migration.run({
    sourcePath: input.sourcePath,
    steps: input.steps,
    exclude: input.exclude,
    dryRun: input.dryRun,
    conflict: input.conflict,
  })

  // Polling loop: emit one onStep callback per visible step in each
  // snapshot so the renderer can show progress. Break as soon as the
  // daemon transitions to a terminal status — anything else risks
  // looping forever if the daemon hangs. The watchdog (timeoutMs > 0)
  // throws `MigratePollingTimeoutError` once total elapsed exceeds the
  // cap so the CLI can surface a resume hint instead of hanging
  // indefinitely. Set timeoutMs=0 to disable (tests).
  const startedAt = Date.now()
  while (true) {
    if (pollMs > 0) {
      await new Promise((resolve) => setTimeout(resolve, pollMs))
    }
    const snapshot = await migration.get(submitted.migrationId)
    if (onStep) {
      for (const step of snapshot.steps) {
        onStep({
          name: step.name,
          status: step.status,
          copied: step.copied,
          skipped: step.skipped,
          errors: step.errors.length,
        })
      }
    }
    if (TERMINAL_STATES.has(snapshot.status)) break
    if (timeoutMs > 0 && Date.now() - startedAt >= timeoutMs) {
      throw new MigratePollingTimeoutError(
        submitted.migrationId,
        Date.now() - startedAt,
      )
    }
  }

  const report = await migration.getReport(submitted.migrationId)
  return {
    migrationId: submitted.migrationId,
    status: report.status,
    summary: report.summary,
  }
}

export interface MigrateFromDesktopFlags {
  url?: string
  source?: string
  steps?: string
  exclude?: string
  dryRun?: boolean
  conflict?: string
  reportPath?: string
}

/**
 * Operator-facing `sepilot migrate` verb. Resolves the source path
 * (explicit `--source`, otherwise platform-conventional sepilot-desktop
 * userData), then hands off to `runMigrateFacade`. Exit codes:
 *   0 — completed
 *   1 — daemon unreachable / no source detected / report write failure
 *   2 — daemon-side migration finished in a non-completed state
 */
export async function migrateFromSepilotDesktopCommand(
  flags: MigrateFromDesktopFlags,
): Promise<void> {
  const client = new DaemonClient(flags.url)
  try {
    await ensureDaemon(client, { url: flags.url })
  } catch (err) {
    console.error(chalk.red(err instanceof Error ? err.message : String(err)))
    process.exit(1)
  }

  const detected = flags.source ?? detectSource()?.root
  if (!detected) {
    console.error(
      chalk.red('No legacy sepilot-desktop source detected. Pass --source <path>.'),
    )
    process.exit(1)
  }

  const steps = flags.steps
    ?.split(',')
    .map((s) => s.trim())
    .filter(Boolean)
  const exclude = flags.exclude
    ?.split(',')
    .map((s) => s.trim())
    .filter(Boolean)
  const conflict = flags.conflict === 'overwrite' ? 'overwrite' : 'skip'

  console.log(chalk.gray(`Source: ${detected}\n`))

  let result: MigrateFacadeResult
  try {
    result = await runMigrateFacade({
      sourcePath: detected,
      steps,
      exclude,
      dryRun: Boolean(flags.dryRun),
      conflict,
      migration: client.migration,
      onStep: (event) => {
        const tag =
          event.status === 'succeeded'
            ? chalk.green('✓')
            : event.status === 'failed'
              ? chalk.red('✗')
              : event.status === 'skipped'
                ? chalk.yellow('-')
                : chalk.gray('·')
        console.log(
          `  ${tag} ${event.name.padEnd(16)} copied=${event.copied} skipped=${event.skipped} errors=${event.errors}`,
        )
      },
    })
  } catch (err) {
    if (err instanceof MigratePollingTimeoutError) {
      const seconds = Math.round(err.elapsedMs / 1000)
      // operator hint: stay non-zero (2) so scripts catch it; mirror
      // the resume affordance batch exposes — `migrate:status <id>`
      // reattaches to the daemon-side run without restarting it.
      console.error(
        chalk.red(
          `migration polling timed out after ${seconds}s (jobId ${err.migrationId}) — daemon may still be running. resume with: sepilotd migrate:status ${err.migrationId}`,
        ),
      )
      process.exit(2)
    }
    throw err
  }

  console.log(
    chalk.gray(
      `\nstatus: ${result.status} | copied=${result.summary.copied} skipped=${result.summary.skipped} errors=${result.summary.errors}`,
    ),
  )

  if (flags.reportPath) {
    try {
      const full = await client.migration.getReport(result.migrationId)
      writeFileSync(flags.reportPath, JSON.stringify(full, null, 2), 'utf-8')
      console.log(chalk.green(`report written to ${flags.reportPath}`))
    } catch (err) {
      console.error(chalk.red(`Failed to write report: ${errorMessage(err)}`))
      process.exit(1)
    }
  }

  if (result.status !== 'completed') {
    process.exit(2)
  }
}

/**
 * `sepilot migrate:status <id>` — reattach to a daemon-resident
 * migration and print its snapshot. Mirrors batch:status in shape:
 * one terse top line for the daemon-side run status, then one
 * indented row per step with the visible counters operators key on.
 */
export async function migrateStatusCommand(
  id: string,
  options: { url?: string },
): Promise<void> {
  const client = new DaemonClient(options.url)
  try {
    const snap = await client.migration.get(id)
    console.log(snap.status)
    for (const step of snap.steps) {
      console.log(
        `  ${step.name}: ${step.status} copied=${step.copied} skipped=${step.skipped} errors=${step.errors.length}`,
      )
    }
  } catch (err) {
    console.error(chalk.red(`status failed: ${errorMessage(err)}`))
    process.exit(1)
  }
}

/**
 * `sepilot migrate:cancel <id>` — DELETE /api/v1/migration/:id.
 * Echoes "canceled: <id>" so an operator scanning stdout confirms
 * the request landed without parsing the daemon snapshot.
 */
export async function migrateCancelCommand(
  id: string,
  options: { url?: string },
): Promise<void> {
  const client = new DaemonClient(options.url)
  try {
    await client.migration.cancel(id)
    console.log(chalk.gray(`canceled: ${id}`))
  } catch (err) {
    console.error(chalk.red(`cancel failed: ${errorMessage(err)}`))
    process.exit(1)
  }
}
