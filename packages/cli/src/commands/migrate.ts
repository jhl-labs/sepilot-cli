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
