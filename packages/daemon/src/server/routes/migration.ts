import { existsSync, statSync } from 'node:fs'
import type { FastifyInstance } from 'fastify'
import type { MigrationRepo } from '../../migration/repo.js'
import { runMigration } from '../../migration/runner.js'
import { ALL_STEPS, STEP_BY_NAME } from '../../migration/steps/index.js'
import type {
  MigrationContext,
  MigrationStepDefinition,
} from '../../migration/types.js'

export interface MigrationDeps {
  repo: MigrationRepo
  selfOrigin: string
  selfToken: string | null
  targetHome: string
  cancellers: Map<string, AbortController>
}

export function registerMigrationRoutes(
  app: FastifyInstance,
  deps: MigrationDeps,
): void {
  app.post<{
    Body: {
      sourcePath: string
      steps?: string[]
      exclude?: string[]
      dryRun?: boolean
      conflict?: 'skip' | 'overwrite'
    }
  }>('/migration/run', async (req) => {
    const {
      sourcePath,
      steps,
      exclude,
      dryRun = false,
      conflict = 'skip',
    } = req.body
    if (!sourcePath || typeof sourcePath !== 'string') {
      const e = new Error('sourcePath required') as Error & {
        statusCode: number
      }
      e.statusCode = 400
      throw e
    }
    if (!existsSync(sourcePath) || !statSync(sourcePath).isDirectory()) {
      const e = new Error(
        'sourcePath does not exist or is not a directory',
      ) as Error & { statusCode: number }
      e.statusCode = 400
      throw e
    }
    const requested =
      steps && steps.length > 0 ? steps : ALL_STEPS.map((s) => s.name)
    const filtered = requested.filter((n) => !(exclude ?? []).includes(n))
    const stepDefs: MigrationStepDefinition[] = filtered
      .map((n) => STEP_BY_NAME.get(n))
      .filter((s): s is MigrationStepDefinition => Boolean(s))

    const run = deps.repo.create({ sourcePath, dryRun })
    deps.repo.insertSteps(
      run.id,
      stepDefs.map((s) => s.name),
    )
    const ctx: MigrationContext = {
      sourcePath,
      targetHome: deps.targetHome,
      dryRun,
      conflict,
      daemonOrigin: deps.selfOrigin,
      daemonToken: deps.selfToken,
    }
    const abort = new AbortController()
    deps.cancellers.set(run.id, abort)
    void runMigration({
      runId: run.id,
      repo: deps.repo,
      steps: stepDefs,
      ctx,
      signal: abort.signal,
    }).finally(() => deps.cancellers.delete(run.id))

    return {
      migrationId: run.id,
      status: 'pending' as const,
      dryRun,
      steps: stepDefs.map((s) => ({
        name: s.name,
        status: 'queued' as const,
      })),
    }
  })

  app.get<{ Params: { id: string } }>('/migration/:id', async (req) => {
    const run = deps.repo.get(req.params.id)
    if (!run) {
      const e = new Error('not found') as Error & { statusCode: number }
      e.statusCode = 404
      throw e
    }
    const stepRows = deps.repo.listSteps(req.params.id)
    return { ...run, steps: stepRows }
  })

  app.get<{ Params: { id: string } }>(
    '/migration/:id/report',
    async (req) => {
      const run = deps.repo.get(req.params.id)
      if (!run) {
        const e = new Error('not found') as Error & { statusCode: number }
        e.statusCode = 404
        throw e
      }
      const stepRows = deps.repo.listSteps(req.params.id)
      const summary = stepRows.reduce(
        (acc, s) => ({
          copied: acc.copied + s.copied,
          skipped: acc.skipped + s.skipped,
          errors: acc.errors + s.errors.length,
        }),
        { copied: 0, skipped: 0, errors: 0 },
      )
      return {
        migrationId: run.id,
        sourcePath: run.sourcePath,
        status: run.status,
        dryRun: run.dryRun,
        startedAt: run.startedAt,
        finishedAt: run.finishedAt,
        perStep: stepRows,
        summary,
      }
    },
  )

  app.delete<{ Params: { id: string } }>('/migration/:id', async (req) => {
    const ac = deps.cancellers.get(req.params.id)
    if (ac) ac.abort()
    return { ok: true }
  })
}
