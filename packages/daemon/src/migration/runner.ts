import type { MigrationRepo } from './repo.js'
import type {
  MigrationContext,
  MigrationStepDefinition,
} from './types.js'

export interface RunMigrationInput {
  runId: string
  repo: MigrationRepo
  steps: MigrationStepDefinition[]
  ctx: MigrationContext
  signal?: AbortSignal
}

export async function runMigration(input: RunMigrationInput): Promise<void> {
  const { runId, repo, steps, ctx, signal } = input
  repo.updateRunStatus(runId, 'running')
  const order = topoSort(steps)
  const failed = new Set<string>()

  for (const step of order) {
    if (signal?.aborted) {
      repo.completeStep(runId, step.name, {
        status: 'skipped',
        copied: 0,
        skipped: 0,
        errors: [],
      })
      continue
    }
    if ((step.dependsOn ?? []).some((d) => failed.has(d))) {
      repo.completeStep(runId, step.name, {
        status: 'skipped',
        copied: 0,
        skipped: 0,
        errors: [{ path: '', error: 'prerequisite failed' }],
      })
      continue
    }
    repo.startStep(runId, step.name)
    try {
      await step.validate(ctx)
      const r = await step.execute(ctx)
      repo.completeStep(runId, step.name, {
        status: 'succeeded',
        copied: r.copied,
        skipped: r.skipped,
        errors: r.errors,
      })
      if (r.errors.length > 0) failed.add(step.name)
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      repo.completeStep(runId, step.name, {
        status: 'failed',
        copied: 0,
        skipped: 0,
        errors: [{ path: '', error: msg }],
      })
      failed.add(step.name)
    }
  }

  const finalStatus = signal?.aborted
    ? 'canceled'
    : failed.size > 0
      ? 'failed'
      : 'completed'
  repo.updateRunStatus(runId, finalStatus)
}

function topoSort(
  steps: MigrationStepDefinition[],
): MigrationStepDefinition[] {
  const byName = new Map(steps.map((s) => [s.name, s] as const))
  const visited = new Set<string>()
  const result: MigrationStepDefinition[] = []
  function visit(name: string, stack: Set<string>): void {
    if (visited.has(name)) return
    if (stack.has(name)) {
      throw new Error(`migration: circular dependency at ${name}`)
    }
    stack.add(name)
    const s = byName.get(name)
    if (!s) throw new Error(`migration: unknown step ${name}`)
    for (const d of s.dependsOn ?? []) visit(d, stack)
    stack.delete(name)
    visited.add(name)
    result.push(s)
  }
  for (const s of steps) visit(s.name, new Set())
  return result
}
