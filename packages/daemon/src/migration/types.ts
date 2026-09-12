export type MigrationStatus =
  | 'pending'
  | 'running'
  | 'completed'
  | 'failed'
  | 'canceled'
export type StepStatus =
  | 'queued'
  | 'running'
  | 'succeeded'
  | 'failed'
  | 'skipped'

export interface MigrationRun {
  id: string
  sourcePath: string
  status: MigrationStatus
  dryRun: boolean
  startedAt: number | null
  finishedAt: number | null
  error: string | null
  createdAt: number
}

export interface MigrationStepRow {
  migrationId: string
  name: string
  status: StepStatus
  copied: number
  skipped: number
  errors: { path: string; error: string }[]
  startedAt: number | null
  finishedAt: number | null
}

export interface MigrationContext {
  sourcePath: string
  targetHome: string
  dryRun: boolean
  conflict: 'skip' | 'overwrite'
  daemonOrigin?: string
  daemonToken?: string | null
}

export interface MigrationStepDefinition {
  name: string
  dependsOn?: string[]
  validate(ctx: MigrationContext): Promise<void>
  execute(ctx: MigrationContext): Promise<{
    copied: number
    skipped: number
    errors: { path: string; error: string }[]
  }>
}
