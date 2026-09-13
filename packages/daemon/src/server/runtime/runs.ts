import { mkdir, readFile, readdir, rm, stat, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import type {
  ActiveSkillExecutionPolicy,
  AgentRunContract,
  Message,
  TokenUsage,
  ToolCall,
} from '@sepilotd/core'
import { createLogger } from '../../logger.js'
import { isNodeFsError } from '../../utils/fs-error.js'
import { writeFileAtomic } from '../../utils/atomic-write.js'
import type { AgentState as GraphAgentState } from '../../agent/graph/types.js'

const log = createLogger('run-checkpoints')

export type RunResumeStage = 'thinking' | 'acting' | 'observing'

export interface PendingToolExecutionSnapshot {
  toolCalls: ToolCall[]
  startIndex: number
  batchSize?: number
  currentExecutionId?: string
}

export interface SessionRunCheckpoint {
  autonomy?: import('@sepilotd/core').AutonomyLevel
  sessionId: string
  provider: string
  model: string
  mode?: string
  modeControlState?: import('@sepilotd/core').AgentContext['modeControlState']
  systemPrompt?: string
  cwd?: string
  workspaceRoot?: string
  workspaceIsolation?: 'policy' | 'strict'
  /** Caller memory scope preserved across crash/cancellation resumes. */
  scopeTags?: string[]
  /** Structured skill execution invariants preserved across crash/cancellation resumes. */
  executionSkillIds?: string[]
  /** Immutable loaded-skill runtime profile preserved across crash resumes. */
  skillToolNames?: string[]
  /** Exact tool registry surface used before the run was checkpointed. */
  toolAllowlist?: string[]
  skillExecutionPolicies?: ActiveSkillExecutionPolicy[]
  /** Preserve the per-turn explicit tool-approval guard across resumes. */
  requireToolApproval?: boolean
  messages: Message[]
  totalUsage: TokenUsage
  iteration: number
  maxIterations: number
  thinkingLevel?: string
  textDeltaMode?: 'buffered' | 'live'
  stage: RunResumeStage
  checkpointedAt: string
  pendingToolExecution?: PendingToolExecutionSnapshot
  graphState?: GraphAgentState
  runContract?: AgentRunContract
}

export type RunCheckpointClaim =
  | {
      status: 'claimed'
      checkpoint: SessionRunCheckpoint
      release: () => Promise<void>
    }
  | {
      status: 'busy'
    }
  | {
      status: 'missing'
    }
  | {
      status: 'unavailable'
      issue: RunCheckpointIssue
    }

export interface RunCheckpointIssue {
  status: 'corrupt' | 'unreadable'
  message: string
  path: string
}

export type RunCheckpointInspection =
  | {
      status: 'available'
      checkpoint: SessionRunCheckpoint
    }
  | {
      status: 'missing'
    }
  | {
      status: 'unavailable'
      issue: RunCheckpointIssue
    }

export type RunCheckpointSummary =
  | {
      status: 'available'
      sessionId: string
      checkpointedAt: string
      stage: RunResumeStage
      locked: boolean
    }
  | {
      status: 'unavailable'
      sessionId: string
      locked: boolean
      issue: RunCheckpointIssue
    }

export interface RunCheckpointPruneResult {
  deletedCheckpoints: number
  deletedClaims: number
  skippedClaimedCheckpoints: number
  unavailableCheckpoints: number
}

export class RunCheckpointStore {
  private initPromise: Promise<void> | null = null

  constructor(private readonly checkpointDir: string) {}

  async init(): Promise<void> {
    this.initPromise ??= this.initialize()
    await this.initPromise
  }

  private async initialize(): Promise<void> {
    await mkdir(this.checkpointDir, { recursive: true })
    await this.clearStaleClaims()
  }

  async save(checkpoint: SessionRunCheckpoint): Promise<void> {
    await this.init()
    await writeFileAtomic(
      this.filePath(checkpoint.sessionId),
      JSON.stringify(checkpoint, null, 2),
    )
  }

  async get(sessionId: string): Promise<SessionRunCheckpoint | null> {
    const inspection = await this.inspect(sessionId)
    return inspection.status === 'available' ? inspection.checkpoint : null
  }

  async inspect(sessionId: string): Promise<RunCheckpointInspection> {
    await this.init()
    const path = this.filePath(sessionId)
    let raw: string
    try {
      raw = await readFile(path, 'utf-8')
    } catch (err) {
      if (isNodeFsError(err, 'ENOENT')) {
        return { status: 'missing' }
      }
      log.warn('run checkpoint unreadable', {
        sessionId,
        path,
        error: err instanceof Error ? err.message : String(err),
      })
      return {
        status: 'unavailable',
        issue: {
          status: 'unreadable',
          message: 'Run checkpoint could not be read',
          path,
        },
      }
    }
    try {
      return {
        status: 'available',
        checkpoint: JSON.parse(raw) as SessionRunCheckpoint,
      }
    } catch (err) {
      // A user-facing run that vanishes after a daemon restart is
      // confusing if there's no diagnostic. Log so the operator can
      // correlate with whatever caused the partial write.
      log.warn('run checkpoint unparseable; treating as no resume available', {
        sessionId,
        path,
        error: err instanceof Error ? err.message : String(err),
      })
      return {
        status: 'unavailable',
        issue: {
          status: 'corrupt',
          message: 'Run checkpoint could not be parsed',
          path,
        },
      }
    }
  }

  async claim(sessionId: string): Promise<RunCheckpointClaim> {
    await this.init()
    const lockPath = this.lockFilePath(sessionId)
    try {
      await writeFile(
        lockPath,
        JSON.stringify({
          sessionId,
          claimedAt: new Date().toISOString(),
          pid: process.pid,
        }),
        { encoding: 'utf-8', flag: 'wx' },
      )
    } catch (err) {
      if (isNodeFsError(err, 'EEXIST')) {
        return { status: 'busy' }
      }
      throw err
    }

    let released = false
    const release = async () => {
      if (released) {
        return
      }
      released = true
      await rm(lockPath, { force: true })
    }

    const inspection = await this.inspect(sessionId)
    if (inspection.status === 'missing') {
      await release()
      return { status: 'missing' }
    }
    if (inspection.status === 'unavailable') {
      await release()
      return {
        status: 'unavailable',
        issue: inspection.issue,
      }
    }

    return {
      status: 'claimed',
      checkpoint: inspection.checkpoint,
      release,
    }
  }

  async has(sessionId: string): Promise<boolean> {
    return (await this.get(sessionId)) !== null
  }

  async delete(sessionId: string): Promise<void> {
    await this.init()
    await rm(this.filePath(sessionId), { force: true })
    await rm(this.lockFilePath(sessionId), { force: true })
  }

  async list(): Promise<RunCheckpointSummary[]> {
    await this.init()
    const entries = await this.readEntries()
    const entrySet = new Set(entries)
    const summaries: RunCheckpointSummary[] = []

    for (const entry of entries) {
      if (!entry.endsWith('.json')) {
        continue
      }
      const sessionId = entry.slice(0, -'.json'.length)
      const locked = entrySet.has(`${sessionId}.lock`)
      const inspection = await this.inspect(sessionId)
      if (inspection.status === 'available') {
        summaries.push({
          status: 'available',
          sessionId,
          checkpointedAt: inspection.checkpoint.checkpointedAt,
          stage: inspection.checkpoint.stage,
          locked,
        })
      } else if (inspection.status === 'unavailable') {
        summaries.push({
          status: 'unavailable',
          sessionId,
          locked,
          issue: inspection.issue,
        })
      }
    }

    return summaries.sort((left, right) => left.sessionId.localeCompare(right.sessionId))
  }

  async pruneOlderThan(
    maxAgeMs: number,
    now = new Date(),
  ): Promise<RunCheckpointPruneResult> {
    await this.init()
    const cutoffMs = now.getTime() - maxAgeMs
    const entries = await this.readEntries()
    const entrySet = new Set(entries)
    const result: RunCheckpointPruneResult = {
      deletedCheckpoints: 0,
      deletedClaims: 0,
      skippedClaimedCheckpoints: 0,
      unavailableCheckpoints: 0,
    }

    for (const entry of entries) {
      if (!entry.endsWith('.json')) {
        continue
      }
      const sessionId = entry.slice(0, -'.json'.length)
      if (entrySet.has(`${sessionId}.lock`)) {
        result.skippedClaimedCheckpoints += 1
        continue
      }

      const inspection = await this.inspect(sessionId)
      const timestampMs = inspection.status === 'available'
        ? Date.parse(inspection.checkpoint.checkpointedAt)
        : await this.fileModifiedAtMs(this.filePath(sessionId))
      if (inspection.status === 'unavailable') {
        result.unavailableCheckpoints += 1
      }
      if (!Number.isFinite(timestampMs) || timestampMs > cutoffMs) {
        continue
      }

      await this.delete(sessionId)
      result.deletedCheckpoints += 1
    }

    for (const entry of entries) {
      if (!entry.endsWith('.lock')) {
        continue
      }
      const sessionId = entry.slice(0, -'.lock'.length)
      if (entrySet.has(`${sessionId}.json`)) {
        continue
      }
      const timestampMs = await this.fileModifiedAtMs(this.lockFilePath(sessionId))
      if (!Number.isFinite(timestampMs) || timestampMs > cutoffMs) {
        continue
      }
      await rm(this.lockFilePath(sessionId), { force: true })
      result.deletedClaims += 1
    }

    return result
  }

  private filePath(sessionId: string): string {
    return join(this.checkpointDir, `${sessionId}.json`)
  }

  private lockFilePath(sessionId: string): string {
    return join(this.checkpointDir, `${sessionId}.lock`)
  }

  private async readEntries(): Promise<string[]> {
    try {
      return await readdir(this.checkpointDir)
    } catch (err) {
      if (isNodeFsError(err, 'ENOENT')) {
        return []
      }
      throw err
    }
  }

  private async fileModifiedAtMs(path: string): Promise<number> {
    try {
      return (await stat(path)).mtimeMs
    } catch {
      return Number.NaN
    }
  }

  private async clearStaleClaims(): Promise<void> {
    const entries = await this.readEntries()

    await Promise.all(
      entries
        .filter((entry) => entry.endsWith('.lock'))
        .map(async (entry) => {
          const path = join(this.checkpointDir, entry)
          await rm(path, { force: true })
          log.warn('cleared stale run checkpoint claim after startup', { path })
        }),
    )
  }
}
