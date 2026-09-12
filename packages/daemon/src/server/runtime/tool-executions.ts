import { mkdir, readFile, readdir, rm, stat } from 'node:fs/promises'
import { join } from 'node:path'
import type { ToolExecutionPosture, ToolResultMetadata } from '@sepilotd/core'
import { createLogger } from '../../logger.js'
import { isNodeFsError } from '../../utils/fs-error.js'
import { writeFileAtomic } from '../../utils/atomic-write.js'

const log = createLogger('tool-executions')

export interface ToolExecutionResultSnapshot {
  output: string
  status: 'success' | 'error'
  durationMs: number
  recovery?: 'journal' | 'probe'
  executionPosture?: ToolExecutionPosture
  metadata?: ToolResultMetadata
}

export interface ToolExecutionRecord {
  executionId: string
  sessionId: string
  toolCallId: string
  tool: string
  input: Record<string, unknown>
  startedAt: string
  completedAt?: string
  status: 'running' | 'completed'
  result?: ToolExecutionResultSnapshot
}

export interface ToolExecutionPruneResult {
  deletedRecords: number
  skippedRunningRecords: number
  unavailableRecords: number
}

export type ToolExecutionSummary =
  | {
      status: 'running' | 'completed'
      sessionId: string
      executionId: string
      startedAt: string
      completedAt?: string
    }
  | {
      status: 'unavailable'
      sessionId: string
    }

export class ToolExecutionStore {
  constructor(private readonly executionDir: string) {}

  async init(): Promise<void> {
    await mkdir(this.executionDir, { recursive: true })
  }

  async save(record: ToolExecutionRecord): Promise<void> {
    await this.init()
    await writeFileAtomic(
      this.filePath(record.sessionId),
      JSON.stringify(record, null, 2),
    )
  }

  async get(sessionId: string): Promise<ToolExecutionRecord | null> {
    const path = this.filePath(sessionId)
    let raw: string
    try {
      raw = await readFile(path, 'utf-8')
    } catch (err) {
      if (isNodeFsError(err, 'ENOENT')) {
        return null
      }
      log.warn('tool execution record unreadable', {
        sessionId,
        path,
        error: err instanceof Error ? err.message : String(err),
      })
      return null
    }
    try {
      return JSON.parse(raw) as ToolExecutionRecord
    } catch (err) {
      log.warn('tool execution record unparseable; clearActive will be a no-op', {
        sessionId,
        path,
        error: err instanceof Error ? err.message : String(err),
      })
      return null
    }
  }

  async clearActive(sessionId: string): Promise<void> {
    const record = await this.get(sessionId)
    if (!record || record.status === 'completed') {
      return
    }

    await this.delete(sessionId)
  }

  async delete(sessionId: string): Promise<void> {
    await rm(this.filePath(sessionId), { force: true })
  }

  async list(): Promise<ToolExecutionSummary[]> {
    await this.init()
    const summaries: ToolExecutionSummary[] = []
    for (const entry of await this.readEntries()) {
      if (!entry.endsWith('.json')) {
        continue
      }
      const sessionId = entry.slice(0, -'.json'.length)
      const record = await this.get(sessionId)
      summaries.push(record
        ? {
            status: record.status,
            sessionId,
            executionId: record.executionId,
            startedAt: record.startedAt,
            completedAt: record.completedAt,
          }
        : {
            status: 'unavailable',
            sessionId,
          })
    }
    return summaries.sort((left, right) => left.sessionId.localeCompare(right.sessionId))
  }

  async pruneOlderThan(
    maxAgeMs: number,
    now = new Date(),
    options: { includeRunning?: boolean } = {},
  ): Promise<ToolExecutionPruneResult> {
    await this.init()
    const cutoffMs = now.getTime() - maxAgeMs
    const result: ToolExecutionPruneResult = {
      deletedRecords: 0,
      skippedRunningRecords: 0,
      unavailableRecords: 0,
    }

    for (const entry of await this.readEntries()) {
      if (!entry.endsWith('.json')) {
        continue
      }
      const sessionId = entry.slice(0, -'.json'.length)
      const record = await this.get(sessionId)
      if (record?.status === 'running' && options.includeRunning !== true) {
        result.skippedRunningRecords += 1
        continue
      }
      if (!record) {
        result.unavailableRecords += 1
      }
      const timestampMs = record
        ? Date.parse(record.completedAt ?? record.startedAt)
        : await this.fileModifiedAtMs(this.filePath(sessionId))
      if (!Number.isFinite(timestampMs) || timestampMs > cutoffMs) {
        continue
      }

      await this.delete(sessionId)
      result.deletedRecords += 1
    }

    return result
  }

  private filePath(sessionId: string): string {
    return join(this.executionDir, `${sessionId}.json`)
  }

  private async readEntries(): Promise<string[]> {
    try {
      return await readdir(this.executionDir)
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
}
