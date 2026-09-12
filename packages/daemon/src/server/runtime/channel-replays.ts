import { createHash } from 'node:crypto'
import { mkdir, readdir, readFile, rename, rm, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { createLogger } from '../../logger.js'

const log = createLogger('channel-replays')

export type ChannelReplayState = 'processing' | 'processed'

export interface ChannelReplayRecord {
  key: string
  channelType: string
  channelId: string
  messageId: string
  state: ChannelReplayState
  claimedAt: string
  expiresAt: string
  processedAt?: string
}

export interface ChannelReplayClaimInput {
  key: string
  channelType: string
  channelId: string
  messageId: string
  ttlMs: number
  now?: Date
}

export interface ChannelReplayClaimResult {
  key: string
  duplicate: boolean
  state?: ChannelReplayState
}

export interface ChannelReplayChannelStats {
  channelType: string
  totalRecords: number
  processingRecords: number
  processedRecords: number
  staleProcessingRecords: number
}

export interface ChannelReplayStats {
  totalRecords: number
  processingRecords: number
  processedRecords: number
  staleProcessingRecords: number
  byChannelType: ChannelReplayChannelStats[]
}

export class ChannelReplayStore {
  private lastPruneAt = 0

  constructor(private readonly replayDir: string) {}

  async init(): Promise<void> {
    await mkdir(this.replayDir, { recursive: true })
  }

  async claim(
    input: ChannelReplayClaimInput,
  ): Promise<ChannelReplayClaimResult> {
    await this.init()

    const now = input.now ?? new Date()
    await this.pruneExpiredIfNeeded(now)

    const claimedAt = now.toISOString()
    const record: ChannelReplayRecord = {
      key: input.key,
      channelType: input.channelType,
      channelId: input.channelId,
      messageId: input.messageId,
      state: 'processing',
      claimedAt,
      expiresAt: new Date(now.getTime() + input.ttlMs).toISOString(),
    }

    const filePath = this.filePath(input.key)
    try {
      await writeFile(filePath, JSON.stringify(record, null, 2), {
        encoding: 'utf-8',
        flag: 'wx',
      })
      return { key: input.key, duplicate: false }
    } catch (error) {
      if (!isFileAlreadyExistsError(error)) {
        throw error
      }
    }

    const existing = await this.get(input.key)
    if (!existing) {
      await this.release(input.key)
      return this.claim(input)
    }

    if (new Date(existing.expiresAt).getTime() <= now.getTime()) {
      await this.release(input.key)
      return this.claim(input)
    }

    return {
      key: input.key,
      duplicate: true,
      state: existing.state,
    }
  }

  async get(key: string): Promise<ChannelReplayRecord | null> {
    try {
      const raw = await readFile(this.filePath(key), 'utf-8')
      return JSON.parse(raw) as ChannelReplayRecord
    } catch {
      return null
    }
  }

  async markProcessed(
    key: string,
    ttlMs: number,
    now = new Date(),
  ): Promise<void> {
    const existing = await this.get(key)
    if (!existing) {
      return
    }

    const updated: ChannelReplayRecord = {
      ...existing,
      state: 'processed',
      processedAt: now.toISOString(),
      expiresAt: new Date(now.getTime() + ttlMs).toISOString(),
    }
    await writeFile(
      this.filePath(key),
      JSON.stringify(updated, null, 2),
      'utf-8',
    )
  }

  async release(key: string): Promise<void> {
    await rm(this.filePath(key), { force: true })
  }

  async list(now = new Date()): Promise<ChannelReplayRecord[]> {
    await this.pruneExpired(now)

    const records: ChannelReplayRecord[] = []
    for (const name of await readdir(this.replayDir)) {
      // Skip rotated-aside corrupted records — they are forensic
      // copies, not live dedupe state.
      if (name.includes('.broken-')) continue
      const filePath = join(this.replayDir, name)
      try {
        const raw = await readFile(filePath, 'utf-8')
        records.push(JSON.parse(raw) as ChannelReplayRecord)
      } catch (err) {
        // A bad file here is dedupe state (channel message
        // already-processed marker). Silently `rm` would
        // resurrect the message and let the channel handler
        // process it twice. Rotate aside instead so the operator
        // sees the corruption and the dedupe key stays out of
        // service until they investigate.
        const aside = `${filePath}.broken-${Date.now()}`
        log.warn('replay record unreadable; rotating aside', {
          path: filePath,
          rotated: aside,
          error: err instanceof Error ? err.message : String(err),
        })
        await rename(filePath, aside).catch((renameErr) => {
          log.warn('failed to rotate replay record aside', {
            path: filePath,
            error: renameErr instanceof Error ? renameErr.message : String(renameErr),
          })
        })
      }
    }

    return records
  }

  async getStats(
    now = new Date(),
    staleAfterMs = 2 * 60 * 1000,
  ): Promise<ChannelReplayStats> {
    const records = await this.list(now)
    const byChannelType = new Map<string, ChannelReplayChannelStats>()

    let processingRecords = 0
    let processedRecords = 0
    let staleProcessingRecords = 0

    for (const record of records) {
      const stats = byChannelType.get(record.channelType) ?? {
        channelType: record.channelType,
        totalRecords: 0,
        processingRecords: 0,
        processedRecords: 0,
        staleProcessingRecords: 0,
      }
      stats.totalRecords++

      if (record.state === 'processing') {
        processingRecords++
        stats.processingRecords++
        if (
          now.getTime() - new Date(record.claimedAt).getTime() > staleAfterMs
        ) {
          staleProcessingRecords++
          stats.staleProcessingRecords++
        }
      } else {
        processedRecords++
        stats.processedRecords++
      }

      byChannelType.set(record.channelType, stats)
    }

    return {
      totalRecords: records.length,
      processingRecords,
      processedRecords,
      staleProcessingRecords,
      byChannelType: Array.from(byChannelType.values()).sort((left, right) =>
        left.channelType.localeCompare(right.channelType),
      ),
    }
  }

  async pruneExpired(now = new Date()): Promise<number> {
    await this.init()

    let removed = 0
    for (const name of await readdir(this.replayDir)) {
      if (name.includes('.broken-')) continue
      const filePath = join(this.replayDir, name)
      try {
        const raw = await readFile(filePath, 'utf-8')
        const record = JSON.parse(raw) as ChannelReplayRecord
        if (new Date(record.expiresAt).getTime() <= now.getTime()) {
          await rm(filePath, { force: true })
          removed++
        }
      } catch (err) {
        // Same rationale as list(): unreadable file is dedupe
        // state, not garbage. Rotate aside instead of deleting
        // so the channel can't re-deliver a previously-processed
        // message just because the marker became unreadable.
        const aside = `${filePath}.broken-${Date.now()}`
        log.warn('replay record unreadable during prune; rotating aside', {
          path: filePath,
          rotated: aside,
          error: err instanceof Error ? err.message : String(err),
        })
        await rename(filePath, aside).catch((renameErr) => {
          log.warn('failed to rotate replay record aside', {
            path: filePath,
            error: renameErr instanceof Error ? renameErr.message : String(renameErr),
          })
        })
      }
    }

    this.lastPruneAt = now.getTime()
    return removed
  }

  private async pruneExpiredIfNeeded(now: Date): Promise<void> {
    if (now.getTime() - this.lastPruneAt < 60_000) {
      return
    }
    await this.pruneExpired(now)
  }

  private filePath(key: string): string {
    const digest = createHash('sha256').update(key).digest('hex')
    return join(this.replayDir, `${digest}.json`)
  }
}

function isFileAlreadyExistsError(error: unknown): boolean {
  return typeof error === 'object'
    && error !== null
    && 'code' in error
    && error.code === 'EEXIST'
}
