import { createHash } from 'node:crypto'
import { mkdir, readFile, readdir, rename, rm, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { createLogger } from '../../logger.js'

const log = createLogger('channel-sessions')

export interface ChannelSessionBinding {
  key: string
  sessionId: string
  channelType: string
  channelId: string
  updatedAt: string
}

export interface ChannelSessionChannelStats {
  channelType: string
  totalBindings: number
  staleBindings: number
}

export interface ChannelSessionStats {
  totalBindings: number
  staleBindings: number
  byChannelType: ChannelSessionChannelStats[]
}

export type ChannelSessionLookup = (
  sessionId: string,
) => Promise<{ status?: string } | null>

export class ChannelSessionStore {
  constructor(private readonly sessionDir: string) {}

  async init(): Promise<void> {
    await mkdir(this.sessionDir, { recursive: true })
  }

  async get(key: string): Promise<ChannelSessionBinding | null> {
    try {
      const raw = await readFile(this.filePath(key), 'utf-8')
      return JSON.parse(raw) as ChannelSessionBinding
    } catch {
      return null
    }
  }

  async bind(binding: ChannelSessionBinding): Promise<void> {
    await this.init()
    await writeFile(
      this.filePath(binding.key),
      JSON.stringify(binding, null, 2),
      'utf-8',
    )
  }

  async delete(key: string): Promise<void> {
    await rm(this.filePath(key), { force: true })
  }

  async list(): Promise<ChannelSessionBinding[]> {
    await this.init()
    const bindings: ChannelSessionBinding[] = []
    for (const name of await readdir(this.sessionDir)) {
      // Live bindings are stored as "<sha256>.json". Backup and
      // rotated-aside copies are forensic artifacts, not active
      // "Telegram chat <-> session" pairings.
      if (!name.endsWith('.json')) continue
      const filePath = join(this.sessionDir, name)
      try {
        const raw = await readFile(filePath, 'utf-8')
        bindings.push(JSON.parse(raw) as ChannelSessionBinding)
      } catch (err) {
        // The original code silently `rm`d unreadable bindings —
        // a transient EBUSY/EAGAIN could destroy the user's
        // chat-to-session mapping forever. Rotate aside instead so
        // the operator can recover or re-pair, and the broken
        // file is preserved for forensics.
        const aside = `${filePath}.broken-${Date.now()}`
        log.warn('channel session binding unreadable; rotating aside', {
          path: filePath,
          rotated: aside,
          error: err instanceof Error ? err.message : String(err),
        })
        await rename(filePath, aside).catch((renameErr) => {
          log.warn('failed to rotate channel session binding aside', {
            path: filePath,
            error: renameErr instanceof Error ? renameErr.message : String(renameErr),
          })
        })
      }
    }
    return bindings
  }

  async getStats(
    now = new Date(),
    staleAfterMs = 30 * 24 * 60 * 60 * 1000,
  ): Promise<ChannelSessionStats> {
    const bindings = await this.list()
    const byChannelType = new Map<string, ChannelSessionChannelStats>()
    let staleBindings = 0

    for (const binding of bindings) {
      const stats = byChannelType.get(binding.channelType) ?? {
        channelType: binding.channelType,
        totalBindings: 0,
        staleBindings: 0,
      }
      stats.totalBindings++

      if (now.getTime() - new Date(binding.updatedAt).getTime() > staleAfterMs) {
        staleBindings++
        stats.staleBindings++
      }

      byChannelType.set(binding.channelType, stats)
    }

    return {
      totalBindings: bindings.length,
      staleBindings,
      byChannelType: Array.from(byChannelType.values()).sort((left, right) =>
        left.channelType.localeCompare(right.channelType),
      ),
    }
  }

  async pruneStale(
    now = new Date(),
    staleAfterMs = 30 * 24 * 60 * 60 * 1000,
  ): Promise<number> {
    const bindings = await this.list()
    let removed = 0

    for (const binding of bindings) {
      if (now.getTime() - new Date(binding.updatedAt).getTime() > staleAfterMs) {
        await this.delete(binding.key)
        removed++
      }
    }

    return removed
  }

  async pruneDetached(
    lookupSession: ChannelSessionLookup,
  ): Promise<number> {
    const bindings = await this.list()
    let removed = 0

    for (const binding of bindings) {
      const session = await lookupSession(binding.sessionId)
      if (!session || session.status === 'completed') {
        await this.delete(binding.key)
        removed++
      }
    }

    return removed
  }

  private filePath(key: string): string {
    const digest = createHash('sha256').update(key).digest('hex')
    return join(this.sessionDir, `${digest}.json`)
  }
}
