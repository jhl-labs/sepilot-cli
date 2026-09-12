import { createHash } from 'node:crypto'
import { mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises'
import { join } from 'node:path'

export type ChannelOriginKind = 'approval' | 'question'

export interface ChannelOriginRecord {
  kind: ChannelOriginKind
  id: string
  sessionId?: string
  channelType: string
  channelId: string
  senderId: string
  requestedAt: number
  sequence?: number
  toolName?: string
}

export class ChannelOriginStore {
  constructor(private readonly dir: string) {}

  async init(): Promise<void> {
    await mkdir(this.dir, { recursive: true })
  }

  async upsert(record: ChannelOriginRecord): Promise<void> {
    await this.init()
    await writeFile(
      this.filePath(record.kind, record.id),
      JSON.stringify(record, null, 2),
      'utf-8',
    )
  }

  async delete(kind: ChannelOriginKind, id: string): Promise<void> {
    await rm(this.filePath(kind, id), { force: true })
  }

  async list(): Promise<ChannelOriginRecord[]> {
    await this.init()
    const records: ChannelOriginRecord[] = []
    for (const name of await readdir(this.dir)) {
      if (!name.endsWith('.json')) continue
      try {
        const raw = await readFile(join(this.dir, name), 'utf-8')
        const parsed = JSON.parse(raw) as ChannelOriginRecord
        if (
          (parsed.kind === 'approval' || parsed.kind === 'question')
          && typeof parsed.id === 'string'
          && typeof parsed.channelType === 'string'
          && typeof parsed.channelId === 'string'
          && typeof parsed.senderId === 'string'
          && typeof parsed.requestedAt === 'number'
        ) {
          records.push(parsed)
        }
      } catch {
        // Ignore unreadable forensic leftovers; the in-memory origin guard
        // remains fail-closed when a record cannot be restored.
      }
    }
    return records
  }

  async pruneOlderThan(maxAgeMs: number, now = Date.now()): Promise<number> {
    const records = await this.list()
    let removed = 0
    for (const record of records) {
      if (now - record.requestedAt > maxAgeMs) {
        await this.delete(record.kind, record.id)
        removed += 1
      }
    }
    return removed
  }

  private filePath(kind: ChannelOriginKind, id: string): string {
    const digest = createHash('sha256').update(`${kind}:${id}`).digest('hex')
    return join(this.dir, `${digest}.json`)
  }
}
