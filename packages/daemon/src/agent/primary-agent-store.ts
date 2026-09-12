import { mkdir, readFile, rename, writeFile } from 'node:fs/promises'
import { dirname } from 'node:path'
import { createLogger } from '../logger.js'
import { isNodeFsError } from '../utils/fs-error.js'

const log = createLogger('primary-agent-store')

function normalizePersistedEntries(raw: unknown): Record<string, string> {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    return {}
  }

  return Object.fromEntries(
    Object.entries(raw).filter((entry): entry is [string, string] =>
      typeof entry[1] === 'string'),
  )
}

export class PrimaryAgentStore {
  private readonly byId: Map<string, string>
  private readonly filePath?: string
  private saveQueue: Promise<void> = Promise.resolve()

  constructor(filePath?: string, initialEntries: Record<string, string> = {}) {
    this.filePath = filePath
    this.byId = new Map(Object.entries(initialEntries))
  }

  static async create(filePath: string): Promise<PrimaryAgentStore> {
    await mkdir(dirname(filePath), { recursive: true })
    let text: string
    try {
      text = await readFile(filePath, 'utf-8')
    } catch (err) {
      if (!isNodeFsError(err, 'ENOENT')) {
        log.warn('primary agent store unreadable; starting empty', {
          path: filePath,
          error: err instanceof Error ? err.message : String(err),
        })
      }
      return new PrimaryAgentStore(filePath)
    }
    try {
      return new PrimaryAgentStore(filePath, normalizePersistedEntries(JSON.parse(text)))
    } catch (err) {
      // Corrupt file means every existing session loses its
      // primary-agent binding silently. Operator gets a log entry
      // so "session forgot its agent" is correlatable with the
      // parse failure.
      log.warn('primary agent store unparseable; starting empty', {
        path: filePath,
        error: err instanceof Error ? err.message : String(err),
      })
      return new PrimaryAgentStore(filePath)
    }
  }

  async set(sessionId: string, agentId: string): Promise<void> {
    this.byId.set(sessionId, agentId)
    await this.persist()
  }

  get(sessionId: string): string | undefined {
    return this.byId.get(sessionId)
  }

  async clear(sessionId: string): Promise<void> {
    if (!this.byId.delete(sessionId)) {
      return
    }
    await this.persist()
  }

  private persist(): Promise<void> {
    if (!this.filePath) {
      return Promise.resolve()
    }

    const snapshot = Object.fromEntries(this.byId)
    const persistJob = this.saveQueue
      .catch(() => {})
      .then(async () => {
        const tempPath = `${this.filePath}.tmp`
        await mkdir(dirname(this.filePath!), { recursive: true })
        await writeFile(
          tempPath,
          JSON.stringify(snapshot, null, 2),
          { encoding: 'utf-8', mode: 0o600 },
        )
        await rename(tempPath, this.filePath!)
      })

    this.saveQueue = persistJob
    return persistJob
  }
}
