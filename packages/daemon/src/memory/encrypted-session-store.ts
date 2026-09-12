import type { ISessionStore, SessionEvent, SessionMeta, PaginatedResult, PaginationParams } from '@sepilotd/core'
import { JsonlSessionStore } from './session-store.js'
import type { EncryptionManager } from '../security/encryption.js'

type EncryptedPathSegment = string | number

interface EncryptedContent {
  content?: string
  _encrypted?: boolean
  _encryptedPaths?: EncryptedPathSegment[][]
}

const PRESERVED_TOP_LEVEL_KEYS = new Set(['type', 'id', 'timestamp'])

export class EncryptedSessionStore implements ISessionStore {
  private inner: JsonlSessionStore
  private encryption: EncryptionManager

  constructor(sessionsDir: string, encryption: EncryptionManager) {
    this.inner = new JsonlSessionStore(sessionsDir)
    this.encryption = encryption
  }

  async init(): Promise<void> {
    await this.inner.init()
  }

  async create(meta: Omit<SessionMeta, 'messageCount' | 'totalTokens' | 'totalCost'>): Promise<SessionMeta> {
    return this.inner.create(meta)
  }

  async get(id: string): Promise<SessionMeta | null> {
    return this.inner.get(id)
  }

  async list(params?: PaginationParams & { query?: string; workspaceRoot?: string }): Promise<PaginatedResult<SessionMeta>> {
    const query = normalizeSearchQuery(params?.query)
    if (query) {
      const all = await this.inner.list({
        page: 1,
        perPage: Number.MAX_SAFE_INTEGER,
        workspaceRoot: params?.workspaceRoot,
      })
      const matches = await Promise.all(
        all.items.map(async (session) => {
          if (sessionMetaMatchesQuery(session, query)) {
            return session
          }

          const events = await this.getEvents(session.id)
          return events.some((event) => searchTextMatches(event, query))
            ? session
            : null
        }),
      )
      const filtered = matches.filter((session): session is SessionMeta => session !== null)
      filtered.sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))
      const page = params?.page ?? 1
      const perPage = params?.perPage ?? 20
      const start = (page - 1) * perPage
      return {
        items: filtered.slice(start, start + perPage),
        totalCount: filtered.length,
        page,
        perPage,
        hasNextPage: start + perPage < filtered.length,
      }
    }

    return this.inner.list(params)
  }

  async delete(id: string): Promise<void> {
    return this.inner.delete(id)
  }

  async appendEvent(sessionId: string, event: SessionEvent): Promise<void> {
    return this.inner.appendEvent(sessionId, this.encryptEvent(event))
  }

  async getEvents(sessionId: string): Promise<SessionEvent[]> {
    const events = await this.inner.getEvents(sessionId)
    // Decrypt encrypted content
    return events.map((event) => this.decryptEvent(event))
  }

  async replaceEvents(sessionId: string, events: SessionEvent[]): Promise<void> {
    await this.inner.replaceEvents(sessionId, events.map((event) => this.encryptEvent(event)))
  }

  async updateMeta(
    sessionId: string,
    patch: {
      title?: string
      status?: SessionMeta['status']
      cwd?: string | null
      workspaceIsolation?: 'policy' | 'strict'
      provider?: string
      model?: string
      personaIds?: string[]
      /** Server-bound isolated persona memory identity. */
      memoryNamespace?: string
      preferPromptReact?: boolean
      starred?: boolean
      tags?: string[]
    },
  ): Promise<SessionMeta | null> {
    return this.inner.updateMeta(sessionId, patch)
  }

  private encryptEvent(event: SessionEvent): SessionEvent {
    if (!this.encryption.isEnabled()) {
      return event
    }
    const encryptedPaths: EncryptedPathSegment[][] = []
    const encrypted = this.encryptValue(event, [], encryptedPaths) as SessionEvent & EncryptedContent
    if (encryptedPaths.length > 0) {
      encrypted._encrypted = true
      encrypted._encryptedPaths = encryptedPaths
    }
    return encrypted
  }

  private decryptEvent(event: SessionEvent): SessionEvent {
    const tagged = event as SessionEvent & Partial<EncryptedContent>
    if (!this.encryption.isEnabled() || !tagged._encrypted) {
      return event
    }
    try {
      const decrypted = cloneJsonLike(tagged) as SessionEvent & Partial<EncryptedContent>
      if (Array.isArray(decrypted._encryptedPaths)) {
        for (const path of decrypted._encryptedPaths) {
          this.decryptPath(decrypted, path)
        }
      } else if ('content' in decrypted && typeof decrypted.content === 'string') {
        decrypted.content = this.encryption.decrypt(decrypted.content)
      }
      delete decrypted._encrypted
      delete decrypted._encryptedPaths
      return decrypted
    } catch {
      return event
    }
  }

  private encryptValue(
    value: unknown,
    path: EncryptedPathSegment[],
    encryptedPaths: EncryptedPathSegment[][],
  ): unknown {
    if (typeof value === 'string') {
      if (path.length === 1 && PRESERVED_TOP_LEVEL_KEYS.has(String(path[0]))) {
        return value
      }
      encryptedPaths.push([...path])
      return this.encryption.encrypt(value)
    }
    if (Array.isArray(value)) {
      return value.map((item, index) => this.encryptValue(item, [...path, index], encryptedPaths))
    }
    if (value && typeof value === 'object') {
      const out: Record<string, unknown> = {}
      for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
        out[key] = this.encryptValue(child, [...path, key], encryptedPaths)
      }
      return out
    }
    return value
  }

  private decryptPath(target: unknown, path: EncryptedPathSegment[]): void {
    if (!target || path.length === 0) return
    let cursor = target as Record<string, unknown> | unknown[]
    for (const segment of path.slice(0, -1)) {
      const next = Array.isArray(cursor)
        ? cursor[Number(segment)]
        : (cursor as Record<string, unknown>)[String(segment)]
      if (!next || typeof next !== 'object') return
      cursor = next as Record<string, unknown> | unknown[]
    }
    const last = path[path.length - 1]
    if (Array.isArray(cursor)) {
      const index = Number(last)
      if (typeof cursor[index] === 'string') {
        cursor[index] = this.encryption.decrypt(cursor[index] as string)
      }
      return
    }
    const key = String(last)
    const record = cursor as Record<string, unknown>
    if (typeof record[key] === 'string') {
      record[key] = this.encryption.decrypt(record[key] as string)
    }
  }
}

function cloneJsonLike<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T
}

function normalizeSearchQuery(query: string | undefined): string {
  return query?.trim().toLowerCase() ?? ''
}

function searchTextMatches(value: unknown, query: string): boolean {
  return stringifySearchValue(value).toLowerCase().includes(query)
}

function sessionMetaMatchesQuery(session: SessionMeta, query: string): boolean {
  return (
    searchTextMatches(session.title, query)
    || searchTextMatches(session.provider, query)
    || searchTextMatches(session.model, query)
    || searchTextMatches(session.device, query)
    || (session.tags ?? []).some((tag) => searchTextMatches(tag, query))
  )
}

function stringifySearchValue(value: unknown): string {
  if (typeof value === 'string') {
    return value
  }
  if (value == null) {
    return ''
  }
  if (Array.isArray(value)) {
    return value.map((item) => stringifySearchValue(item)).join(' ')
  }
  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value)
  }
  try {
    return JSON.stringify(value)
  } catch {
    return ''
  }
}
