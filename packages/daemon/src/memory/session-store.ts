import { readFile, unlink, mkdir, rename, readdir, open } from 'node:fs/promises'
import { join, resolve, sep } from 'node:path'
import type {
  ISessionStore,
  SessionEvent,
  SessionMeta,
  PaginatedResult,
  PaginationParams,
  ToolCallEvent,
  ToolResultEvent,
} from '@sepilotd/core'
import { throwIfAborted } from '../abort.js'
import { createLogger } from '../logger.js'
import { isNodeFsError } from '../utils/fs-error.js'
import { assertSafeId } from '../utils/safe-id.js'
import { secureFileAsync } from '../utils/secure-file.js'
import { writeFileAtomic } from '../utils/atomic-write.js'
import { retryTransientFsOperation } from '../utils/fs-retry.js'

const log = createLogger('session-store')

const SESSION_INDEX_VERSION = 2
const MAX_SESSION_PERSONAS = 6

function normalizeSessionPersonaIds(personaIds: readonly string[]): string[] {
  const normalized: string[] = []
  const seen = new Set<string>()
  for (const value of personaIds) {
    const id = value.trim()
    if (!id || seen.has(id)) continue
    seen.add(id)
    normalized.push(id)
    if (normalized.length === MAX_SESSION_PERSONAS) break
  }
  return normalized
}

interface IndexedSessionMeta extends SessionMeta {
  searchText?: string
}

interface SessionIndex {
  version: number
  sessions: IndexedSessionMeta[]
}

function buildSessionMetaFromJournalEvents(
  sessionId: string,
  events: SessionEvent[],
): IndexedSessionMeta {
  const firstEvent = events[0] as (typeof events)[number] & {
    content?: string
    metadata?: { provider?: string; model?: string; device?: string }
  }
  const lastEvent = events[events.length - 1]!
  const msgCount = events.filter(
    (e) => e.type === 'user_message' || e.type === 'assistant_message',
  ).length
  const meta: IndexedSessionMeta = {
    id: sessionId,
    title: firstEvent.content?.slice(0, 50) ?? sessionId,
    createdAt: firstEvent.timestamp,
    updatedAt: lastEvent.timestamp,
    provider: firstEvent.metadata?.provider ?? 'unknown',
    model: firstEvent.metadata?.model ?? 'unknown',
    device: firstEvent.metadata?.device ?? 'unknown',
    status: lastEvent.type === 'session_end' ? 'completed' : 'active',
    messageCount: msgCount,
    totalTokens: { input: 0, output: 0 },
    totalCost: 0,
    tags: [],
  }
  meta.searchText = buildSessionSearchText(meta, events)
  return meta
}

type SessionMetaPatch = {
  title?: string
  status?: SessionMeta['status']
  cwd?: string | null
  workspaceIsolation?: 'policy' | 'strict'
  provider?: string
  model?: string
  personaIds?: string[]
  memoryNamespace?: string
  preferPromptReact?: boolean
  starred?: boolean
  tags?: string[]
}

export class JsonlSessionStore implements ISessionStore {
  private sessionsDir: string
  private index: SessionIndex = { version: SESSION_INDEX_VERSION, sessions: [] }
  private indexWriteQueue: Promise<void> = Promise.resolve()
  private readonly sessionWriteQueues = new Map<string, Promise<void>>()

  constructor(sessionsDir: string) {
    this.sessionsDir = sessionsDir
  }

  private withSessionWrite<T>(sessionId: string, operation: () => Promise<T>): Promise<T> {
    const previous = this.sessionWriteQueues.get(sessionId) ?? Promise.resolve()
    const current = previous.catch(() => undefined).then(operation)
    const tail = current.then(
      () => undefined,
      () => undefined,
    )
    this.sessionWriteQueues.set(sessionId, tail)

    return current.finally(() => {
      if (this.sessionWriteQueues.get(sessionId) === tail) {
        this.sessionWriteQueues.delete(sessionId)
      }
    })
  }

  private withIndexWrite<T>(operation: () => Promise<T>): Promise<T> {
    const current = this.indexWriteQueue.catch(() => undefined).then(operation)
    this.indexWriteQueue = current.then(
      () => undefined,
      () => undefined,
    )
    return current
  }

  private sessionFile(sessionId: string): string {
    assertSafeId(sessionId, 'sessionId')
    const root = resolve(this.sessionsDir)
    const file = resolve(join(root, `${sessionId}.jsonl`))
    const prefix = root.endsWith(sep) ? root : `${root}${sep}`
    if (!file.startsWith(prefix)) {
      throw new Error('invalid sessionId')
    }
    return file
  }

  private assertSessionExists(sessionId: string): void {
    if (!this.index.sessions.some((session) => session.id === sessionId)) {
      throw new Error(`Session not found: ${sessionId}`)
    }
  }

  async init(): Promise<void> {
    await mkdir(this.sessionsDir, { recursive: true })
    const indexPath = join(this.sessionsDir, 'index.json')
    let data: string
    try {
      data = await retryTransientFsOperation(() => readFile(indexPath, 'utf-8'))
    } catch (err) {
      if (isNodeFsError(err, 'ENOENT')) {
        this.index = { version: SESSION_INDEX_VERSION, sessions: [] }
        await this.rebuildIndex()
        return
      }
      log.error('failed to read session index — refusing to silently rebuild and lose meta', {
        path: indexPath,
        error: err instanceof Error ? err.message : String(err),
      })
      throw err
    }

    try {
      this.index = parseSessionIndex(data)
      await this.ensureSearchIndex()
    } catch (err) {
      const aside = `${indexPath}.broken-${Date.now()}`
      log.error('session index unparseable; rotating aside before rebuild', {
        path: indexPath,
        rotated: aside,
        error: err instanceof Error ? err.message : String(err),
      })
      try {
        await retryTransientFsOperation(() => rename(indexPath, aside))
      } catch (renameErr) {
        log.warn('failed to rotate broken session index aside', {
          path: indexPath,
          error: renameErr instanceof Error ? renameErr.message : String(renameErr),
        })
      }
      this.index = { version: SESSION_INDEX_VERSION, sessions: [] }
      await this.rebuildIndex()
    }
  }

  private async rebuildIndex(): Promise<void> {
    let files: string[]
    try {
      files = await retryTransientFsOperation(() => readdir(this.sessionsDir))
    } catch (err) {
      log.error('failed to list session JSONL files during rebuild', {
        sessionsDir: this.sessionsDir,
        error: err instanceof Error ? err.message : String(err),
      })
      throw err
    }

    for (const file of files) {
      if (!file.endsWith('.jsonl')) continue
      const sessionId = file.replace('.jsonl', '')
      let events: SessionEvent[]
      try {
        events = await this.getEvents(sessionId)
      } catch (err) {
        log.error('failed to read session during index rebuild', {
          sessionId,
          error: err instanceof Error ? err.message : String(err),
        })
        throw err
      }
      if (events.length === 0) continue
      this.index.sessions.push(buildSessionMetaFromJournalEvents(sessionId, events))
    }
    await this.saveIndex()
  }

  private async recoverIndexEntryFromJournal(sessionId: string): Promise<boolean> {
    const events = await this.getEvents(sessionId)
    if (events.length === 0) return false
    log.warn('session journal exists but index entry is missing — recovering index entry', {
      sessionId,
      events: events.length,
    })
    const meta = buildSessionMetaFromJournalEvents(sessionId, events)
    await this.withIndexWrite(async () => {
      if (this.index.sessions.some((session) => session.id === sessionId)) return
      this.index.sessions.push(meta)
      await this._flushIndex()
    })
    return true
  }

  private async ensureSessionIndexed(sessionId: string): Promise<void> {
    if (this.index.sessions.some((session) => session.id === sessionId)) return
    if (await this.recoverIndexEntryFromJournal(sessionId)) return
    this.assertSessionExists(sessionId)
  }

  async create(
    meta: Omit<SessionMeta, 'messageCount' | 'totalTokens' | 'totalCost'>,
  ): Promise<SessionMeta> {
    return this.withSessionWrite(meta.id, () => this.createUnlocked(meta))
  }

  private async createUnlocked(
    meta: Omit<SessionMeta, 'messageCount' | 'totalTokens' | 'totalCost'>,
  ): Promise<SessionMeta> {
    const session: IndexedSessionMeta = {
      ...meta,
      ...(meta.personaIds !== undefined
        ? { personaIds: normalizeSessionPersonaIds(meta.personaIds) }
        : {}),
      messageCount: 0,
      totalTokens: { input: 0, output: 0 },
      totalCost: 0,
    }
    session.searchText = buildSessionSearchText(session, [])
    return this.withIndexWrite(async () => {
      this.index.sessions.push(session)
      try {
        await this._flushIndex()
      } catch (error) {
        this.index.sessions = this.index.sessions.filter((candidate) => candidate !== session)
        throw error
      }
      return toSessionMeta(session)
    })
  }

  async get(id: string): Promise<SessionMeta | null> {
    const session = this.index.sessions.find((s) => s.id === id)
    return session ? toSessionMeta(session) : null
  }

  async list(
    params?: PaginationParams & { query?: string; workspaceRoot?: string },
  ): Promise<PaginatedResult<SessionMeta>> {
    let filtered = [...this.index.sessions]
    if (params?.workspaceRoot) {
      filtered = filtered.filter((session) => session.cwd === params.workspaceRoot)
    }
    const query = normalizeSearchQuery(params?.query)
    if (query) {
      let populatedLegacySearchText = false
      const matches = await Promise.all(
        filtered.map(async (session) => {
          if (sessionSearchIndexMatchesQuery(session, query)) {
            return session
          }

          if (typeof session.searchText === 'string') {
            return null
          }

          const events = await this.getEvents(session.id)
          session.searchText = buildSessionSearchText(session, events)
          populatedLegacySearchText = true
          return sessionSearchIndexMatchesQuery(session, query) ? session : null
        }),
      )
      filtered = matches.filter((session): session is IndexedSessionMeta => session !== null)
      if (populatedLegacySearchText) {
        await this.saveIndex()
      }
    }
    filtered.sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))
    const page = params?.page ?? 1
    const perPage = params?.perPage ?? 20
    const start = (page - 1) * perPage
    const items = filtered.slice(start, start + perPage)
    return {
      items: items.map(toSessionMeta),
      totalCount: filtered.length,
      page,
      perPage,
      hasNextPage: start + perPage < filtered.length,
    }
  }

  async delete(id: string): Promise<void> {
    return this.withSessionWrite(id, () => this.deleteUnlocked(id))
  }

  private async deleteUnlocked(id: string): Promise<void> {
    let removed: IndexedSessionMeta[] = []
    await this.withIndexWrite(async () => {
      const previous = this.index.sessions
      removed = previous.filter((session) => session.id === id)
      if (removed.length === 0) return

      this.index.sessions = previous.filter((session) => session.id !== id)
      try {
        await this._flushIndex()
      } catch (error) {
        this.index.sessions = previous
        throw error
      }
    })

    try {
      await retryTransientFsOperation(() => unlink(this.sessionFile(id)))
    } catch (error) {
      if (isNodeFsError(error, 'ENOENT')) return

      if (removed.length > 0) {
        try {
          await this.withIndexWrite(async () => {
            this.index.sessions = [
              ...this.index.sessions.filter((session) => session.id !== id),
              ...removed,
            ]
            await this._flushIndex()
          })
        } catch (rollbackError) {
          log.error('failed to restore session index after JSONL delete failed', {
            sessionId: id,
            deleteError: error instanceof Error ? error.message : String(error),
            rollbackError:
              rollbackError instanceof Error ? rollbackError.message : String(rollbackError),
          })
          throw new AggregateError(
            [error, rollbackError],
            'Failed to delete session and restore its index entry',
          )
        }
      }
      throw error
    }
  }

  async updateMeta(sessionId: string, patch: SessionMetaPatch): Promise<SessionMeta | null> {
    return this.withSessionWrite(sessionId, () => this.updateMetaUnlocked(sessionId, patch))
  }

  async updateTitleIfUnchanged(
    sessionId: string,
    title: string,
    expected: Pick<SessionMeta, 'title' | 'createdAt'>,
  ): Promise<SessionMeta | null> {
    return this.withSessionWrite(sessionId, () => this.updateMetaUnlocked(sessionId, { title }, expected))
  }

  private async updateMetaUnlocked(
    sessionId: string,
    patch: SessionMetaPatch,
    expectedTitle?: Pick<SessionMeta, 'title' | 'createdAt'>,
  ): Promise<SessionMeta | null> {
    const events = await this.getEvents(sessionId)
    return this.withIndexWrite(async () => {
      const index = this.index.sessions.findIndex((session) => session.id === sessionId)
      if (index < 0) return null

      const previous = this.index.sessions[index]
      if (expectedTitle && (previous.title !== expectedTitle.title || previous.createdAt !== expectedTitle.createdAt)) {
        return null
      }
      const meta: IndexedSessionMeta = { ...previous }
      if (typeof patch.title === 'string') {
        const trimmed = patch.title.trim()
        if (trimmed.length > 0) {
          meta.title = trimmed.slice(0, 200)
        }
      }
      if (
        patch.status === 'active' ||
        patch.status === 'completed' ||
        patch.status === 'abandoned'
      ) {
        meta.status = patch.status
      }
      if (patch.cwd === null) {
        delete meta.cwd
      } else if (typeof patch.cwd === 'string') {
        const trimmed = patch.cwd.trim()
        if (trimmed.length > 0) {
          meta.cwd = trimmed
        }
      }
      if (patch.workspaceIsolation !== undefined) {
        meta.workspaceIsolation = patch.workspaceIsolation
      }
      if (typeof patch.provider === 'string') {
        const trimmed = patch.provider.trim()
        if (trimmed.length > 0) {
          meta.provider = trimmed
        }
      }
      if (typeof patch.model === 'string') {
        const trimmed = patch.model.trim()
        if (trimmed.length > 0) {
          meta.model = trimmed
        }
      }
      if (patch.memoryNamespace !== undefined) {
        if ((meta.memoryNamespace && meta.memoryNamespace !== patch.memoryNamespace) || (!meta.memoryNamespace && meta.messageCount > 0)) throw new Error('Cannot change conversation memory space')
        meta.memoryNamespace = patch.memoryNamespace
      }
      if (patch.personaIds !== undefined) {
        meta.personaIds = normalizeSessionPersonaIds(patch.personaIds)
      }
      if (typeof patch.preferPromptReact === 'boolean') {
        meta.preferPromptReact = patch.preferPromptReact
      }
      if (typeof patch.starred === 'boolean') {
        meta.starred = patch.starred
      }
      if (Array.isArray(patch.tags)) {
        meta.tags = normalizeSessionTags(patch.tags)
      }
      meta.searchText = buildSessionSearchText(meta, events)
      this.index.sessions[index] = meta
      try {
        await this._flushIndex()
      } catch (error) {
        this.index.sessions[index] = previous
        throw error
      }
      return toSessionMeta(meta)
    })
  }

  async appendEvent(sessionId: string, event: SessionEvent): Promise<void> {
    return this.withSessionWrite(sessionId, () => this.appendEventUnlocked(sessionId, event))
  }

  private async appendEventUnlocked(sessionId: string, event: SessionEvent): Promise<void> {
    await this.ensureSessionIndexed(sessionId)

    const line = JSON.stringify(event) + '\n'
    const path = this.sessionFile(sessionId)
    try {
      const handle = await retryTransientFsOperation(() => open(path, 'a', 0o600))
      try {
        await handle.writeFile(line, 'utf-8')
      } finally {
        await handle.close()
      }
      await secureFileAsync(path)
    } catch (error) {
      log.error('failed to append session event', {
        sessionId,
        ...fileSystemErrorContext(error),
      })
      throw error
    }

    await this.withIndexWrite(async () => {
      const index = this.index.sessions.findIndex((session) => session.id === sessionId)
      if (index < 0) return

      const meta: IndexedSessionMeta = { ...this.index.sessions[index] }
      meta.updatedAt = event.timestamp
      if (event.type === 'user_message' || event.type === 'assistant_message') {
        meta.messageCount++
      }
      if (event.type === 'session_end') {
        meta.status = 'completed'
        meta.totalTokens = event.totalTokens
        meta.totalCost = event.totalCost
      }
      meta.searchText = appendSessionSearchText(
        meta.searchText ?? buildSessionSearchText(meta, []),
        event,
      )
      this.index.sessions[index] = meta
      await this._flushIndex()
    })
  }

  async getToolResult(
    sessionId: string,
    toolCallId: string,
    options: { signal?: AbortSignal } = {},
    decode: (event: SessionEvent) => SessionEvent = (event) => event,
  ): Promise<{ call: ToolCallEvent; result: ToolResultEvent } | null> {
    throwIfAborted(options.signal)
    if (!toolCallId || toolCallId.length > 128) return null
    const handle = await open(this.sessionFile(sessionId), 'r').catch((error: unknown) => {
      if (isNodeFsError(error, 'ENOENT')) return null
      throw error
    })
    if (!handle) return null
    try {
      const { size } = await handle.stat()
      if (!size || size > 32 * 1024 * 1024) return null
      let call: ToolCallEvent | undefined
      let result: ToolResultEvent | undefined
      let pending = ''
      const consume = (line: string): boolean => {
        if (!line.trim()) return true
        if (line.length > 2 * 1024 * 1024) return false
        let event: SessionEvent
        try {
          const raw = JSON.parse(line) as SessionEvent
          if (!raw || typeof raw !== 'object') return false
          if (raw.type !== 'tool_result' && !(raw.type === 'tool_call' && raw.id === toolCallId)) return true
          event = decode(raw)
          if ('_encrypted' in event) return false
        } catch { return false }
        if (event.type === 'tool_call' && event.id === toolCallId) {
          if (call || result || typeof event.tool !== 'string' || event.tool.length > 256) return false
          call = event
        }
        if (event.type === 'tool_result' && event.toolCallId === toolCallId) {
          if (!call || result || typeof event.output !== 'string'
            || !['success', 'error', 'timeout', 'cancelled'].includes(event.status)
            || typeof event.timestamp !== 'string' || event.timestamp.length > 64) return false
          result = event
        }
        return true
      }
      const stream = handle.createReadStream({
        start: 0, end: size - 1, encoding: 'utf8', highWaterMark: 64 * 1024,
        autoClose: false, signal: options.signal,
      })
      try {
        for await (const chunk of stream) {
          throwIfAborted(options.signal)
          pending += String(chunk)
          let newline: number
          while ((newline = pending.indexOf('\n')) >= 0) {
            if (!consume(pending.slice(0, newline))) return null
            pending = pending.slice(newline + 1)
          }
          if (pending.length > 2 * 1024 * 1024) return null
        }
        if (pending && !consume(pending)) return null
      } finally {
        stream.destroy()
      }
      throwIfAborted(options.signal)
      return call && result ? { call, result } : null
    } finally {
      await handle.close()
    }
  }

  async getEvents(sessionId: string): Promise<SessionEvent[]> {
    try {
      const path = this.sessionFile(sessionId)
      const data = await retryTransientFsOperation(() => readFile(path, 'utf-8'))
      const events: SessionEvent[] = []
      let tornLines = 0
      for (const line of data.trim().split('\n')) {
        if (!line) continue
        try {
          events.push(JSON.parse(line) as SessionEvent)
        } catch {
          tornLines += 1
        }
      }
      if (tornLines > 0) {
        log.warn(
          'dropped unparseable session event line(s) — session jsonl may be torn/truncated',
          {
            sessionId,
            tornLines,
            totalLines: events.length + tornLines,
          },
        )
      }
      return events
    } catch (error) {
      if (isNodeFsError(error, 'ENOENT')) return []
      log.warn('failed to read session events', {
        sessionId,
        error: error instanceof Error ? error.message : String(error),
      })
      throw error
    }
  }

  async replaceEvents(sessionId: string, events: SessionEvent[]): Promise<void> {
    return this.withSessionWrite(sessionId, () => this.replaceEventsUnlocked(sessionId, events))
  }

  private async replaceEventsUnlocked(sessionId: string, events: SessionEvent[]): Promise<void> {
    await this.ensureSessionIndexed(sessionId)

    const path = this.sessionFile(sessionId)
    const body = events.map((event) => JSON.stringify(event)).join('\n')
    const trailing = events.length > 0 ? '\n' : ''
    try {
      await writeFileAtomic(path, body + trailing)
    } catch (error) {
      log.error('failed to replace session journal', {
        sessionId,
        ...fileSystemErrorContext(error),
      })
      throw error
    }
    await this.withIndexWrite(async () => {
      const index = this.index.sessions.findIndex((session) => session.id === sessionId)
      if (index < 0) return

      const meta: IndexedSessionMeta = { ...this.index.sessions[index] }
      meta.messageCount = events.filter(
        (event) => event.type === 'user_message' || event.type === 'assistant_message',
      ).length
      const last = events.at(-1)
      if (last) meta.updatedAt = last.timestamp
      meta.searchText = buildSessionSearchText(meta, events)
      this.index.sessions[index] = meta
      await this._flushIndex()
    })
  }

  private async ensureSearchIndex(): Promise<void> {
    let changed = this.index.version !== SESSION_INDEX_VERSION
    this.index.version = SESSION_INDEX_VERSION
    for (const session of this.index.sessions) {
      if (typeof session.searchText !== 'string') {
        session.searchText = buildSessionSearchText(session, await this.getEvents(session.id))
        changed = true
      }
    }
    if (changed) {
      await this.saveIndex()
    }
  }

  private saveIndex(): Promise<void> {
    return this.withIndexWrite(() => this._flushIndex())
  }

  private async _flushIndex(): Promise<void> {
    const indexPath = join(this.sessionsDir, 'index.json')
    try {
      await writeFileAtomic(indexPath, JSON.stringify(this.index, null, 2))
    } catch (error) {
      log.error('failed to persist session index', {
        indexPath,
        ...fileSystemErrorContext(error),
      })
      throw error
    }
  }
}

function fileSystemErrorContext(error: unknown): {
  error: string
  code?: string
  syscall?: string
  path?: string
  dest?: string
} {
  const fsError = error as NodeJS.ErrnoException & { dest?: unknown }
  return {
    error: error instanceof Error ? error.message : String(error),
    ...(typeof fsError?.code === 'string' ? { code: fsError.code } : {}),
    ...(typeof fsError?.syscall === 'string' ? { syscall: fsError.syscall } : {}),
    ...(typeof fsError?.path === 'string' ? { path: fsError.path } : {}),
    ...(typeof fsError?.dest === 'string' ? { dest: fsError.dest } : {}),
  }
}

function parseSessionIndex(data: string): SessionIndex {
  const parsed = JSON.parse(data) as Partial<SessionIndex>
  return {
    version: typeof parsed.version === 'number' ? parsed.version : 1,
    sessions: Array.isArray(parsed.sessions) ? (parsed.sessions as IndexedSessionMeta[]) : [],
  }
}

function toSessionMeta(session: IndexedSessionMeta): SessionMeta {
  const { searchText: _searchText, ...meta } = session
  return {
    ...meta,
    tags: [...meta.tags],
    ...(meta.personaIds !== undefined ? { personaIds: [...meta.personaIds] } : {}),
    totalTokens: { ...meta.totalTokens },
  }
}

function normalizeSearchQuery(query: string | undefined): string {
  return query?.trim().toLowerCase() ?? ''
}

function normalizeSearchIndexText(text: string): string {
  return text.toLowerCase()
}

function normalizeSessionTags(tags: string[]): string[] {
  const seen = new Set<string>()
  const normalized: string[] = []
  for (const tag of tags) {
    const trimmed = tag.trim()
    if (!trimmed) continue
    const key = trimmed.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    normalized.push(trimmed.slice(0, 64))
    if (normalized.length >= 16) break
  }
  return normalized
}

function sessionSearchIndexMatchesQuery(session: IndexedSessionMeta, query: string): boolean {
  return typeof session.searchText === 'string' && session.searchText.includes(query)
}

function buildSessionSearchText(session: SessionMeta, events: SessionEvent[]): string {
  return normalizeSearchIndexText(
    [...sessionMetaSearchValues(session), ...events.flatMap(sessionEventSearchValues)]
      .map(stringifySearchValue)
      .join(' '),
  )
}

function appendSessionSearchText(current: string, event: SessionEvent): string {
  const eventText = normalizeSearchIndexText(
    sessionEventSearchValues(event).map(stringifySearchValue).join(' '),
  )
  if (!eventText) {
    return current
  }
  return current ? `${current} ${eventText}` : eventText
}

function sessionMetaSearchValues(session: SessionMeta): unknown[] {
  return [session.title, session.provider, session.model, session.device, session.cwd, session.tags]
}

function sessionEventSearchValues(event: SessionEvent): unknown[] {
  switch (event.type) {
    case 'user_message':
    case 'assistant_message':
      return [event.content]
    case 'memory_context':
      return event.items.flatMap((item) => [
        item.title,
        item.snippet,
        item.citationLabel,
        item.documentTitle,
        item.documentPath,
      ])
    case 'llm_request':
      return [
        event.turnId,
        event.iteration,
        event.requestDigest.model,
        event.requestDigest.toolNames,
        event.requestDigest.traceRef,
      ]
    case 'mode_route_decision':
      return [
        event.chosen,
        event.persona,
        event.candidates,
        event.reason,
        event.confidence,
        event.fallback,
      ]
    case 'quality_gate_verdict':
      return [event.phase, event.decision, event.blockingReason, event.backtrackCount]
    case 'backtrack':
      return [event.phase, event.reason, event.attempt]
    case 'node_trace':
      return [event.node, event.durationMs, event.nextEdge]
    case 'tool_call':
      return [event.tool, event.input]
    case 'tool_result':
      return [event.output]
    case 'approval_request':
      return [event.tool, event.input]
    case 'approval_response':
      return [event.approvedBy, event.note, event.decision]
    case 'auto_approval':
      return [event.tool, event.rule.pattern, event.decision]
    case 'context_compact':
      return [event.summary]
    case 'memory_summary':
      return [
        event.source,
        event.stage,
        event.turnId,
        event.lightCaptured,
        event.semanticMemoriesExtracted,
        event.ragContextPromotions,
        event.ragPromotedContextIds,
      ]
    case 'provider_attempt':
      return [
        event.provider,
        event.model,
        event.source,
        event.status,
        event.errorCode,
        event.errorMessage,
        event.nextProvider,
        event.nextModel,
      ]
    case 'run_contract':
      return [
        event.contract.summary,
        event.contract.acceptanceCriteria.map((criterion) => criterion.text),
        event.contract.constraints,
        event.contract.outOfScope,
        event.contract.source,
      ]
    case 'todo_list':
      return event.items.map((item) => item.content)
    case 'cowork_plan':
      return event.plan.flatMap((step) => [step.role, step.instruction])
    case 'cowork_task_start':
    case 'cowork_task_complete':
    case 'cowork_task_failed':
      return [
        event.role,
        event.instruction,
        'result' in event ? event.result : undefined,
        'error' in event ? event.error : undefined,
      ]
    case 'cowork_synthesizing':
      return [event.summary]
    case 'cowork_discuss_request':
    case 'cowork_discuss_response':
      return [
        event.prompt,
        'response' in event ? event.response : undefined,
        'choices' in event ? event.choices : undefined,
      ]
    case 'delegation_state':
      return [event.targetDevice, event.detail, event.claimHealth, event.source]
    case 'delegation_result':
      return [event.targetDevice, event.status, event.result, event.artifactHandles, event.source]
    case 'session_start':
      return [event.metadata]
    case 'session_end':
      return [event.totalTokens, event.totalCost, event.duration_ms]
    case 'edit_checkpoint_opened':
    case 'edit_checkpoint_resolved':
      return [event.checkpoint]
    case 'debate_round':
      return [event.round]
    case 'planner_working_memory_updated':
      return [event.workingMemory]
    case 'phase_change':
      return [event.enteredPhase, event.closedPhase, event.phaseUsages]
    case 'post_edit_findings':
      return [
        event.editedFiles,
        event.impactedExternalModules,
        event.impactedLocalModules,
        event.reverseCallers,
        event.diagnostics.map((d) => `${d.file}: ${d.summary}`),
      ]
    default:
      return []
  }
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
