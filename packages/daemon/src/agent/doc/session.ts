import { randomUUID, createHash } from 'node:crypto'
import { EventEmitter } from 'node:events'
import { readFile, writeFile, stat } from 'node:fs/promises'
import type {
  DocSession,
  DocChange,
  DocChangeAuthor,
  DocOutlineEntry,
  DocUpdateEvent,
  DocDiffPreview,
  DocDiffPendingEvent,
  DocDiffResolvedEvent,
  DocClosedEvent,
  DocHistoryEntry,
} from '@sepilotd/core'
import { parseOutline, findSection } from './parse.js'

const DIFF_PREVIEW_TTL_MS = 10 * 60 * 1000

export interface DocRegistryEvents {
  updated: (e: DocUpdateEvent) => void
  diffPending: (e: DocDiffPendingEvent) => void
  diffResolved: (e: DocDiffResolvedEvent) => void
  closed: (e: DocClosedEvent) => void
}

interface MutableSession extends DocSession {
  history: DocHistoryEntry[]
  pendingPreviews: Map<string, DocDiffPreview>
}

export class DocRegistry {
  private sessions = new Map<string, MutableSession>()
  private pathToSession = new Map<string, string>()
  private activeId: string | null = null
  public events = new EventEmitter()

  getActiveId(): string | null {
    return this.activeId && this.sessions.has(this.activeId) ? this.activeId : null
  }

  setActive(id: string | null): void {
    if (id == null) {
      this.activeId = null
      return
    }
    if (!this.sessions.has(id)) throw new Error(`doc session not found: ${id}`)
    this.activeId = id
  }

  list(): DocSession[] {
    return [...this.sessions.values()].map(stripInternal)
  }

  get(id: string): DocSession | null {
    const s = this.sessions.get(id)
    return s ? stripInternal(s) : null
  }

  async open(input: { path: string | null; initialContent?: string }): Promise<DocSession> {
    if (input.path) {
      const existingId = this.pathToSession.get(input.path)
      if (existingId) {
        const s = this.sessions.get(existingId)!
        return stripInternal(s)
      }
    }

    let content = input.initialContent ?? ''
    let diskMtimeMs: number | null = null
    if (input.path) {
      try {
        content = await readFile(input.path, 'utf-8')
        const st = await stat(input.path)
        diskMtimeMs = st.mtimeMs
      } catch (err) {
        if (input.initialContent == null) throw err
      }
    }

    const id = randomUUID()
    const now = Date.now()
    const session: MutableSession = {
      id,
      path: input.path,
      content,
      version: 1,
      dirty: false,
      diskMtimeMs,
      createdAt: now,
      updatedAt: now,
      history: [],
      pendingPreviews: new Map(),
    }
    this.sessions.set(id, session)
    if (input.path) this.pathToSession.set(input.path, id)
    this.activeId = id
    return stripInternal(session)
  }

  close(id: string, reason?: string): boolean {
    const s = this.sessions.get(id)
    if (!s) return false
    this.sessions.delete(id)
    if (s.path) this.pathToSession.delete(s.path)
    if (this.activeId === id) this.activeId = null
    const ev: DocClosedEvent = { type: 'doc.closed', sessionId: id, reason }
    this.events.emit('closed', ev)
    return true
  }

  outline(id: string): DocOutlineEntry[] {
    const s = this.require(id)
    return parseOutline(s.content)
  }

  undoLastLlmChange(id: string): { version: number } | null {
    const s = this.require(id)
    let idx = -1
    for (let i = s.history.length - 1; i >= 0; i--) {
      if (s.history[i].author === 'llm' || s.history[i].author === 'external') {
        idx = i
        break
      }
    }
    if (idx < 0) return null
    const entry = s.history[idx]
    s.history.splice(idx, 1)
    return this.apply(id, entry.reverseChanges, 'external', {
      toolName: 'doc.undo',
      label: `undo ${entry.toolName ?? 'change'} (v${entry.version})`,
    })
  }

  apply(
    id: string,
    changes: DocChange[],
    author: DocChangeAuthor,
    meta?: { toolName?: string; label?: string; expectedVersion?: number },
  ): { version: number } {
    const s = this.require(id)
    if (meta?.expectedVersion != null && meta.expectedVersion !== s.version) {
      throw new Error(
        `doc.version mismatch: expected ${meta.expectedVersion}, current ${s.version}`,
      )
    }
    if (changes.length === 0) return { version: s.version }

    const sorted = [...changes].sort((a, b) => b.start - a.start)
    let content = s.content
    const reverseChanges: DocChange[] = []
    for (const ch of sorted) {
      const before = content.slice(0, ch.start)
      const replaced = content.slice(ch.start, ch.end)
      const after = content.slice(ch.end)
      content = before + ch.newText + after
      reverseChanges.push({
        start: ch.start,
        end: ch.start + ch.newText.length,
        newText: replaced,
      })
    }
    s.content = content
    s.version += 1
    s.dirty = true
    s.updatedAt = Date.now()
    s.history.push({
      version: s.version,
      reverseChanges,
      author,
      toolName: meta?.toolName,
      at: s.updatedAt,
    })

    const ev: DocUpdateEvent = {
      type: 'doc.updated',
      sessionId: id,
      changes: [...sorted].reverse(),
      version: s.version,
      author,
      toolName: meta?.toolName,
      label: meta?.label,
    }
    this.events.emit('updated', ev)
    return { version: s.version }
  }

  createPreview(
    sessionId: string,
    changes: DocChange[],
    summary: string,
  ): DocDiffPreview {
    const s = this.require(sessionId)
    const preview: DocDiffPreview = {
      id: randomUUID(),
      sessionId,
      baseVersion: s.version,
      summary,
      changes,
      createdAt: Date.now(),
    }
    s.pendingPreviews.set(preview.id, preview)
    setTimeout(() => {
      if (s.pendingPreviews.delete(preview.id)) {
        const ev: DocDiffResolvedEvent = {
          type: 'doc.diff_resolved',
          sessionId,
          previewId: preview.id,
          resolution: 'expired',
        }
        this.events.emit('diffResolved', ev)
      }
    }, DIFF_PREVIEW_TTL_MS).unref?.()
    const ev: DocDiffPendingEvent = {
      type: 'doc.diff_pending',
      sessionId,
      preview,
    }
    this.events.emit('diffPending', ev)
    return preview
  }

  acceptPreview(sessionId: string, previewId: string): { version: number } {
    const s = this.require(sessionId)
    const preview = s.pendingPreviews.get(previewId)
    if (!preview) throw new Error(`preview not found: ${previewId}`)
    if (preview.baseVersion !== s.version) {
      throw new Error(
        `preview is stale (base v${preview.baseVersion}, current v${s.version}); ` +
          'doc changed since the preview was created. discard and ask the LLM to redo.',
      )
    }
    s.pendingPreviews.delete(previewId)
    const result = this.apply(sessionId, preview.changes, 'llm', {
      toolName: 'doc.diff_preview/accept',
      label: preview.summary,
    })
    const ev: DocDiffResolvedEvent = {
      type: 'doc.diff_resolved',
      sessionId,
      previewId,
      resolution: 'accepted',
    }
    this.events.emit('diffResolved', ev)
    return result
  }

  cancelPreview(sessionId: string, previewId: string): boolean {
    const s = this.require(sessionId)
    const ok = s.pendingPreviews.delete(previewId)
    if (ok) {
      const ev: DocDiffResolvedEvent = {
        type: 'doc.diff_resolved',
        sessionId,
        previewId,
        resolution: 'cancelled',
      }
      this.events.emit('diffResolved', ev)
    }
    return ok
  }

  listPreviews(sessionId: string): DocDiffPreview[] {
    const s = this.require(sessionId)
    return [...s.pendingPreviews.values()]
  }

  async saveToDisk(sessionId: string, savePath?: string): Promise<{ path: string; mtimeMs: number }> {
    const s = this.require(sessionId)
    const target = savePath ?? s.path
    if (!target) throw new Error('save: no path (call open with a path or pass savePath)')
    await writeFile(target, s.content, 'utf-8')
    const st = await stat(target)
    s.diskMtimeMs = st.mtimeMs
    s.dirty = false
    if (s.path !== target) {
      if (s.path) this.pathToSession.delete(s.path)
      s.path = target
      this.pathToSession.set(target, s.id)
    }
    return { path: target, mtimeMs: st.mtimeMs }
  }

  async reloadFromDisk(sessionId: string): Promise<{ version: number }> {
    const s = this.require(sessionId)
    if (!s.path) throw new Error('reload: session has no path')
    const fresh = await readFile(s.path, 'utf-8')
    const st = await stat(s.path)
    s.diskMtimeMs = st.mtimeMs
    const fullRange: DocChange = {
      start: 0,
      end: s.content.length,
      newText: fresh,
    }
    return this.apply(sessionId, [fullRange], 'external', {
      toolName: 'doc.reload',
      label: 'reload from disk',
    })
  }

  findSection(sessionId: string, selector: string | number): DocOutlineEntry | null {
    return findSection(this.outline(sessionId), selector)
  }

  contentOf(sessionId: string): string {
    return this.require(sessionId).content
  }

  contentHash(sessionId: string): string {
    return createHash('sha256').update(this.require(sessionId).content).digest('hex')
  }

  private require(id: string): MutableSession {
    const s = this.sessions.get(id)
    if (!s) throw new Error(`doc session not found: ${id}`)
    return s
  }
}

function stripInternal(s: MutableSession): DocSession {
  return {
    id: s.id,
    path: s.path,
    content: s.content,
    version: s.version,
    dirty: s.dirty,
    diskMtimeMs: s.diskMtimeMs,
    createdAt: s.createdAt,
    updatedAt: s.updatedAt,
  }
}

let _instance: DocRegistry | null = null
export function getDocRegistry(): DocRegistry {
  if (!_instance) _instance = new DocRegistry()
  return _instance
}
