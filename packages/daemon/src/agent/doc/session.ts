// "글쓰기 모드 (canvas)" 의 daemon-측 문서 세션 owner.
//
// 책임:
//   - DocSession 메모리 보관 + 변경 적용 (apply)
//   - doc.* tool과 desktop user_edit endpoint 모두 한 곳에서 처리
//   - 변경 시마다 EventEmitter로 'updated' broadcast → server/ws.ts가 desktop에 push
//   - diff_preview pending 보관 + accept/cancel
//   - history 보관 (undo/redo는 future; 현재는 history entry만 push)
//
// 영속:
//   - 디스크는 사용자의 명시적 save 시점에만 write
//   - file watcher는 desktop 측에서 외부 편집(Obsidian 등) 감지용으로 보조 유지 가능
//
// 동시성:
//   - 단일 process 내 single-threaded JS. version 증가만으로 race 감지 (apply에 expectedVersion 인자 둠)

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
  /** 가장 최근에 open되었거나 사용자가 명시적으로 set한 session id. doc.* tool이
   *  context로 어떤 doc을 쓸지 결정할 때 사용. mode='writing'인 desktop 1개 가정. */
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
    this.activeId = id // 가장 최근 open이 active
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

  /**
   * 가장 최근 LLM-author change를 undo. user-author 변경은 editor 측 native undo가
   * 처리하므로 daemon undo는 author='llm' 또는 'external' entry만 대상.
   * 반환된 version은 적용 후. undo할 entry 없으면 null.
   */
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
    // entry.reverseChanges는 그 version으로 가게 한 변경의 역. 적용해서 이전 상태 복원.
    s.history.splice(idx, 1)
    return this.apply(id, entry.reverseChanges, 'external', {
      toolName: 'doc.undo',
      label: `undo ${entry.toolName ?? 'change'} (v${entry.version})`,
    })
  }

  /**
   * 변경 적용. expectedVersion 주면 version mismatch 시 거부.
   * changes는 start ascending 정렬되어야 함 (호출자 책임).
   */
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

    // descending 순서로 적용해야 offset이 안 깨짐
    const sorted = [...changes].sort((a, b) => b.start - a.start)
    let content = s.content
    const reverseChanges: DocChange[] = []
    for (const ch of sorted) {
      const before = content.slice(0, ch.start)
      const replaced = content.slice(ch.start, ch.end)
      const after = content.slice(ch.end)
      content = before + ch.newText + after
      // reverse: 새 영역(start..start+newText.length)을 원래 텍스트로
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
      changes: [...sorted].reverse(), // 호출자에게는 ascending이 자연
      version: s.version,
      author,
      toolName: meta?.toolName,
      label: meta?.label,
    }
    this.events.emit('updated', ev)
    return { version: s.version }
  }

  // ------- diff preview -------

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

  // ------- 디스크 sync -------

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

  /** 외부에서 디스크가 바뀐 경우(다른 에디터 등) — content 통째 reload */
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

  // ------- helpers for tools -------

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

/** Process-wide singleton (daemon 전체에서 같은 registry 공유) */
let _instance: DocRegistry | null = null
export function getDocRegistry(): DocRegistry {
  if (!_instance) _instance = new DocRegistry()
  return _instance
}
