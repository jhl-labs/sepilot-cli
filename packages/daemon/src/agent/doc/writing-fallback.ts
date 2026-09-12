import { getDocRegistry } from './session.js'
import { strictWorkspacePathViolation } from '../../security/policy-engine.js'

export interface WritingDocStartSnapshot {
  id: string
  version: number
  contentLength: number
  workspaceRoot?: string
}

export interface WritingDocFallbackResult {
  applied: boolean
  chatContent?: string
}

export function snapshotWritingDocStart(
  mode: string | undefined,
  writingDocId?: string,
  workspaceRoot?: string,
): WritingDocStartSnapshot | null {
  if (mode !== 'writing') return null
  const registry = getDocRegistry()
  const id = writingDocId ?? registry.getActiveId()
  if (!id) return null
  const session = registry.get(id)
  if (!session) return null
  if (
    session.path
    && workspaceRoot
    && strictWorkspacePathViolation(session.path, workspaceRoot)
  ) return null
  return {
    id,
    version: session.version,
    contentLength: session.content.length,
    workspaceRoot,
  }
}

export function applyWritingDocFallback(
  mode: string | undefined,
  snapshot: WritingDocStartSnapshot | null,
  finalContent: string,
): WritingDocFallbackResult {
  if (mode !== 'writing' || !snapshot) return { applied: false }
  const text = finalContent.trim()
  if (!text) return { applied: false }
  const registry = getDocRegistry()
  const current = registry.get(snapshot.id)
  if (!current || current.version !== snapshot.version) return { applied: false }
  if (
    current.path
    && snapshot.workspaceRoot
    && strictWorkspacePathViolation(current.path, snapshot.workspaceRoot)
  ) return { applied: false }

  if (snapshot.contentLength === 0 || current.content.trim().length === 0) {
    registry.apply(
      snapshot.id,
      [{ start: 0, end: current.content.length, newText: text }],
      'llm',
      { toolName: 'doc.rewrite', label: '채팅 답변을 빈 문서에 작성' },
    )
    return {
      applied: true,
      chatContent: '오른쪽 문서에 초안을 작성했습니다.',
    }
  }

  const prefix = current.content.endsWith('\n') ? '\n' : '\n\n'
  registry.apply(
    snapshot.id,
    [{
      start: current.content.length,
      end: current.content.length,
      newText: `${prefix}${text}`,
    }],
    'llm',
    { toolName: 'doc.append', label: '채팅 답변을 문서 끝에 추가' },
  )
  return {
    applied: true,
    chatContent: '오른쪽 문서 끝에 내용을 추가했습니다.',
  }
}
