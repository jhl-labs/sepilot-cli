// "글쓰기 모드 (writing canvas)" 의 daemon-측 문서 세션 표현.
// daemon이 메모리에서 현재 활성 문서를 owner로 갖고, LLM의 doc.* tool 호출 +
// 사용자 직접 편집 둘 다 같은 세션에 적용한다. desktop renderer는 WS event로
// 변화를 받아 즉시 editor에 반영한다. 디스크는 영속용으로만 read/write.

export interface DocSession {
  /** session id; desktop이 open 응답으로 받음 */
  id: string
  /** 디스크 절대경로. null이면 미저장 새 문서 */
  path: string | null
  /** 현재 본문 (UTF-8) */
  content: string
  /** 변경마다 1씩 증가. desktop이 race 감지에 사용 */
  version: number
  /** 디스크 본문과 다른 미저장 사용자/LLM 편집이 있는지 */
  dirty: boolean
  /** 마지막 disk save 시점의 mtimeMs (사용자/외부 편집 충돌 감지용) */
  diskMtimeMs: number | null
  /** ms epoch */
  createdAt: number
  updatedAt: number
}

/** Markdown 헤딩 기반 outline 항목. doc.outline tool 응답. */
export interface DocOutlineEntry {
  /** 1~6 (#~######) */
  level: number
  /** 헤딩 텍스트 (마크업 제거 전) */
  title: string
  /** 0-based 시작 char offset */
  start: number
  /** 다음 같은-또는-상위 헤딩 직전까지의 char offset (exclusive) */
  end: number
  /** 순번 (top-level 헤딩 외에도 전부 포함) */
  index: number
}

/** doc 한 번의 변경. start..end (char offset, half-open) 영역을 newText로 교체 */
export interface DocChange {
  start: number
  end: number
  newText: string
}

/**
 * doc 변경의 출처. desktop이 author='self'를 echo 받았을 때
 * editor에 다시 적용하면 cursor가 깨질 수 있어 skip 판단에 사용.
 */
export type DocChangeAuthor = 'user' | 'llm' | 'external'

/** daemon → desktop WS broadcast. mode='writing' session 활성 동안 흐름. */
export interface DocUpdateEvent {
  type: 'doc.updated'
  sessionId: string
  /** 적용된 변경. 여러 개를 한 번에 묶을 수 있음 (rewrite 등) */
  changes: DocChange[]
  /** 적용 *후* version */
  version: number
  author: DocChangeAuthor
  /** highlight UX를 위한 메타: tool 이름 (e.g. 'doc.replace_section') */
  toolName?: string
  /** outline 라벨 (highlight 메시지에 표시) */
  label?: string
}

/**
 * doc.diff_preview tool이 만든 pending 변경. desktop이 inline diff를 표시하고
 * 사용자 수락/취소를 받음. 수락 = `POST /api/v1/doc/{id}/diff/{previewId}/accept`,
 * 취소 = `.../cancel`. 수락 시점에 비로소 doc.updated가 broadcast됨.
 */
export interface DocDiffPreview {
  id: string
  sessionId: string
  /** preview 시점 doc version (race 감지: accept 시 mismatch면 거부) */
  baseVersion: number
  /** preview 시점 description (사용자에게 표시) */
  summary: string
  changes: DocChange[]
  createdAt: number
}

/** WS broadcast: 새 diff preview가 만들어졌음 */
export interface DocDiffPendingEvent {
  type: 'doc.diff_pending'
  sessionId: string
  preview: DocDiffPreview
}

/** WS broadcast: diff preview가 수락/취소/만료됨 */
export interface DocDiffResolvedEvent {
  type: 'doc.diff_resolved'
  sessionId: string
  previewId: string
  resolution: 'accepted' | 'cancelled' | 'expired'
}

/** WS broadcast: doc session이 닫혔음 (서버측 정리) */
export interface DocClosedEvent {
  type: 'doc.closed'
  sessionId: string
  reason?: string
}

export type DocEvent =
  | DocUpdateEvent
  | DocDiffPendingEvent
  | DocDiffResolvedEvent
  | DocClosedEvent

/** undo/redo용 history entry. daemon에 보관, 사용자가 endpoint로 nav. */
export interface DocHistoryEntry {
  version: number
  /** 이 version 이전 → version 으로 가게 한 변경들의 reverse (undo 적용용) */
  reverseChanges: DocChange[]
  author: DocChangeAuthor
  toolName?: string
  at: number
}
