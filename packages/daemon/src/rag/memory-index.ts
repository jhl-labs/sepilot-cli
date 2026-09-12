/**
 * In-memory RAG vector index (D3 MVP).
 *
 * sqlite-vec native binding 의존성을 피하기 위해 청크 벡터를 메모리에 보관하고
 * 단순 cosine similarity로 top-k 검색한다. 프로덕션용 sqlite-vec 기반 영속
 * 벡터 인덱스는 후속 iteration에서 교체된다.
 */

export interface IndexedChunk {
  rowid: number
  documentId: string
  chunkIndex: number
  text: string
  vector: number[]
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0
  let na = 0
  let nb = 0
  const len = Math.min(a.length, b.length)
  for (let i = 0; i < len; i++) {
    const x = a[i]!
    const y = b[i]!
    dot += x * y
    na += x * x
    nb += y * y
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb)
  return denom === 0 ? 0 : dot / denom
}

export interface MemoryIndex {
  replaceDocument(
    documentId: string,
    chunks: Array<{ chunkIndex: number; text: string; vector: number[] }>,
  ): void
  removeDocument(documentId: string): void
  search(query: number[], limit: number): Array<{ rowid: number; score: number }>
  get(rowid: number): IndexedChunk | undefined
  snapshot(): {
    dimension: number
    documents: number
    chunks: number
  }
}

export function createMemoryIndex(): MemoryIndex {
  const chunks = new Map<number, IndexedChunk>()
  const byDocument = new Map<string, Set<number>>()
  let nextRowId = 1
  let dimension = 0

  return {
    replaceDocument(documentId, input) {
      const existing = byDocument.get(documentId)
      if (existing) {
        for (const rid of existing) chunks.delete(rid)
      }
      const fresh = new Set<number>()
      for (const c of input) {
        if (!dimension) dimension = c.vector.length
        const rowid = nextRowId++
        chunks.set(rowid, {
          rowid,
          documentId,
          chunkIndex: c.chunkIndex,
          text: c.text,
          vector: c.vector,
        })
        fresh.add(rowid)
      }
      byDocument.set(documentId, fresh)
    },
    removeDocument(documentId) {
      const existing = byDocument.get(documentId)
      if (!existing) return
      for (const rid of existing) chunks.delete(rid)
      byDocument.delete(documentId)
    },
    search(query, limit) {
      const all = Array.from(chunks.values()).map((c) => ({
        rowid: c.rowid,
        score: cosineSimilarity(query, c.vector),
      }))
      all.sort((a, b) => b.score - a.score)
      return all.slice(0, limit)
    },
    get(rowid) {
      return chunks.get(rowid)
    },
    snapshot() {
      return {
        dimension,
        documents: byDocument.size,
        chunks: chunks.size,
      }
    },
  }
}
