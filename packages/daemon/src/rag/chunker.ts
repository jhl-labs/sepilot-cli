export interface Chunk {
  index: number
  text: string
}

export interface ChunkOptions {
  size: number
  overlap: number
}

export function chunkText(text: string, opts: ChunkOptions): Chunk[] {
  if (text.length <= opts.size) return [{ index: 0, text }]
  const out: Chunk[] = []
  let i = 0
  let idx = 0
  while (i < text.length) {
    const end = Math.min(text.length, i + opts.size)
    out.push({ index: idx++, text: text.slice(i, end) })
    if (end === text.length) break
    i = end - opts.overlap
    if (i <= 0) i = end
  }
  return out
}
