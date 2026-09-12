import type { DocumentIngestInput } from '@sepilotd/core'

export interface DocumentChunk {
  content: string
  chunkIndex: number
  chunkCount: number
  startOffset: number
  endOffset: number
  chunkTitle?: string
}

interface Section {
  title?: string
  start: number
  end: number
  body: string
}

interface Piece {
  text: string
  start: number
  end: number
  preOverlapped?: boolean
}

interface TokenSpan {
  text: string
  start: number
  end: number
}

const DEFAULT_TARGET_TOKENS = 220
const DEFAULT_MAX_TOKENS = 320
const DEFAULT_OVERLAP_TOKENS = 48
const CJK_TOKEN_PATTERN = /[\u3000-\u9fff\uac00-\ud7af\uf900-\ufaff]|[^\s\u3000-\u9fff\uac00-\ud7af\uf900-\ufaff]+/gu

function tokenizeForChunking(text: string): TokenSpan[] {
  return Array.from(text.matchAll(CJK_TOKEN_PATTERN), (match) => ({
    text: match[0],
    start: match.index ?? 0,
    end: (match.index ?? 0) + match[0].length,
  }))
}

function countTokens(text: string): number {
  const trimmed = text.trim()
  if (!trimmed) return 0
  return tokenizeForChunking(trimmed).length
}

function normalizeText(text: string): string {
  return text.replace(/\r\n/g, '\n').trim()
}

function splitMarkdownSections(content: string, fallbackTitle: string): Section[] {
  const normalized = normalizeText(content)
  if (!normalized) return []

  const headingMatches = Array.from(
    normalized.matchAll(/^(#{1,6})\s+(.+)$/gm),
  )

  if (headingMatches.length === 0) {
    return [{
      title: fallbackTitle,
      start: 0,
      end: normalized.length,
      body: normalized,
    }]
  }

  const sections: Section[] = []
  let cursor = 0
  let currentTitle = fallbackTitle

  for (let index = 0; index < headingMatches.length; index++) {
    const match = headingMatches[index]
    const headingStart = match.index ?? 0
    if (headingStart > cursor) {
      const body = normalized.slice(cursor, headingStart).trim()
      if (body) {
        sections.push({
          title: currentTitle,
          start: cursor,
          end: headingStart,
          body,
        })
      }
    }

    currentTitle = match[2]?.trim() || fallbackTitle
    cursor = headingStart
    const nextHeadingStart = headingMatches[index + 1]?.index ?? normalized.length
    const sectionBody = normalized.slice(headingStart, nextHeadingStart).trim()
    if (sectionBody) {
      sections.push({
        title: currentTitle,
        start: headingStart,
        end: nextHeadingStart,
        body: sectionBody,
      })
    }
    cursor = nextHeadingStart
  }

  return sections
}

function splitPlainSections(content: string, fallbackTitle: string): Section[] {
  const normalized = normalizeText(content)
  if (!normalized) return []
  return [{
    title: fallbackTitle,
    start: 0,
    end: normalized.length,
    body: normalized,
  }]
}

function splitPieces(section: Section): Piece[] {
  const pieces: Piece[] = []
  const regex = /\n\s*\n+/g
  let lastIndex = 0

  for (const match of section.body.matchAll(regex)) {
    const raw = section.body.slice(lastIndex, match.index ?? lastIndex)
    const trimmed = raw.trim()
    if (trimmed) {
      const startOffset = raw.indexOf(trimmed)
      pieces.push({
        text: trimmed,
        start: section.start + lastIndex + Math.max(startOffset, 0),
        end: section.start + lastIndex + Math.max(startOffset, 0) + trimmed.length,
      })
    }
    lastIndex = (match.index ?? lastIndex) + match[0].length
  }

  const finalRaw = section.body.slice(lastIndex)
  const finalTrimmed = finalRaw.trim()
  if (finalTrimmed) {
    const startOffset = finalRaw.indexOf(finalTrimmed)
    pieces.push({
      text: finalTrimmed,
      start: section.start + lastIndex + Math.max(startOffset, 0),
      end: section.start + lastIndex + Math.max(startOffset, 0) + finalTrimmed.length,
    })
  }

  return pieces
}

function splitLongPiece(piece: Piece, maxTokens: number, overlapTokens: number): Piece[] {
  if (countTokens(piece.text) <= maxTokens) return [piece]

  const tokens = tokenizeForChunking(piece.text)
  const chunks: Piece[] = []
  const step = Math.max(1, maxTokens - overlapTokens)

  for (let start = 0; start < tokens.length; start += step) {
    const end = Math.min(tokens.length, start + maxTokens)
    const first = tokens[start]
    const last = tokens[end - 1]
    if (!first || !last) continue
    const rawText = piece.text.slice(first.start, last.end)
    const chunkText = rawText.trim()
    if (!chunkText) continue
    const leadingTrim = rawText.indexOf(chunkText)
    const chunkStart = piece.start + first.start + Math.max(leadingTrim, 0)
    chunks.push({
      text: chunkText,
      start: chunkStart,
      end: chunkStart + chunkText.length,
      preOverlapped: true,
    })
    if (end >= tokens.length) break
  }

  return chunks
}

export function chunkDocumentForRag(
  input: Pick<DocumentIngestInput, 'title' | 'content' | 'mimeType'>,
): DocumentChunk[] {
  const fallbackTitle = input.title.trim() || 'Document'
  const isMarkdown = (input.mimeType ?? '').includes('markdown')
    || /\n#{1,6}\s+/.test(input.content)
  const sections = isMarkdown
    ? splitMarkdownSections(input.content, fallbackTitle)
    : splitPlainSections(input.content, fallbackTitle)

  const chunks: Omit<DocumentChunk, 'chunkIndex' | 'chunkCount'>[] = []

  for (const section of sections) {
    const pieces = splitPieces(section)
      .flatMap((piece) => splitLongPiece(piece, DEFAULT_MAX_TOKENS, DEFAULT_OVERLAP_TOKENS))
    if (pieces.length === 0) continue

    let bucket: Piece[] = []
    let bucketTokens = 0

    const flush = () => {
      if (bucket.length === 0) return
      const first = bucket[0]
      const last = bucket[bucket.length - 1]
      const body = bucket.map((piece) => piece.text).join('\n\n').trim()
      if (!body) {
        bucket = []
        bucketTokens = 0
        return
      }
      const prefix = section.title ? `# ${section.title}\n\n` : ''
      chunks.push({
        content: `${prefix}${body}`.trim(),
        startOffset: first.start,
        endOffset: last.end,
        chunkTitle: section.title,
      })

      const overlap: Piece[] = []
      let overlapTokens = 0
      for (let index = bucket.length - 1; index >= 0; index--) {
        if (bucket[index].preOverlapped) continue
        overlap.unshift(bucket[index])
        overlapTokens += countTokens(bucket[index].text)
        if (overlapTokens >= DEFAULT_OVERLAP_TOKENS) break
      }
      bucket = overlap
      bucketTokens = overlapTokens
    }

    for (const piece of pieces) {
      const pieceTokens = countTokens(piece.text)
      if (bucket.length > 0 && bucketTokens + pieceTokens > DEFAULT_TARGET_TOKENS) {
        flush()
      }
      bucket.push(piece)
      bucketTokens += pieceTokens
    }

    if (bucket.length > 0) {
      const first = bucket[0]
      const last = bucket[bucket.length - 1]
      const body = bucket.map((piece) => piece.text).join('\n\n').trim()
      const prefix = section.title ? `# ${section.title}\n\n` : ''
      chunks.push({
        content: `${prefix}${body}`.trim(),
        startOffset: first.start,
        endOffset: last.end,
        chunkTitle: section.title,
      })
    }
  }

  return chunks.map((chunk, index) => ({
    ...chunk,
    chunkIndex: index,
    chunkCount: chunks.length,
  }))
}
