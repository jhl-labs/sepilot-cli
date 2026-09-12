import type { MemoryDocumentChunk } from '@sepilotd/core'
import { tokenizeMemoryText } from './relevance.js'

const DEFAULT_SNIPPET_LENGTH = 220
const SNIPPET_CONTEXT_RADIUS = 72

function stripLeadingMarkdownHeading(content: string): string {
  return content
    .replace(/^\s*#{1,6}\s+[^\n]+\n+/u, '')
    .trim()
}

function compactWhitespace(content: string): string {
  return content.replace(/\s+/g, ' ').trim()
}

function ellipsize(content: string, start: number, end: number): string {
  const prefix = start > 0 ? '...' : ''
  const suffix = end < content.length ? '...' : ''
  return `${prefix}${content.slice(start, end).trim()}${suffix}`.trim()
}

export function buildDocumentCitationLabel(
  chunk: Pick<MemoryDocumentChunk, 'documentTitle' | 'documentPath' | 'chunkTitle' | 'chunkIndex' | 'chunkCount'>,
): string {
  const parts = [chunk.documentTitle]
  if (chunk.chunkTitle?.trim() && chunk.chunkTitle.trim() !== chunk.documentTitle.trim()) {
    parts.push(chunk.chunkTitle.trim())
  }

  const base = parts.join(' > ')
  const path = chunk.documentPath?.trim() ? ` [${chunk.documentPath.trim()}]` : ''
  const chunkOrdinal = chunk.chunkCount > 1 ? ` (chunk ${chunk.chunkIndex + 1}/${chunk.chunkCount})` : ''
  return `${base}${path}${chunkOrdinal}`
}

export function buildDocumentSnippet(
  query: string,
  content: string,
  maxLength = DEFAULT_SNIPPET_LENGTH,
): string {
  const normalized = compactWhitespace(stripLeadingMarkdownHeading(content))
  if (!normalized) return ''
  if (normalized.length <= maxLength) return normalized

  const lower = normalized.toLowerCase()
  const queryTokens = tokenizeMemoryText(query)
  const matchIndex = queryTokens
    .map((token) => lower.indexOf(token))
    .filter((index) => index >= 0)
    .sort((left, right) => left - right)[0]

  if (matchIndex == null) {
    return ellipsize(normalized, 0, Math.min(normalized.length, maxLength))
  }

  const start = Math.max(0, matchIndex - SNIPPET_CONTEXT_RADIUS)
  const end = Math.min(
    normalized.length,
    Math.max(matchIndex + maxLength - SNIPPET_CONTEXT_RADIUS, start + maxLength),
  )
  return ellipsize(normalized, start, end)
}

export function enrichDocumentChunk(
  chunk: MemoryDocumentChunk,
  query: string,
): MemoryDocumentChunk {
  return {
    ...chunk,
    snippet: buildDocumentSnippet(query, chunk.content),
    citationLabel: buildDocumentCitationLabel(chunk),
  }
}

export function formatDocumentChunkForPrompt(chunk: MemoryDocumentChunk): string {
  const label = chunk.citationLabel ?? buildDocumentCitationLabel(chunk)
  const snippet = chunk.snippet ?? buildDocumentSnippet('', chunk.content)
  return `[Document] ${label}\n${snippet || chunk.content}`
}
