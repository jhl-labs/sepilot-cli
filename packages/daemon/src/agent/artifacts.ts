/**
 * Artifact extraction from assistant responses.
 *
 * Parses code blocks, HTML blocks, and document blocks from LLM output,
 * returning structured artifacts that clients can render in separate panels.
 */

import { createHash } from 'node:crypto'

export interface Artifact {
  id: string
  key?: string
  version?: number
  type: 'code' | 'html' | 'document' | 'mermaid' | 'svg' | 'image'
  title?: string
  language?: string
  content: string
}

interface ParsedResponse {
  text: string
  artifacts: Artifact[]
}

export interface ArtifactExtractionContext {
  sessionId?: string
}

interface ArtifactInput {
  type: Artifact['type']
  content: string
  title?: string
  language?: string
}

function artifactHash(input: ArtifactInput, context: ArtifactExtractionContext): string {
  return createHash('sha256')
    .update(context.sessionId ?? 'global')
    .update('\0')
    .update(input.type)
    .update('\0')
    .update(input.language ?? '')
    .update('\0')
    .update(input.title ?? '')
    .update('\0')
    .update(input.content)
    .digest('hex')
}

function createArtifact(
  input: ArtifactInput,
  context: ArtifactExtractionContext,
): Artifact {
  const key = `artifact-${artifactHash(input, context).slice(0, 24)}`
  return {
    id: key,
    key,
    version: 1,
    ...input,
  }
}

/**
 * Extract artifacts from assistant response content.
 *
 * Detects:
 * - Code blocks with language tags (```lang ... ```)
 * - HTML artifacts (```html with full page structure)
 * - Mermaid diagrams (```mermaid)
 * - SVG content (```svg)
 *
 * Small inline code snippets are kept inline unless the model clearly marked
 * them as a typed, multi-line artifact.
 */
export function extractArtifacts(
  content: string,
  context: ArtifactExtractionContext = {},
): ParsedResponse {
  const artifacts: Artifact[] = []
  const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g

  const text = content.replace(codeBlockRegex, (match, lang, code) => {
    const trimmedCode = code.trim()
    const lineCount = trimmedCode.split('\n').length

    // Keep small inline snippets
    if (shouldKeepInline(lang, trimmedCode, lineCount)) {
      return match
    }

    const type = classifyArtifact(lang, trimmedCode)
    const title = inferTitle(lang, trimmedCode)
    const artifact = createArtifact({
      type,
      language: lang,
      title,
      content: trimmedCode,
    }, context)

    artifacts.push(artifact)

    return `[Artifact: ${title ?? lang ?? 'code'} (${artifact.id})]`
  })

  return { text, artifacts }
}

export function extractImageArtifacts(
  content: string,
  context: ArtifactExtractionContext = {},
): Artifact[] {
  const artifacts: Artifact[] = []
  const imageRegex = /!\[([^\]]*)\]\((data:image\/(?:png|jpe?g|gif|webp|avif|bmp);base64,[^)]+)\)/gi
  for (const match of content.matchAll(imageRegex)) {
    const title = match[1]?.trim() || 'Pasted image'
    const dataUrl = match[2]
    if (!dataUrl) continue
    artifacts.push(createArtifact({
      type: 'image',
      title,
      content: dataUrl,
    }, context))
  }
  return artifacts
}

function shouldKeepInline(
  lang: string | undefined,
  code: string,
  lineCount: number,
): boolean {
  if (isFullDocument(lang, code)) return false
  if (lineCount >= 4) return false

  // Language-tagged multi-line blocks are usually deliberate outputs that the
  // desktop should surface in the artifact gallery even when they are short.
  if (lang && lineCount > 1) return false

  return true
}

function classifyArtifact(
  lang: string | undefined,
  code: string,
): Artifact['type'] {
  if (lang === 'html' && isFullDocument('html', code)) return 'html'
  if (lang === 'mermaid') return 'mermaid'
  if (lang === 'svg' || code.trimStart().startsWith('<svg')) return 'svg'
  return 'code'
}

function isFullDocument(lang: string | undefined, code: string): boolean {
  if (lang === 'html') {
    return (
      code.includes('<!DOCTYPE') ||
      code.includes('<html') ||
      code.includes('<body') ||
      code.includes('<head')
    )
  }
  return false
}

function inferTitle(
  lang: string | undefined,
  code: string,
): string | undefined {
  // Try to find a title comment at the top
  const firstLine = code.split('\n')[0]
  const commentMatch = firstLine.match(
    /^(?:\/\/|#|<!--)\s*(.+?)(?:-->)?$/,
  )
  if (commentMatch) return commentMatch[1].trim()

  // For HTML, try <title>
  const titleMatch = code.match(/<title>([^<]+)<\/title>/i)
  if (titleMatch) return titleMatch[1].trim()

  return lang ? `${lang} snippet` : undefined
}
