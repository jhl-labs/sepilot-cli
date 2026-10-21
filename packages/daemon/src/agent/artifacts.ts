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

export function extractArtifacts(
  content: string,
  context: ArtifactExtractionContext = {},
): ParsedResponse {
  const artifacts: Artifact[] = []
  const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g

  const text = content.replace(codeBlockRegex, (match, lang, code) => {
    const trimmedCode = code.trim()
    const lineCount = trimmedCode.split('\n').length

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
  const commentTitle = leadingCommentTitle(code.split('\n')[0] ?? '')
  if (commentTitle) return commentTitle

  const titleMatch = code.match(/<title>([^<]+)<\/title>/i)
  if (titleMatch) return titleMatch[1].trim()

  return lang ? `${lang} snippet` : undefined
}

const TITLE_COMMENT_PREFIXES = ['//', '#', '<!--'] as const
const HTML_COMMENT_SUFFIX = '-->'

function leadingCommentTitle(line: string): string | undefined {
  const prefix = TITLE_COMMENT_PREFIXES.find((candidate) => line.startsWith(candidate))
  if (!prefix) return undefined
  let body = line.slice(prefix.length).trimEnd()
  if (body.endsWith(HTML_COMMENT_SUFFIX)) body = body.slice(0, -HTML_COMMENT_SUFFIX.length)
  return body.trim() || undefined
}
