import { basename, extname } from 'node:path'

const MARKDOWN_EXTENSIONS = new Set(['.md', '.markdown', '.mdx'])
const MARKDOWN_H1_RE = /^\s{0,3}#\s+(.+?)\s*#*\s*$/mu

export class WikiMarkdownError extends Error {
  readonly statusCode = 400
  readonly code = 'INVALID_MARKDOWN_FILE'

  constructor(message: string) {
    super(message)
    this.name = 'WikiMarkdownError'
  }
}

export function isMarkdownPath(relativePath: string): boolean {
  return MARKDOWN_EXTENSIONS.has(extname(relativePath).toLowerCase())
}

export function validateMarkdownName(name: string): void {
  if (
    name.includes('/') ||
    name.includes('\\') ||
    name.includes('\0') ||
    name === '.' ||
    name === '..' ||
    !isMarkdownPath(name)
  ) {
    throw new WikiMarkdownError(`Invalid Markdown file name: ${name}`)
  }
}

export function normalizeMarkdown(content: string): string {
  return content.replace(/^\uFEFF/u, '')
}

export function headingFromMarkdown(content: string): string | null {
  return normalizeMarkdown(content).match(MARKDOWN_H1_RE)?.[1]?.trim() || null
}

export function titleFromMarkdown(name: string, content: string): string {
  validateMarkdownName(name)
  const heading = headingFromMarkdown(content)
  const fallback = basename(name, extname(name)).trim()
  const title = heading || fallback
  if (!title) {
    throw new WikiMarkdownError(
      `Markdown file does not have a usable title: ${name}`,
    )
  }
  return title.slice(0, 500)
}

/** Wiki stores the title separately, so remove the H1 used as that title. */
export function bodyFromMarkdown(content: string): string {
  const normalized = normalizeMarkdown(content)
  const match = normalized.match(MARKDOWN_H1_RE)
  if (!match || match.index === undefined) return normalized
  const before = normalized.slice(0, match.index)
  const after = normalized
    .slice(match.index + match[0].length)
    .replace(/^(?:\r?\n){1,2}/u, '')
  return `${before}${after}`
}

export function portableMarkdownHeading(title: string): string {
  const oneLine = title
    .normalize('NFC')
    .replace(/[\r\n\t]+/gu, ' ')
    .replace(/\s+/gu, ' ')
    .trim()
  return oneLine || 'Untitled'
}

export function renderWikiMarkdown(title: string, body: string): string {
  return `# ${portableMarkdownHeading(title)}\n\n${body}`
}
