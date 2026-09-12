import { Lexer, type Token, type Tokens } from 'marked'
import stringWidth from 'string-width'
import wrapAnsi from 'wrap-ansi'
import { highlightCode } from './highlight.js'

const BOLD = '\x1b[1m'
const DIM = '\x1b[2m'
const ITALIC = '\x1b[3m'
const UNDERLINE = '\x1b[4m'
const RESET = '\x1b[0m'
const CYAN = '\x1b[36m'
const YELLOW = '\x1b[33m'
const GRAY = '\x1b[90m'
const BG_GRAY = '\x1b[48;5;236m'

const DEFAULT_RULE_WIDTH = 40
const MIN_TABLE_CELL_WIDTH = 3

export interface MarkdownCodeBlock {
  language: string | null
  content: string
}

export interface RenderMarkdownOptions {
  /**
   * Target terminal column budget. When provided every block is wrapped to
   * fit (string-width aware, so CJK/Hangul count as 2 columns) with hanging
   * indents for lists and blockquotes. When omitted no wrapping is applied
   * and the caller (e.g. Ink) is responsible for it.
   */
  width?: number
}

export function renderMarkdown(text: string, options: RenderMarkdownOptions = {}): string {
  const width = normalizeWidth(options.width)
  const blocks = renderBlockTokens(lexMarkdown(text), { listDepth: 0, width })
  return joinBlocks(blocks)
}

export function extractMarkdownCodeBlocks(text: string): MarkdownCodeBlock[] {
  return collectCodeBlocks(lexMarkdown(text))
}

interface RenderContext {
  listDepth: number
  width: number | undefined
}

function normalizeWidth(width: number | undefined): number | undefined {
  if (typeof width !== 'number' || !Number.isFinite(width)) return undefined
  return Math.max(10, Math.floor(width))
}

function lexMarkdown(text: string): Token[] {
  return new Lexer().lex(text)
}

/**
 * Blocks are rendered as arrays of physical lines and joined with exactly one
 * blank line, so paragraph/space token combinations can never stack up
 * irregular vertical gaps.
 */
function joinBlocks(blocks: string[][]): string {
  return blocks
    .filter((block) => block.length > 0)
    .map((block) => block.join('\n'))
    .join('\n\n')
}

function renderBlockTokens(tokens: Token[], context: RenderContext): string[][] {
  return tokens
    .map((token) => renderBlockToken(token, context))
    .filter((block): block is string[] => block !== null)
}

function renderBlockToken(token: Token, context: RenderContext): string[] | null {
  switch (token.type) {
    case 'heading': {
      const t = token as Tokens.Heading
      const prefix = '#'.repeat(t.depth) + ' '
      const style =
        t.depth === 1 ? `${BOLD}${UNDERLINE}${YELLOW}` : t.depth === 2 ? `${BOLD}${YELLOW}` : BOLD
      return wrapStyled(`${prefix}${renderInline(t.tokens)}`, context.width, style)
    }
    case 'paragraph': {
      const t = token as Tokens.Paragraph
      return wrapText(renderInline(t.tokens), context.width)
    }
    case 'code': {
      const t = token as Tokens.Code
      return renderCodeBlock(t, context.width)
    }
    case 'blockquote': {
      const t = token as Tokens.Blockquote
      const innerWidth = context.width === undefined ? undefined : Math.max(10, context.width - 2)
      const inner = joinBlocks(
        renderBlockTokens(t.tokens, { ...context, width: innerWidth }),
      ).split('\n')
      return inner.map((line) => `${GRAY}${BOLD}│${RESET} ${DIM}${line}${RESET}`)
    }
    case 'list': {
      const t = token as Tokens.List
      return renderList(t, context.listDepth, context.width)
    }
    case 'space':
      // Vertical rhythm is owned by joinBlocks; explicit blank runs collapse.
      return null
    case 'hr': {
      const ruleWidth = Math.min(context.width ?? DEFAULT_RULE_WIDTH, DEFAULT_RULE_WIDTH)
      return [`${GRAY}${'─'.repeat(ruleWidth)}${RESET}`]
    }
    case 'html':
      return null
    case 'table': {
      const t = token as Tokens.Table
      return renderTable(t, context.width)
    }
    default: {
      if ('tokens' in token && Array.isArray(token.tokens)) {
        return wrapText(renderInline(token.tokens), context.width)
      }
      if ('text' in token) {
        return wrapText((token as Tokens.Text).text, context.width)
      }
      return null
    }
  }
}

function renderCodeBlock(block: Tokens.Code, width: number | undefined): string[] {
  const lang = block.lang || ''
  const highlighted = highlightCode(block.text, lang)
  const codeLabel = lang || 'code'
  const innerWidth = width === undefined ? undefined : Math.max(10, width - 2)
  const codeLines = highlighted
    .split('\n')
    .flatMap((line) => wrapLine(line, innerWidth))
    .map((line) => `${GRAY}│${RESET} ${line}`)
  const header = `${GRAY}╭─ ${codeLabel} · Copy Ctrl+Y${RESET}`
  const footer = `${GRAY}╰─${RESET}`
  return [header, ...codeLines, footer]
}

function renderList(list: Tokens.List, depth: number, width: number | undefined): string[] {
  const indent = '  '.repeat(depth + 1)
  const lines: string[] = []
  list.items.forEach((item, i) => {
    const bullet = list.ordered ? `${i + 1}.` : '•'
    const bulletIndent = ' '.repeat(stringWidth(bullet) + 1)
    const continuationIndent = `${indent}${bulletIndent}`
    const contentWidth =
      width === undefined ? undefined : Math.max(10, width - stringWidth(continuationIndent))
    const itemLines = renderListItemLines(item, depth, contentWidth, continuationIndent)
    const [firstLine = '', ...restLines] = itemLines
    lines.push(`${indent}${GRAY}${bullet}${RESET}${firstLine ? ` ${firstLine}` : ''}`)
    lines.push(...restLines)
  })
  return lines
}

/**
 * Render one list item's blocks. The first line is returned bare (the caller
 * attaches the bullet); every following line is already indented — nested
 * lists carry their own absolute indent, other blocks get the hanging indent.
 */
function renderListItemLines(
  item: Tokens.ListItem,
  depth: number,
  width: number | undefined,
  continuationIndent: string,
): string[] {
  const blocks: { lines: string[]; isNestedList: boolean }[] = []
  for (const token of item.tokens) {
    if (token.type === 'list') {
      blocks.push({ lines: renderList(token as Tokens.List, depth + 1, width), isNestedList: true })
      continue
    }
    const rendered = renderBlockToken(token, { listDepth: depth, width })
    if (rendered && rendered.length > 0) {
      blocks.push({ lines: rendered.map((line) => line.trimEnd()), isNestedList: false })
    }
  }

  const out: string[] = []
  let isFirstLine = true
  blocks.forEach((block, index) => {
    // Loose items keep a blank line between their paragraphs; tight items pack.
    if (index > 0 && item.loose) out.push('')
    for (const line of block.lines) {
      if (isFirstLine) {
        out.push(line)
        isFirstLine = false
        continue
      }
      if (block.isNestedList || line === '') {
        out.push(line)
        continue
      }
      out.push(`${continuationIndent}${line.trimStart()}`)
    }
  })
  return out
}

function renderTable(table: Tokens.Table, width: number | undefined): string[] {
  const headers = table.header.map((h) => renderInline(h.tokens))
  const rows = table.rows.map((row) => row.map((cell) => renderInline(cell.tokens)))
  const allRows = [headers, ...rows]
  let colWidths = headers.map((_, i) =>
    Math.max(MIN_TABLE_CELL_WIDTH, ...allRows.map((row) => stringWidth(row[i] || ''))),
  )

  if (width !== undefined) {
    colWidths = shrinkColumnsToFit(colWidths, width)
  }

  const renderRow = (cells: string[]) =>
    cells.map((cell, i) => fitCell(cell, colWidths[i] ?? MIN_TABLE_CELL_WIDTH)).join(' │ ')

  const headerLine = `${BOLD}${renderRow(headers)}${RESET}`
  const sepLine = `${GRAY}${colWidths.map((w) => '─'.repeat(w)).join('─┼─')}${RESET}`
  const bodyLines = rows.map((row) => renderRow(row))
  return [headerLine, sepLine, ...bodyLines]
}

function shrinkColumnsToFit(colWidths: number[], width: number): number[] {
  const separatorWidth = (colWidths.length - 1) * 3
  const available = Math.max(colWidths.length * MIN_TABLE_CELL_WIDTH, width - separatorWidth)
  let total = colWidths.reduce((sum, w) => sum + w, 0)
  if (total <= available) return colWidths

  const result = [...colWidths]
  // Repeatedly trim the widest column until the table fits (or nothing can shrink).
  while (total > available) {
    let widest = 0
    for (let i = 1; i < result.length; i += 1) {
      if (result[i]! > result[widest]!) widest = i
    }
    if (result[widest]! <= MIN_TABLE_CELL_WIDTH) break
    result[widest] -= 1
    total -= 1
  }
  return result
}

/** Pad or truncate a cell to an exact display width (string-width aware). */
function fitCell(cell: string, width: number): string {
  const visible = stringWidth(stripAnsi(cell))
  if (visible <= width) {
    return cell + ' '.repeat(width - visible)
  }
  const plain = stripAnsi(cell)
  let truncated = ''
  let used = 0
  for (const char of Array.from(plain)) {
    const charWidth = Math.max(1, stringWidth(char))
    if (used + charWidth > width - 1) break
    truncated += char
    used += charWidth
  }
  return `${truncated}…${' '.repeat(Math.max(0, width - used - 1))}`
}

function wrapText(text: string, width: number | undefined): string[] {
  return text.split('\n').flatMap((line) => wrapLine(line, width))
}

function wrapLine(line: string, width: number | undefined): string[] {
  if (width === undefined || stringWidth(stripAnsi(line)) <= width) {
    return [line]
  }
  return wrapAnsi(line, width, { hard: true, trim: false }).split('\n')
}

function wrapStyled(text: string, width: number | undefined, style: string): string[] {
  return wrapText(text, width).map((line) => `${style}${line}${RESET}`)
}

function renderInline(tokens: Token[]): string {
  return tokens
    .map((t) => {
      switch (t.type) {
        case 'text':
          if ('tokens' in t && Array.isArray(t.tokens)) {
            return renderInline(t.tokens)
          }
          return (t as Tokens.Text).text
        case 'strong': {
          const s = t as Tokens.Strong
          return `${BOLD}${renderInline(s.tokens)}${RESET}`
        }
        case 'em': {
          const e = t as Tokens.Em
          return `${ITALIC}${renderInline(e.tokens)}${RESET}`
        }
        case 'codespan': {
          const c = t as Tokens.Codespan
          return `${BG_GRAY}${CYAN} ${c.text} ${RESET}`
        }
        case 'link': {
          const l = t as Tokens.Link
          const text = renderInline(l.tokens)
          return `${UNDERLINE}${text}${RESET} ${GRAY}(${l.href})${RESET}`
        }
        case 'br':
          return '\n'
        case 'del': {
          const d = t as Tokens.Del
          return `${DIM}${renderInline(d.tokens)}${RESET}`
        }
        case 'escape':
          return (t as Tokens.Escape).text
        default: {
          const candidate = t as { text?: unknown }
          if (typeof candidate.text === 'string') return candidate.text
          return ''
        }
      }
    })
    .join('')
}

function stripAnsi(str: string): string {
  return str.replace(/\x1b\[[0-9;]*m/g, '')
}

function collectCodeBlocks(tokens: Token[]): MarkdownCodeBlock[] {
  return tokens.flatMap((token) => {
    switch (token.type) {
      case 'code': {
        const block = token as Tokens.Code
        return [
          {
            language: block.lang || null,
            content: block.text,
          },
        ]
      }
      case 'blockquote': {
        const blockquote = token as Tokens.Blockquote
        return collectCodeBlocks(blockquote.tokens)
      }
      case 'list': {
        const list = token as Tokens.List
        return list.items.flatMap((item) => collectCodeBlocks(item.tokens))
      }
      default:
        return []
    }
  })
}
