import stringWidth from 'string-width'
import { renderMarkdown } from '../renderer/markdown.js'
import type { Message } from '../types.js'
import {
  buildDiffLines,
  previewTextLines,
  splitTerminalOutput,
  terminalCommandLabel,
  truncatePreviewItems,
} from './tooling.js'

export interface TranscriptViewport {
  messages: Message[]
  hiddenCount: number
  hiddenAboveCount: number
  hiddenBelowCount: number
  usedLines: number
  reservedFooterLines: number
}

export interface TranscriptReaderRow {
  key: string
  text: string
  tone: 'primary' | 'success' | 'warning' | 'info' | 'dimText' | 'text'
}

export interface TranscriptReaderViewport {
  rows: TranscriptReaderRow[]
  hiddenAboveCount: number
  hiddenBelowCount: number
  usedLines: number
  reservedFooterLines: number
}

interface TranscriptViewportInput {
  messages: Message[]
  height: number
  width: number
  isStreaming: boolean
  currentMessage: string
  isThinking: boolean
  error: string | null
  scrollOffset?: number
}

const MIN_CONTENT_WIDTH = 20
const MIN_STREAMING_RESERVE = 6
const TOOL_PREVIEW_MAX_LINES = 12
const ANSI_PATTERN = /\u001B\[[0-?]*[ -/]*[@-~]/g
const OPERATIONAL_SYSTEM_MESSAGE_PATTERNS = [
  /^Mode set to /,
  /^Theme set to /,
  /^Model set to /,
  /^Provider set to /,
  /^Current provider set to /,
  /^Current session now uses /,
  /^Daemon default /,
  /^Loaded session /,
  /^Copied last code block/,
  /^Autonomy set to /,
  /^Thinking level set to /,
  /^Max tokens set to /,
  /^Project (set to|cleared)/,
  /^Context usage is above 80%\. Consider running \/compact\b/,
]

function countWrappedLines(text: string, width: number): number {
  const normalizedWidth = Math.max(MIN_CONTENT_WIDTH, width)
  // string-width strips ANSI codes internally and correctly measures
  // full-width CJK / Hangul characters as 2 columns each, so the wrap
  // estimate matches Ink's actual terminal output.
  const lines = text.split('\n')
  return lines.reduce((total, line) => {
    if (!line) {
      return total + 1
    }

    return total + Math.max(1, Math.ceil(stringWidth(line) / normalizedWidth))
  }, 0)
}

export function stripAnsi(text: string): string {
  return text.replace(ANSI_PATTERN, '')
}

function wrapPlainLine(text: string, width: number): string[] {
  const normalizedWidth = Math.max(MIN_CONTENT_WIDTH, width)
  if (!text) {
    return ['']
  }

  const wrapped: string[] = []
  let current = ''
  let currentWidth = 0

  for (const char of Array.from(text.normalize('NFC'))) {
    const charWidth = Math.max(1, stringWidth(char))
    if (current && currentWidth + charWidth > normalizedWidth) {
      wrapped.push(current)
      current = char
      currentWidth = charWidth
      continue
    }

    current += char
    currentWidth += charWidth
  }

  wrapped.push(current)
  return wrapped
}

export function wrapPlainText(text: string, width: number): string[] {
  return text.split('\n').flatMap((line) => wrapPlainLine(line, width))
}

export function tailWrappedText(
  text: string,
  width: number,
  maxLines: number,
): {
  visible: string[]
  omitted: number
} {
  const wrapped = wrapPlainText(stripAnsi(text), width)

  if (maxLines <= 0) {
    return { visible: [], omitted: wrapped.length }
  }

  if (wrapped.length <= maxLines) {
    return { visible: wrapped, omitted: 0 }
  }

  return {
    visible: wrapped.slice(-maxLines),
    omitted: wrapped.length - maxLines,
  }
}

function pushWrappedRows(
  rows: TranscriptReaderRow[],
  keyPrefix: string,
  text: string,
  tone: TranscriptReaderRow['tone'],
  width: number,
  indent = '',
) {
  const wrapped = wrapPlainText(text, width)
  wrapped.forEach((line, index) => {
    rows.push({
      key: `${keyPrefix}:${index}`,
      text: `${indent}${line}`.trimEnd(),
      tone,
    })
  })
}

function buildToolReaderLines(message: Message): TranscriptReaderRow[] {
  const tool = message.toolCall
  if (!tool) {
    return []
  }

  const rows: TranscriptReaderRow[] = []
  const summary = [
    tool.name,
    tool.meta,
    tool.collapsed ? JSON.stringify(tool.input ?? {}) : null,
  ].filter(Boolean).join(' ')

  rows.push({
    key: `${message.id}:tool:summary`,
    text: summary,
    tone: 'info',
  })

  if (tool.collapsed) {
    return rows
  }

  if (tool.name === 'fs.write' && typeof tool.input.content === 'string') {
    if (typeof tool.input.path === 'string') {
      rows.push({
        key: `${message.id}:tool:path`,
        text: `file ${tool.input.path}`,
        tone: 'info',
      })
    }

    if (tool.previousContent === undefined) {
      rows.push({
        key: `${message.id}:tool:previous`,
        text: 'Previous file contents unavailable. Showing requested write payload.',
        tone: 'dimText',
      })
    }

    const diffLines = buildDiffLines(tool.previousContent, tool.input.content)
    const visibleDiff = truncatePreviewItems(diffLines, TOOL_PREVIEW_MAX_LINES)
    if (visibleDiff.visible.length === 0) {
      rows.push({
        key: `${message.id}:tool:no-diff`,
        text: 'No textual changes detected.',
        tone: 'dimText',
      })
    } else {
      visibleDiff.visible.forEach((line, index) => {
        rows.push({
          key: `${message.id}:tool:diff:${index}`,
          text: `${line.prefix} ${line.value || ' '}`,
          tone: line.prefix === '+'
            ? 'success'
            : line.prefix === '-'
              ? 'warning'
              : 'dimText',
        })
      })
    }

    if (visibleDiff.omitted > 0) {
      rows.push({
        key: `${message.id}:tool:diff:omitted`,
        text: `... ${visibleDiff.omitted} more diff line${visibleDiff.omitted === 1 ? '' : 's'}`,
        tone: 'dimText',
      })
    }

    return rows
  }

  if (tool.name === 'terminal.run') {
    rows.push({
      key: `${message.id}:tool:command`,
      text: `$ ${terminalCommandLabel(tool.input)}`,
      tone: 'info',
    })

    if (typeof tool.input.cwd === 'string') {
      rows.push({
        key: `${message.id}:tool:cwd`,
        text: `cwd ${tool.input.cwd}`,
        tone: 'dimText',
      })
    }

    if (typeof tool.input.timeoutMs === 'number') {
      rows.push({
        key: `${message.id}:tool:timeout`,
        text: `timeout ${tool.input.timeoutMs}ms`,
        tone: 'dimText',
      })
    }

    const { stdout, stderr } = splitTerminalOutput(tool.output)
    const previewLines: Array<{
      text: string
      tone: TranscriptReaderRow['tone']
    }> = stdout
      ? previewTextLines(stdout).map((line) => ({
          text: line || ' ',
          tone: 'text',
        }))
      : [{
          text: 'No stdout',
          tone: 'dimText',
        }]

    if (stderr) {
      previewLines.push({
        text: 'stderr',
        tone: 'warning',
      })
      previewLines.push(
        ...previewTextLines(stderr).map((line) => ({
          text: line || ' ',
          tone: 'warning' as TranscriptReaderRow['tone'],
        })),
      )
    }

    const visiblePreview = truncatePreviewItems(previewLines, TOOL_PREVIEW_MAX_LINES)
    visiblePreview.visible.forEach((line, index) => {
      rows.push({
        key: `${message.id}:tool:preview:${index}`,
        text: line.text,
        tone: line.tone,
      })
    })

    if (visiblePreview.omitted > 0) {
      rows.push({
        key: `${message.id}:tool:preview:omitted`,
        text: `... ${visiblePreview.omitted} more output line${visiblePreview.omitted === 1 ? '' : 's'}`,
        tone: 'dimText',
      })
    }

    rows.push({
      key: `${message.id}:tool:status`,
      text: tool.status === 'success'
        ? 'exit 0'
        : tool.status === 'pending'
          ? 'awaiting approval'
          : tool.status === 'running'
            ? 'running'
            : 'command failed',
      tone: tool.status === 'success'
        ? 'success'
        : tool.status === 'pending'
          ? 'warning'
          : tool.status === 'running'
            ? 'info'
            : 'warning',
    })

    return rows
  }

  rows.push({
    key: `${message.id}:tool:arguments`,
    text: tool.arguments,
    tone: 'dimText',
  })

  if (tool.output) {
    rows.push({
      key: `${message.id}:tool:output`,
      text: stripAnsi(tool.output.slice(0, 500)),
      tone: tool.status === 'error' ? 'warning' : 'text',
    })
  }

  return rows
}

function estimateToolLines(message: Message, width: number): number {
  const contentWidth = Math.max(MIN_CONTENT_WIDTH, width - 6)
  const tool = message.toolCall
  if (!tool) {
    return 2
  }

  let lines = 2
  for (const row of buildToolReaderLines(message)) {
    lines += countWrappedLines(row.text, contentWidth)
  }
  return lines
}

export function buildTranscriptReaderRows(
  messages: Message[],
  width: number,
): TranscriptReaderRow[] {
  const rows: TranscriptReaderRow[] = []
  const contentWidth = Math.max(MIN_CONTENT_WIDTH, width - 2)

  for (const message of messages) {
    const normalizedContent = (message.content ?? '').normalize('NFC')
    const isUser = message.role === 'user'
    const isSystem = message.role === 'system'
    const isTool = message.role === 'tool'
    const label = isSystem
      ? 'System'
      : isUser
        ? 'You'
        : isTool
          ? 'Tool'
          : 'Assistant'
    const labelTone: TranscriptReaderRow['tone'] = isSystem
      ? 'warning'
      : isUser
        ? 'primary'
        : isTool
          ? 'info'
          : 'success'

    rows.push({
      key: `${message.id}:header`,
      text: label,
      tone: labelTone,
    })

    if (isSystem) {
      pushWrappedRows(
        rows,
        `${message.id}:body`,
        normalizedContent,
        'dimText',
        contentWidth,
        '  ',
      )
    } else if (isUser) {
      if (normalizedContent) {
        pushWrappedRows(
          rows,
          `${message.id}:body`,
          normalizedContent,
          'text',
          contentWidth,
          '  ',
        )
      }
      if ((message.attachments?.length ?? 0) > 0) {
        rows.push({
          key: `${message.id}:attachments:label`,
          text: '  Attached files',
          tone: 'dimText',
        })
        message.attachments?.forEach((attachment, index) => {
          pushWrappedRows(
            rows,
            `${message.id}:attachments:${index}`,
            `- ${attachment.path}`,
            'info',
            contentWidth,
            '  ',
          )
        })
      }
    } else if (isTool) {
      buildToolReaderLines(message).forEach((row) => {
        pushWrappedRows(
          rows,
          row.key,
          row.text,
          row.tone,
          contentWidth,
          '  ',
        )
      })
    } else {
      pushWrappedRows(
        rows,
        `${message.id}:body`,
        stripAnsi(renderMarkdown(normalizedContent, { width: contentWidth })),
        'text',
        contentWidth,
        '  ',
      )
      if ((message.citations?.length ?? 0) > 0) {
        rows.push({
          key: `${message.id}:sources:label`,
          text: '  Sources used',
          tone: 'info',
        })
        message.citations?.forEach((citation, index) => {
          pushWrappedRows(
            rows,
            `${message.id}:citation:${index}:label`,
            `- ${citation.citationLabel}`,
            'info',
            contentWidth,
            '  ',
          )
          pushWrappedRows(
            rows,
            `${message.id}:citation:${index}:snippet`,
            citation.snippet,
            'dimText',
            contentWidth,
            '  ',
          )
        })
      }
    }

    rows.push({
      key: `${message.id}:gap`,
      text: '',
      tone: 'text',
    })
  }

  return rows
}

function estimateStreamingReserve(
  isStreaming: boolean,
  currentMessage: string,
  width: number,
  height: number,
): number {
  if (!isStreaming) {
    return 0
  }

  const desired = currentMessage.trim()
    ? 2 + countWrappedLines(renderMarkdown(currentMessage, { width: width - 2 }), width - 2)
    : 2

  return Math.min(
    Math.max(MIN_STREAMING_RESERVE, desired),
    Math.max(MIN_STREAMING_RESERVE, Math.floor(height * 0.45)),
  )
}

export function estimateMessageLines(message: Message, width: number): number {
  const contentWidth = Math.max(MIN_CONTENT_WIDTH, width - 4)

  if (message.role === 'tool') {
    return estimateToolLines(message, width)
  }

  let lines = 2

  if (message.role === 'assistant') {
    lines += countWrappedLines(renderMarkdown(message.content, { width: contentWidth }), contentWidth)
    if ((message.citations?.length ?? 0) > 0) {
      lines += 1
      for (const citation of message.citations ?? []) {
        lines += countWrappedLines(`- ${citation.citationLabel}`, contentWidth)
        lines += countWrappedLines(citation.snippet, contentWidth)
      }
    }
    return lines
  }

  lines += countWrappedLines(message.content, contentWidth)

  if (message.role === 'user' && (message.attachments?.length ?? 0) > 0) {
    lines += 1 + (message.attachments?.length ?? 0)
  }

  return lines
}

export function isOperationalSystemMessage(message: Message): boolean {
  return message.role === 'system'
    && OPERATIONAL_SYSTEM_MESSAGE_PATTERNS.some((pattern) => pattern.test(message.content))
}

export function filterTranscriptMessages(
  messages: Message[],
  clearedAt: number | null = null,
): Message[] {
  return messages.filter((message) => (
    !isOperationalSystemMessage(message)
    && (clearedAt === null || message.timestamp >= clearedAt)
  ))
}

export function buildTranscriptViewport(
  input: TranscriptViewportInput,
): TranscriptViewport {
  const reservedFooterLines =
    estimateStreamingReserve(
      input.isStreaming,
      input.currentMessage,
      input.width,
      input.height,
    )
    + (input.isThinking ? 1 : 0)
    + (input.error ? countWrappedLines(`Error: ${input.error}`, input.width - 2) : 0)

  const availableLines = Math.max(0, input.height - reservedFooterLines)
  const estimates = input.messages.map((message) => estimateMessageLines(message, input.width))
  const maxScrollOffset = Math.max(0, input.messages.length - 1)
  const scrollOffset = Math.max(
    0,
    Math.min(
      maxScrollOffset,
      Math.floor(input.scrollOffset ?? 0),
    ),
  )
  const endExclusive = Math.max(0, input.messages.length - scrollOffset)

  let usedLines = 0
  let selectedStart = endExclusive

  for (let index = endExclusive - 1; index >= 0; index -= 1) {
    const nextLines = estimates[index]
    if (selectedStart === endExclusive || usedLines + nextLines <= availableLines) {
      usedLines += nextLines
      selectedStart = index
      continue
    }
    break
  }

  if (selectedStart === endExclusive && endExclusive > 0 && availableLines > 0) {
    selectedStart = endExclusive - 1
    usedLines = estimates[selectedStart]
  }

  let messages = selectedStart < endExclusive
    ? input.messages.slice(selectedStart, endExclusive)
    : []
  let hiddenAboveCount = selectedStart
  const hiddenBelowCount = Math.max(0, input.messages.length - endExclusive)

  const indicatorLines = () => (
    (hiddenAboveCount > 0 ? 1 : 0)
    + (hiddenBelowCount > 0 ? 1 : 0)
  )

  while (
    indicatorLines() > 0
    && messages.length > 1
    && usedLines + indicatorLines() > availableLines
  ) {
    usedLines -= estimateMessageLines(messages[0], input.width)
    messages = messages.slice(1)
    hiddenAboveCount += 1
  }

  return {
    messages,
    hiddenCount: hiddenAboveCount,
    hiddenAboveCount,
    hiddenBelowCount,
    usedLines,
    reservedFooterLines,
  }
}

export function buildTranscriptReaderViewport(
  input: Omit<TranscriptViewportInput, 'messages'> & {
    rows: TranscriptReaderRow[]
  },
): TranscriptReaderViewport {
  const reservedFooterLines =
    estimateStreamingReserve(
      input.isStreaming,
      input.currentMessage,
      input.width,
      input.height,
    )
    + (input.isThinking ? 1 : 0)
    + (input.error ? countWrappedLines(`Error: ${input.error}`, input.width - 2) : 0)

  const totalRows = input.rows.length
  const availableLines = Math.max(0, input.height - reservedFooterLines)
  const maxScrollOffset = Math.max(0, totalRows - availableLines)
  const scrollOffset = Math.max(
    0,
    Math.min(
      maxScrollOffset,
      Math.floor(input.scrollOffset ?? 0),
    ),
  )
  const visibleEnd = Math.max(0, totalRows - scrollOffset)
  let hiddenBelowCount = Math.max(0, totalRows - visibleEnd)
  let rowBudget = availableLines
  let visibleStart = Math.max(0, visibleEnd - rowBudget)
  let hiddenAboveCount = visibleStart
  rowBudget = Math.max(
    hiddenAboveCount > 0 || hiddenBelowCount > 0 ? 1 : 0,
    availableLines
      - (hiddenAboveCount > 0 ? 1 : 0)
      - (hiddenBelowCount > 0 ? 1 : 0),
  )
  visibleStart = Math.max(0, visibleEnd - rowBudget)
  hiddenAboveCount = visibleStart
  hiddenBelowCount = Math.max(0, totalRows - visibleEnd)
  const rows = input.rows.slice(visibleStart, visibleEnd)

  return {
    rows,
    hiddenAboveCount,
    hiddenBelowCount,
    usedLines: rows.length,
    reservedFooterLines,
  }
}

export function tailLines(text: string, maxLines: number): {
  visible: string[]
  omitted: number
} {
  if (maxLines <= 0) {
    return { visible: [], omitted: text.length > 0 ? text.split('\n').length : 0 }
  }

  const lines = text.split('\n')
  if (lines.length <= maxLines) {
    return { visible: lines, omitted: 0 }
  }

  return {
    visible: lines.slice(-maxLines),
    omitted: lines.length - maxLines,
  }
}
