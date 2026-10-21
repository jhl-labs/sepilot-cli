const DEFAULT_MAX_CONTEXT_LINES = 200
const DEFAULT_MAX_CONTEXT_CHARS = 12000
const FILESYSTEM_MAX_CONTEXT_CHARS = 8000
const TERMINAL_MAX_CONTEXT_CHARS = 4000
const WEBFETCH_MAX_CONTEXT_CHARS = 20000
const BROWSER_SNAPSHOT_MAX_CONTEXT_CHARS = 24000
const CONTEXT_HEAD_LINES = 32
const CONTEXT_TAIL_LINES = 16
const TERMINAL_DIAGNOSTIC_HEAD_LINES = 12
const TERMINAL_DIAGNOSTIC_TAIL_LINES = 10
const TERMINAL_DIAGNOSTIC_MAX_LINES = 48
const FS_READ_MAX_VISIBLE_LINES = 200
const TOOL_CONTROL_DIRECTIVE_MAX_CHARS = 1400

const ANSI_ESCAPE_PATTERN = /\u001B(?:\[[0-?]*[ -/]*[@-~]|\][^\u0007]*(?:\u0007|\u001B\\))/gu

function isTerminalDiagnosticLine(line: string): boolean {
  const plain = line.replace(ANSI_ESCAPE_PATTERN, '')
  return (
    /^\s*(?:●|×|✗|❌|❯)\s+/u.test(plain)
    || /(?:^|\s)(?:fail(?:ed|ure)?|error|fatal|panic|exception|assertionerror|not ok)(?:\s|:|\[|$)/iu.test(plain)
    || /\b[A-Za-z][A-Za-z0-9_]*(?:Error|Exception)\b/u.test(plain)
    || /\bexpected\b.*\b(?:received|got|to (?:be|equal|have))\b/iu.test(plain)
    || /\b(?:tests?|suites?|checks?)\b.*\bfailed\b/iu.test(plain)
  )
}

function summarizeTerminalDiagnostics(
  lines: string[],
  normalized: string,
  maxChars: number,
): string {
  const diagnostics = lines
    .filter(isTerminalDiagnosticLine)
    .filter((line, index, all) => index === 0 || line !== all[index - 1])
    .slice(0, TERMINAL_DIAGNOSTIC_MAX_LINES)
  if (diagnostics.length === 0) return ''

  const header = `[terminal.run output summarized for agent context: ${lines.length} lines, ${normalized.length} chars]`
  const diagnosticBudget = Math.max(1_200, Math.floor(maxChars * 0.58))
  const diagnosticSection = truncateByChars(diagnostics.join('\n'), diagnosticBudget)
  const representativeBudget = Math.max(700, maxChars - diagnosticBudget - header.length - 180)
  const head = lines.slice(0, TERMINAL_DIAGNOSTIC_HEAD_LINES)
  const tail = lines.slice(-TERMINAL_DIAGNOSTIC_TAIL_LINES)
  const omittedLines = Math.max(0, lines.length - head.length - tail.length)
  const representative = truncateByChars([
    ...head,
    ...(omittedLines > 0
      ? [`...[${omittedLines} omitted line${omittedLines === 1 ? '' : 's'}; diagnostic highlights preserved above]...`]
      : []),
    ...tail,
  ].join('\n'), representativeBudget)

  return truncateByChars([
    header,
    '[Diagnostic highlights preserved from the full output]',
    diagnosticSection,
    '[Representative beginning/end]',
    representative,
  ].join('\n'), maxChars)
}

function summarizeGitDiffStructure(
  lines: string[],
  normalized: string,
  maxChars: number,
): string {
  const structuralLines = lines.filter((line) => (
    line.startsWith('diff --git ')
    || line.startsWith('index ')
    || line.startsWith('--- ')
    || line.startsWith('+++ ')
    || line.startsWith('@@ ')
  ))
  const changedFileCount = structuralLines.filter((line) => line.startsWith('diff --git ')).length
  if (changedFileCount < 2) return ''

  const structureBudget = Math.max(1_000, Math.floor(maxChars * 0.55))
  const representativeBudget = Math.max(1_000, maxChars - structureBudget - 320)
  const structure = truncateByChars(structuralLines.join('\n'), structureBudget)
  const representativeLines = [
    ...lines.slice(0, CONTEXT_HEAD_LINES),
    ...lines.slice(-CONTEXT_TAIL_LINES),
  ]
  const representative = truncateByChars(
    representativeLines.join('\n'),
    representativeBudget,
  )
  return truncateByChars([
    `[git.diff output summarized for agent context: ${lines.length} lines, ${normalized.length} chars, ${changedFileCount} changed-file sections]`,
    '[Changed files and hunk locations]',
    structure,
    '[Representative patch lines]',
    representative,
  ].join('\n'), maxChars)
}

function summarizeFsReadContiguousPage(
  lines: string[],
  normalized: string,
  maxChars: number,
): string {
  const firstMatch = lines[0]?.match(/^\s*(\d+)\t/u)
  if (!firstMatch) return ''

  const startLine = Number(firstMatch[1])
  if (!Number.isSafeInteger(startLine) || startLine < 1) return ''

  const bodyBudget = Math.max(1_000, maxChars - 320)
  const visible: string[] = []
  let expectedLine = startLine
  let visibleChars = 0
  for (const line of lines) {
    if (visible.length >= FS_READ_MAX_VISIBLE_LINES) break
    const match = line.match(/^\s*(\d+)\t/u)
    if (!match || Number(match[1]) !== expectedLine) break
    const addedChars = line.length + (visible.length > 0 ? 1 : 0)
    if (visibleChars + addedChars > bodyBudget) break
    visible.push(line)
    visibleChars += addedChars
    expectedLine += 1
  }
  if (visible.length === 0) return ''

  const endLine = startLine + visible.length - 1
  const nextOffset = endLine + 1
  const header = [
    `[fs.read agent-visible range: lines ${startLine}-${endLine}; continue with offset=${nextOffset}; original result ${lines.length} lines, ${normalized.length} chars]`,
    'Only this contiguous range is reusable evidence; omitted source lines were not shown to the model.',
  ].join('\n')
  return `${header}\n${visible.join('\n')}`
}

function truncateByChars(text: string, maxChars: number): string {
  if (text.length <= maxChars) {
    return text
  }

  const headBudget = Math.max(0, Math.floor(maxChars * 0.7))
  const tailBudget = Math.max(0, maxChars - headBudget - 32)
  const head = text.slice(0, headBudget).trimEnd()
  const tail = tailBudget > 0 ? text.slice(-tailBudget).trimStart() : ''
  const marker = '\n...[truncated for agent context]...\n'
  return `${head}${marker}${tail}`.trim()
}

function preserveToolControlDirectives(
  lines: string[],
  summary: string,
  maxChars: number,
): string {
  const directives = lines
    .filter((line) => /^\[(?:recovery:[a-z0-9][a-z0-9-]*|hint)\]\s*/iu.test(line))
    .filter((line, index, all) => all.indexOf(line) === index)
  if (directives.length === 0) return summary

  const directiveBlock = truncateByChars([
    '[Tool control directives preserved from the full output]',
    ...directives,
  ].join('\n'), Math.min(TOOL_CONTROL_DIRECTIVE_MAX_CHARS, maxChars))
  const summaryBudget = Math.max(800, maxChars - directiveBlock.length - 1)
  const combined = `${directiveBlock}\n${truncateByChars(summary, summaryBudget)}`
  return combined.length <= maxChars
    ? combined
    : combined.slice(0, maxChars).trimEnd()
}

function compactEvidenceText(text: string, limit: number): string {
  if (text.length <= limit) return text
  const marker = '\n... [middle omitted; tail preserved] ...\n'
  if (limit <= marker.length) return text.slice(0, limit)
  const available = limit - marker.length
  const headLength = Math.floor(available / 2)
  return `${text.slice(0, headLength)}${marker}${text.slice(-(available - headLength))}`
}

export function summarizeToolOutputForEvidence(
  toolName: string,
  output: string,
  maxChars = 900,
): string {
  if (!Number.isFinite(maxChars) || maxChars < 0) {
    throw new RangeError('Evidence budget must be a finite non-negative number')
  }
  const limit = Math.floor(maxChars)
  const normalized = output.trim()
  if (normalized.length <= limit) return normalized
  if (toolName !== 'terminal.run' || limit < 256) {
    return compactEvidenceText(normalized, limit)
  }
  const diagnostics = [...new Set(normalized.split('\n').filter(isTerminalDiagnosticLine))]
  if (diagnostics.length === 0) return compactEvidenceText(normalized, limit)

  const header = '[Diagnostic highlights — excerpts, not a verdict]\n'
  const separator = '\n[Beginning/end excerpts]\n'
  const diagnosticBudget = Math.floor((limit - header.length - separator.length) * 0.6)
  const highlights = compactEvidenceText(diagnostics.join('\n'), diagnosticBudget)
  const remainder = limit - header.length - highlights.length - separator.length
  return `${header}${highlights}${separator}${compactEvidenceText(normalized, remainder)}`
}

function summarizeHeadFirst(
  toolName: string,
  lines: string[],
  normalized: string,
  maxContextChars: number,
): string {
  const header = `[${toolName} output summarized for agent context: ${lines.length} lines, ${normalized.length} chars; shown from the top, the rest is readable with tool.output.read]`
  const kept: string[] = []
  let used = header.length + 1
  for (const line of lines) {
    const cost = line.length + 1
    if (used + cost > maxContextChars - 64) break
    kept.push(line)
    used += cost
  }
  const omitted = lines.length - kept.length
  return [header, ...kept, ...(omitted > 0 ? [`...[${omitted} omitted line${omitted === 1 ? '' : 's'} at the end]...`] : [])].join('\n')
}

export function summarizeToolOutputForAgentContext(
  toolName: string,
  output: string,
  toolCallId?: string,
): string {
  const normalized = output.trim()
  if (!normalized) {
    return output
  }

  const lines = normalized.split('\n')
  const maxContextChars = toolName === 'terminal.run'
    ? TERMINAL_MAX_CONTEXT_CHARS
    : toolName.startsWith('fs.')
      ? FILESYSTEM_MAX_CONTEXT_CHARS
      : toolName === 'webfetch'
        ? WEBFETCH_MAX_CONTEXT_CHARS
        : toolName === 'browser.remote_snapshot'
          ? BROWSER_SNAPSHOT_MAX_CONTEXT_CHARS
          : DEFAULT_MAX_CONTEXT_CHARS
  const shouldCompact = (
    (!['tool.output.read', 'webfetch', 'browser.extract', 'browser.remote_snapshot'].includes(toolName) && lines.length > DEFAULT_MAX_CONTEXT_LINES)
    || normalized.length > maxContextChars
  )

  if (!shouldCompact) {
    return normalized
  }

  const reference = toolCallId && /^[a-zA-Z0-9_-]{1,128}$/.test(toolCallId)
    ? `[Captured output reference: ${toolCallId}]\n`
    : ''
  const preserveControlDirectives = (summary: string): string =>
    reference + preserveToolControlDirectives(lines, summary, maxContextChars)

  if (toolName === 'git.diff') {
    const structuredDiff = summarizeGitDiffStructure(lines, normalized, maxContextChars)
    if (structuredDiff) return preserveControlDirectives(structuredDiff)
  }
  if (toolName === 'fs.read') {
    const contiguousPage = summarizeFsReadContiguousPage(lines, normalized, maxContextChars)
    if (contiguousPage) return preserveControlDirectives(contiguousPage)
  }
  if (toolName === 'terminal.run') {
    const diagnosticSummary = summarizeTerminalDiagnostics(lines, normalized, maxContextChars)
    if (diagnosticSummary) return preserveControlDirectives(diagnosticSummary)
  }

  if (toolName === 'browser.remote_snapshot') {
    const headOnly = summarizeHeadFirst(toolName, lines, normalized, maxContextChars)
    return preserveControlDirectives(headOnly)
  }

  const head = lines.slice(0, CONTEXT_HEAD_LINES)
  const tail = lines.length > CONTEXT_HEAD_LINES
    ? lines.slice(-CONTEXT_TAIL_LINES)
    : []
  const omittedLines = Math.max(0, lines.length - head.length - tail.length)
  const summary = [
    `[${toolName} output summarized for agent context: ${lines.length} lines, ${normalized.length} chars]`,
    ...head,
    ...(omittedLines > 0
      ? [`...[${omittedLines} omitted line${omittedLines === 1 ? '' : 's'}]...`]
      : []),
    ...tail,
  ].join('\n')

  return preserveControlDirectives(truncateByChars(summary, maxContextChars))
}
