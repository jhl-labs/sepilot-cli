const DEFAULT_MAX_CONTEXT_LINES = 200
const DEFAULT_MAX_CONTEXT_CHARS = 12000
const FILESYSTEM_MAX_CONTEXT_CHARS = 8000
const TERMINAL_MAX_CONTEXT_CHARS = 4000
// Web content is denser than a `cat` of one source file — a benchmark page or
// docs page is mostly the body the user wants to read, not noise. A tight cap
// like 12k forces aggressive truncation, the model sees the "[truncated]"
// marker, and falls back to training-data answers (which is how the qwen3.5
// vs gemini3.5 hallucination on mobile happened). Give webfetch more room.
const WEBFETCH_MAX_CONTEXT_CHARS = 20000
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

  // A plain head/tail summary hides every middle file in a multi-file commit.
  // Keep the structural map (exact paths and hunk ranges) so the next focused
  // read can only use observed paths, then spend the remainder on
  // representative patch content. Tool output remains ordinary untrusted
  // evidence; only its compaction shape changes.
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

/**
 * Source reads are range evidence. A generic head/tail summary is actively
 * misleading here: the observation cache would remember the requested range
 * even though the model never saw the omitted middle. Keep one contiguous
 * prefix and expose its exact absolute line range so continuation and cache
 * coverage can agree on what was actually visible to the model.
 */
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

/**
 * Tool adapters append daemon-owned recovery and hint records after their raw
 * diagnostic payload. Those records are control-plane evidence: dropping a
 * recovery marker while retaining only part of its prose can turn an
 * executable capability transition into a false blocker. Preserve the
 * bounded records atomically for every verbose tool result, independent of
 * the command, repository, language, provider, or recovery scenario.
 */
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

export function summarizeToolOutputForAgentContext(
  toolName: string,
  output: string,
): string {
  const normalized = output.trim()
  if (!normalized) {
    return output
  }

  const lines = normalized.split('\n')
  // terminal.run output can be verbose, so it gets a tighter byte budget —
  // but only *large* output should be summarized. Small command output
  // (e.g. `cloc --json` for a repo) must pass through verbatim; otherwise
  // the "[…summarized…]" header and "…omitted lines…" marker make the model
  // believe the result was truncated and it keeps re-running the command
  // instead of answering with the data it already has.
  const maxContextChars = toolName === 'terminal.run'
    ? TERMINAL_MAX_CONTEXT_CHARS
    : toolName.startsWith('fs.')
      ? FILESYSTEM_MAX_CONTEXT_CHARS
      : toolName === 'webfetch'
        ? WEBFETCH_MAX_CONTEXT_CHARS
        : DEFAULT_MAX_CONTEXT_CHARS
  const shouldCompact = (
    lines.length > DEFAULT_MAX_CONTEXT_LINES
    || normalized.length > maxContextChars
  )

  if (!shouldCompact) {
    return normalized
  }

  const preserveControlDirectives = (summary: string): string =>
    preserveToolControlDirectives(lines, summary, maxContextChars)

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
