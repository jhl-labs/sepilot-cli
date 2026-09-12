import type { Message } from '@sepilotd/core'
import { clearlyReportsLatestPermanentFailure } from './outcome-recovery.js'
import { TOOL_RESULT_STATUS_METADATA_KEY } from './memory-write-completion.js'
import { currentTurnMessages } from './policy-failure.js'

export const MAX_EMPTY_FINAL_REPAIRS = 2
export const MAX_INTERIM_PROGRESS_REPAIRS = 4
// The stem is a presentation contract, not semantic work. One provider turn
// is enough to repair it; after that, preserve the latest usable candidate
// instead of multiplying final-synthesis calls after tool work is complete.
export const MAX_MISSING_ANSWER_PROTOCOL_REPAIRS = 1
export const MAX_UNSUPPORTED_CITATION_REPAIRS = 2
const MAX_EMPTY_FINAL_TOOL_LINES = 3
const MAX_EMPTY_FINAL_LINE_CHARS = 180
const FILE_EDIT_TOOL_NAMES = new Set(['fs.write', 'fs.append', 'fs.edit', 'apply_patch'])
const PLANNER_FENCE_PATTERN = /\n?```(?:planner|json:planner)\s*\n[\s\S]*?\n?```\s*/gi
const JSON_FENCE_PATTERN = /\n?```json\s*\n([\s\S]*?)\n?```\s*/gi
export const ANSWER_PROTOCOL_SYSTEM_PROMPT = [
  'Final-answer protocol: when you send user-visible assistant text instead of a tool call, start the response with `ANSWER:`.',
  'These stems are literal runtime markers: never translate them. After the marker, follow the user-requested body format directly, without another answer label or preamble.',
  'If you cannot provide the finished answer yet and no tool call is being made in this reply, start with `INCOMPLETE:` and state the concrete blocker or next required action.',
  'If the active specialist prompt explicitly requires another terminal stem such as `VERIFIED:` or `UNVERIFIED:`, use that specialist stem instead of `ANSWER:`.',
  'Do not send progress-only prose without one of these stems.',
  'If you intend to inspect, search, read, write, or run something, include the tool call in the same assistant response; do not only say what you will do next.',
  'When the tool-use progress contract is present, its structured action annotation may accompany that same tool response; do not send the annotation as a separate prose-only turn.',
].join(' ')
const ANSWER_PROTOCOL_MARKER = 'Final-answer protocol:'
const ANSWER_PROTOCOL_STEM_PATTERN = /^[ \t]*(ANSWER|INCOMPLETE):[ \t]*(.*)$/i
// Completion-gate bookkeeping lines the model is instructed to emit before the
// final stem (`CRITERION AC1: MET`). They are loop-control protocol, never part
// of the user-facing answer, so the stem scanner skips them and the final
// sanitizers drop them outside code fences.
const CRITERION_PROTOCOL_LINE_PATTERN = /^[ \t]*CRITERION\s+\S+:\s*(?:MET|UNMET)\b.*$/i
const APPROXIMATE_LINE_REFERENCE_PATTERN =
  /(?:\bline(?:s)?\s*~\s*\d+\b|\b~\s*line(?:s)?\s*\d+\b|:[ \t]*~\s*\d+\b|\|\s*~\s*\d+\s*\||\b(?:around|approximately|roughly)\s+line(?:s)?\s+\d+\b)/i
const BLOCKED_SHELL_WRAPPER_PATTERN = /Tool terminal\.run blocked: Executable '(?:bash|dash|sh|zsh|fish|csh|tcsh)' is blocked/i
const SHELL_WRAPPER_BLOCK_CONTEXT_PATTERNS = [
  BLOCKED_SHELL_WRAPPER_PATTERN,
  /(?:sh|bash|dash|zsh|fish|csh|tcsh)\s*\/\s*(?:sh|bash|dash|zsh|fish|csh|tcsh).*차단/i,
  /(?:sh|bash|dash|zsh|fish|csh|tcsh).*차단(?:되었|되었습|되었는|되었습니다|됐|됨)/i,
  /shell wrapper/i,
]

type AnswerProtocolStem = 'ANSWER' | 'INCOMPLETE'

interface FoundAnswerProtocolStem {
  stem: AnswerProtocolStem
  body: string
  lineIndex: number
  lines: string[]
}

interface MarkdownFence {
  character: '`' | '~'
  length: number
}

function findOpeningMarkdownFence(line: string): MarkdownFence | null {
  const match = line.match(/^ {0,3}(`{3,}|~{3,})/)
  if (!match?.[1]) return null
  return {
    character: match[1][0] as MarkdownFence['character'],
    length: match[1].length,
  }
}

function isClosingMarkdownFence(line: string, fence: MarkdownFence): boolean {
  const match = line.match(/^ {0,3}(`+|~+)[ \t]*$/)
  return Boolean(
    match?.[1]
    && match[1][0] === fence.character
    && match[1].length >= fence.length,
  )
}

function isMarkdownIndentedCodeLine(line: string): boolean {
  return /^(?: {4,}| {0,3}\t)/.test(line)
}

function trimPresentationContent(content: string): string {
  const withoutTrailingWhitespace = content.trimEnd()
  const withoutLeadingBlankLines = withoutTrailingWhitespace.replace(/^(?:[ \t]*\r?\n)+/, '')
  return isMarkdownIndentedCodeLine(withoutLeadingBlankLines)
    ? withoutLeadingBlankLines
    : withoutLeadingBlankLines.trimStart()
}

function stripRepeatedProtocolStem(value: string, stem: AnswerProtocolStem): string {
  let body = value
  for (;;) {
    const marker = /^[ \\t]*([A-Z]+):[ \\t]*/i.exec(body)
    if (!marker || marker[1]?.toUpperCase() !== stem) break
    body = body.slice(marker[0].length)
  }
  return body
}

function findAnswerProtocolStem(content: string): FoundAnswerProtocolStem | null {
  const lines = content.split(/\r?\n/)
  const candidates: FoundAnswerProtocolStem[] = []
  let fence: MarkdownFence | null = null
  let hasSubstantivePreamble = false

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i]!
    if (fence) {
      if (isClosingMarkdownFence(line, fence)) fence = null
      continue
    }

    const openingFence = findOpeningMarkdownFence(line)
    if (openingFence) {
      fence = openingFence
      if (candidates.length === 0) hasSubstantivePreamble = true
      continue
    }

    if (line.trim().length === 0) continue
    if (CRITERION_PROTOCOL_LINE_PATTERN.test(line)) continue
    if (isMarkdownIndentedCodeLine(line)) {
      if (candidates.length === 0) hasSubstantivePreamble = true
      continue
    }

    const match = line.match(ANSWER_PROTOCOL_STEM_PATTERN)
    if (!match) {
      if (candidates.length === 0) hasSubstantivePreamble = true
      continue
    }

    const stem = match[1]!.toUpperCase() as AnswerProtocolStem
    candidates.push({
      stem,
      body: match[2] ?? '',
      lineIndex: i,
      lines,
    })
  }

  if (candidates.length === 0) return null

  // A protocol block at the beginning owns the whole response, including any
  // later duplicated marker. When prose or a rendered preview precedes the
  // protocol, the last independent marker is the terminal handoff.
  return hasSubstantivePreamble ? candidates.at(-1)! : candidates[0]!
}

function stripProtocolStemsOutsideCodeFences(lines: string[]): string[] {
  let fence: MarkdownFence | null = null
  const result: string[] = []
  for (const line of lines) {
    if (fence) {
      if (isClosingMarkdownFence(line, fence)) fence = null
      result.push(line)
      continue
    }

    const openingFence = findOpeningMarkdownFence(line)
    if (openingFence) {
      fence = openingFence
      result.push(line)
      continue
    }
    if (CRITERION_PROTOCOL_LINE_PATTERN.test(line) && !isMarkdownIndentedCodeLine(line)) {
      continue
    }

    const match = line.match(ANSWER_PROTOCOL_STEM_PATTERN)
    if (!match || isMarkdownIndentedCodeLine(line)) {
      result.push(line)
      continue
    }
    const stem = match[1]!.toUpperCase() as AnswerProtocolStem
    result.push(stripRepeatedProtocolStem(match[2] ?? '', stem))
  }
  return result
}

export function appendAnswerProtocolSystemPrompt(systemPrompt: string | undefined): string {
  const trimmed = systemPrompt?.trim() ?? ''
  if (trimmed.includes(ANSWER_PROTOCOL_MARKER)) {
    return trimmed
  }

  return [trimmed, ANSWER_PROTOCOL_SYSTEM_PROMPT].filter(Boolean).join('\n\n')
}

export function hasFinalAnswerStem(content: string): boolean {
  return findAnswerProtocolStem(content)?.stem === 'ANSWER'
}

export function hasIncompleteAnswerStem(content: string): boolean {
  return findAnswerProtocolStem(content)?.stem === 'INCOMPLETE'
}

export function hasAnyAnswerProtocolStem(content: string): boolean {
  return findAnswerProtocolStem(content) !== null
}

export function stripAnswerProtocolStem(content: string): string {
  const found = findAnswerProtocolStem(content)
  if (!found) {
    return trimPresentationContent(
      stripProtocolStemsOutsideCodeFences(content.split(/\r?\n/)).join('\n'),
    )
  }

  return stripFoundAnswerProtocolStem(found)
}

export function stripFinalAnswerStem(content: string): string {
  const found = findAnswerProtocolStem(content)
  if (!found) return trimPresentationContent(content)

  const body = stripFoundAnswerProtocolStem(found)
  return found.stem === 'ANSWER'
    ? body
    : ['INCOMPLETE:', body].filter(Boolean).join(' ')
}

function stripFoundAnswerProtocolStem(found: FoundAnswerProtocolStem): string {
  const rest = stripProtocolStemsOutsideCodeFences(found.lines.slice(found.lineIndex + 1))
  return trimPresentationContent(
    [stripRepeatedProtocolStem(found.body, found.stem), ...rest].join('\n'),
  )
}

function isPlannerJsonPayload(value: string): boolean {
  try {
    const parsed = JSON.parse(value) as unknown
    return Boolean(
      parsed
      && typeof parsed === 'object'
      && !Array.isArray(parsed)
      && (
        'taskSummary' in parsed
        || 'currentSubtaskId' in parsed
        || (
          'plan' in parsed
          && Array.isArray((parsed as { plan?: unknown }).plan)
        )
      ),
    )
  } catch {
    return false
  }
}

export function stripInternalPlannerBlocks(content: string): string {
  return content
    .replace(PLANNER_FENCE_PATTERN, '\n')
    .replace(JSON_FENCE_PATTERN, (match, body: string) => (
      isPlannerJsonPayload(body.trim()) ? '\n' : match
    ))
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

export function isLikelyInterimProgressUpdate(content: string): boolean {
  return hasIncompleteAnswerStem(content)
}

export function shouldRepairInterimProgressReply(options: {
  content: string
  messages: Message[]
  repairedCount: number
  isLastIteration?: boolean
  maxRepairs?: number
}): boolean {
  const {
    content,
    repairedCount,
    isLastIteration = false,
    maxRepairs = MAX_INTERIM_PROGRESS_REPAIRS,
  } = options

  return !isLastIteration
    && repairedCount < maxRepairs
    && isLikelyInterimProgressUpdate(content)
    && !clearlyReportsLatestPermanentFailure(options.messages, content)
}

export function shouldRepairMissingAnswerProtocolReply(options: {
  content: string
  repairedCount: number
  strict?: boolean
  isLastIteration?: boolean
  maxRepairs?: number
}): boolean {
  const {
    content,
    repairedCount,
    strict = false,
    isLastIteration = false,
    maxRepairs = MAX_MISSING_ANSWER_PROTOCOL_REPAIRS,
  } = options

  return strict
    && !isLastIteration
    && repairedCount < maxRepairs
    && content.trim().length > 0
    && !hasAnyAnswerProtocolStem(content)
}

export function hasUnsupportedCitationPattern(content: string): boolean {
  return APPROXIMATE_LINE_REFERENCE_PATTERN.test(content)
}

// Exact `path/to/file.ext:123` citations. Requires a real file extension so we
// do not match prose like "step 3:12" or bare `foo:1`. Used to cross-check the
// cited files against the evidence ledger's observed reads.
const EXACT_FILE_LINE_CITATION_PATTERN =
  /\b([A-Za-z0-9_][A-Za-z0-9_./-]*\.[A-Za-z0-9]{1,10}):(\d+)\b/g

export function findExactFileLineCitations(content: string): string[] {
  const paths = new Set<string>()
  for (const match of content.matchAll(EXACT_FILE_LINE_CITATION_PATTERN)) {
    if (match[1]) paths.add(match[1])
  }
  return [...paths]
}

export function stripUnsupportedCitationReferences(content: string): string {
  return content
    .replace(/\|\s*~\s*\d+\s*\|/g, '| Unknown |')
    .replace(/\bline(?:s)?\s*~\s*\d+\b/gi, 'line reference omitted')
    .replace(/\b~\s*line(?:s)?\s*\d+\b/gi, 'line reference omitted')
    .replace(/\b(?:around|approximately|roughly)\s+line(?:s)?\s+\d+\b/gi, 'with no verified line number')
    .replace(/:[ \t]*~\s*\d+\b/g, '')
}

export function shouldRepairUnsupportedCitationReply(options: {
  content: string
  repairedCount: number
  strict?: boolean
  isLastIteration?: boolean
  maxRepairs?: number
}): boolean {
  const {
    content,
    repairedCount,
    strict = false,
    isLastIteration = false,
    maxRepairs = MAX_UNSUPPORTED_CITATION_REPAIRS,
  } = options

  return strict
    && !isLastIteration
    && repairedCount < maxRepairs
    && hasUnsupportedCitationPattern(content)
}

export function buildUnsupportedCitationRepairMessage(): Message {
  return {
    role: 'system',
    content: [
      'Your previous assistant reply used approximate or unsupported line references.',
      'For this run, cite line numbers only when a tool result included line numbers, such as fs.search, fs.read NNN<TAB> prefixes, or terminal output from rg -n/nl -ba.',
      'fs.read with lineNumbers=false supports file/symbol citations, not exact or approximate line numbers.',
      'Either call a line-numbered tool now if line precision matters, or restate the answer with file/symbol evidence and no approximate line references.',
      'Start the final text with ANSWER: when complete.',
    ].join(' '),
  }
}

export function buildMissingAnswerProtocolRepairMessage(): Message {
  return {
    role: 'system',
    content: [
      'Your previous assistant reply did not follow the final-answer protocol.',
      'For this run, user-visible assistant text is accepted only when it starts with ANSWER: or INCOMPLETE:.',
      'If the active validator or reviewer prompt requires VERIFIED: or UNVERIFIED:, end with that required specialist stem instead.',
      'If you can satisfy the user request now, reply with ANSWER: followed by the complete answer.',
      'If the task is not finished, call the next required tool now; do not describe the next tool in prose.',
      'If no valid tool can continue, reply with INCOMPLETE: and the concrete blocker.',
      'Do not send untagged progress, planning, or next-step prose.',
    ].join(' '),
  }
}

export function buildInterimProgressRepairMessage(): Message {
  return buildInterimProgressRepairMessageForMessages([])
}

function buildInterimProgressRepairMessageForMessages(messages: Message[]): Message {
  const hasBlockedShellWrapperFailure = messages.some((message) => (
    SHELL_WRAPPER_BLOCK_CONTEXT_PATTERNS.some((pattern) => (
      pattern.test(extractMessageText(message))
    ))
  ))

  return {
    role: 'system',
    content: [
      'Your previous assistant reply used INCOMPLETE:, so it was not a completed answer.',
      'Continue the task now: either call the next required tool(s) or provide the',
      'finished answer if you already have enough information.',
      'If you provide the finished answer, start it with ANSWER:.',
      'If more repository search is required, use terminal.run with rg or use fs.search.',
      hasBlockedShellWrapperFailure
        ? 'The previous attempt used a blocked shell wrapper such as sh -c or bash -c. Do not retry with shell wrappers. Call terminal.run with a direct executable + args pair, or use fs.read when you only need to read a file.'
        : '',
    ].join(' '),
  }
}

export function buildInterimProgressRepairMessageWithContext(messages: Message[]): Message {
  return buildInterimProgressRepairMessageForMessages(messages)
}

function extractMessageText(message: Message): string {
  if (typeof message.content === 'string') {
    return message.content
  }

  return message.content
    .filter((part): part is { type: 'text'; text: string } => part.type === 'text')
    .map((part) => part.text)
    .join('\n')
}

function compactFallbackToolLine(text: string): string {
  const normalized = text.trim().replace(/\s+/g, ' ')
  if (!normalized) {
    return 'No output captured.'
  }

  return normalized.length > MAX_EMPTY_FINAL_LINE_CHARS
    ? `${normalized.slice(0, MAX_EMPTY_FINAL_LINE_CHARS - 1)}…`
    : normalized
}

function collectRecentToolOutputs(messages: Message[]): string[] {
  const latestUserIndex = messages.findLastIndex((message) => message.role === 'user')
  const currentTurnTools = messages
    .slice(latestUserIndex + 1)
    .filter((message) => message.role === 'tool')
  const successful = currentTurnTools.filter((message) => (
    message.metadata?.[TOOL_RESULT_STATUS_METADATA_KEY] === 'success'
  ))
  const selected = successful.length > 0 ? successful : currentTurnTools

  return selected
    .slice(-MAX_EMPTY_FINAL_TOOL_LINES)
    .map((message) => {
      const output = compactFallbackToolLine(extractMessageText(message))
      return message.name ? `${message.name}: ${output}` : output
    })
    .filter(Boolean)
}

export function isFileEditToolName(name: string): boolean {
  return FILE_EDIT_TOOL_NAMES.has(name)
}

export function shouldRepairEmptyFinalReply(options: {
  content: string
  repairedCount: number
  isLastIteration?: boolean
  maxRepairs?: number
}): boolean {
  const {
    content,
    repairedCount,
    isLastIteration = false,
    maxRepairs = MAX_EMPTY_FINAL_REPAIRS,
  } = options

  return !isLastIteration
    && repairedCount < maxRepairs
    && content.trim().length === 0
}

export function buildEmptyFinalRepairMessage(): Message {
  return {
    role: 'system',
    content: [
      'Your previous assistant reply was empty.',
      'Do not end the run with a blank response.',
      'Continue the task now: either call the next required tool(s) or provide',
      'the finished answer using the information already available.',
      'If the current run requires the ANSWER:/INCOMPLETE: protocol, follow it.',
      'If the user asked for a concrete action such as writing a file, continue',
      'that work instead of stopping early.',
    ].join(' '),
  }
}

export function buildBoundedEmptyFinalRepairMessage(): Message {
  return {
    role: 'system',
    content: [
      'Your previous tool-free final reply was empty.',
      'The evidence-gathering phase is closed; do not request or describe another tool call.',
      'Synthesize a concise, substantive answer from the successful tool evidence already present in this conversation.',
      'Separate verified findings from unavailable or failed optional checks.',
      'If the current run requires the ANSWER:/INCOMPLETE: protocol, follow it.',
    ].join(' '),
  }
}

export function buildInvalidToolResponseRepairMessage(options: {
  evidenceAvailable: boolean
  availableToolNames: readonly string[]
}): Message {
  const availableTools = options.availableToolNames.length > 0
    ? options.availableToolNames.join(', ')
    : '(none)'
  return {
    role: 'system',
    content: options.evidenceAvailable
      ? [
          'Your previous response attempted a malformed or unavailable tool call.',
          'Successful tool evidence is already available, so the evidence-gathering phase is now closed.',
          'Do not request another tool. Synthesize a concise, substantive final answer from the verified evidence already present.',
          'Separate verified findings from optional checks that were not run.',
        ].join(' ')
      : [
          'Your previous response attempted a malformed or unavailable tool call.',
          `Retry once using only a currently available tool (${availableTools}), or provide a final answer if no tool is needed.`,
          'Do not name or emit a tool that is absent from the available-tool list.',
        ].join(' '),
  }
}

const MAX_INTERIM_FALLBACK_BODY_CHARS = 6000

export function buildInterimProgressFallbackMessage(
  content: string,
  messages: Message[],
): string {
  const progressText = stripAnswerProtocolStem(content.trim())
  const recentToolOutputs = collectRecentToolOutputs(messages)
  const header = recentToolOutputs.length > 0
    ? 'INCOMPLETE: The run completed some tool work but ended with a progress-only reply instead of a finished answer.'
    : 'INCOMPLETE: The run ended with a progress-only reply instead of a finished answer.'

  // A multi-line progress body is often a substantive draft (e.g. the
  // completion gate's explicitly-unverified draft). Keep its structure and
  // cap generously instead of squashing it into one 180-char line — real
  // tool work must not be reduced to an ellipsis.
  const progressBlock = !progressText
    ? ''
    : progressText.includes('\n')
      ? [
          'Latest progress update:',
          progressText.length > MAX_INTERIM_FALLBACK_BODY_CHARS
            ? `${progressText.slice(0, MAX_INTERIM_FALLBACK_BODY_CHARS).trimEnd()}\n…`
            : progressText,
        ].join('\n')
      : `Latest progress update: ${compactFallbackToolLine(progressText)}`

  return [
    header,
    progressBlock,
    recentToolOutputs.length > 0 ? 'Latest tool outputs:' : '',
    ...recentToolOutputs.map((line) => `- ${line}`),
  ].filter(Boolean).join('\n')
}

export function buildEmptyFinalFallbackMessage(messages: Message[]): string {
  const recentToolOutputs = collectRecentToolOutputs(messages)

  if (recentToolOutputs.length > 0) {
    return [
      'INCOMPLETE: The model returned an empty final reply after its tool attempts.',
      'Latest tool outputs (these alone do not establish task completion):',
      ...recentToolOutputs.map((line) => `- ${line}`),
    ].join('\n')
  }

  return 'INCOMPLETE: The run ended with an empty final reply.'
}

export function buildInvalidFinalResponseMessage(messages: Message[]): string {
  const recentTools = currentTurnMessages(messages)
    .filter((message) => message.role === 'tool')
    .slice(-MAX_EMPTY_FINAL_TOOL_LINES)
  const hasFailure = recentTools.some((message) => message.metadata?.[TOOL_RESULT_STATUS_METADATA_KEY] === 'error')
  return [
    'INCOMPLETE: Request incomplete because the model did not return a valid final response after its tool attempt.',
    recentTools.length > 0 ? 'Retained current-turn tool results:' : '',
    ...recentTools.map((message) => `- [${message.metadata?.[TOOL_RESULT_STATUS_METADATA_KEY] ?? 'unknown'}] ${message.name ?? 'tool'}: ${compactFallbackToolLine(extractMessageText(message))}`),
    hasFailure
      ? 'A failed tool result records a concrete blocker; successful actions remain completed and must not be repeated blindly.'
      : 'The response failure does not undo successful actions. Resume from the retained evidence without repeating completed side effects.',
  ].filter(Boolean).join('\n')
}
