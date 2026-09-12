import type { Message, ToolCall } from '@sepilotd/core'
import type {
  ToolInputNormalizationContext,
  ToolRegistry,
} from '../tools/registry.js'
import { TOOL_RESULT_STATUS_METADATA_KEY } from './memory-write-completion.js'
import { CURRENT_AGENT_TURN_USER_METADATA_KEY } from './turn-context.js'
import { resolveToolPath } from '../tools/path-utils.js'
import { FS_READ_OBSERVATION_EOF_LINE } from '../tools/fs-read.js'
import { summarizeToolOutputForAgentContext } from './tool-output.js'

export interface CoveredObservationCall {
  requested: ToolCall
  observed?: ToolCall
  observedOutput?: string
  observedEvidence?: Array<{ call: ToolCall; output: string }>
}

export interface NarrowedObservationCall {
  requested: ToolCall
  replacements: ToolCall[]
}

export interface ObservationCoveragePartition {
  executableCalls: ToolCall[]
  coveredCalls: CoveredObservationCall[]
  narrowedCalls: NarrowedObservationCall[]
}

export interface ObservationHistoryEntry {
  tool: string
  input: Record<string, unknown>
  status: 'success' | 'error'
  ts?: number
  output?: string
}

/**
 * Store exactly the bounded evidence that was eligible for the model context.
 * `fs.read` pages carry an explicit agent-visible line range, so truncating
 * them a second time can leave a header that claims more lines than the
 * retained body actually contains. Other tool outputs keep the compact
 * ledger cap because they do not participate in line-range coverage.
 */
export function observationHistoryOutput(
  toolName: string,
  output: string,
): string {
  const summarized = summarizeToolOutputForAgentContext(toolName, output)
  return toolName === 'fs.read' ? summarized : summarized.slice(0, 4_000)
}

function observedOutputForCall(
  messages: readonly Message[],
  history: readonly ObservationHistoryEntry[] | undefined,
  observed: ToolCall | undefined,
): string | undefined {
  if (!observed) return undefined
  const messageOutput = [...currentTurnMessages(messages)].reverse().find((message) => (
    message.role === 'tool'
    && message.toolCallId === observed.id
    && message.metadata?.[TOOL_RESULT_STATUS_METADATA_KEY] === 'success'
  ))?.content
  if (typeof messageOutput === 'string' && messageOutput.trim()) return messageOutput

  const historyIndex = observed.id.match(/^observation-history-(\d+)$/u)?.[1]
  if (historyIndex === undefined) return undefined
  const output = history?.[Number(historyIndex)]?.output
  return typeof output === 'string' && output.trim() ? output : undefined
}

const MAX_REPLAYABLE_OBSERVATION_CHARS = 20_000

interface FsReadLine {
  number: number
  content: string
}

function fsReadRequestedRange(call: ToolCall): { start: number; end?: number } {
  const offset = Number(call.arguments.offset)
  const limit = Number(call.arguments.limit)
  const start = Number.isSafeInteger(offset) && offset >= 1 ? offset : 1
  return {
    start,
    ...(Number.isSafeInteger(limit) && limit > 0
      ? { end: start + limit - 1 }
      : {}),
  }
}

function parseFsReadLines(output: string): FsReadLine[] {
  const lines: FsReadLine[] = []
  for (const content of output.replace(/\r\n?/gu, '\n').split('\n')) {
    const match = content.match(/^\s*(\d+)\t/u)
    if (!match) continue
    const number = Number(match[1])
    if (!Number.isSafeInteger(number) || number < 1) continue
    lines.push({ number, content })
  }
  return lines
}

/**
 * A cache hit should answer the range the model requested, not replay every
 * page that happened to prove coverage. Reconstruct only exact numbered
 * source lines. If compaction removed any line from a bounded requested range,
 * fail open so the real read executes instead of presenting a misleading
 * head/tail summary as complete evidence.
 */
function projectFsReadEvidence(
  output: string,
  requested: ToolCall,
  observed: ToolCall,
): string | undefined {
  const parsed = parseFsReadLines(output)
  const eofLine = fsReadObservationEofLine(observed)
  if (parsed.length === 0) {
    return summarizeToolOutputForAgentContext('fs.read', output)
  }

  const { start, end } = fsReadRequestedRange(requested)
  const selected = parsed.filter((line) => (
    line.number >= start && (end === undefined || line.number <= end)
  ))
  if (selected.length === 0) {
    return eofLine !== undefined && start > eofLine
      ? `[cached fs.read EOF: requested offset ${start} is past end of file (${eofLine} lines total)]`
      : undefined
  }
  return selected.map((line) => line.content).join('\n')
}

function fsReadObservationEofLine(call: ToolCall): number | undefined {
  const raw = call.arguments[FS_READ_OBSERVATION_EOF_LINE]
  return typeof raw === 'number' && Number.isSafeInteger(raw) && raw >= 0
    ? raw
    : undefined
}

/**
 * Return the exact bounded evidence that can be replayed on the next model
 * turn. Coverage is only an optimization: if compaction removed an output, or
 * a collective result would itself exceed a bounded tool-result unit, fail
 * open and execute a fresh observation instead of hiding the requested read.
 */
function replayableEvidenceForCalls(
  messages: readonly Message[],
  history: readonly ObservationHistoryEntry[] | undefined,
  calls: readonly ToolCall[],
  requested: ToolCall,
): Array<{ call: ToolCall; output: string }> | undefined {
  const evidence: Array<{ call: ToolCall; output: string }> = []
  let totalChars = 0
  for (const call of calls) {
    const output = observedOutputForCall(messages, history, call)
    if (!output) return undefined
    const bounded = requested.name === 'fs.read'
      ? projectFsReadEvidence(output, requested, call)
      : summarizeToolOutputForAgentContext(call.name, output)
    if (!bounded) continue
    totalChars += bounded.length
    if (totalChars > MAX_REPLAYABLE_OBSERVATION_CHARS) return undefined
    evidence.push({ call, output: bounded })
  }
  if (requested.name === 'fs.read') {
    const { start, end } = fsReadRequestedRange(requested)
    if (end !== undefined) {
      const parsedEvidence = evidence.flatMap(({ output }) => parseFsReadLines(output))
      const eofLine = calls.reduce<number | undefined>((latest, call) => {
        const current = fsReadObservationEofLine(call)
        return current === undefined ? latest : Math.max(latest ?? 0, current)
      }, undefined)
      const expectedEnd = eofLine === undefined ? end : Math.min(end, eofLine)
      if (parsedEvidence.length > 0 && start <= expectedEnd) {
        const present = new Set(parsedEvidence.map((line) => line.number))
        for (let line = start; line <= expectedEnd; line += 1) {
          if (!present.has(line)) return undefined
        }
      }
    }
  }
  return evidence.length > 0 ? evidence : undefined
}

function observationsRelevantToRequestedCall(
  observations: readonly ToolCall[],
  requested: ToolCall,
  context: ToolInputNormalizationContext,
): ToolCall[] {
  const sameTool = observations.filter((candidate) => candidate.name === requested.name)
  if (requested.name !== 'fs.read') return sameTool
  const requestedPath = typeof requested.arguments.path === 'string'
    ? resolveToolPath(requested.arguments.path, context.cwd)
    : ''
  if (!requestedPath) return sameTool
  return sameTool.filter((candidate) => {
    const candidatePath = typeof candidate.arguments.path === 'string'
      ? resolveToolPath(candidate.arguments.path, context.cwd)
      : ''
    return candidatePath === requestedPath
  })
}

export function observationHistoryFromCurrentTurnMessages(
  messages: readonly Message[],
): ObservationHistoryEntry[] {
  const callsById = new Map<string, ToolCall>()
  const history: ObservationHistoryEntry[] = []
  for (const message of currentTurnMessages(messages)) {
    for (const call of message.toolCalls ?? []) callsById.set(call.id, call)
    if (message.role !== 'tool' || !message.toolCallId) continue
    const status = message.metadata?.[TOOL_RESULT_STATUS_METADATA_KEY]
    if (status !== 'success' && status !== 'error') continue
    const call = callsById.get(message.toolCallId)
    if (!call) continue
    history.push({
      tool: call.name,
      input: { ...call.arguments },
      status,
      ...(typeof message.content === 'string'
        ? { output: observationHistoryOutput(call.name, message.content) }
        : {}),
    })
  }
  return history
}

const SINGLE_PATH_FILE_MUTATION_TOOLS = new Set([
  'fs.append',
  'fs.edit',
  'fs.write',
])

const FS_READ_VISIBLE_RANGE_PATTERN = /\[fs\.read agent-visible range: lines (\d+)-(\d+); continue with offset=(\d+)(?:;[^\]]*)?\]/u
const FS_READ_PAST_EOF_PATTERN = /^\[fs\.read:\s*offset \d+ is past (?:the scanned )?end of file \((\d+) lines? total\)\]$/iu
const FS_READ_NON_CONTENT_RESULT_PATTERN = /^\[fs\.read:\s.*(?:is a directory|appears to be binary|requires a non-empty)/iu

/**
 * Observation reuse must be bounded by evidence the model actually received,
 * not by the wider range the filesystem tool read before context compaction.
 * Large fs.read results carry an explicit contiguous visible-range marker;
 * narrow the cached call to that range. Non-content sentinels and legacy
 * generic summaries provide no safe range and therefore are not reusable.
 */
function agentVisibleObservationCall(
  call: ToolCall,
  output: string | undefined,
): ToolCall | null {
  if (call.name !== 'fs.read' || !output?.trim()) return call
  const normalized = output.trim()
  const pastEof = normalized.match(FS_READ_PAST_EOF_PATTERN)
  if (pastEof) {
    const eofLine = Number(pastEof[1])
    if (Number.isSafeInteger(eofLine) && eofLine >= 0) {
      return {
        ...call,
        arguments: {
          ...call.arguments,
          [FS_READ_OBSERVATION_EOF_LINE]: eofLine,
        },
      }
    }
  }
  if (FS_READ_NON_CONTENT_RESULT_PATTERN.test(normalized)) return null

  const visibleOutput = summarizeToolOutputForAgentContext(call.name, normalized)
  const match = visibleOutput.match(FS_READ_VISIBLE_RANGE_PATTERN)
  if (match) {
    const start = Number(match[1])
    const end = Number(match[2])
    const next = Number(match[3])
    if (
      Number.isSafeInteger(start)
      && Number.isSafeInteger(end)
      && Number.isSafeInteger(next)
      && start >= 1
      && end >= start
      && next === end + 1
    ) {
      return {
        ...call,
        arguments: {
          ...call.arguments,
          offset: start,
          limit: end - start + 1,
        },
      }
    }
  }
  if (/^\[fs\.read output summarized for agent context:/u.test(visibleOutput)) {
    return null
  }
  const parsedLines = parseFsReadLines(visibleOutput)
  if (parsedLines.length === 0) return call
  const firstLine = parsedLines[0]!.number
  const lastLine = parsedLines.at(-1)!.number
  const hasUnreadTail = /call fs\.read again\b[^\]]*\bto continue\]/iu.test(visibleOutput)
  return {
    ...call,
    arguments: {
      ...call.arguments,
      offset: firstLine,
      limit: lastLine - firstLine + 1,
      ...(!hasUnreadTail
        ? { [FS_READ_OBSERVATION_EOF_LINE]: lastLine }
        : {}),
    },
  }
}

/**
 * A successful mutation does not make every earlier file observation stale.
 * Keep exact-file reads when a known single-path write changed some other
 * file, but retain the conservative full barrier for patches, moves, shell
 * commands, plugins, and other mutations whose affected paths are not encoded
 * by this small stable contract.
 */
function retainObservationsAcrossMutation(
  observations: readonly ToolCall[],
  mutation: ToolCall,
  context: ToolInputNormalizationContext,
): ToolCall[] {
  if (mutation.name === 'terminal.run') {
    const network = mutation.arguments.network
    const writesWorkspace = network === 'external'
      || Boolean(
        network
        && typeof network === 'object'
        && !Array.isArray(network)
        && (network as Record<string, unknown>).mode === 'external',
      )
    // The normal terminal sandbox mounts the workspace read-only, so tests,
    // builds, and inspection commands cannot stale filesystem observations.
    // The explicit external-network mode is the exception: it is the audited
    // workspace-writing sandbox used by generators/installers/crawlers.
    return writesWorkspace ? [] : [...observations]
  }
  if (!SINGLE_PATH_FILE_MUTATION_TOOLS.has(mutation.name)) return []
  const mutationPath = typeof mutation.arguments.path === 'string'
    ? resolveToolPath(mutation.arguments.path, context.cwd)
    : ''
  if (!mutationPath) return []
  return observations.filter((observation) => {
    if (observation.name !== 'fs.read') return false
    const observedPath = typeof observation.arguments.path === 'string'
      ? resolveToolPath(observation.arguments.path, context.cwd)
      : ''
    return Boolean(observedPath && observedPath !== mutationPath)
  })
}

function currentTurnMessages(messages: readonly Message[]): readonly Message[] {
  let start = -1
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]
    if (
      message?.role === 'user'
      && message.metadata?.[CURRENT_AGENT_TURN_USER_METADATA_KEY] === true
    ) {
      start = index
      break
    }
  }
  if (start < 0) {
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      if (messages[index]?.role === 'user') {
        start = index
        break
      }
    }
  }
  return start >= 0 ? messages.slice(start) : messages
}

/**
 * Successful observations since the latest successful state-changing tool.
 * Unknown/plugin tools form a conservative barrier through isReadOnlyTool.
 */
export function reusableObservationsInCurrentTurn(
  messages: readonly Message[],
  tools: ToolRegistry,
  isReadOnlyTool: (toolName: string) => boolean,
  context: ToolInputNormalizationContext = {},
): ToolCall[] {
  const callsById = new Map<string, ToolCall>()
  const observations: ToolCall[] = []

  for (const message of currentTurnMessages(messages)) {
    for (const call of message.toolCalls ?? []) callsById.set(call.id, call)
    if (
      message.role !== 'tool'
      || !message.toolCallId
      || message.metadata?.[TOOL_RESULT_STATUS_METADATA_KEY] !== 'success'
    ) {
      continue
    }
    const call = callsById.get(message.toolCallId)
    if (!call) continue
    if (!isReadOnlyTool(call.name)) {
      observations.splice(
        0,
        observations.length,
        ...retainObservationsAcrossMutation(observations, call, context),
      )
      continue
    }
    const visibleCall = agentVisibleObservationCall(
      call,
      typeof message.content === 'string' ? message.content : undefined,
    )
    if (visibleCall && tools.get(call.name)?.observationCoverage) {
      observations.push(visibleCall)
    }
  }

  return observations
}

/**
 * Rebuild reusable observations from the structured execution ledger.
 *
 * Message compaction is allowed to summarize or remove old assistant/tool
 * protocol pairs, so messages alone are not a durable record of which reads
 * already succeeded in the active run. The execution ledger survives that
 * compaction and is scoped to the current graph/React run. Process it in
 * order so successful mutations invalidate observations with exactly the
 * same semantics as the message-backed path.
 */
export function reusableObservationsFromHistory(
  history: readonly ObservationHistoryEntry[],
  tools: ToolRegistry,
  isReadOnlyTool: (toolName: string) => boolean,
  context: ToolInputNormalizationContext = {},
): ToolCall[] {
  const observations: ToolCall[] = []

  for (const [index, entry] of history.entries()) {
    if (entry.status !== 'success') continue
    const call: ToolCall = {
      id: `observation-history-${index}`,
      name: entry.tool,
      arguments: { ...entry.input },
    }
    if (!isReadOnlyTool(call.name)) {
      observations.splice(
        0,
        observations.length,
        ...retainObservationsAcrossMutation(observations, call, context),
      )
      continue
    }
    const visibleCall = agentVisibleObservationCall(call, entry.output)
    if (visibleCall && tools.get(call.name)?.observationCoverage) {
      observations.push(visibleCall)
    }
  }

  return observations
}

export function partitionCallsCoveredByCurrentTurnObservations(
  messages: readonly Message[],
  requestedCalls: readonly ToolCall[],
  tools: ToolRegistry,
  context: ToolInputNormalizationContext,
  isReadOnlyTool: (toolName: string) => boolean,
  history?: readonly ObservationHistoryEntry[],
): ObservationCoveragePartition {
  // A non-empty structured ledger is authoritative because compaction can
  // remove protocol messages while retaining the executed-call history. Fall
  // back to messages for callers that do not maintain such a ledger yet.
  const observations = history?.length
    ? reusableObservationsFromHistory(history, tools, isReadOnlyTool, context)
    : reusableObservationsInCurrentTurn(messages, tools, isReadOnlyTool, context)
  const executableCalls: ToolCall[] = []
  const coveredCalls: CoveredObservationCall[] = []
  const narrowedCalls: NarrowedObservationCall[] = []

  for (const requested of requestedCalls) {
    const coverage = tools.get(requested.name)?.observationCoverage
    const relevantObservations = observationsRelevantToRequestedCall(
      observations,
      requested,
      context,
    )
    const observed = coverage
      ? relevantObservations.findLast((candidate) => {
          try {
            return coverage.covers(candidate.arguments, requested.arguments, context)
          } catch {
            // Coverage is an optimization contract. A buggy comparator must
            // fail open and execute the requested tool, never hide evidence.
            return false
          }
        })
      : undefined
    let collectivelyCovered = false
    if (!observed && coverage?.coversCollectively) {
      try {
        collectivelyCovered = coverage.coversCollectively(
          relevantObservations.map((candidate) => candidate.arguments),
          requested.arguments,
          context,
        )
      } catch {
        // As with pairwise coverage, fail open and execute the request.
        collectivelyCovered = false
      }
    }
    if (observed || collectivelyCovered) {
      const coveredObservations = observed ? [observed] : relevantObservations
      const observedEvidence = replayableEvidenceForCalls(
        messages,
        history,
        coveredObservations,
        requested,
      )
      if (!observedEvidence) {
        executableCalls.push(requested)
        continue
      }
      coveredCalls.push({
        requested,
        observed,
        observedOutput: observedEvidence[0]!.output,
        observedEvidence,
      })
      continue
    }

    let uncoveredInputs: readonly Record<string, unknown>[] | undefined
    if (coverage?.uncoveredInputs) {
      try {
        uncoveredInputs = coverage.uncoveredInputs(
          relevantObservations.map((candidate) => candidate.arguments),
          requested.arguments,
          context,
        )
      } catch {
        // Subtraction is an optimization contract. Any comparator failure
        // executes the original request so missing evidence is never hidden.
        uncoveredInputs = undefined
      }
    }
    if (uncoveredInputs?.length === 0) {
      const observedEvidence = replayableEvidenceForCalls(
        messages,
        history,
        relevantObservations,
        requested,
      )
      if (!observedEvidence) {
        executableCalls.push(requested)
        continue
      }
      coveredCalls.push({
        requested,
        observed: undefined,
        observedOutput: observedEvidence[0]!.output,
        observedEvidence,
      })
      continue
    }
    if (uncoveredInputs && uncoveredInputs.length > 0) {
      const replacements = uncoveredInputs.map((arguments_, index): ToolCall => ({
        ...requested,
        id: index === 0 ? requested.id : `${requested.id}:uncovered:${index + 1}`,
        arguments: { ...arguments_ },
      }))
      narrowedCalls.push({ requested, replacements })
      executableCalls.push(...replacements)
      continue
    }
    executableCalls.push(requested)
  }

  return { executableCalls, coveredCalls, narrowedCalls }
}

export function buildObservationReuseMessage(
  coveredCalls: readonly CoveredObservationCall[],
  narrowedCalls: readonly NarrowedObservationCall[] = [],
): Message {
  const tools = [...new Set([
    ...coveredCalls.map(({ requested }) => requested.name),
    ...narrowedCalls.map(({ requested }) => requested.name),
  ])]
  return {
    role: 'system',
    metadata: { reminderKind: 'observation-reuse' },
    content: [
      '[Current-turn observation reuse guard]',
      coveredCalls.length > 0
        ? `Skipped ${coveredCalls.length} read call(s) (${tools.join(', ')}) because successful results in this user turn already cover their requested scope and detail.`
        : '',
      narrowedCalls.length > 0
        ? `Narrowed ${narrowedCalls.length} overlapping read call(s) to ${narrowedCalls.reduce((sum, entry) => sum + entry.replacements.length, 0)} still-unobserved slice(s); combine those new results with the successful results already present in this user turn.`
        : '',
      'Use those existing tool results as evidence. Do not retry them with narrower limits or presentation-only argument changes.',
      'Continue with one genuinely new tool call or a final answer; do not repeat an immutable completed-action history record as though it were a new action.',
      'Attribute facts only to tool calls that have successful result messages; never claim that a skipped, blocked, failed, or unrequested tool ran.',
    ].filter(Boolean).join('\n'),
  }
}

/**
 * Preserve the assistant/tool protocol when a read is satisfied from current-
 * turn evidence without executing it again. A silent cache hit leaves weaker
 * models waiting for a tool result that never arrives and can create a loop.
 * These messages explicitly say that no new execution occurred; they are not
 * marked as successful tool executions, so ledgers and validation gates do
 * not double-count them.
 */
export function buildObservationReuseToolResultMessages(
  coveredCalls: readonly CoveredObservationCall[],
): Message[] {
  return coveredCalls.map(({ requested, observed, observedOutput, observedEvidence }) => ({
    role: 'tool',
    name: requested.name,
    toolCallId: requested.id,
    content: [
      '[observation reuse: no new tool execution]',
      observed
        ? `A prior successful ${observed.name} call (${observed.id}) already covers this requested scope.`
        : `Prior successful ${requested.name} observations collectively cover this requested scope.`,
      observedEvidence?.length
        ? [
            'Cached result (same bounded agent-visible evidence as the successful call(s)):',
            ...observedEvidence.map(({ call, output }, index) => [
              `[evidence ${index + 1}/${observedEvidence.length}: ${call.name} ${call.id}]`,
              output,
            ].join('\n')),
          ].join('\n')
        : observedOutput
          ? `Cached result (same bounded agent-visible evidence as the successful call):\n${summarizeToolOutputForAgentContext(requested.name, observedOutput)}`
          : 'Use the prior successful tool result already present in the conversation as evidence.',
      'Choose a genuinely new observation, an implementation/validation action, or a final answer; do not request the same covered scope again.',
    ].join('\n'),
    metadata: { observationReuse: true },
  }))
}

/** Keep the assistant/tool protocol consistent after a requested observation
 * is replaced by smaller uncovered slices. */
export function applyObservationNarrowingToLatestAssistantMessage(
  messages: Message[],
  narrowedCalls: readonly NarrowedObservationCall[],
): void {
  if (narrowedCalls.length === 0) return
  const replacementsById = new Map(
    narrowedCalls.map(({ requested, replacements }) => [requested.id, replacements]),
  )
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]!
    if (
      message.role !== 'assistant'
      || !(message.toolCalls ?? []).some((call) => replacementsById.has(call.id))
    ) continue
    messages[index] = {
      ...message,
      toolCalls: (message.toolCalls ?? []).flatMap(
        (call) => replacementsById.get(call.id) ?? [call],
      ),
    }
    return
  }
}
