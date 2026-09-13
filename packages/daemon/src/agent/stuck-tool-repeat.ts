import type { Message, ToolSecurityEffect } from '@sepilotd/core'

/**
 * Detect a stuck agent loop: the same tool called with the same
 * arguments K or more times within the most recent window of the
 * tool-call history. The graph's duplicate-tool-call repair only
 * catches dupes *within a single turn*; this catches the
 * cross-turn case (e.g. `fs.glob` re-issued every turn with no
 * progress) that burns the iteration budget on long-horizon tasks.
 *
 * Terminal-Bench `modernize-fortran-build` surfaced the pattern —
 * see docs/plans/2026-05-13-agent-loop-readonly-repeat.md.
 *
 * This is a general "you're stuck" signal, not a dataset-fitting
 * heuristic: it triggers on the *structural* property "same tool +
 * same args, K times in the recent window", with no special-casing
 * for any tool name, task, or dataset.
 */

export interface StuckToolRepeatEntry {
  tool: string
  input: Record<string, unknown>
  status: 'success' | 'error'
  failureCode?: string
  /**
   * Policy or approval refused the call before execution. Such entries are
   * friction, not failures: they never feed the repeated-failure or
   * permanent-failure buckets. The exact-repeat threshold still counts them,
   * so an identical blocked call issued over and over is caught there.
   */
  blocked?: boolean
  /** Executor evidence, when available. Different results are not an exact loop. */
  output?: string
  outputHash?: string
  executionObserved?: boolean
  securityEffect?: ToolSecurityEffect
  ts: number
}

export const DEFAULT_STUCK_REPEAT_THRESHOLD = 4
export const DEFAULT_STUCK_REPEAT_WINDOW = 8
export const DEFAULT_LOW_NOVELTY_WINDOW = 12
export const DEFAULT_LOW_NOVELTY_MIN_CALLS = 10
export const DEFAULT_LOW_NOVELTY_MAX_UNIQUE_SIGNATURES = 6
export const DEFAULT_TARGET_REPEAT_THRESHOLD = 4
export const DEFAULT_FAILURE_REPEAT_THRESHOLD = 2
export const DEFAULT_PERMANENT_FAILURE_CHURN_THRESHOLD = 4
export const DEFAULT_NO_OP_MUTATION_REPEAT_THRESHOLD = 3
export const MAX_STUCK_REPEAT_REPAIRS = 2

const LOW_NOVELTY_VOLATILE_KEYS = new Set([
  'after',
  'before',
  'count',
  'contextChars',
  'context_chars',
  'end',
  'endLine',
  'end_line',
  'limit',
  'maxBytes',
  'max_bytes',
  'offset',
  'page',
  'pageSize',
  'page_size',
  'perPage',
  'per_page',
  'start',
  'startLine',
  'start_line',
  'timeout',
  'timeout-ms',
  'timeoutMs',
  'timeout_ms',
])

function stableSerialize(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map((v) => stableSerialize(v)).join(',')}]`
  }
  if (value && typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>)
      .sort(([a], [b]) => a.localeCompare(b))
    return `{${entries.map(([k, v]) => `${JSON.stringify(k)}:${stableSerialize(v)}`).join(',')}}`
  }
  if (typeof value === 'number' && !Number.isFinite(value)) return 'null'
  return JSON.stringify(value) ?? 'null'
}

/**
 * Structural signature of a tool invocation: tool name + stable-serialized
 * arguments. Shared with the failed-attempt guard so "same action" is decided
 * by one deterministic definition instead of content matching.
 */
export function signatureOf(entry: Pick<StuckToolRepeatEntry, 'tool' | 'input'>): string {
  return `${entry.tool}:${stableSerialize(entry.input ?? {})}`
}

function normalizeLowNoveltyString(value: string): string {
  if (!/[\\/]/.test(value)) {
    return value
  }
  const parts = value
    .replace(/\\/g, '/')
    .replace(/\/+/g, '/')
    .split('/')
    .filter((part) => part && part !== '.')
  return parts.slice(-4).join('/')
}

function stableSerializeLowNovelty(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map((v) => stableSerializeLowNovelty(v)).join(',')}]`
  }
  if (value && typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>)
      .filter(([key]) => !LOW_NOVELTY_VOLATILE_KEYS.has(key))
      .sort(([a], [b]) => a.localeCompare(b))
    return `{${entries.map(([k, v]) => `${JSON.stringify(k)}:${stableSerializeLowNovelty(v)}`).join(',')}}`
  }
  if (typeof value === 'string') return JSON.stringify(normalizeLowNoveltyString(value))
  if (typeof value === 'number' && !Number.isFinite(value)) return 'null'
  return JSON.stringify(value) ?? 'null'
}

function lowNoveltySignatureOf(entry: StuckToolRepeatEntry): string {
  return `${entry.tool}:${stableSerializeLowNovelty(entry.input ?? {})}`
}

const SHELL_EXECUTABLES = new Set(['bash', 'dash', 'sh', 'zsh', 'fish', 'csh', 'tcsh'])
const SHELL_GRAMMAR_WORDS = new Set([
  'case', 'do', 'done', 'elif', 'else', 'esac', 'fi', 'for', 'function', 'if',
  'in', 'select', 'then', 'time', 'until', 'while',
])

function commandBasename(value: string): string {
  return value.trim().replace(/\\/g, '/').split('/').at(-1)?.toLowerCase() ?? ''
}

function shellChildExecutable(input: Record<string, unknown>): string {
  if (!Array.isArray(input.args)) return ''
  const args = input.args.filter((value): value is string => typeof value === 'string')
  const commandFlagIndex = args.findIndex((arg) => /^-[a-z]*c[a-z]*$/i.test(arg))
  const command = commandFlagIndex >= 0 ? args[commandFlagIndex + 1]?.trim() : ''
  if (!command) return ''
  const match = command.match(/^(?:[A-Za-z_][A-Za-z0-9_]*=\S+\s+)*([^\s;&|()]+)/u)
  const candidate = commandBasename(match?.[1] ?? '')
  return candidate && !SHELL_GRAMMAR_WORDS.has(candidate) ? candidate : ''
}

function terminalFailureScope(input: Record<string, unknown>): string {
  const executable = typeof input.executable === 'string'
    ? commandBasename(input.executable)
    : typeof input.command === 'string'
      ? commandBasename(input.command.trim().split(/\s+/, 1)[0] ?? '')
      : typeof input.cmd === 'string'
        ? commandBasename(input.cmd.trim().split(/\s+/, 1)[0] ?? '')
        : ''
  if (SHELL_EXECUTABLES.has(executable)) {
    return shellChildExecutable(input) || lowNoveltySignatureOf({
      tool: 'terminal.run', input, status: 'error', ts: 0,
    })
  }
  return executable || 'unknown-executable'
}

function failureScopeOf(entry: StuckToolRepeatEntry): string {
  if (entry.tool === 'terminal.run') {
    // A program/test exiting nonzero does not mean its interpreter or shell
    // is unavailable. Different commands can fail for unrelated assertions.
    if (entry.failureCode === 'EXIT_NONZERO_PERMANENT') return signatureOf(entry)
    return `${entry.tool}:${terminalFailureScope(entry.input)}`
  }
  if (entry.tool.startsWith('memory.')) return entry.tool
  return lowNoveltySignatureOf(entry)
}

/**
 * A mutation that repeats an earlier successful call with byte-identical
 * arguments leaves the target exactly as it already is, so it is not progress
 * — yet it does reset the low-novelty barrier below, and mutation tools are
 * not tracked for low novelty at all. That is what lets a write / re-write
 * cycle longer than the exact-repeat window run unchecked until the iteration
 * budget is gone. Counted across the whole run rather than a window, because
 * the point of the cycle is that it is too long for the window to see.
 */
function detectRepeatedNoOpMutation(
  history: StuckToolRepeatEntry[],
  mutationTools: ReadonlySet<string>,
  threshold: number,
): StuckToolRepeatResult | null {
  const counts = new Map<string, { count: number; tool: string }>()
  for (const entry of history) {
    if (entry.status !== 'success' || !mutationTools.has(entry.tool)) continue
    const sig = signatureOf(entry)
    const existing = counts.get(sig)
    if (existing) existing.count += 1
    else counts.set(sig, { count: 1, tool: entry.tool })
  }
  for (const { count, tool } of counts.values()) {
    if (count >= threshold) return { stuck: true, tool, count, kind: 'no_op_mutation' }
  }
  return null
}

export interface StuckToolRepeatResult {
  stuck: boolean
  tool?: string
  count?: number
  kind?: 'exact' | 'target_repeat' | 'low_novelty' | 'no_op_mutation' | 'repeated_failure' | 'permanent_failure_churn'
  uniqueSignatures?: number
  tools?: string[]
  /**
   * The repair budget is spent and the loop persists anyway. `stuck` stays
   * false (no further repair message should be injected), but the caller must
   * escalate: close the evidence phase and force a final synthesis instead of
   * letting the loop run the remaining iteration budget unchecked.
   */
  exhausted?: boolean
}

/**
 * Returns `{ stuck: true, tool, count }` when, within the last
 * `window` history entries, some (tool, args) signature appears at
 * least `threshold` times. Otherwise `{ stuck: false }`.
 */
export function detectStuckToolRepeat(
  history: StuckToolRepeatEntry[] | undefined,
  options: {
    threshold?: number
    window?: number
    trackedTools?: ReadonlySet<string>
    lowNoveltyBarrierTools?: ReadonlySet<string>
    lowNoveltyWindow?: number
    lowNoveltyMinCalls?: number
    lowNoveltyMaxUniqueSignatures?: number
    targetRepeatThreshold?: number
    noOpMutationRepeatThreshold?: number
    failureRepeatThreshold?: number
    /**
     * Optional run-wide ceiling for distinct structured permanent failures.
     * Callers should enable this only for an observe-only phase: without a
     * mutation boundary, permanent source failures cannot be repaired by
     * guessing more endpoints, executables, or discovery targets forever.
     */
    permanentFailureChurnThreshold?: number
  } = {},
): StuckToolRepeatResult {
  const threshold = options.threshold ?? DEFAULT_STUCK_REPEAT_THRESHOLD
  const failureRepeatThreshold = options.failureRepeatThreshold
    ?? DEFAULT_FAILURE_REPEAT_THRESHOLD
  const window = options.window ?? DEFAULT_STUCK_REPEAT_WINDOW
  if (!history || history.length < Math.min(threshold, failureRepeatThreshold)) {
    return { stuck: false }
  }
  const recent = history.slice(-window)
  const counts = new Map<string, { count: number; tool: string; outcome: string }>()
  for (const entry of recent) {
    // An error followed by a successful retry is a changed outcome, not four
    // copies of one unchanged result. Prefer the executor's full-result hash;
    // legacy/checkpoint entries fall back to their retained output or status.
    const sig = signatureOf(entry)
    const outcome = JSON.stringify([entry.status, entry.failureCode, entry.outputHash ?? entry.output])
    const existing = counts.get(sig)
    if (existing?.outcome === outcome) existing.count += 1
    else counts.set(sig, { count: 1, tool: entry.tool, outcome })
  }
  for (const { count, tool } of counts.values()) {
    if (count >= threshold) return { stuck: true, tool, count, kind: 'exact' }
  }

  const failureCounts = new Map<string, { count: number; tool: string; scope: string; code: string; outcome?: string }>()
  const seenMutations = new Set<string>()
  for (const entry of recent) {
    if (entry.status === 'success' && entry.executionObserved !== false && !entry.blocked) {
      const mutation = entry.securityEffect === 'workspace-write'
        || options.lowNoveltyBarrierTools?.has(entry.tool)
      const signature = signatureOf(entry)
      if (mutation && !seenMutations.has(signature)) {
        // A real edit invalidates old command outcomes, not authentication,
        // policy, missing-executable, or remote-service failures.
        for (const [key, failure] of failureCounts) {
          if (failure.tool === 'terminal.run' && failure.code === 'EXIT_NONZERO_PERMANENT') {
            failureCounts.delete(key)
          }
        }
        seenMutations.add(signature)
      }
      for (const [key, failure] of failureCounts) {
        const scope = failureScopeOf({ ...entry, failureCode: failure.code })
        if (failure.scope === scope) failureCounts.delete(key)
      }
      continue
    }
    if (entry.status !== 'error' || !entry.failureCode || entry.blocked) continue
    const scope = failureScopeOf(entry)
    const key = `${scope}:${entry.failureCode}`
    const outcome = entry.failureCode === 'EXIT_NONZERO_PERMANENT'
      ? entry.outputHash ?? entry.output
      : undefined
    const existing = failureCounts.get(key)
    if (existing && existing.outcome === outcome) existing.count += 1
    else failureCounts.set(key, { count: 1, tool: entry.tool, scope, code: entry.failureCode, outcome })
  }
  for (const { count, tool } of failureCounts.values()) {
    if (count >= failureRepeatThreshold) {
      return { stuck: true, tool, count, kind: 'repeated_failure' }
    }
  }

  const permanentFailureChurnThreshold = options.permanentFailureChurnThreshold
  if (
    permanentFailureChurnThreshold !== undefined
    && permanentFailureChurnThreshold > 0
  ) {
    const permanentFailures = history.filter((entry) => (
      entry.status === 'error'
      && !entry.blocked
      && entry.failureCode?.endsWith('_PERMANENT') === true
      // Executed tests/programs with nonzero status are task evidence, not
      // permanent loss of a capability (even inside a read-only reviewer).
      && entry.failureCode !== 'EXIT_NONZERO_PERMANENT'
    ))
    if (permanentFailures.length >= permanentFailureChurnThreshold) {
      const toolCounts = new Map<string, number>()
      for (const entry of permanentFailures) {
        toolCounts.set(entry.tool, (toolCounts.get(entry.tool) ?? 0) + 1)
      }
      const tools = [...toolCounts.keys()].sort((a, b) => (
        (toolCounts.get(b) ?? 0) - (toolCounts.get(a) ?? 0)
        || a.localeCompare(b)
      ))
      return {
        stuck: true,
        tool: tools[0],
        count: permanentFailures.length,
        kind: 'permanent_failure_churn',
        uniqueSignatures: new Set(permanentFailures.map((entry) => (
          `${failureScopeOf(entry)}:${entry.failureCode}`
        ))).size,
        tools,
      }
    }
  }

  // Pagination changes (`offset`, `limit`, page size) do not make a fourth
  // read of the same target novel. Catch this earlier than the broad
  // low-novelty window while still allowing several legitimate chunks.
  if (
    options.trackedTools
    && options.trackedTools.size > 0
    && options.targetRepeatThreshold !== undefined
  ) {
    const targetCounts = new Map<string, { count: number; tool: string }>()
    for (const entry of recent) {
      if (!options.trackedTools.has(entry.tool)) continue
      const signature = lowNoveltySignatureOf(entry)
      const existing = targetCounts.get(signature)
      if (existing) existing.count += 1
      else targetCounts.set(signature, { count: 1, tool: entry.tool })
    }
    const targetRepeatThreshold = options.targetRepeatThreshold
    for (const { count, tool } of targetCounts.values()) {
      if (count >= targetRepeatThreshold) {
        return { stuck: true, tool, count, kind: 'target_repeat' }
      }
    }
  }

  if (options.lowNoveltyBarrierTools && options.lowNoveltyBarrierTools.size > 0) {
    const noOpMutation = detectRepeatedNoOpMutation(
      history,
      options.lowNoveltyBarrierTools,
      options.noOpMutationRepeatThreshold ?? DEFAULT_NO_OP_MUTATION_REPEAT_THRESHOLD,
    )
    if (noOpMutation) return noOpMutation
  }

  if (!options.trackedTools || options.trackedTools.size === 0) {
    return { stuck: false }
  }
  const lowNoveltyWindow = options.lowNoveltyWindow ?? DEFAULT_LOW_NOVELTY_WINDOW
  const lowNoveltyMinCalls = options.lowNoveltyMinCalls ?? DEFAULT_LOW_NOVELTY_MIN_CALLS
  const lowNoveltyMaxUniqueSignatures =
    options.lowNoveltyMaxUniqueSignatures ?? DEFAULT_LOW_NOVELTY_MAX_UNIQUE_SIGNATURES
  if (history.length < lowNoveltyMinCalls) {
    return { stuck: false }
  }
  let lowNoveltyRecent = history.slice(-lowNoveltyWindow)
  if (lowNoveltyRecent.length < lowNoveltyMinCalls) {
    return { stuck: false }
  }
  if (options.lowNoveltyBarrierTools && options.lowNoveltyBarrierTools.size > 0) {
    const barrierTools = options.lowNoveltyBarrierTools
    // Only the first of a byte-identical mutation changed anything; later
    // repeats are no-ops and must not pass as the progress that resets the
    // window below.
    const seenMutations = new Set<string>()
    const windowStart = history.length - lowNoveltyRecent.length
    const isNoOpRepeat: boolean[] = []
    for (let index = 0; index < history.length; index += 1) {
      const entry = history[index]!
      let noOp = false
      if (entry.status === 'success' && barrierTools.has(entry.tool)) {
        const sig = signatureOf(entry)
        noOp = seenMutations.has(sig)
        seenMutations.add(sig)
      }
      if (index >= windowStart) isNoOpRepeat.push(noOp)
    }
    for (let index = lowNoveltyRecent.length - 1; index >= 0; index -= 1) {
      const entry = lowNoveltyRecent[index]!
      if (entry.status === 'success' && barrierTools.has(entry.tool) && !isNoOpRepeat[index]) {
        lowNoveltyRecent = lowNoveltyRecent.slice(index + 1)
        break
      }
    }
  }
  const lowNoveltyEntries = lowNoveltyRecent.filter((entry) => options.trackedTools!.has(entry.tool))
  if (lowNoveltyEntries.length < lowNoveltyMinCalls) {
    return { stuck: false }
  }
  const signatures = new Set(lowNoveltyEntries.map(lowNoveltySignatureOf))
  if (signatures.size <= lowNoveltyMaxUniqueSignatures) {
    const toolCounts = new Map<string, number>()
    for (const entry of lowNoveltyEntries) {
      toolCounts.set(entry.tool, (toolCounts.get(entry.tool) ?? 0) + 1)
    }
    const tools = [...toolCounts.keys()].sort((a, b) => (toolCounts.get(b) ?? 0) - (toolCounts.get(a) ?? 0))
    return {
      stuck: true,
      tool: tools[0],
      count: lowNoveltyEntries.length,
      kind: 'low_novelty',
      uniqueSignatures: signatures.size,
      tools,
    }
  }
  return { stuck: false }
}

export function buildStuckToolRepeatMessage(
  tool: string,
  count: number,
  kind: StuckToolRepeatResult['kind'] = 'exact',
): Message {
  if (kind === 'permanent_failure_churn') {
    return {
      role: 'system',
      content: [
        `Observe-only evidence gathering has accumulated ${count} structured permanent failures across available read sources.`,
        'You are stuck in permanent-failure churn. Stop guessing alternate endpoints, executables, namespaces, or discovery targets.',
        'Synthesize the successful evidence already retained and mark unsupported criteria UNMET/INCOMPLETE with the exact unavailable capability or credential boundary.',
        'Do not issue another read call unless the current conversation contains a concrete state change that can invalidate one of those failures.',
      ].join(' '),
    }
  }
  if (kind === 'low_novelty') {
    return {
      role: 'system',
      content: [
        `Recent tool use is low-novelty: ${count} recent read/discovery calls are cycling across a small repeated evidence set.`,
        'You are stuck in a loop. Stop repository discovery for now.',
        'Synthesize the evidence you already have, use a non-read action such as writing/updating the requested artifact, or state the concrete blocker as INCOMPLETE.',
        'Do not issue more read-only exploration calls until a non-read action has changed the state or the user provides new scope.',
      ].join(' '),
    }
  }
  if (kind === 'target_repeat') {
    return {
      role: 'system',
      content: [
        `You have read the same target ${count} times with only pagination or display arguments changing.`,
        'The evidence already gathered is sufficient for this pass. Stop rereading this target and synthesize the answer now.',
        'If a specific missing fact still blocks the answer, state that fact as INCOMPLETE instead of issuing another broad read.',
      ].join(' '),
    }
  }
  if (kind === 'no_op_mutation') {
    return {
      role: 'system',
      content: [
        `You have called ${tool} ${count} times with byte-identical arguments, so the target already holds exactly that content and the repeats changed nothing.`,
        'You are cycling over work that is already done. Do not issue that call again.',
        'Check the current state before rewriting, then either take the different action the task still needs (a distinct write, a move/delete, a validation step) or state the concrete blocker as INCOMPLETE.',
      ].join(' '),
    }
  }
  if (kind === 'repeated_failure') {
    return {
      role: 'system',
      content: [
        `${tool} has failed ${count} times in the same execution scope with the same structured failure class.`,
        'Treat that capability path as unavailable for the rest of this turn instead of varying the query and retrying it.',
        'Use successful evidence already collected, switch to a genuinely different capability scope, or state the concrete blocker as INCOMPLETE.',
      ].join(' '),
    }
  }
  return {
    role: 'system',
    content: [
      `You have called ${tool} with the same arguments ${count} times in the recent window with the same success/error status.`,
      'You are stuck in a loop. Stop repeating this call.',
      'Either: (a) act on the information you already have with a different tool (e.g. a write tool, terminal.run, or a more specific read), (b) call the tool with *different* arguments if you genuinely need other data, or (c) state the concrete blocker to the user and stop.',
      'Do not issue the same read-only call again.',
    ].join(' '),
  }
}

/**
 * Decide whether to inject the repair message this turn.
 */
export function shouldRepairStuckToolRepeat(options: {
  history: StuckToolRepeatEntry[] | undefined
  repairedCount: number
  isLastIteration?: boolean
  maxRepairs?: number
  threshold?: number
  window?: number
  trackedTools?: ReadonlySet<string>
  lowNoveltyBarrierTools?: ReadonlySet<string>
  lowNoveltyWindow?: number
  lowNoveltyMinCalls?: number
  lowNoveltyMaxUniqueSignatures?: number
  targetRepeatThreshold?: number
  permanentFailureChurnThreshold?: number
}): StuckToolRepeatResult {
  const {
    history,
    repairedCount,
    isLastIteration = false,
    maxRepairs = MAX_STUCK_REPEAT_REPAIRS,
    threshold,
    window,
    trackedTools,
    lowNoveltyBarrierTools,
    lowNoveltyWindow,
    lowNoveltyMinCalls,
    lowNoveltyMaxUniqueSignatures,
    targetRepeatThreshold,
    permanentFailureChurnThreshold,
  } = options
  if (isLastIteration) return { stuck: false }
  if (repairedCount >= maxRepairs) {
    // Out of repair messages. If the loop persists regardless, report
    // exhaustion so the caller can foreclose the run instead of silently
    // letting the loop consume the remaining iteration budget.
    const verdict = detectStuckToolRepeat(history, {
      threshold,
      window,
      trackedTools,
      lowNoveltyBarrierTools,
      lowNoveltyWindow,
      lowNoveltyMinCalls,
      lowNoveltyMaxUniqueSignatures,
      targetRepeatThreshold,
      permanentFailureChurnThreshold,
    })
    return verdict.stuck ? { ...verdict, stuck: false, exhausted: true } : { stuck: false }
  }
  return detectStuckToolRepeat(history, {
    threshold,
    window,
    trackedTools,
    lowNoveltyBarrierTools,
    lowNoveltyWindow,
    lowNoveltyMinCalls,
    lowNoveltyMaxUniqueSignatures,
    targetRepeatThreshold,
    permanentFailureChurnThreshold,
  })
}

/**
 * Resolve the concrete history entry behind an `exact` stuck verdict so the
 * caller can record it as a failed attempt (same structural signature the
 * failed-attempt guard blocks on). Low-novelty verdicts span many distinct
 * signatures, so they have no single blockable entry and return undefined.
 */
export function findStuckRepeatEntry(
  history: StuckToolRepeatEntry[] | undefined,
  result: StuckToolRepeatResult,
  window: number = DEFAULT_STUCK_REPEAT_WINDOW,
): StuckToolRepeatEntry | undefined {
  if (!result.stuck || !result.tool) return undefined
  if (result.kind !== 'exact' && result.kind !== 'no_op_mutation') return undefined
  // A no-op mutation cycle is detected across the whole run precisely because
  // it is longer than the exact-repeat window, so scan the whole run for it.
  const recent = result.kind === 'no_op_mutation' ? (history ?? []) : (history ?? []).slice(-window)
  const counts = new Map<string, { count: number; entry: StuckToolRepeatEntry; outcome?: string }>()
  for (const entry of recent) {
    if (entry.tool !== result.tool) continue
    const sig = signatureOf(entry)
    const outcome = result.kind === 'exact'
      ? JSON.stringify([entry.status, entry.failureCode, entry.outputHash ?? entry.output])
      : undefined
    const existing = counts.get(sig)
    if (existing && existing.outcome === outcome) {
      existing.count += 1
      existing.entry = entry
    } else {
      counts.set(sig, { count: 1, entry, outcome })
    }
  }
  let best: { count: number; entry: StuckToolRepeatEntry } | undefined
  for (const candidate of counts.values()) {
    if (!best || candidate.count > best.count) best = candidate
  }
  return best?.entry
}
