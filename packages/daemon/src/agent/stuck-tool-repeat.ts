import type { Message, ToolSecurityEffect } from '@sepilotd/core'
import { createHash } from 'node:crypto'

export interface StuckToolRepeatEntry {
  tool: string
  input: Record<string, unknown>
  status: 'success' | 'error'
  failureCode?: string
  blocked?: boolean
  output?: string
  outputFingerprint?: string
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
export const DEFAULT_EXECUTION_FAILURE_CHURN_THRESHOLD = 6
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

export function signatureOf(entry: Pick<StuckToolRepeatEntry, 'tool' | 'input'>): string {
  return `${entry.tool}:${stableSerialize(entry.input ?? {})}`
}

export function findRepeatedInvalidToolCall(
  history: readonly StuckToolRepeatEntry[],
  proposed: readonly Pick<StuckToolRepeatEntry, 'tool' | 'input'>[],
): Pick<StuckToolRepeatEntry, 'tool' | 'input'> | undefined {
  const counts = new Map<string, number>()
  for (const entry of history.slice(-DEFAULT_STUCK_REPEAT_WINDOW)) {
    if (entry.status === 'success' && entry.executionObserved === true
      && !entry.blocked && entry.securityEffect !== 'internal-state') {
      counts.clear()
      continue
    }
    const signature = signatureOf(entry)
    if (entry.status === 'error' && entry.failureCode === 'INVALID_INPUT_PERMANENT'
      && entry.executionObserved === false && !entry.blocked) {
      counts.set(signature, (counts.get(signature) ?? 0) + 1)
    } else {
      counts.delete(signature)
    }
  }
  return proposed.find(entry => (counts.get(signatureOf(entry)) ?? 0) >= DEFAULT_FAILURE_REPEAT_THRESHOLD)
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

function observationSignatureOf(entry: StuckToolRepeatEntry): string {
  const target = lowNoveltySignatureOf(entry)
  if (entry.status !== 'success' || entry.blocked || entry.executionObserved !== true
    || entry.securityEffect !== 'observe' || !entry.output?.trim()) return target
  const fingerprint = entry.outputFingerprint?.trim()
    || createHash('sha256').update(entry.output).digest('hex')
  return `${target}:observation:${fingerprint}`
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
    if (entry.failureCode === 'EXIT_NONZERO_PERMANENT') return signatureOf(entry)
    return `${entry.tool}:${terminalFailureScope(entry.input)}`
  }
  if (entry.tool.startsWith('memory.')) return entry.tool
  return lowNoveltySignatureOf(entry)
}

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
  kind?: 'exact' | 'target_repeat' | 'low_novelty' | 'no_op_mutation' | 'repeated_failure' | 'permanent_failure_churn' | 'execution_failure_churn'
  uniqueSignatures?: number
  tools?: string[]
  exhausted?: boolean
}

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
    permanentFailureChurnThreshold?: number
    executionFailureChurnThreshold?: number
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
    const sig = signatureOf(entry)
    const outcome = JSON.stringify([entry.status, entry.failureCode, entry.outputFingerprint ?? entry.output])
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
      ? entry.outputFingerprint ?? entry.output
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
  const executionFailureChurnThreshold = options.executionFailureChurnThreshold
  if (
    permanentFailureChurnThreshold !== undefined
    && permanentFailureChurnThreshold > 0
  ) {
    const permanentFailures = history.filter((entry) => (
      entry.status === 'error'
      && !entry.blocked
      && entry.failureCode?.endsWith('_PERMANENT') === true
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

  if (
    executionFailureChurnThreshold !== undefined
    && executionFailureChurnThreshold > 0
  ) {
    const executionFailures = recent.filter((entry) => (
      entry.tool === 'terminal.run'
      && entry.status === 'error'
      && !entry.blocked
      && entry.failureCode === 'EXIT_NONZERO_PERMANENT'
    ))
    if (executionFailures.length >= executionFailureChurnThreshold) {
      const tools = [...new Set(executionFailures.map((entry) => entry.tool))]
      return {
        stuck: true,
        tool: tools[0] ?? 'terminal.run',
        count: executionFailures.length,
        kind: 'execution_failure_churn',
        uniqueSignatures: new Set(executionFailures.map((entry) => signatureOf(entry))).size,
        tools,
      }
    }
  }

  if (
    options.trackedTools
    && options.trackedTools.size > 0
    && options.targetRepeatThreshold !== undefined
  ) {
    const targetCounts = new Map<string, { count: number; tool: string }>()
    for (const entry of recent) {
      if (!options.trackedTools.has(entry.tool)) continue
      const signature = observationSignatureOf(entry)
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
  const signatures = new Set(lowNoveltyEntries.map(observationSignatureOf))
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
  if (kind === 'execution_failure_churn') {
    return {
      role: 'system',
      content: [
        `Recent validation/execution attempts have failed ${count} times with non-zero results across varied commands.`,
        'Stop guessing additional test commands or wrappers. Inspect the latest failure evidence and either make one concrete product correction, run the smallest directly relevant check, or report INCOMPLETE with the exact unresolved failure.',
        'Do not spend more turns re-running equivalent failing checks without a new workspace mutation or a materially different prerequisite.',
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
        `You have read the same target ${count} times without distinct executor-confirmed observations.`,
        'Stop repeating this read and synthesize the retained evidence; repetition does not establish that the evidence is sufficient.',
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
  executionFailureChurnThreshold?: number
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
    executionFailureChurnThreshold,
  } = options
  if (isLastIteration) return { stuck: false }
  if (repairedCount >= maxRepairs) {
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
      executionFailureChurnThreshold,
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
    executionFailureChurnThreshold,
  })
}

export function findStuckRepeatEntry(
  history: StuckToolRepeatEntry[] | undefined,
  result: StuckToolRepeatResult,
  window: number = DEFAULT_STUCK_REPEAT_WINDOW,
): StuckToolRepeatEntry | undefined {
  if (!result.stuck || !result.tool) return undefined
  if (result.kind !== 'exact' && result.kind !== 'no_op_mutation') return undefined
  const recent = result.kind === 'no_op_mutation' ? (history ?? []) : (history ?? []).slice(-window)
  const counts = new Map<string, { count: number; entry: StuckToolRepeatEntry; outcome?: string }>()
  for (const entry of recent) {
    if (entry.tool !== result.tool) continue
    const sig = signatureOf(entry)
    const outcome = result.kind === 'exact'
      ? JSON.stringify([entry.status, entry.failureCode, entry.outputFingerprint ?? entry.output])
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
