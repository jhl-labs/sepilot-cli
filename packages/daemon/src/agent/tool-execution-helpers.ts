// Helpers extracted from tool-execution.ts so the main runToolExecution
// generator can shrink. The contract for living here is: no agent-
// runtime *state* is owned (no sessions map, no checkpoint store, no
// hook registry held in the module). Pure functions are the default;
// stateless side-effect adapters that take their dependencies as
// arguments are also fine — they're still trivially testable. If you
// reach for owned state, the helper belongs back in tool-execution.ts
// or a dedicated coordinator module.

import type {
  AgentEvent,
  ApprovalDecision,
  ApprovalDecisionStatus,
  AutonomyLevel,
  ContentPart,
  Message,
  TokenUsage,
  ToolExecutionPosture,
  ToolCall,
  ToolResultMetadata,
} from '@sepilotd/core'
import type { HookRegistry } from '../hook/registry.js'
import type { PolicyEngine } from '../security/policy-engine.js'
import {
  resolveToolBatchFailureCouplingKey,
  resolveToolResumeSafety,
  resolveToolSchedulingHint,
  type ResolvedToolSchedulingHint,
  type ToolDefinitionRuntime,
  type ToolRegistry,
} from '../tools/registry.js'
import type {
  ToolExecutionRecord,
  ToolExecutionResultSnapshot,
} from '../server/runtime/tool-executions.js'
import {
  APPROVAL_FAILURE_STATUS_METADATA_KEY,
  APPROVAL_DENIAL_NOTE_METADATA_KEY,
  APPROVAL_STOP_REQUESTED_METADATA_KEY,
  approvalFailureStatusFromDecision,
  TOOL_RESULT_BLOCK_SOURCE_METADATA_KEY,
  type ToolResultBlockSource,
} from './approval-failure.js'
import {
  POLICY_FAILURE_REASON_METADATA_KEY,
  TOOL_RESULT_EXECUTION_OBSERVED_METADATA_KEY,
} from './policy-failure.js'

const AUDIT_INPUT_SUMMARY_MAX = 200
const PARALLEL_TOOL_MAX_FLOOR = 1
const PARALLEL_TOOL_MAX_CEILING = 16
const DEFAULT_PARALLEL_TOOL_MAX = 4
const WRITE_CAPABLE_SUBAGENT_CATEGORIES = new Set([
  'implementation',
  'validation',
  'writing',
  'visual',
])

// Compatible with @sepilotd/core's AuditEvent (open index signature). The
// open signature lets the daemon attach session-/event-specific fields
// without forcing every audit entry to declare them.
export interface ToolExecutionAuditEntry {
  timestamp: string
  event: 'tool.execute' | 'tool.autonomous_approval_block'
  device: string
  session: string
  tool: string
  input_summary: string
  // Capture the safety posture under which the tool ran. Without these
  // fields an after-the-fact audit can't tell whether a destructive call
  // was a deliberate user click or a side-effect of running with
  // `--yes-to-everything` in CI.
  autonomy: AutonomyLevel
  auto_approve_flag: boolean
  [key: string]: unknown
}

/**
 * Build the audit log entry for a tool.execute event. Pure: callers
 * provide the timestamp (defaults to `new Date().toISOString()` when
 * omitted) so unit tests can assert exact records without mocking
 * Date.now.
 */
export function buildToolExecutionAuditEntry(opts: {
  toolCall: ToolCall
  sessionId: string
  deviceName?: string
  autonomy: AutonomyLevel
  autoApprove?: boolean
  timestamp?: string
}): ToolExecutionAuditEntry {
  return {
    timestamp: opts.timestamp ?? new Date().toISOString(),
    event: 'tool.execute',
    device: opts.deviceName ?? 'unknown',
    session: opts.sessionId,
    tool: opts.toolCall.name,
    input_summary: JSON.stringify(opts.toolCall.arguments).slice(0, AUDIT_INPUT_SUMMARY_MAX),
    autonomy: opts.autonomy,
    auto_approve_flag: opts.autoApprove === true,
  }
}

/**
 * Build the audit entry for the Autonomous approval-block case: a tool whose
 * policy says supervised was requested under Autonomous autonomy, where no
 * interactive approval prompt is available. The tool is blocked and the audit
 * log records the attempted escalation.
 */
export function buildAutonomousApprovalBlockAuditEntry(opts: {
  toolCall: ToolCall
  sessionId: string
  deviceName?: string
  autonomy: AutonomyLevel
  reason?: string
  timestamp?: string
}): ToolExecutionAuditEntry {
  return {
    timestamp: opts.timestamp ?? new Date().toISOString(),
    event: 'tool.autonomous_approval_block',
    device: opts.deviceName ?? 'unknown',
    session: opts.sessionId,
    tool: opts.toolCall.name,
    input_summary: JSON.stringify({
      arguments: opts.toolCall.arguments,
      policy_reason: opts.reason ?? 'requires_approval',
    }).slice(0, AUDIT_INPUT_SUMMARY_MAX),
    autonomy: opts.autonomy,
    auto_approve_flag: false,
  }
}

/**
 * Tools that mutate files on disk. Centralised so the post-execution
 * file-edit hook and any future audit stays in lock-step with whichever
 * tools are considered editing tools.
 */
const FILE_EDIT_TOOLS = new Set<string>(['fs.write', 'fs.append', 'fs.edit', 'apply_patch'])

export function isFileEditTool(name: string): boolean {
  return FILE_EDIT_TOOLS.has(name)
}

/**
 * Deep-clone a ToolCall via structured JSON so callers can safely mutate
 * the result without leaking changes back into the original tool-call
 * record persisted on the conversation thread.
 */
export function cloneToolCall(toolCall: ToolCall): ToolCall {
  return {
    id: toolCall.id,
    name: toolCall.name,
    arguments: JSON.parse(JSON.stringify(toolCall.arguments ?? {})) as Record<string, unknown>,
  }
}

/**
 * Deep-clone a Message including any nested toolCalls. Array content
 * blocks are JSON-cloned to prevent shared references; scalar string
 * content is copied by spread.
 */
export function cloneMessage(message: Message): Message {
  return {
    ...message,
    content: Array.isArray(message.content)
      ? JSON.parse(JSON.stringify(message.content))
      : message.content,
    toolCalls: message.toolCalls?.map(cloneToolCall),
  }
}

/**
 * Coerce the multiple shapes an approval source can return (boolean,
 * partial ApprovalDecision, undefined) into a fully populated decision so
 * the caller does not have to defend against missing fields. An undefined
 * input is treated as a denial — keeping callers in a fail-safe default
 * when the approval pipeline drops a value.
 */
export function normalizeApprovalDecision(
  decision: boolean | ApprovalDecision | undefined,
): ApprovalDecision {
  if (typeof decision === 'boolean') {
    return {
      decision: decision ? 'approved' : 'denied',
      approved: decision,
    }
  }

  if (decision) {
    const normalizedDecision: ApprovalDecisionStatus =
      decision.decision ?? (decision.approved ? 'approved' : 'denied')
    return {
      decision: normalizedDecision,
      approved: normalizedDecision === 'approved',
      note: decision.note,
      // Preserve the auto-approval marker so tool-execution can decide to
      // emit `auto_approval` instead of `approval_request`.
      autoApproval: decision.autoApproval,
      timedOut: decision.timedOut,
      // `stop` only has meaning for a denial: an approved call cannot
      // simultaneously end the run.
      ...(decision.stop === true && normalizedDecision === 'denied' ? { stop: true } : {}),
    }
  }

  return {
    decision: 'denied',
    approved: false,
  }
}

/**
 * Two scheduling hints conflict when they cannot safely run in parallel.
 * Two parallel-safe hints with the same resource conflict only if they
 * also touch the same key (e.g. two fs.read calls on the same path);
 * different keys on the same resource may still proceed concurrently.
 */
export function schedulingConflicts(
  left: ResolvedToolSchedulingHint,
  right: ResolvedToolSchedulingHint,
): boolean {
  if (left.mode !== 'parallel-safe' || right.mode !== 'parallel-safe') {
    return true
  }

  if (!left.resource || !right.resource) {
    return true
  }

  if (left.resource !== right.resource) {
    return false
  }

  if (!left.key || !right.key) {
    return true
  }

  return left.key === right.key
}

/**
 * Prefix structured-error tool outputs with their error code and a
 * one-line retry hint. The convention `*_TRANSIENT` vs `*_PERMANENT`
 * (set by individual tools) lets the model decide whether retrying is
 * even worth a turn — without this, an EACCES on fs.read and a timeout
 * on terminal.run looked identical to the agent and both got retried
 * with the same arguments. Successful outputs are returned unchanged.
 */
export function decorateErrorOutput(result: {
  output: string
  status: 'success' | 'error'
  code?: string
  executionPosture?: ToolExecutionPosture
}): string {
  if (result.status !== 'error') return result.output
  const decorated = result.code
    ? (() => {
        const transient = /_TRANSIENT$/.test(result.code!)
        const hint = transient
          ? '\n[hint] This error class is usually transient — make at most one focused retry with materially adjusted arguments, or switch to an equivalent capability when one is available; do not repeat or merely extend an already-expired tool deadline.'
          : '\n[hint] This error class is permanent for the same arguments — switch tools or change the inputs rather than retrying.'
        return `[error: ${result.code}] ${result.output}${hint}`
      })()
    : result.output
  if (!result.executionPosture) return decorated
  const posture = result.executionPosture
  return [
    decorated,
    `[execution boundary] sandbox=${posture.sandbox.mode}; filesystem=${posture.filesystem.boundary}; network=${posture.network.mode}. This result proves only what was observable inside that boundary; it does not by itself prove host or external-service state outside it.`,
  ].join('\n')
}

export interface PostToolExecuteResult {
  output: string
  status: 'success' | 'error'
  durationMs?: number
  recovery?: 'journal' | 'probe'
  contentParts?: ContentPart[]
  /** Opt-in visual parts forwarded only on the live AgentEvent. */
  clientContentParts?: ContentPart[]
  executionPosture?: ToolExecutionPosture
  metadata?: ToolResultMetadata
}

export interface PreToolExecuteOutcome {
  blocked: boolean
  /** Tool-result error output when blocked. */
  output?: string
  /** Rewritten tool arguments when a handler returned a modified payload. */
  modifiedArguments?: Record<string, unknown>
}

/**
 * Fire the `pre:tool:execute` gating hook right before a tool runs.
 * Handlers can abort the execution (`action: 'abort'` with an optional
 * `reason`) or rewrite `data.toolCall.arguments` via `modifiedPayload`.
 * Handler exceptions are logged-and-ignored by contract: a crashing hook
 * must not take down all tool execution, so a throw counts as `continue`.
 */
export async function emitPreToolExecute(opts: {
  signal?: AbortSignal
  hookRegistry?: HookRegistry
  sessionId: string
  provider: string
  model: string
  toolCall: ToolCall
  cwd?: string
}): Promise<PreToolExecuteOutcome> {
  if (!opts.hookRegistry) return { blocked: false }
  let result
  try {
    result = await opts.hookRegistry.trigger({
      event: 'pre:tool:execute',
      data: {
        sessionId: opts.sessionId,
        provider: opts.provider,
        model: opts.model,
        toolCall: opts.toolCall,
        cwd: opts.cwd,
      },
    }, opts.signal)
  } catch {
    return { blocked: false }
  }

  if (result.action === 'abort' || result.action === 'skip') {
    const reason = result.reason ?? 'a pre:tool:execute hook blocked this call'
    return {
      blocked: true,
      output: `Tool ${opts.toolCall.name} blocked by hook: ${reason}`,
    }
  }

  const modifiedToolCall = result.modifiedPayload?.data?.toolCall as ToolCall | undefined
  if (
    modifiedToolCall
    && modifiedToolCall.id === opts.toolCall.id
    && modifiedToolCall.arguments
    && modifiedToolCall.arguments !== opts.toolCall.arguments
  ) {
    return { blocked: false, modifiedArguments: modifiedToolCall.arguments }
  }
  return { blocked: false }
}

/**
 * Fire the `post:tool:execute` hook event. Stateless adapter — the
 * caller owns the HookRegistry, sessionId, provider/model, and any
 * extra fields. Errors raised by observation hooks are swallowed
 * deliberately: hooks must never break tool execution.
 *
 * Returns void; the caller does not need to await an outcome.
 */
export async function emitPostToolExecute(opts: {
  hookRegistry?: HookRegistry
  sessionId: string
  provider: string
  model: string
  toolCall: ToolCall
  result: PostToolExecuteResult
  extras?: Record<string, unknown>
}): Promise<void> {
  if (!opts.hookRegistry) return
  try {
    await opts.hookRegistry.trigger({
      event: 'post:tool:execute',
      data: {
        sessionId: opts.sessionId,
        provider: opts.provider,
        model: opts.model,
        toolCall: opts.toolCall,
        result: opts.result,
        ...(opts.extras ?? {}),
      },
    })
  } catch {
    // Observation hooks must not break tool execution.
  }
}

/**
 * Build the running ToolExecutionRecord persisted right before the tool
 * fires. Pure: callers feed timestamp + cloned input.
 */
export function buildRunningToolExecutionRecord(opts: {
  executionId: string
  sessionId: string
  toolCall: ToolCall
  startedAt: string
}): ToolExecutionRecord {
  return {
    executionId: opts.executionId,
    sessionId: opts.sessionId,
    toolCallId: opts.toolCall.id,
    tool: opts.toolCall.name,
    input: cloneToolCallInput(opts.toolCall.arguments),
    startedAt: opts.startedAt,
    status: 'running',
  }
}

/**
 * Build the completed ToolExecutionRecord persisted after the tool
 * resolves. Carries the result snapshot and any recovery source so
 * resume logic knows whether the value came from journal/probe.
 */
export function buildCompletedToolExecutionRecord(opts: {
  executionId: string
  sessionId: string
  toolCall: ToolCall
  startedAt: string
  completedAt: string
  result: ToolExecutionResultSnapshot
}): ToolExecutionRecord {
  return {
    executionId: opts.executionId,
    sessionId: opts.sessionId,
    toolCallId: opts.toolCall.id,
    tool: opts.toolCall.name,
    input: cloneToolCallInput(opts.toolCall.arguments),
    startedAt: opts.startedAt,
    completedAt: opts.completedAt,
    status: 'completed',
    result: opts.result,
  }
}

function cloneToolCallInput(args: ToolCall['arguments'] | undefined): Record<string, unknown> {
  return JSON.parse(JSON.stringify(args ?? {})) as Record<string, unknown>
}

/**
 * Build the tool-result output that runToolExecution emits when an
 * operator denies an approval (or asks for changes). Pure: caller
 * supplies the toolName, decision flavour, and optional note. The
 * output is shaped so the agent can reason about the outcome by
 * matching on the bracketed status tag instead of parsing free-form
 * prose, and the trailing guidance line nudges the model away from
 * the most common failure mode (re-emitting the same tool call after
 * a denial).
 */
export function buildDenialToolOutput(opts: {
  toolName: string
  decision: 'denied' | 'feedback'
  note?: string
  /**
   * True when the prompt expired instead of the user answering it. An
   * unanswered prompt is not a refusal, and conflating the two is actively
   * harmful: the agent is told the user declined, is forbidden from retrying,
   * and the user — who may have simply been away from the screen — has no way
   * back to the action except starting the request over.
   */
  timedOut?: boolean
}): string {
  if (opts.decision !== 'feedback' && opts.timedOut) {
    return [
      '[approval:timeout] The approval prompt for '
        + `${opts.toolName} expired before anyone answered it.`,
      'This is not a refusal — the user may have been away from the screen.',
      'Say that the action is still pending approval and offer to try again;'
        + ' retry the same call if they confirm.',
    ].join('\n')
  }
  const statusTag = opts.decision === 'feedback' ? '[approval:needs-changes]' : '[approval:denied]'
  const headline =
    opts.decision === 'feedback'
      ? `${statusTag} The user wants ${opts.toolName} adjusted before it runs.`
      : `${statusTag} The user declined to run ${opts.toolName} with these arguments.`
  const noteLine = opts.note ? `\nuser note: ${opts.note}` : ''
  const guidance =
    opts.decision === 'feedback'
      ? '\nApply the requested changes and propose a revised tool call, or ask a clarifying question.'
      : '\nThis denial ends the current turn. Do not retry, rewrite, or work around the rejected action with another side-effecting tool.'
  return `${headline}${noteLine}${guidance}`
}

export interface PreparedParallelToolCall {
  toolCall: ToolCall
  tool: ToolDefinitionRuntime
  scheduling: ResolvedToolSchedulingHint
}

/**
 * Decide whether a tool call is eligible for the parallel-execution
 * batch. Returns null when:
 *   - the tool is not registered
 *   - policy denies or asks for approval
 *   - the tool is replay-risky (so we cannot safely retry it on resume)
 *   - the tool's scheduling hint isn't parallel-safe
 *
 * Pure: every dependency is supplied as an argument so this can be
 * unit-tested without spinning up a full agent runtime.
 */
export function missingRequiredToolArguments(
  tool: Pick<ToolDefinitionRuntime, 'inputSchema'>,
  input: ToolCall['arguments'],
): string[] {
  const required = tool.inputSchema.required
  return Array.isArray(required)
    ? required.filter((key): key is string => typeof key === 'string'
      && (!Object.hasOwn(input, key) || input[key] === undefined))
    : []
}

export function prepareParallelToolCall(opts: {
  toolCall: ToolCall
  tools: ToolRegistry
  policy: PolicyEngine
  autonomy: AutonomyLevel
  cwd?: string
  workspaceRoot?: string
  primaryAgentId?: string
  autoApprove?: boolean
}): PreparedParallelToolCall | null {
  const tool = opts.tools.get(opts.toolCall.name)
  if (!tool) return null
  // Invalid siblings must reach the ordinary validation result path, rather
  // than bypassing it when a valid first call starts a parallel batch.
  if (missingRequiredToolArguments(tool, opts.toolCall.arguments).length > 0) return null

  const policyResult = opts.policy.check(
    {
      tool: opts.toolCall.name,
      input: opts.toolCall.arguments,
      cwd: opts.cwd,
      workspaceRoot: opts.workspaceRoot,
      registrationSource: opts.tools.registrationSource(opts.toolCall.name),
      security: opts.tools.securityDescriptor(opts.toolCall.name),
    },
    opts.autonomy,
    opts.primaryAgentId,
    opts.autoApprove,
  )
  if (
    !policyResult.allowed ||
    policyResult.requiresApproval
  ) {
    return null
  }

  if (resolveToolResumeSafety(tool, opts.toolCall.arguments) !== 'replay-safe') {
    return null
  }

  // A failure-coupled fan-out must observe each result before starting the
  // next sibling. Otherwise every sibling can repeat one deterministic input
  // template error concurrently before the executor has a chance to repair.
  if (resolveToolBatchFailureCouplingKey(tool, opts.toolCall.arguments)) {
    return null
  }

  if (
    opts.toolCall.name === 'subagent.dispatch'
    && typeof opts.toolCall.arguments?.category === 'string'
    && WRITE_CAPABLE_SUBAGENT_CATEGORIES.has(opts.toolCall.arguments.category)
  ) {
    return null
  }

  const scheduling = resolveToolSchedulingHint(tool, opts.toolCall.arguments)
  if (scheduling.mode !== 'parallel-safe') return null

  return { toolCall: opts.toolCall, tool, scheduling }
}

export function resolveParallelToolMax(): number {
  const raw = Number(process.env.SEPILOTD_PARALLEL_TOOL_MAX ?? '')
  if (Number.isFinite(raw) && raw >= PARALLEL_TOOL_MAX_FLOOR) {
    return Math.min(PARALLEL_TOOL_MAX_CEILING, Math.floor(raw))
  }
  return DEFAULT_PARALLEL_TOOL_MAX
}

export function delegatedUsageFromMetadata(metadata: ToolResultMetadata | undefined): TokenUsage | undefined {
  const delegated = metadata?.delegatedUsage
  if (!delegated || typeof delegated !== 'object') {
    return undefined
  }
  const record = delegated as Record<string, unknown>
  const inputTokens = typeof record.inputTokens === 'number' && Number.isFinite(record.inputTokens)
    ? record.inputTokens
    : 0
  const outputTokens = typeof record.outputTokens === 'number' && Number.isFinite(record.outputTokens)
    ? record.outputTokens
    : 0
  return { inputTokens, outputTokens }
}

/**
 * Walk forward from startAt collecting consecutive tool calls that
 * (a) prepareParallelToolCall accepts and (b) do not conflict with any
 * tool already in the batch by scheduling resource/key. The first
 * rejection breaks the batch — we never reorder calls.
 */
export function collectParallelBatch(opts: {
  toolCalls: ToolCall[]
  startAt: number
  current: PreparedParallelToolCall
  tools: ToolRegistry
  policy: PolicyEngine
  autonomy: AutonomyLevel
  cwd?: string
  workspaceRoot?: string
  primaryAgentId?: string
  autoApprove?: boolean
  /**
   * Per-turn consent guard. A candidate that needs its own fresh approval
   * must remain sequential so an earlier approved/read-only call cannot carry
   * it into the same parallel batch.
   */
  requiresFreshApproval?: (toolName: string) => boolean
}): PreparedParallelToolCall[] {
  const maxBatchSize = resolveParallelToolMax()
  const batch = [opts.current]
  for (let nextIndex = opts.startAt + 1; nextIndex < opts.toolCalls.length; nextIndex++) {
    if (batch.length >= maxBatchSize) break
    if (opts.requiresFreshApproval?.(opts.toolCalls[nextIndex]!.name)) break
    const candidate = prepareParallelToolCall({
      toolCall: opts.toolCalls[nextIndex]!,
      tools: opts.tools,
      policy: opts.policy,
      autonomy: opts.autonomy,
      cwd: opts.cwd,
      workspaceRoot: opts.workspaceRoot,
      primaryAgentId: opts.primaryAgentId,
      autoApprove: opts.autoApprove,
    })
    if (!candidate) break
    if (batch.some((existing) => schedulingConflicts(existing.scheduling, candidate.scheduling))) {
      break
    }
    batch.push(candidate)
  }
  return batch
}

/**
 * Fire the `post:file:edit` hook event for tools that mutate files
 * (fs.write, fs.append, fs.edit, apply_patch). Stateless adapter — caller owns
 * the HookRegistry. Errors are swallowed; observation hooks must not
 * break tool execution. The caller decides whether to invoke this
 * (typically only when the tool succeeded and isFileEditTool returns
 * true).
 */
export async function emitPostFileEdit(opts: {
  hookRegistry?: HookRegistry
  sessionId: string
  toolCall: ToolCall
  output: string
}): Promise<void> {
  if (!opts.hookRegistry) return
  try {
    await opts.hookRegistry.trigger({
      event: 'post:file:edit',
      data: {
        tool: opts.toolCall.name,
        sessionId: opts.sessionId,
        input: opts.toolCall.arguments,
        output: opts.output,
      },
    })
  } catch {
    // Observation hooks must not fail tool execution.
  }
}

/**
 * Walk the toolCalls array from startIndex onward and let any tool
 * with a normalizeInput hook canonicalize its arguments. Mutates the
 * array in place when normalizeInput returns a different value (an
 * identity check is enough — tools return the same reference when no
 * normalisation was needed). Tools without a normalizeInput hook are
 * left untouched.
 *
 * Pulled out of runToolExecution so callers (and tests) don't have to
 * re-derive the loop semantics. Stays "in place" because the
 * generator already passes its toolCalls slot around as a mutable
 * scratch buffer; returning a fresh array would require threading the
 * new reference through every yield.
 */
export async function normalizeToolCallInputsInPlace(opts: {
  toolCalls: ToolCall[]
  startIndex: number
  tools: ToolRegistry
  cwd?: string
  workspaceRoot?: string
}): Promise<void> {
  for (let index = opts.startIndex; index < opts.toolCalls.length; index++) {
    const toolCall = opts.toolCalls[index]!
    const tool = opts.tools.get(toolCall.name)
    if (!tool?.normalizeInput) continue
    const normalizedArguments = await tool.normalizeInput(toolCall.arguments, {
      cwd: opts.cwd,
      workspaceRoot: opts.workspaceRoot,
    })
    if (normalizedArguments !== toolCall.arguments) {
      opts.toolCalls[index] = {
        ...toolCall,
        arguments: normalizedArguments,
      }
    }
  }
}

/**
 * Persist the run checkpoint pointing at the next tool call when one
 * remains. Called after each `continue` in runToolExecution so a
 * crash mid-way through a multi-tool batch resumes from the right
 * spot without re-running tools that already completed. No-ops when
 * `nextIndex` is past the last tool (the caller's loop will exit on
 * the next iteration anyway) and when no checkpoint sink is wired
 * up. The narrow signature — just the writer plus the array length
 * and target index — is intentional: this helper has no business
 * touching pending.currentExecutionId or initialApprovalDecision,
 * which only the resume-from-mid-tool branches care about.
 */
export async function advanceRunCheckpoint(opts: {
  persistRunCheckpoint?: (pending: { toolCalls: ToolCall[]; startIndex: number }) => Promise<void>
  toolCalls: ToolCall[]
  nextIndex: number
}): Promise<void> {
  if (opts.nextIndex < opts.toolCalls.length) {
    await opts.persistRunCheckpoint?.({
      toolCalls: opts.toolCalls,
      startIndex: opts.nextIndex,
    })
  }
}

interface ToolResultTailDeps {
  emitPostToolExecute: (
    toolCall: ToolCall,
    result: PostToolExecuteResult,
    extras: Record<string, unknown>,
  ) => Promise<void>
  appendToolResultMessage: (
    toolCall: ToolCall,
    output: string,
    contentParts?: ContentPart[],
    metadata?: Record<string, unknown>,
  ) => void
  clearApprovalCheckpoint?: (id: string) => Promise<void>
}

/**
 * Shared post-yield-push-clear sequence for every branch of
 * runToolExecution that emits a tool_result. The order — fire post
 * hook → yield to the consumer → push the tool message → clear
 * the approval checkpoint — matches the original inline blocks
 * exactly so consumers observe the tool_result event between the
 * hook fire and the in-memory cleanup, preserving pre-extraction
 * behaviour bit-for-bit.
 *
 * `eventRecoveryOverride` lets callers (the journal-hit branch)
 * surface a different recovery tag on the event from the one the
 * hook saw — observers consuming post:tool:execute want to know
 * "we read this from the on-disk journal", but the agent-side event
 * preserves the journaled record's own recovery field so consumers
 * still see when the original execution was probe-recovered.
 */
async function* emitToolResultTail(
  opts: ToolResultTailDeps & {
    toolCall: ToolCall
    result: PostToolExecuteResult
    extras: Record<string, unknown>
    eventRecoveryOverride?: 'journal' | 'probe'
    trustedApprovalFailureStatus?: 'denied'
    trustedPolicyFailureReason?: string
    /** Set for calls refused before execution by policy or by a human. */
    trustedBlockSource?: ToolResultBlockSource
    /** Operator asked to end the run with the denial (`deny & stop`). */
    trustedApprovalStopRequested?: boolean
    /** Free-text note the operator attached to a denial. */
    trustedApprovalDenialNote?: string
    executionObserved?: boolean
  },
): AsyncGenerator<AgentEvent> {
  const hookResult: PostToolExecuteResult = {
    output: opts.result.output,
    status: opts.result.status,
    ...(opts.result.durationMs !== undefined ? { durationMs: opts.result.durationMs } : {}),
    ...(opts.result.recovery ? { recovery: opts.result.recovery } : {}),
    ...(opts.result.executionPosture ? { executionPosture: opts.result.executionPosture } : {}),
    ...(opts.result.metadata ? { metadata: opts.result.metadata } : {}),
  }
  await opts.emitPostToolExecute(opts.toolCall, hookResult, opts.extras)
  const eventRecovery = opts.eventRecoveryOverride ?? opts.result.recovery
  const eventMetadata = {
    ...opts.result.metadata,
    ...(typeof opts.executionObserved === 'boolean'
      ? { [TOOL_RESULT_EXECUTION_OBSERVED_METADATA_KEY]: opts.executionObserved }
      : {}),
    ...(opts.trustedApprovalFailureStatus
      ? { [APPROVAL_FAILURE_STATUS_METADATA_KEY]: opts.trustedApprovalFailureStatus }
      : {}),
    ...(opts.trustedPolicyFailureReason
      ? { [POLICY_FAILURE_REASON_METADATA_KEY]: opts.trustedPolicyFailureReason }
      : {}),
    ...(opts.trustedBlockSource
      ? { [TOOL_RESULT_BLOCK_SOURCE_METADATA_KEY]: opts.trustedBlockSource }
      : {}),
    ...(opts.trustedApprovalStopRequested
      ? { [APPROVAL_STOP_REQUESTED_METADATA_KEY]: true }
      : {}),
    ...(opts.trustedApprovalDenialNote
      ? { [APPROVAL_DENIAL_NOTE_METADATA_KEY]: opts.trustedApprovalDenialNote }
      : {}),
  }
  yield {
    type: 'tool_result',
    toolCallId: opts.toolCall.id,
    output: opts.result.output,
    status: opts.result.status,
    ...(opts.result.clientContentParts?.length
      ? { contentParts: opts.result.clientContentParts }
      : {}),
    ...(eventRecovery ? { recovery: eventRecovery } : {}),
    ...(opts.result.executionPosture ? { executionPosture: opts.result.executionPosture } : {}),
    ...(Object.keys(eventMetadata).length > 0 ? { metadata: eventMetadata } : {}),
  }
  // Tool-supplied result metadata is intentionally not copied into the
  // provider-facing Message. Only internal approval/policy branches can set
  // this provenance, preventing a plugin/MCP result from spoofing a trusted
  // terminal denial or policy block with matching text and metadata.
  const messageMetadata = {
    toolResultStatus: opts.result.status,
    ...(opts.result.executionPosture
      ? { executionPosture: opts.result.executionPosture }
      : {}),
    ...(typeof opts.executionObserved === 'boolean'
      ? { [TOOL_RESULT_EXECUTION_OBSERVED_METADATA_KEY]: opts.executionObserved }
      : {}),
    ...(opts.trustedApprovalFailureStatus
      ? { [APPROVAL_FAILURE_STATUS_METADATA_KEY]: opts.trustedApprovalFailureStatus }
      : {}),
    ...(opts.trustedPolicyFailureReason
      ? { [POLICY_FAILURE_REASON_METADATA_KEY]: opts.trustedPolicyFailureReason }
      : {}),
    ...(opts.trustedBlockSource
      ? { [TOOL_RESULT_BLOCK_SOURCE_METADATA_KEY]: opts.trustedBlockSource }
      : {}),
    ...(opts.trustedApprovalStopRequested
      ? { [APPROVAL_STOP_REQUESTED_METADATA_KEY]: true }
      : {}),
    ...(opts.trustedApprovalDenialNote
      ? { [APPROVAL_DENIAL_NOTE_METADATA_KEY]: opts.trustedApprovalDenialNote }
      : {}),
  }
  if (opts.result.contentParts) {
    opts.appendToolResultMessage(
      opts.toolCall,
      opts.result.output,
      opts.result.contentParts,
      messageMetadata,
    )
  } else {
    opts.appendToolResultMessage(opts.toolCall, opts.result.output, undefined, messageMetadata)
  }
  await opts.clearApprovalCheckpoint?.(opts.toolCall.id)
}

/**
 * Tool failed before execution — policy block, missing approval
 * handler, denied approval, unknown tool. status is forced to
 * 'error' so callers don't have to spell it out.
 */
export async function* yieldToolError(
  opts: ToolResultTailDeps & {
    toolCall: ToolCall
    output: string
    extras: Record<string, unknown>
  },
): AsyncGenerator<AgentEvent> {
  const approvalFailureStatus = opts.extras.source === 'approval' && opts.extras.timedOut !== true
    ? approvalFailureStatusFromDecision(opts.extras.decision)
    : null
  const policyFailureReason = opts.extras.source === 'policy'
    && typeof opts.extras.reason === 'string'
    && opts.extras.reason.trim().length > 0
    ? opts.extras.reason
    : null
  // Policy and approval refusals are friction, not tool failures: loop-control
  // counters read this provenance to keep them out of failure accounting.
  const blockSource: ToolResultBlockSource | null =
    opts.extras.source === 'policy' || opts.extras.source === 'approval'
      ? opts.extras.source
      : null
  const stopRequested = approvalFailureStatus === 'denied' && opts.extras.stop === true
  const denialNote = approvalFailureStatus === 'denied'
    && typeof opts.extras.note === 'string'
    && opts.extras.note.trim().length > 0
    ? opts.extras.note.trim()
    : null
  yield* emitToolResultTail({
    toolCall: opts.toolCall,
    result: { output: opts.output, status: 'error' },
    extras: opts.extras,
    executionObserved: false,
    ...(approvalFailureStatus
      ? { trustedApprovalFailureStatus: approvalFailureStatus }
      : {}),
    ...(policyFailureReason
      ? { trustedPolicyFailureReason: policyFailureReason }
      : {}),
    ...(blockSource ? { trustedBlockSource: blockSource } : {}),
    ...(stopRequested ? { trustedApprovalStopRequested: true } : {}),
    ...(denialNote ? { trustedApprovalDenialNote: denialNote } : {}),
    emitPostToolExecute: opts.emitPostToolExecute,
    appendToolResultMessage: opts.appendToolResultMessage,
    clearApprovalCheckpoint: opts.clearApprovalCheckpoint,
  })
}

/**
 * Tool ran (or was recovered from journal/probe) and produced a
 * result. The caller threads through the result's status, durationMs
 * and optional recovery tag — both the post hook payload and the
 * yielded event surface them. `eventRecoveryOverride` is only used
 * by the journal-hit branch where the hook recovery field
 * intentionally differs from the event recovery field.
 */
export async function* yieldToolSuccessOrRecovery(
  opts: ToolResultTailDeps & {
    toolCall: ToolCall
    result: PostToolExecuteResult
    extras: Record<string, unknown>
    eventRecoveryOverride?: 'journal' | 'probe'
  },
): AsyncGenerator<AgentEvent> {
  yield* emitToolResultTail({ ...opts, executionObserved: true })
}
