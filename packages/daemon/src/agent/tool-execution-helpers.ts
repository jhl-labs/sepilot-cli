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

export interface ToolExecutionAuditEntry {
  timestamp: string
  event: 'tool.execute' | 'tool.autonomous_approval_block'
  device: string
  session: string
  tool: string
  input_summary: string
  autonomy: AutonomyLevel
  auto_approve_flag: boolean
  [key: string]: unknown
}

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

const FILE_EDIT_TOOLS = new Set<string>(['fs.write', 'fs.append', 'fs.edit', 'apply_patch'])

export function isFileEditTool(name: string): boolean {
  return FILE_EDIT_TOOLS.has(name)
}

export function cloneToolCall(toolCall: ToolCall): ToolCall {
  return {
    id: toolCall.id,
    name: toolCall.name,
    arguments: JSON.parse(JSON.stringify(toolCall.arguments ?? {})) as Record<string, unknown>,
  }
}

export function cloneMessage(message: Message): Message {
  return {
    ...message,
    content: Array.isArray(message.content)
      ? JSON.parse(JSON.stringify(message.content))
      : message.content,
    toolCalls: message.toolCalls?.map(cloneToolCall),
  }
}

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
      autoApproval: decision.autoApproval,
      timedOut: decision.timedOut,
      ...(decision.stop === true && normalizedDecision === 'denied' ? { stop: true } : {}),
    }
  }

  return {
    decision: 'denied',
    approved: false,
  }
}

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
  clientContentParts?: ContentPart[]
  executionPosture?: ToolExecutionPosture
  metadata?: ToolResultMetadata
}

export interface PreToolExecuteOutcome {
  blocked: boolean
  output?: string
  modifiedArguments?: Record<string, unknown>
}

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
  }
}

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

export function buildDenialToolOutput(opts: {
  toolName: string
  decision: 'denied' | 'feedback'
  note?: string
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
  }
}

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

async function* emitToolResultTail(
  opts: ToolResultTailDeps & {
    toolCall: ToolCall
    result: PostToolExecuteResult
    extras: Record<string, unknown>
    eventRecoveryOverride?: 'journal' | 'probe'
    trustedApprovalFailureStatus?: 'denied'
    trustedPolicyFailureReason?: string
    trustedBlockSource?: ToolResultBlockSource
    trustedApprovalStopRequested?: boolean
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
    result: {
      output: opts.output,
      status: 'error',
      ...(policyFailureReason && opts.extras.userActionRequiredCode === 'workspace_execution_unavailable'
        ? { metadata: {
            userActionRequired: policyFailureReason,
            userActionRequiredCode: 'workspace_execution_unavailable',
          } }
        : {}),
    },
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
