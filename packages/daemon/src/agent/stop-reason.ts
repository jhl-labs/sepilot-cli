import type {
  AgentRunContract,
  RunStopCode,
  RunStopKind,
  RunStopNextAction,
  RunStopReason,
} from '@sepilotd/core'

export const INCOMPLETE_OUTPUT_PREFIX = 'INCOMPLETE:'
const INCOMPLETE_OUTPUT_STEM = /^(?:INCOMPLETE|UNVERIFIED):/

export function isIncompleteOutput(content: string | undefined): boolean {
  return typeof content === 'string' && INCOMPLETE_OUTPUT_STEM.test(content.trimStart())
}

export type BudgetStopLayer =
  | 'iteration'
  | 'node'
  | 'observation'
  | 'exact_tool'
  | 'cycle'
  | 'completion_gate'

function criteriaIds(contract?: AgentRunContract): string[] | undefined {
  const ids = contract?.acceptanceCriteria?.map((criterion) => criterion.id) ?? []
  return ids.length > 0 ? ids : undefined
}

function withDetail(
  detail: RunStopReason['detail'] | undefined,
): Pick<RunStopReason, 'detail'> {
  if (!detail) return {}
  const entries = Object.entries(detail).filter(([, value]) => value !== undefined)
  return entries.length > 0 ? { detail: Object.fromEntries(entries) } : {}
}

export function stopReasonCompleted(input?: {
  summary?: string
  layer?: string
}): RunStopReason {
  return {
    kind: 'completed',
    code: 'completed',
    summary: input?.summary ?? 'Run completed.',
    ...withDetail(input?.layer ? { layer: input.layer } : undefined),
    resumable: false,
    nextActions: [],
  }
}

export function stopReasonBudget(
  layer: 'iteration' | 'node' | 'exact_tool',
  budget: number | undefined,
  used: number | undefined,
  contract?: AgentRunContract,
): RunStopReason {
  return {
    kind: 'incomplete',
    code: layer === 'node' ? 'node_budget' : 'iteration_budget',
    summary: layer === 'node'
      ? `Node execution budget (${budget ?? '?'}) exhausted before completion.`
      : `Iteration budget (${budget ?? '?'}) exhausted before completion.`,
    ...withDetail({
      budget,
      used,
      criteria: criteriaIds(contract),
      layer: layer === 'iteration' ? undefined : layer,
    }),
    resumable: true,
    nextActions: ['resume', 'raise_budget'],
  }
}

export function stopReasonObservationBudget(input: {
  incomplete: boolean
  budget?: number
  used?: number
  contract?: AgentRunContract
}): RunStopReason {
  if (!input.incomplete) {
    return stopReasonCompleted({
      summary: 'Run completed after the observation budget closed tool access.',
      layer: 'observation_budget',
    })
  }
  return {
    kind: 'incomplete',
    code: 'observation_budget',
    summary: 'Observation budget closed tool access before the run could complete.',
    ...withDetail({
      budget: input.budget,
      used: input.used,
      criteria: criteriaIds(input.contract),
    }),
    resumable: true,
    nextActions: ['resume', 'raise_budget'],
  }
}

export function stopReasonNoProgress(input?: {
  layer?: 'cycle' | 'recovery' | 'provider' | 'continuation' | 'question'
  contract?: AgentRunContract
  budget?: number
  used?: number
}): RunStopReason {
  return {
    kind: 'incomplete',
    code: 'no_progress',
    summary: input?.layer === 'continuation'
      ? 'Run stopped at its iteration budget because the last cycle produced no new evidence or artifacts.'
      : 'Run stopped because it kept revisiting the same state without advancing.',
    ...withDetail({
      layer: input?.layer,
      budget: input?.budget,
      used: input?.used,
      criteria: criteriaIds(input?.contract),
    }),
    resumable: true,
    nextActions: ['resume', 'retry'],
  }
}

export function stopReasonStuckRepeat(input?: {
  tool?: string
  layer?: 'question'
  contract?: AgentRunContract
}): RunStopReason {
  return {
    kind: 'incomplete',
    code: 'stuck_repeat',
    summary: input?.layer === 'question'
      ? 'Run stopped at the user\'s request after the same tool call kept repeating.'
      : 'Run stopped after the same tool call repeated past the repair limit.',
    ...withDetail({
      tool: input?.tool,
      layer: input?.layer,
      criteria: criteriaIds(input?.contract),
    }),
    resumable: true,
    nextActions: ['resume', 'retry'],
  }
}

export type CompletionGateSource =
  | 'gate'
  | 'self_reported'
  | 'provider_no_progress'
  | 'answer_verification'
  | 'quality_phase'

const MAX_REPORTED_REASON_CHARS = 300

export function incompleteOutputReason(content: string | undefined): string | undefined {
  if (!isIncompleteOutput(content)) return undefined
  const body = content!.trimStart().replace(INCOMPLETE_OUTPUT_STEM, '')
  const line = body.split('\n').map((part) => part.trim()).find(Boolean)
  if (!line) return undefined
  return line.length > MAX_REPORTED_REASON_CHARS ? `${line.slice(0, MAX_REPORTED_REASON_CHARS - 1)}…` : line
}

function completionGateSummary(source: CompletionGateSource, reason: string | undefined): string {
  switch (source) {
    case 'self_reported':
      return reason ? `Agent reported the work incomplete: ${reason}` : 'Agent reported the work incomplete.'
    case 'provider_no_progress':
      return 'Provider responses made no progress; the run ended without a usable final response.'
    case 'answer_verification':
      return reason ? `Final answer verification did not pass: ${reason}` : 'Final answer verification did not pass.'
    case 'quality_phase':
      return 'The required quality phase did not establish acceptance.'
    case 'gate':
      return reason
        ? `Completion gate block budget exhausted with unmet acceptance criteria: ${reason}`
        : 'Completion gate block budget exhausted with unmet acceptance criteria.'
  }
}

export function stopReasonCompletionGate(input: {
  unmet?: string[]
  budget?: number
  used?: number
  source?: CompletionGateSource
}): RunStopReason {
  const source = input.source ?? 'gate'
  return {
    kind: 'incomplete',
    code: 'completion_gate',
    summary: completionGateSummary(source, input.unmet?.[0]),
    ...withDetail({
      budget: input.budget,
      used: input.used,
      criteria: input.unmet && input.unmet.length > 0 ? [...input.unmet] : undefined,
      layer: source === 'gate' ? undefined : source,
    }),
    resumable: true,
    nextActions: ['resume', 'raise_budget'],
  }
}

export function stopReasonCostGate(input: {
  budget?: number
  used?: number
  layer?: string
}): RunStopReason {
  return {
    kind: 'incomplete',
    code: 'cost_gate',
    summary: 'Token cost gate refused further retries.',
    ...withDetail({ budget: input.budget, used: input.used, layer: input.layer }),
    resumable: true,
    nextActions: ['raise_budget', 'resume'],
  }
}

export function stopReasonSpendBudget(input?: {
  budget?: number
  used?: number
}): RunStopReason {
  return {
    kind: 'blocked',
    code: 'spend_budget',
    summary: 'Priced spend budget exceeded; the provider call was refused.',
    ...withDetail({ budget: input?.budget, used: input?.used }),
    resumable: true,
    nextActions: ['raise_budget', 'resume'],
  }
}

export function stopReasonUserActionRequired(message: string): RunStopReason {
  return {
    kind: 'blocked',
    code: 'user_action_required',
    summary: message,
    resumable: true,
    nextActions: ['resume'],
  }
}

export function stopReasonApprovalDenied(tool: string, requestId?: string): RunStopReason {
  return {
    kind: 'blocked',
    code: 'approval_denied',
    summary: `Approval for '${tool}' was denied; the run stopped without executing it.`,
    ...withDetail({ tool, requestId }),
    resumable: true,
    nextActions: ['resume', 'switch_autonomy'],
  }
}

export function stopReasonApprovalTimeout(input: {
  requestId: string
  tool?: string
  layer?: 'approval' | 'question'
}): RunStopReason {
  const layer = input.layer ?? 'approval'
  return {
    kind: 'blocked',
    code: 'approval_timeout',
    summary: layer === 'question'
      ? 'The run waited for an answer to a pending question and none arrived.'
      : 'The run waited for a pending approval and no decision arrived.',
    ...withDetail({ requestId: input.requestId, tool: input.tool, layer }),
    resumable: true,
    nextActions: ['approve_pending', 'resume'],
  }
}

export function stopReasonPolicyBlocked(input: {
  tool?: string
  layer: string
}): RunStopReason {
  return {
    kind: 'blocked',
    code: 'policy_blocked',
    summary: `Tool policy (${input.layer}) blocked the action the run needed.`,
    ...withDetail({ tool: input.tool, layer: input.layer }),
    resumable: true,
    nextActions: ['switch_autonomy', 'resume'],
  }
}

export function stopReasonWallClock(budgetMs?: number): RunStopReason {
  return {
    kind: 'incomplete',
    code: 'wall_clock',
    summary: 'Run stopped by the wall-clock deadline.',
    ...withDetail({ budget: budgetMs }),
    resumable: true,
    nextActions: ['resume', 'raise_budget'],
  }
}

export function stopReasonInactivity(input?: {
  budgetMs?: number
  layer?: 'no_model_output' | 'stalled'
}): RunStopReason {
  return {
    kind: 'error',
    code: 'inactivity',
    summary: 'Run stopped after the inactivity watchdog fired.',
    ...withDetail({ budget: input?.budgetMs, layer: input?.layer }),
    resumable: true,
    nextActions: ['retry', 'resume'],
  }
}

export function stopReasonUserAbort(): RunStopReason {
  return {
    kind: 'cancelled',
    code: 'user_abort',
    summary: 'Run cancelled by the user.',
    resumable: true,
    nextActions: ['resume'],
  }
}

export function stopReasonProviderError(input?: {
  layer?: string
  summary?: string
}): RunStopReason {
  return {
    kind: 'error',
    code: 'provider_error',
    summary: input?.summary ?? 'The provider call failed.',
    ...withDetail({ layer: input?.layer }),
    resumable: true,
    nextActions: ['retry'],
  }
}

export function isTruncatedStopReason(reason: RunStopReason | undefined): boolean {
  return reason != null && reason.kind !== 'completed' && reason.kind !== 'error'
    && reason.kind !== 'cancelled'
}

const STOP_KIND_BY_CODE: Record<RunStopCode, RunStopKind> = {
  completed: 'completed',
  iteration_budget: 'incomplete',
  node_budget: 'incomplete',
  observation_budget: 'incomplete',
  no_progress: 'incomplete',
  stuck_repeat: 'incomplete',
  completion_gate: 'incomplete',
  cost_gate: 'incomplete',
  spend_budget: 'blocked',
  approval_denied: 'blocked',
  approval_timeout: 'blocked',
  user_action_required: 'blocked',
  policy_blocked: 'blocked',
  wall_clock: 'incomplete',
  inactivity: 'error',
  user_abort: 'cancelled',
  provider_error: 'error',
  agent_incomplete: 'incomplete',
}

export function stopKindForCode(code: RunStopCode): RunStopKind {
  return STOP_KIND_BY_CODE[code]
}

export type { RunStopNextAction }
