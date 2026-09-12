import type {
  ActiveSkillExecutionPolicy,
  Message,
  SkillExecutionStage,
  ToolCall,
} from '@sepilotd/core'
import { TOOL_RESULT_EXECUTION_OBSERVED_METADATA_KEY } from './policy-failure.js'

export interface SkillExecutionToolHistoryEntry {
  tool: string
  status: 'success' | 'error'
  input?: Record<string, unknown>
  executionObserved?: boolean
}

export interface SkillExecutionToolCallPartition {
  executableCalls: ToolCall[]
  rejectedCalls: ToolCall[]
  prioritizedToolNames: string[]
}

export interface SkillExecutionToolCallPartitionOptions {
  /**
   * Reserve a bounded evidence phase for the currently actionable required
   * skill stage before allowing optional observations. Outside those phases,
   * unrelated calls retain their existing permissive behavior.
   */
  prioritizeRequiredStages?: boolean
  /** Only prioritize required stages whose canonical tool is actually visible. */
  availableToolNames?: ReadonlySet<string>
}

export interface MissingSkillExecutionStage {
  skillId: string
  stageId: string
  tools: string[]
}

export interface SkillExecutionCompletion {
  missing: MissingSkillExecutionStage[]
  maxRetries: number
}

interface ObservedToolCall {
  name: string
  resultIndex: number
  input: Record<string, unknown>
  status: 'success' | 'error'
  executionObserved: boolean
}

interface PolicyState {
  active: ActiveSkillExecutionPolicy
  stageByTool: Map<string, { stage: SkillExecutionStage; index: number }>
  counts: Map<string, number>
  satisfied: Set<string>
  satisfiedBeforeBatch: ReadonlySet<string>
  furthestStageIndex: number
  bindingValues: Map<string, unknown>
}

const IMPLICIT_CURRENT_ARGUMENT = Symbol('implicit-current-argument')

function stageCallLimit(stage: SkillExecutionStage): number {
  return stage.maxCallsPerTurn ?? Number.POSITIVE_INFINITY
}

function currentTurnStart(messages: readonly Message[]): number {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (
      messages[index]?.role === 'user'
      && messages[index]?.metadata?.currentAgentTurnUserInput === true
    ) {
      return index
    }
  }
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === 'user') return index
  }
  return -1
}

function observedToolCalls(messages: readonly Message[]): ObservedToolCall[] {
  const start = currentTurnStart(messages)
  const calls = new Map<string, ToolCall>()
  const observed: ObservedToolCall[] = []

  for (let index = start + 1; index < messages.length; index += 1) {
    const message = messages[index]!
    for (const call of message.toolCalls ?? []) calls.set(call.id, call)
    if (message.role !== 'tool' || typeof message.toolCallId !== 'string') continue
    const rawStatus = message.metadata?.toolResultStatus ?? message.metadata?.status
    if (rawStatus !== 'success' && rawStatus !== 'error') continue
    const call = calls.get(message.toolCallId)
    if (call) {
      observed.push({
        name: call.name,
        resultIndex: index,
        input: call.arguments,
        status: rawStatus,
        executionObserved:
          rawStatus === 'success'
          || message.metadata?.[TOOL_RESULT_EXECUTION_OBSERVED_METADATA_KEY] === true,
      })
    }
  }
  return observed
}

function createPolicyState(
  active: ActiveSkillExecutionPolicy,
  observed: readonly ObservedToolCall[],
): PolicyState {
  const stageByTool = new Map<string, { stage: SkillExecutionStage; index: number }>()
  active.policy.stages.forEach((stage, index) => {
    for (const tool of stage.tools) stageByTool.set(tool, { stage, index })
  })
  const counts = new Map<string, number>()
  const satisfied = new Set<string>()
  const bindingValues = new Map<string, unknown>()
  let furthestStageIndex = -1
  for (const call of observed) {
    const matched = stageByTool.get(call.name)
    if (!matched) continue
    const satisfies = call.status === 'success'
      || (
        matched.stage.satisfyOn === 'executed-outcome'
        && call.executionObserved
      )
    if (!satisfies) continue
    counts.set(matched.stage.id, (counts.get(matched.stage.id) ?? 0) + 1)
    satisfied.add(matched.stage.id)
    furthestStageIndex = Math.max(furthestStageIndex, matched.index)
    recordBindingValues(active, matched.stage.id, call.input, bindingValues)
  }
  return {
    active,
    stageByTool,
    counts,
    satisfied,
    satisfiedBeforeBatch: new Set(satisfied),
    furthestStageIndex,
    bindingValues,
  }
}

function bindingTargetArgument(
  active: ActiveSkillExecutionPolicy,
  bindingId: string,
  stageId: string,
): string | undefined {
  return active.policy.argumentBindings
    ?.find((binding) => binding.id === bindingId)
    ?.targets.find((target) => target.stage === stageId)
    ?.argument
}

function readExplicitArgument(
  input: Record<string, unknown>,
  argument: string,
): { present: boolean; value?: unknown } {
  if (!Object.hasOwn(input, argument) || input[argument] === undefined) {
    return { present: false }
  }
  return { present: true, value: input[argument] }
}

function boundValuesEqual(left: unknown, right: unknown): boolean {
  if (Object.is(left, right)) return true
  if (
    left === null
    || right === null
    || typeof left !== 'object'
    || typeof right !== 'object'
  ) {
    return false
  }
  try {
    return JSON.stringify(left) === JSON.stringify(right)
  } catch {
    return false
  }
}

function violatesArgumentBinding(
  state: PolicyState,
  stageId: string,
  input: Record<string, unknown>,
): boolean {
  return (state.active.policy.argumentBindings ?? []).some((binding) => {
    const argument = bindingTargetArgument(state.active, binding.id, stageId)
    if (!argument) return false
    const candidate = readExplicitArgument(input, argument)
    if (!candidate.present) return binding.allowMissing !== true
    return state.bindingValues.has(binding.id)
      && !boundValuesEqual(state.bindingValues.get(binding.id), candidate.value)
  })
}

function recordBindingValues(
  active: ActiveSkillExecutionPolicy,
  stageId: string,
  input: Record<string, unknown>,
  values: Map<string, unknown>,
): void {
  for (const binding of active.policy.argumentBindings ?? []) {
    const argument = bindingTargetArgument(active, binding.id, stageId)
    if (!argument) continue
    const candidate = readExplicitArgument(input, argument)
    if (!values.has(binding.id)) {
      if (candidate.present) {
        values.set(binding.id, candidate.value)
      } else if (binding.allowMissing === true) {
        // Missing is a concrete cursor constraint, not a wildcard. Once a
        // stage means "the current target", a later explicit target could be
        // a different item and must be rejected. Later omissions remain valid.
        values.set(binding.id, IMPLICIT_CURRENT_ARGUMENT)
      }
    }
  }
}

function partitionFromObservedCalls(
  observed: readonly ObservedToolCall[],
  proposedCalls: readonly ToolCall[],
  policies: readonly ActiveSkillExecutionPolicy[],
  options: SkillExecutionToolCallPartitionOptions = {},
): SkillExecutionToolCallPartition {
  const states = policies.map((policy) => createPolicyState(policy, observed))
  const prioritizedToolNames = new Set<string>()
  if (options.prioritizeRequiredStages) {
    for (const state of states) {
      for (const [index, stage] of state.active.policy.stages.entries()) {
        const stageAvailable = stage.tools.some((toolName) =>
          options.availableToolNames?.has(toolName) ?? true
        )
        if (
          stage.requiredForCompletion === true
          && !state.satisfied.has(stage.id)
          && index >= state.furthestStageIndex
          && (state.counts.get(stage.id) ?? 0) < stageCallLimit(stage)
          && (stage.requires ?? []).every((dependency) =>
            state.satisfiedBeforeBatch.has(dependency)
          )
          && stageAvailable
        ) {
          for (const toolName of stage.tools) {
            if (options.availableToolNames?.has(toolName) ?? true) {
              prioritizedToolNames.add(toolName)
            }
          }
        }
      }
    }
  }
  const partition: SkillExecutionToolCallPartition = {
    executableCalls: [],
    rejectedCalls: [],
    prioritizedToolNames: [...prioritizedToolNames],
  }

  for (const call of proposedCalls) {
    const matches = states.flatMap((state) => {
      const matched = state.stageByTool.get(call.name)
      return matched ? [{ state, ...matched }] : []
    })
    const rejected = (
      prioritizedToolNames.size > 0
      && !prioritizedToolNames.has(call.name)
    ) || matches.some(({ state, stage, index }) => {
      const limit = stageCallLimit(stage)
      return (state.counts.get(stage.id) ?? 0) >= limit
        || index < state.furthestStageIndex
        // A dependency is evidence, not merely an ordering hint. Calls proposed
        // earlier in this same model batch have not executed yet, so they must
        // not unlock dependent calls until a later turn observes success.
        || (stage.requires ?? []).some(
          (dependency) => !state.satisfiedBeforeBatch.has(dependency),
        )
        || violatesArgumentBinding(state, stage.id, call.arguments)
    })
    if (rejected) {
      partition.rejectedCalls.push(call)
      continue
    }

    partition.executableCalls.push(call)
    for (const { state, stage, index } of matches) {
      state.counts.set(stage.id, (state.counts.get(stage.id) ?? 0) + 1)
      state.satisfied.add(stage.id)
      state.furthestStageIndex = Math.max(state.furthestStageIndex, index)
      recordBindingValues(state.active, stage.id, call.arguments, state.bindingValues)
    }
  }
  return partition
}

export function partitionSkillExecutionToolCalls(
  messages: readonly Message[],
  proposedCalls: readonly ToolCall[],
  policies: readonly ActiveSkillExecutionPolicy[],
  options: SkillExecutionToolCallPartitionOptions = {},
): SkillExecutionToolCallPartition {
  return partitionFromObservedCalls(
    observedToolCalls(messages),
    proposedCalls,
    policies,
    options,
  )
}

export function partitionSkillExecutionToolCallsFromHistory(
  history: readonly SkillExecutionToolHistoryEntry[],
  proposedCalls: readonly ToolCall[],
  policies: readonly ActiveSkillExecutionPolicy[],
  options: SkillExecutionToolCallPartitionOptions = {},
): SkillExecutionToolCallPartition {
  const observed = history.map((entry, index) => ({
    name: entry.tool,
    resultIndex: index,
    input: entry.input ?? {},
    status: entry.status,
    executionObserved: entry.status === 'success' || entry.executionObserved === true,
  }))
  return partitionFromObservedCalls(observed, proposedCalls, policies, options)
}

function completionFromObservedCalls(
  observed: readonly ObservedToolCall[],
  policies: readonly ActiveSkillExecutionPolicy[],
): SkillExecutionCompletion {
  const missing: MissingSkillExecutionStage[] = []
  let maxRetries = Number.POSITIVE_INFINITY
  for (const active of policies) {
    const state = createPolicyState(active, observed)
    for (const stage of active.policy.stages) {
      if (stage.requiredForCompletion && !state.satisfied.has(stage.id)) {
        missing.push({
          skillId: active.skillId,
          stageId: stage.id,
          tools: [...stage.tools],
        })
        // Multiple active policies are intersected. A more permissive skill
        // must never expand the retry budget declared by a stricter one.
        maxRetries = Math.min(maxRetries, active.policy.maxCompletionRetries ?? 1)
      }
    }
  }
  return { missing, maxRetries: missing.length > 0 ? maxRetries : 0 }
}

export function evaluateSkillExecutionCompletion(
  messages: readonly Message[],
  policies: readonly ActiveSkillExecutionPolicy[],
): SkillExecutionCompletion {
  return completionFromObservedCalls(observedToolCalls(messages), policies)
}

export function evaluateSkillExecutionCompletionFromHistory(
  history: readonly SkillExecutionToolHistoryEntry[],
  policies: readonly ActiveSkillExecutionPolicy[],
): SkillExecutionCompletion {
  const observed = history.map((entry, index) => ({
    name: entry.tool,
    resultIndex: index,
    input: entry.input ?? {},
    status: entry.status,
    executionObserved: entry.status === 'success' || entry.executionObserved === true,
  }))
  return completionFromObservedCalls(observed, policies)
}

export function hasDeterministicSkillCompletionPolicy(
  policies: readonly ActiveSkillExecutionPolicy[] | undefined,
): boolean {
  return (policies ?? []).some((active) =>
    active.policy.stages.some((stage) => stage.requiredForCompletion === true)
  )
}

export function buildSkillExecutionToolRepairMessage(
  rejectedToolNames: readonly string[],
  executableToolNames: readonly string[] = [],
  prioritizedToolNames: readonly string[] = [],
): Message {
  const continuation = executableToolNames.length > 0
    ? `Continue with only the policy-compliant calls: ${executableToolNames.join(', ')}.`
    : prioritizedToolNames.length > 0
      ? `Call the currently required stage first: ${prioritizedToolNames.join(' or ')}.`
    : 'Call only the next missing policy stage, or answer from already successful tool evidence.'
  return {
    role: 'system',
    metadata: { reminderKind: 'skill-execution-policy' },
    content: [
      '[Skill execution policy guard]',
      `Skipped calls that exceeded or violated the active skill stage order: ${rejectedToolNames.join(', ')}.`,
      continuation,
      'Do not repeat a successful stage in this user turn.',
    ].join('\n'),
  }
}

export function buildSkillExecutionCompletionRecoveryMessage(
  completion: SkillExecutionCompletion,
): Message {
  const requirements = completion.missing.map((missing) =>
    `${missing.skillId}/${missing.stageId}: ${missing.tools.join(' or ')}`
  )
  return {
    role: 'system',
    metadata: { reminderKind: 'skill-execution-completion' },
    content: [
      '[Skill execution completion guard]',
      'The active skill cannot finish this user turn until these stages have a successful tool result:',
      ...requirements.map((requirement) => `- ${requirement}`),
      'Call only the missing stage. Do not claim completion from prose or an unrelated tool result.',
    ].join('\n'),
  }
}

export function buildSkillExecutionCompletionFailureOutput(
  completion: SkillExecutionCompletion,
): string {
  const requirements = completion.missing.map((missing) =>
    `${missing.skillId}/${missing.stageId} (${missing.tools.join(' or ')})`
  )
  return `INCOMPLETE: 활성 스킬의 필수 실행 증거가 없습니다: ${requirements.join(', ')}.`
}

export function countSkillExecutionToolRepairMessages(messages: readonly Message[]): number {
  const start = currentTurnStart(messages)
  const currentTurnMessages = start >= 0 ? messages.slice(start + 1) : messages
  return currentTurnMessages.filter((message) =>
    message.role === 'system'
    && message.metadata?.reminderKind === 'skill-execution-policy'
  ).length
}

export function countSkillExecutionCompletionRecoveryMessages(
  messages: readonly Message[],
): number {
  const start = currentTurnStart(messages)
  const currentTurnMessages = start >= 0 ? messages.slice(start + 1) : messages
  return currentTurnMessages.filter((message) =>
    message.role === 'system'
    && message.metadata?.reminderKind === 'skill-execution-completion'
  ).length
}
