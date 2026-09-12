import type {
  ActiveSkillExecutionPolicy,
  Message,
  ToolCall,
} from '@sepilotd/core'
import {
  partitionSkillExecutionToolCalls,
  partitionSkillExecutionToolCallsFromHistory,
  type SkillExecutionToolHistoryEntry,
} from '../../agent/skill-execution-policy.js'
import {
  resolveActiveSkillExecutionPolicies,
  resolveActiveSkillToolNames,
} from '../../skills/execution-policy.js'
import { withOptionalToolNameAllowlist } from '../../tools/role-filter.js'
import type { ToolRegistry } from '../../tools/registry.js'
import {
  cloneCheckpointExecutionSkillIds,
  cloneCheckpointSkillExecutionPolicies,
  cloneCheckpointSkillToolNames,
  cloneCheckpointToolAllowlist,
} from './checkpoints.js'

export interface CheckpointToolScope {
  executionSkillIds?: unknown
  skillToolNames?: unknown
  toolAllowlist?: unknown
  skillExecutionPolicies?: unknown
}

export interface PendingToolCheckpoint extends CheckpointToolScope {
  messages: Message[]
  graphState?: {
    toolCallHistory?: SkillExecutionToolHistoryEntry[]
  }
}

export interface PendingApprovalIdentity {
  toolCallId: string
  tool: string
  input: Record<string, unknown>
}

export class CheckpointToolScopeError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'CheckpointToolScopeError'
  }
}

function assertUnique(values: readonly string[], label: string): void {
  if (new Set(values).size !== values.length) {
    throw new CheckpointToolScopeError(`${label} contains duplicate entries`)
  }
}

function sameStringSet(left: readonly string[], right: readonly string[]): boolean {
  if (left.length !== right.length) return false
  const rightSet = new Set(right)
  return left.every((value) => rightSet.has(value))
}

function samePolicies(
  left: readonly ActiveSkillExecutionPolicy[],
  right: readonly ActiveSkillExecutionPolicy[],
): boolean {
  return sameJson(left, right)
}

function canonicalJson(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalJson)
  if (typeof value !== 'object' || value === null) return value
  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, entry]) => [key, canonicalJson(entry)]),
  )
}

function sameJson(left: unknown, right: unknown): boolean {
  return JSON.stringify(canonicalJson(left)) === JSON.stringify(canonicalJson(right))
}

/**
 * Rebuild the exact tool surface captured before a run paused.
 *
 * An exact allowlist is mandatory, including for legacy checkpoints. An
 * explicit empty list remains empty; missing scope must never degrade to the
 * full daemon registry. Canonical skill metadata is revalidated as an
 * additional integrity check, not as a substitute for the captured surface.
 */
export function resolveCheckpointToolRegistry(
  source: ToolRegistry,
  checkpoint: CheckpointToolScope,
): ToolRegistry {
  let executionSkillIds: string[] | undefined
  let skillToolNames: string[] | undefined
  let exactAllowlist: string[] | undefined
  let persistedPolicies: ActiveSkillExecutionPolicy[] | undefined
  try {
    executionSkillIds = cloneCheckpointExecutionSkillIds(checkpoint.executionSkillIds)
    skillToolNames = cloneCheckpointSkillToolNames(checkpoint.skillToolNames)
    exactAllowlist = cloneCheckpointToolAllowlist(checkpoint.toolAllowlist)
    persistedPolicies = cloneCheckpointSkillExecutionPolicies(
      checkpoint.skillExecutionPolicies,
    )
  } catch (error) {
    throw new CheckpointToolScopeError(
      error instanceof Error ? error.message : 'Checkpoint tool scope is malformed',
    )
  }

  for (const [values, label] of [
    [executionSkillIds, 'executionSkillIds'],
    [skillToolNames, 'skillToolNames'],
    [exactAllowlist, 'toolAllowlist'],
  ] as const) {
    if (values) assertUnique(values, label)
  }

  if (
    executionSkillIds === undefined
    && (skillToolNames !== undefined || persistedPolicies !== undefined)
  ) {
    throw new CheckpointToolScopeError(
      'Checkpoint skill tool scope exists without executionSkillIds',
    )
  }

  if (executionSkillIds && executionSkillIds.length > 0) {
    const canonicalTools = resolveActiveSkillToolNames(executionSkillIds)
    const canonicalPolicies = resolveActiveSkillExecutionPolicies(executionSkillIds)
    if (canonicalPolicies.length !== executionSkillIds.length) {
      throw new CheckpointToolScopeError(
        'Checkpoint references an unknown trusted execution skill',
      )
    }
    if (skillToolNames && !sameStringSet(skillToolNames, canonicalTools)) {
      throw new CheckpointToolScopeError(
        'Checkpoint skillToolNames do not match the trusted execution skills',
      )
    }
    if (persistedPolicies && !samePolicies(persistedPolicies, canonicalPolicies)) {
      throw new CheckpointToolScopeError(
        'Checkpoint skillExecutionPolicies do not match the trusted execution skills',
      )
    }
    skillToolNames ??= canonicalTools
  } else if (
    executionSkillIds
    && (skillToolNames?.length || persistedPolicies?.length)
  ) {
    throw new CheckpointToolScopeError(
      'Checkpoint has skill execution state but no active execution skill',
    )
  }

  if (
    exactAllowlist
    && skillToolNames?.some((toolName) => !exactAllowlist.includes(toolName))
  ) {
    throw new CheckpointToolScopeError(
      'Checkpoint toolAllowlist omits a declared execution-skill tool',
    )
  }

  if (exactAllowlist === undefined) {
    throw new CheckpointToolScopeError(
      'Checkpoint exact toolAllowlist is missing; legacy unscoped resume is disabled',
    )
  }
  const restoredAllowlist = exactAllowlist

  const available = new Set(source.list().map((tool) => tool.name))
  const unavailable = restoredAllowlist.filter((toolName) => !available.has(toolName))
  if (unavailable.length > 0) {
    throw new CheckpointToolScopeError(
      `Checkpoint tools are unavailable: ${unavailable.join(', ')}`,
    )
  }
  return withOptionalToolNameAllowlist(source, restoredAllowlist)
}

export function assertCheckpointPendingToolCalls(
  checkpoint: PendingToolCheckpoint,
  toolCalls: readonly ToolCall[],
  startIndex: number,
): void {
  if (!Number.isInteger(startIndex) || startIndex < 0 || startIndex >= toolCalls.length) {
    throw new CheckpointToolScopeError('Checkpoint pending tool index is invalid')
  }
  const executionSkillIds = cloneCheckpointExecutionSkillIds(checkpoint.executionSkillIds)
  if (!executionSkillIds?.length) return
  const policies = resolveActiveSkillExecutionPolicies(executionSkillIds)
  if (policies.length !== executionSkillIds.length) {
    throw new CheckpointToolScopeError(
      'Checkpoint references an unknown trusted execution skill',
    )
  }
  const proposedCalls = toolCalls.slice(startIndex)
  const partition = checkpoint.graphState?.toolCallHistory
    ? partitionSkillExecutionToolCallsFromHistory(
        checkpoint.graphState.toolCallHistory,
        proposedCalls,
        policies,
      )
    : partitionSkillExecutionToolCalls(checkpoint.messages, proposedCalls, policies)
  if (partition.rejectedCalls.length > 0) {
    throw new CheckpointToolScopeError(
      `Checkpoint pending tools violate the execution policy: ${partition.rejectedCalls
        .map((toolCall) => toolCall.name)
        .join(', ')}`,
    )
  }
}

export function assertApprovalCheckpointIdentity(
  checkpointToolCall: ToolCall | undefined,
  pendingApproval: PendingApprovalIdentity | null,
): void {
  if (
    !checkpointToolCall
    || !pendingApproval
    || checkpointToolCall.id !== pendingApproval.toolCallId
    || checkpointToolCall.name !== pendingApproval.tool
    || !sameJson(checkpointToolCall.arguments, pendingApproval.input)
  ) {
    throw new CheckpointToolScopeError(
      'The pending approval no longer matches its resumable checkpoint',
    )
  }
}
