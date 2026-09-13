import { mkdir, readFile, readdir, rm, stat } from 'node:fs/promises'
import { join } from 'node:path'
import type {
  ActiveSkillExecutionPolicy,
  AgentRunContract,
  Message,
  TokenUsage,
  ToolCall,
} from '@sepilotd/core'
import { createLogger } from '../../logger.js'
import { isNodeFsError } from '../../utils/fs-error.js'
import { writeFileAtomic } from '../../utils/atomic-write.js'
import type { AgentState as GraphAgentState } from '../../agent/graph/types.js'

const log = createLogger('approval-checkpoints')

/**
 * Clone a checkpoint scope at the trust boundary. A malformed persisted scope
 * must stop resume instead of silently degrading to an unscoped/global run.
 */
export function cloneCheckpointScopeTags(scopeTags: unknown): string[] | undefined {
  if (scopeTags === undefined) return undefined
  if (!Array.isArray(scopeTags) || scopeTags.some((tag) => typeof tag !== 'string')) {
    throw new Error('Checkpoint scopeTags must be an array of strings')
  }
  return [...scopeTags]
}

/**
 * Clone the structured skill ids that define execution invariants for a turn.
 * Persisted checkpoint JSON is an external trust boundary, so malformed ids
 * must fail resume instead of silently disabling the selected skill behavior.
 */
export function cloneCheckpointExecutionSkillIds(
  executionSkillIds: unknown,
): string[] | undefined {
  if (executionSkillIds === undefined) return undefined
  if (
    !Array.isArray(executionSkillIds)
    || executionSkillIds.some((skillId) => typeof skillId !== 'string')
  ) {
    throw new Error('Checkpoint executionSkillIds must be an array of strings')
  }
  return [...executionSkillIds]
}

export function cloneCheckpointSkillToolNames(value: unknown): string[] | undefined {
  if (value === undefined) return undefined
  if (
    !Array.isArray(value)
    || value.some((toolName) => typeof toolName !== 'string' || toolName.trim().length === 0)
  ) {
    throw new Error('Checkpoint skillToolNames must be an array of strings')
  }
  return [...value]
}

export function cloneCheckpointToolAllowlist(value: unknown): string[] | undefined {
  if (value === undefined) return undefined
  if (
    !Array.isArray(value)
    || value.some((toolName) => typeof toolName !== 'string' || toolName.trim().length === 0)
  ) {
    throw new Error('Checkpoint toolAllowlist must be an array of non-empty strings')
  }
  return [...value]
}

export function cloneCheckpointSkillExecutionPolicies(
  value: unknown,
): ActiveSkillExecutionPolicy[] | undefined {
  if (value === undefined) return undefined
  if (!Array.isArray(value)) {
    throw new Error('Checkpoint skillExecutionPolicies must be an array')
  }
  return value.map((entry): ActiveSkillExecutionPolicy => {
    if (
      !isRecord(entry)
      || typeof entry.skillId !== 'string'
      || entry.skillId.trim().length === 0
      || !isRecord(entry.policy)
    ) {
      throw new Error('Checkpoint skillExecutionPolicies contains an invalid policy')
    }
    const stages = entry.policy.stages
    if (!Array.isArray(stages)) {
      throw new Error('Checkpoint skill execution policy stages must be an array')
    }
    if (stages.length === 0) {
      throw new Error('Checkpoint skill execution policy stages must not be empty')
    }
    const stageIds = new Set<string>()
    const stageByTool = new Map<string, string>()
    const clonedStages = stages.map((stage, index) => {
      if (
        !isRecord(stage)
        || typeof stage.id !== 'string'
        || stage.id.trim().length === 0
        || !Array.isArray(stage.tools)
        || stage.tools.length === 0
        || stage.tools.some((tool) => typeof tool !== 'string' || tool.trim().length === 0)
        || (
          stage.requires !== undefined
          && (
            !Array.isArray(stage.requires)
            || stage.requires.some((dependency) =>
              typeof dependency !== 'string' || dependency.trim().length === 0
            )
          )
        )
        || (
          stage.maxCallsPerTurn !== undefined
          && (
            !Number.isInteger(stage.maxCallsPerTurn)
            || Number(stage.maxCallsPerTurn) < 1
            || Number(stage.maxCallsPerTurn) > 10
          )
        )
        || (
          stage.requiredForCompletion !== undefined
          && typeof stage.requiredForCompletion !== 'boolean'
        )
      ) {
        throw new Error('Checkpoint skill execution policy contains an invalid stage')
      }
      if (stageIds.has(stage.id)) {
        throw new Error(`Checkpoint skill execution policy has duplicate stage: ${stage.id}`)
      }
      stageIds.add(stage.id)
      for (const tool of stage.tools) {
        const previousStage = stageByTool.get(tool)
        if (previousStage) {
          throw new Error(
            `Checkpoint skill execution tool ${tool} belongs to multiple stages: ${previousStage}, ${stage.id}`,
          )
        }
        stageByTool.set(tool, stage.id)
      }
      for (const dependency of stage.requires ?? []) {
        const dependencyIndex = stages.findIndex((candidate) =>
          isRecord(candidate) && candidate.id === dependency
        )
        if (dependencyIndex < 0 || dependencyIndex >= index) {
          throw new Error(
            `Checkpoint skill execution stage ${stage.id} requires an unknown or later stage: ${dependency}`,
          )
        }
      }
      return {
        id: stage.id,
        tools: [...stage.tools] as string[],
        ...(stage.maxCallsPerTurn !== undefined
          ? { maxCallsPerTurn: Number(stage.maxCallsPerTurn) }
          : {}),
        ...(stage.requires !== undefined ? { requires: [...stage.requires] as string[] } : {}),
        ...(stage.requiredForCompletion !== undefined
          ? { requiredForCompletion: stage.requiredForCompletion }
          : {}),
      }
    })
    const maxCompletionRetries = entry.policy.maxCompletionRetries
    if (
      maxCompletionRetries !== undefined
      && (
        !Number.isInteger(maxCompletionRetries)
        || Number(maxCompletionRetries) < 0
        || Number(maxCompletionRetries) > 3
      )
    ) {
      throw new Error('Checkpoint skill execution policy has invalid maxCompletionRetries')
    }
    const argumentBindings = entry.policy.argumentBindings
    let clonedArgumentBindings: ActiveSkillExecutionPolicy['policy']['argumentBindings']
    if (argumentBindings !== undefined) {
      if (!Array.isArray(argumentBindings)) {
        throw new Error('Checkpoint skill execution argumentBindings must be an array')
      }
      const bindingIds = new Set<string>()
      clonedArgumentBindings = argumentBindings.map((binding) => {
        if (
          !isRecord(binding)
          || typeof binding.id !== 'string'
          || binding.id.trim().length === 0
          || !Array.isArray(binding.targets)
          || binding.targets.length < 2
          || (
            binding.allowMissing !== undefined
            && typeof binding.allowMissing !== 'boolean'
          )
        ) {
          throw new Error('Checkpoint skill execution policy has an invalid argument binding')
        }
        if (bindingIds.has(binding.id)) {
          throw new Error(
            `Checkpoint skill execution policy has duplicate argument binding: ${binding.id}`,
          )
        }
        bindingIds.add(binding.id)
        const targetStages = new Set<string>()
        const targets = binding.targets.map((target) => {
          if (
            !isRecord(target)
            || typeof target.stage !== 'string'
            || !stageIds.has(target.stage)
            || targetStages.has(target.stage)
            || typeof target.argument !== 'string'
            || target.argument.trim().length === 0
          ) {
            throw new Error(
              `Checkpoint skill execution argument binding ${binding.id} has an invalid target`,
            )
          }
          targetStages.add(target.stage)
          return { stage: target.stage, argument: target.argument }
        })
        return {
          id: binding.id,
          targets,
          ...(binding.allowMissing !== undefined
            ? { allowMissing: binding.allowMissing }
            : {}),
        }
      })
    }
    return {
      skillId: entry.skillId,
      policy: {
        stages: clonedStages,
        ...(clonedArgumentBindings ? { argumentBindings: clonedArgumentBindings } : {}),
        ...(maxCompletionRetries !== undefined
          ? { maxCompletionRetries: Number(maxCompletionRetries) }
          : {}),
      },
    }
  })
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export interface ApprovalRunCheckpoint {
  autonomy?: import('@sepilotd/core').AutonomyLevel
  requestId: string
  sessionId: string
  provider: string
  model: string
  mode?: string
  modeControlState?: import('@sepilotd/core').AgentContext['modeControlState']
  systemPrompt?: string
  cwd?: string
  workspaceRoot?: string
  workspaceIsolation?: 'policy' | 'strict'
  /** Caller memory scope preserved across approval pauses. */
  scopeTags?: string[]
  /** Structured skill execution invariants preserved across approval pauses. */
  executionSkillIds?: string[]
  /** Declared tool surface and execution profile preserved across approval pauses. */
  skillToolNames?: string[]
  /** Exact tool registry surface used before the approval pause. */
  toolAllowlist?: string[]
  skillExecutionPolicies?: ActiveSkillExecutionPolicy[]
  /** Preserve the per-turn explicit tool-approval guard across approval pauses. */
  requireToolApproval?: boolean
  messages: Message[]
  toolCalls: ToolCall[]
  currentToolIndex: number
  totalUsage: TokenUsage
  iteration: number
  maxIterations: number
  thinkingLevel?: string
  textDeltaMode?: 'buffered' | 'live'
  graphState?: GraphAgentState
  runContract?: AgentRunContract
  createdAt: string
}

export type ApprovalCheckpointSummary =
  | {
      status: 'available'
      requestId: string
      sessionId: string
      createdAt: string
    }
  | {
      status: 'unavailable'
      requestId: string
    }

export class ApprovalCheckpointStore {
  constructor(private readonly checkpointDir: string) {}

  async init(): Promise<void> {
    await mkdir(this.checkpointDir, { recursive: true })
  }

  async save(checkpoint: ApprovalRunCheckpoint): Promise<void> {
    await this.init()
    await writeFileAtomic(
      this.filePath(checkpoint.requestId),
      JSON.stringify(checkpoint, null, 2),
    )
  }

  async get(requestId: string): Promise<ApprovalRunCheckpoint | null> {
    const path = this.filePath(requestId)
    let raw: string
    try {
      raw = await readFile(path, 'utf-8')
    } catch (err) {
      if (isNodeFsError(err, 'ENOENT')) {
        return null
      }
      log.warn('approval checkpoint unreadable', {
        requestId,
        path,
        error: err instanceof Error ? err.message : String(err),
      })
      return null
    }
    try {
      return JSON.parse(raw) as ApprovalRunCheckpoint
    } catch (err) {
      log.warn('approval checkpoint unparseable; pending approval cannot resume', {
        requestId,
        path,
        error: err instanceof Error ? err.message : String(err),
      })
      return null
    }
  }

  async has(requestId: string): Promise<boolean> {
    return (await this.get(requestId)) !== null
  }

  async delete(requestId: string): Promise<void> {
    await rm(this.filePath(requestId), { force: true })
  }

  async deleteForSession(
    sessionId: string,
  ): Promise<{ deletedCheckpoints: number; unavailableCheckpoints: number }> {
    let deletedCheckpoints = 0
    let unavailableCheckpoints = 0
    for (const checkpoint of await this.list()) {
      if (checkpoint.status === 'unavailable') {
        unavailableCheckpoints += 1
        continue
      }
      if (checkpoint.sessionId !== sessionId) {
        continue
      }
      await this.delete(checkpoint.requestId)
      deletedCheckpoints += 1
    }
    return { deletedCheckpoints, unavailableCheckpoints }
  }

  async list(): Promise<ApprovalCheckpointSummary[]> {
    await this.init()
    const summaries: ApprovalCheckpointSummary[] = []
    for (const entry of await this.readEntries()) {
      if (!entry.endsWith('.json')) {
        continue
      }
      const requestId = entry.slice(0, -'.json'.length)
      const checkpoint = await this.get(requestId)
      summaries.push(checkpoint
        ? {
            status: 'available',
            requestId,
            sessionId: checkpoint.sessionId,
            createdAt: checkpoint.createdAt,
          }
        : {
            status: 'unavailable',
            requestId,
          })
    }
    return summaries.sort((left, right) => left.requestId.localeCompare(right.requestId))
  }

  async pruneOlderThan(
    maxAgeMs: number,
    now = new Date(),
  ): Promise<{ deletedCheckpoints: number; unavailableCheckpoints: number }> {
    await this.init()
    const cutoffMs = now.getTime() - maxAgeMs
    let deletedCheckpoints = 0
    let unavailableCheckpoints = 0

    for (const entry of await this.readEntries()) {
      if (!entry.endsWith('.json')) {
        continue
      }
      const requestId = entry.slice(0, -'.json'.length)
      const checkpoint = await this.get(requestId)
      const timestampMs = checkpoint
        ? Date.parse(checkpoint.createdAt)
        : await this.fileModifiedAtMs(this.filePath(requestId))
      if (!checkpoint) {
        unavailableCheckpoints += 1
      }
      if (!Number.isFinite(timestampMs) || timestampMs > cutoffMs) {
        continue
      }

      await this.delete(requestId)
      deletedCheckpoints += 1
    }

    return { deletedCheckpoints, unavailableCheckpoints }
  }

  private filePath(requestId: string): string {
    return join(this.checkpointDir, `${requestId}.json`)
  }

  private async readEntries(): Promise<string[]> {
    try {
      return await readdir(this.checkpointDir)
    } catch (err) {
      if (isNodeFsError(err, 'ENOENT')) {
        return []
      }
      throw err
    }
  }

  private async fileModifiedAtMs(path: string): Promise<number> {
    try {
      return (await stat(path)).mtimeMs
    } catch {
      return Number.NaN
    }
  }
}
