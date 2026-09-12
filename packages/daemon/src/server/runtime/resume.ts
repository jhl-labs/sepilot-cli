import type { AutonomyLevel, ToolCall } from '@sepilotd/core'
import type { PolicyEngine } from '../../security/policy-engine.js'
import {
  resolveToolResumeSafety,
  type ToolRegistry,
  type ToolResult,
} from '../../tools/registry.js'
import type { SessionRunCheckpoint } from './runs.js'
import type { ToolExecutionStore } from './tool-executions.js'

export type RunResumeMode = 'exact' | 'replay-safe' | 'replay-risky'

export interface RunResumeAssessment {
  mode: RunResumeMode
  forceRequired: boolean
  currentTool?: string
  currentToolCount?: number
  currentTools?: string[]
  journaledResultAvailable?: boolean
  recoveryProbeAvailable?: boolean
}

function getPendingReplayToolCalls(
  checkpoint: SessionRunCheckpoint,
): ToolCall[] {
  if (checkpoint.stage !== 'acting' || !checkpoint.pendingToolExecution) {
    return []
  }

  const { startIndex, batchSize, toolCalls } = checkpoint.pendingToolExecution
  const pendingCount = Math.max(batchSize ?? 1, 1)
  return toolCalls.slice(startIndex, startIndex + pendingCount)
}

function buildReplayTarget(
  toolCalls: readonly ToolCall[],
): Pick<RunResumeAssessment, 'currentTool' | 'currentToolCount' | 'currentTools'> {
  if (toolCalls.length === 0) {
    return {}
  }

  return {
    currentTool: toolCalls[0]?.name,
    currentToolCount: toolCalls.length,
    currentTools: toolCalls.map((toolCall) => toolCall.name),
  }
}

export function describeRunResumeTarget(
  assessment: Pick<RunResumeAssessment, 'currentTool' | 'currentToolCount' | 'currentTools'>,
): string {
  const currentTools = assessment.currentTools ?? []
  const currentToolCount =
    assessment.currentToolCount
    ?? (currentTools.length > 0 ? currentTools.length : (assessment.currentTool ? 1 : 0))

  if (currentToolCount <= 1) {
    return assessment.currentTool ?? 'the active tool'
  }

  const counts = new Map<string, number>()
  for (const toolName of currentTools) {
    counts.set(toolName, (counts.get(toolName) ?? 0) + 1)
  }

  const summary =
    counts.size > 0
      ? ` (${Array.from(counts.entries())
          .slice(0, 2)
          .map(([toolName, count]) => count > 1 ? `${toolName} x${count}` : toolName)
          .join(', ')}${counts.size > 2 ? `, +${counts.size - 2} more` : ''})`
      : ''

  return `${currentToolCount} tool calls${summary}`
}

export async function attemptInterruptedToolRecovery(
  checkpoint: SessionRunCheckpoint,
  tools: ToolRegistry,
  toolExecutions: ToolExecutionStore,
  policy: PolicyEngine,
  autonomy: AutonomyLevel,
): Promise<ToolResult | null> {
  if (checkpoint.stage !== 'acting' || !checkpoint.pendingToolExecution?.currentExecutionId) {
    return null
  }

  const currentTool =
    checkpoint.pendingToolExecution.toolCalls[checkpoint.pendingToolExecution.startIndex]
  if (!currentTool) {
    return null
  }

  const execution = await toolExecutions.get(checkpoint.sessionId)
  const tool = tools.get(currentTool.name)
  if (
    !execution
    || !tool?.recoverInterruptedExecution
    || execution.executionId !== checkpoint.pendingToolExecution.currentExecutionId
    || execution.toolCallId !== currentTool.id
    || execution.status !== 'running'
  ) {
    return null
  }

  const policyResult = policy.check(
    {
      tool: currentTool.name,
      input: currentTool.arguments,
      cwd: checkpoint.cwd,
      workspaceRoot: checkpoint.workspaceRoot,
      registrationSource: tools.registrationSource(currentTool.name),
      security: tools.securityDescriptor(currentTool.name),
    },
    autonomy,
  )
  if (!policyResult.allowed) return null

  return tool.recoverInterruptedExecution(currentTool.arguments, {
    executionId: execution.executionId,
    startedAt: execution.startedAt,
    sessionId: checkpoint.sessionId,
    cwd: checkpoint.cwd,
    workspaceRoot: checkpoint.workspaceRoot,
  })
}

export async function assessRunResume(
  checkpoint: SessionRunCheckpoint,
  tools: ToolRegistry,
  toolExecutions?: ToolExecutionStore,
): Promise<RunResumeAssessment> {
  if (checkpoint.stage !== 'acting' || !checkpoint.pendingToolExecution) {
    return { mode: 'exact', forceRequired: false }
  }

  const currentToolCalls = getPendingReplayToolCalls(checkpoint)
  const currentTool = currentToolCalls[0]
  const replayTarget = buildReplayTarget(currentToolCalls)

  if (!currentTool) {
    return { mode: 'exact', forceRequired: false }
  }

  if (checkpoint.pendingToolExecution.currentExecutionId && toolExecutions) {
    const execution = await toolExecutions.get(checkpoint.sessionId)
    const tool = tools.get(currentTool.name)
    if (
      execution
      && execution.executionId === checkpoint.pendingToolExecution.currentExecutionId
      && execution.toolCallId === currentTool.id
      && execution.status === 'completed'
      && execution.result
    ) {
      return {
        mode: 'exact',
        forceRequired: false,
        ...replayTarget,
        journaledResultAvailable: true,
      }
    }

    if (
      execution
      && tool?.recoverInterruptedExecution
      && execution.executionId === checkpoint.pendingToolExecution.currentExecutionId
      && execution.toolCallId === currentTool.id
      && execution.status === 'running'
    ) {
      return {
        mode: 'replay-risky',
        forceRequired: true,
        ...replayTarget,
        recoveryProbeAvailable: true,
      }
    }
  }

  if (currentToolCalls.length > 1) {
    const batchMode = currentToolCalls.every(
      (toolCall) =>
        resolveToolResumeSafety(
          tools.get(toolCall.name),
          toolCall.arguments,
        ) === 'replay-safe',
    )
      ? 'replay-safe'
      : 'replay-risky'

    return {
      mode: batchMode,
      forceRequired: batchMode === 'replay-risky',
      ...replayTarget,
    }
  }

  const mode = resolveToolResumeSafety(
    tools.get(currentTool.name),
    currentTool.arguments,
  )
  if (mode === 'replay-safe') {
    return {
      mode,
      forceRequired: false,
      ...replayTarget,
    }
  }

  return {
    mode,
    forceRequired: true,
    ...replayTarget,
  }
}
