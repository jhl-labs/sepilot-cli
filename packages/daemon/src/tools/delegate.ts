import { randomUUID } from 'node:crypto'
import type { DelegationResultEvent } from '@sepilotd/core'
import type { ToolDefinitionRuntime, ToolResult } from './registry.js'
import type { DelegationResult, TaskDelegator } from '../agent/delegator.js'
import { throwIfAborted } from '../abort.js'

export interface DelegateToolOptions {
  appendDelegationResultEvent?: (
    sessionId: string,
    event: DelegationResultEvent,
  ) => Promise<void> | void
}

function extractDelegationArtifactHandles(result: string | undefined): string[] {
  if (!result) {
    return []
  }

  const lines = result.split(/\r?\n/)
  const markerIndex = lines.findIndex((line) => line.trim().toLowerCase() === 'artifact handles:')
  if (markerIndex < 0) {
    return []
  }

  const handles: string[] = []
  for (const line of lines.slice(markerIndex + 1)) {
    const trimmed = line.trim()
    if (!trimmed) {
      if (handles.length > 0) break
      continue
    }
    if (!trimmed.startsWith('- ')) {
      break
    }
    handles.push(trimmed.slice(2).trim())
  }
  return handles
}

function formatDelegationRecoveryOutput(
  delegation: Awaited<ReturnType<TaskDelegator['findDelegationByExecutionId']>>,
): ToolResult {
  if (!delegation) {
    return {
      output: 'Delegation could not be recovered',
      status: 'error',
      durationMs: 0,
    }
  }

  const suffix = delegation.result ? `\n${delegation.result}` : ''
  switch (delegation.status) {
    case 'picked_up':
      return {
        output: `Task delegation picked up: ${delegation.delegationId} → ${delegation.targetDevice}${suffix}`,
        status: 'success',
        durationMs: 0,
      }
    case 'completed':
      return {
        output: `Task delegation completed: ${delegation.delegationId} → ${delegation.targetDevice}${suffix}`,
        status: 'success',
        durationMs: 0,
      }
    case 'failed':
      return {
        output: `Task delegation failed: ${delegation.delegationId} → ${delegation.targetDevice}${suffix}`,
        status: 'error',
        durationMs: 0,
      }
    case 'timeout':
      return {
        output: `Task delegation timed out: ${delegation.delegationId} → ${delegation.targetDevice}${suffix}`,
        status: 'error',
        durationMs: 0,
      }
    case 'cancelled':
      return {
        output: `Task delegation cancelled: ${delegation.delegationId} → ${delegation.targetDevice}${suffix}`,
        status: 'error',
        durationMs: 0,
      }
    default:
      return {
        output: `Task delegated: ${delegation.delegationId} → ${delegation.targetDevice} (${delegation.status})`,
        status: 'success',
        durationMs: 0,
      }
  }
}

function isTerminalDelegationResult(
  delegation: DelegationResult,
): delegation is DelegationResult & { status: 'completed' | 'failed' | 'timeout' | 'cancelled' } {
  return (
    delegation.status === 'completed'
    || delegation.status === 'failed'
    || delegation.status === 'timeout'
    || delegation.status === 'cancelled'
  )
}

async function appendDelegationResultEvent(
  options: DelegateToolOptions | undefined,
  sessionId: string | undefined,
  executionId: string,
  delegation: DelegationResult,
): Promise<void> {
  if (
    !sessionId
    || !options?.appendDelegationResultEvent
    || !isTerminalDelegationResult(delegation)
  ) {
    return
  }

  await options.appendDelegationResultEvent(sessionId, {
    type: 'delegation_result',
    id: randomUUID(),
    timestamp: new Date().toISOString(),
    delegationId: delegation.delegationId,
    targetDevice: delegation.targetDevice,
    executionId,
    status: delegation.status,
    result: delegation.result,
    artifactHandles: extractDelegationArtifactHandles(delegation.result),
    source: 'recovery',
  })
}

export function createDelegateTool(
  delegator: TaskDelegator,
  options?: DelegateToolOptions,
): ToolDefinitionRuntime {
  return {
    name: 'device.delegate',
    description:
      'Delegate a task to another device on the local network (e.g. run a long build on a beefy server, process data on an edge device). Returns once the remote run starts; check session events for completion. Falls back to local execution if no devices are reachable. Avoid for short or read-only tasks — the round-trip overhead negates the benefit.',
    resumeSafety: 'replay-risky',
    inputSchema: {
      type: 'object',
      properties: {
        instruction: { type: 'string', description: 'Task instruction for the remote device' },
        targetDevice: { type: 'string', description: 'Target device name (optional, auto-select if omitted)' },
        priority: { type: 'string', enum: ['high', 'medium', 'low'], description: 'Task priority' },
      },
      required: ['instruction'],
    },
    async recoverInterruptedExecution(_input, context): Promise<ToolResult | null> {
      const delegation = await delegator.findDelegationByExecutionId(context.executionId)
      if (!delegation) {
        return null
      }

      await appendDelegationResultEvent(
        options,
        context.sessionId,
        context.executionId,
        delegation,
      )
      return formatDelegationRecoveryOutput(delegation)
    },
    async execute(input: Record<string, unknown>, context): Promise<ToolResult> {
      const start = Date.now()
      try {
        throwIfAborted(context?.signal, 'Task delegation aborted')
        if (context?.scopeTags && context.scopeTags.length > 0) {
          return {
            output: 'Cross-device delegation is unavailable for scoped sessions because the remote scope cannot yet be preserved safely.',
            status: 'error',
            code: 'SCOPED_DELEGATION_UNSUPPORTED',
            durationMs: Date.now() - start,
          }
        }
        const result = await delegator.delegate({
          instruction: input.instruction as string,
          targetDevice: input.targetDevice as string | undefined,
          priority: input.priority as 'high' | 'medium' | 'low' | undefined,
          executionId: context?.executionId,
          runContract: context?.runContract,
        })
        // A failed gateway write must surface as an error, not a success — the
        // delegation was never persisted for the target to pick up.
        if (result.status === 'failed') {
          return {
            output: `Task delegation failed: ${result.delegationId}${result.result ? ` — ${result.result}` : ''}`,
            status: 'error',
            durationMs: Date.now() - start,
          }
        }
        return {
          output: `Task delegated: ${result.delegationId} → ${result.targetDevice} (${result.status})`,
          status: 'success',
          durationMs: Date.now() - start,
        }
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : String(err)
        return { output: message, status: 'error', durationMs: Date.now() - start }
      }
    },
  }
}
