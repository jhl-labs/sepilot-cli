import type { ToolRegistry } from '../tools/registry.js'
import type { AgentRunContract } from '@sepilotd/core'

export function hasWorkspaceExecutionTool(
  tools: ToolRegistry,
  context: { cwd?: string; workspaceRoot?: string },
): boolean {
  return ['terminal.run', 'process.start'].some((name) => {
    const tool = tools.get(name)
    return tool && !tool.unavailableReason?.(context)
  })
}

export function workspaceExecutionBlocker(
  tools: ToolRegistry,
  context: { cwd?: string; workspaceRoot?: string },
  contract?: AgentRunContract,
): string | undefined {
  if (!context.workspaceRoot || hasWorkspaceExecutionTool(tools, context)) return undefined
  const execution = contract?.executionIntent
  if (!execution) return undefined
  const needsRunner = execution.requestedTerminalCommand
    || execution.requestedProcessStart
    || execution.constrainedProcessStart
    || execution.toolSequence?.some((name) => ['terminal.run', 'process.start', 'service.start'].includes(name))
  if (!needsRunner) return undefined
  return 'The requested execution capability is unavailable inside the strict workspace: no confined terminal or managed-process runner is available.'
}
