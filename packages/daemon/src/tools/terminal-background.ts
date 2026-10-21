import type { ToolDefinitionRuntime, ToolResult } from './registry.js'

const BACKGROUND_FIELD = 'runInBackground'

function withoutBackgroundField(input: Record<string, unknown>): Record<string, unknown> {
  const { [BACKGROUND_FIELD]: _ignored, ...rest } = input
  return rest
}

export function withBackgroundExecution(
  terminal: ToolDefinitionRuntime,
  processStart: ToolDefinitionRuntime,
): ToolDefinitionRuntime {
  const schema = terminal.inputSchema as { properties?: Record<string, unknown> }
  return {
    ...terminal,
    description: `${terminal.description} Set runInBackground=true for a long-running command (server, watcher, long build or test run) that should keep running while you continue: it starts as a managed background process, returns its process id immediately, and this conversation is told when it exits. Read its output with process.read or process.follow.`,
    inputSchema: {
      ...terminal.inputSchema,
      properties: {
        ...(schema.properties ?? {}),
        [BACKGROUND_FIELD]: {
          type: 'boolean',
          description: 'Run as a managed background process and return immediately with its process id. You are notified when it exits. timeoutMs, timeoutOutcome and expectedExitCodes do not apply; external network is not available in background mode.',
        },
      },
    },
    ...(terminal.validateInput ? {
      validateInput: (input: Record<string, unknown>, context: Parameters<NonNullable<ToolDefinitionRuntime['validateInput']>>[1]) =>
        terminal.validateInput!(withoutBackgroundField(input), context),
    } : {}),
    async execute(input, context): Promise<ToolResult> {
      if (input[BACKGROUND_FIELD] !== true) {
        return terminal.execute(withoutBackgroundField(input), context)
      }
      const startedAt = Date.now()
      if (input.network === 'external') {
        return {
          status: 'error',
          code: 'BACKGROUND_NETWORK_UNSUPPORTED_PERMANENT',
          output: 'runInBackground does not support network: external. Run the command in the foreground, or use process.start with a loopback network for a local server.',
          durationMs: Date.now() - startedAt,
        }
      }
      const processInput: Record<string, unknown> = {
        executable: input.executable,
        args: Array.isArray(input.args) ? input.args : [],
        notifyOnExit: true,
      }
      if (typeof input.cwd === 'string' && input.cwd.trim()) processInput.cwd = input.cwd
      const result = await processStart.execute(processInput, context)
      if (result.status !== 'success') return result
      return {
        ...result,
        output: `${result.output}\n\nStarted in the background. Continue with other work; this conversation is notified when the process exits. Read output with process.read or process.follow, stop it with process.stop.`,
      }
    },
  }
}
