import { randomUUID } from 'node:crypto'
import type { ToolDefinitionRuntime, ToolExecutionContext, ToolResult } from './registry.js'
import type { ExternalCliDispatcher } from '../external-cli/dispatcher.js'
import { EXTERNAL_CLI_BACKENDS, EXTERNAL_CLI_PERMISSIONS, isExternalCliBackend, isExternalCliPermission } from '../external-cli/executor.js'

export function createExternalCliRunTool(dispatcher: ExternalCliDispatcher): ToolDefinitionRuntime {
  return {
    name: 'external_cli.run',
    timeoutPolicy: 'inactivity',
    description:
      'Delegate a task to an installed external coding CLI (Claude Code, Codex, or OpenCode) and return its final answer. The CLI works in the given directory under its own account; its permission to edit files or run commands is derived from this run\'s autonomy and cannot exceed it. Pass threadId from a previous result to continue that CLI session.',
    inputSchema: {
      type: 'object',
      properties: {
        prompt: { type: 'string', description: 'Complete, self-contained task instruction for the external CLI. It does not see this conversation.' },
        agent: { type: 'string', enum: [...EXTERNAL_CLI_BACKENDS], description: 'Which external CLI to run' },
        cwd: { type: 'string', description: 'Absolute working directory. Defaults to the current workspace.' },
        model: { type: 'string', description: 'Backend-specific model id, if the CLI should not use its default' },
        threadId: { type: 'string', description: 'Native session id returned by a previous external_cli.run on the same agent, to continue it' },
        timeoutMs: { type: 'number', description: 'Turn timeout in milliseconds (default 20 minutes)' },
        permission: {
          type: 'string',
          enum: [...EXTERNAL_CLI_PERMISSIONS],
          description: 'Optional upper bound on what the CLI may do: "read" for a consultation or review that must not change the workspace. It can only lower the permission this run\'s autonomy allows, never raise it.',
        },
      },
      required: ['prompt', 'agent'],
    },
    async execute(input: Record<string, unknown>, context?: ToolExecutionContext): Promise<ToolResult> {
      const startedAt = Date.now()
      if (!isExternalCliBackend(input.agent)) {
        return {
          output: `[external_cli error] unsupported agent: ${String(input.agent)}. Supported: ${EXTERNAL_CLI_BACKENDS.join(', ')}`,
          status: 'error', durationMs: 0, code: 'EXTERNAL_CLI_UNKNOWN_AGENT',
        }
      }
      if (input.permission !== undefined && !isExternalCliPermission(input.permission)) {
        return {
          output: `[external_cli error] unsupported permission: ${String(input.permission)}. Supported: ${EXTERNAL_CLI_PERMISSIONS.join(', ')}`,
          status: 'error', durationMs: 0, code: 'EXTERNAL_CLI_INVALID_PERMISSION',
        }
      }
      try {
        const sessionId = randomUUID()
        const backend = input.agent
        const result = await dispatcher.dispatch({
          sessionId,
          onEvent: (event) => {
            if (!context?.emitEvent) return
            const inner = event.type === 'text'
              ? { type: 'message' as const, content: event.text }
              : event.type === 'progress'
                ? { type: 'action_progress' as const, summary: event.text, nextStep: '', toolNames: [] }
                : null
            if (inner) context.emitEvent({ type: 'subagent_progress', subagentId: sessionId, label: backend, inner })
          },
          backend,
          prompt: typeof input.prompt === 'string' ? input.prompt : '',
          cwd: typeof input.cwd === 'string' ? input.cwd : context?.cwd,
          model: typeof input.model === 'string' ? input.model : undefined,
          threadId: typeof input.threadId === 'string' ? input.threadId : undefined,
          timeoutMs: typeof input.timeoutMs === 'number' ? input.timeoutMs : undefined,
          autonomy: context?.autonomy,
          maxPermission: input.permission,
          parentSessionId: context?.sessionId,
          scopeTags: context?.scopeTags,
          runContract: context?.runContract,
          signal: context?.signal,
        })
        const head = `[external_cli ${result.backend} permission=${result.permission} session=${result.sessionId} thread=${result.threadId ?? '-'} status=${result.status} tokens=${result.usage.inputTokens}/${result.usage.outputTokens}]`
        return {
          output: `${head}\n\n${result.output}`,
          status: result.status === 'failed' ? 'error' : 'success',
          durationMs: Date.now() - startedAt,
          code: result.status === 'failed' ? 'EXTERNAL_CLI_FAILED' : undefined,
        }
      } catch (error) {
        return {
          output: `[external_cli error] ${error instanceof Error ? error.message : String(error)}`,
          status: 'error', durationMs: Date.now() - startedAt, code: 'EXTERNAL_CLI_ERROR',
        }
      }
    },
  }
}
