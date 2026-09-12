import type { ToolDefinitionRuntime, ToolExecutionContext, ToolResult } from './registry.js'
import {
  EXTERNAL_ACP_AGENT_NAMES,
  type ExternalAcpAgentDispatcher,
  type ExternalAcpAgentName,
} from '../acp/external-agent.js'

function isExternalAcpAgentName(value: unknown): value is ExternalAcpAgentName {
  return typeof value === 'string'
    && (EXTERNAL_ACP_AGENT_NAMES as readonly string[]).includes(value)
}

export function createExternalAcpRunTool(
  dispatcher: ExternalAcpAgentDispatcher,
): ToolDefinitionRuntime {
  return {
    name: 'external_acp.run',
    description:
      'Run a configured external ACP coding agent such as opencode or Codex and return its final answer. The external agent runs in its own ACP session and cannot use sepilotd client-side filesystem or terminal capabilities.',
    inputSchema: {
      type: 'object',
      properties: {
        prompt: { type: 'string', description: 'Task instruction for the external ACP agent' },
        agent: {
          type: 'string',
          enum: [...EXTERNAL_ACP_AGENT_NAMES],
          description: 'External ACP agent preset to run',
        },
        cwd: { type: 'string', description: 'Absolute working directory for the ACP session' },
        timeoutMs: { type: 'number', description: 'Request timeout in milliseconds' },
      },
      required: ['prompt'],
    },
    async execute(
      input: Record<string, unknown>,
      context?: ToolExecutionContext,
    ): Promise<ToolResult> {
      const startedAt = Date.now()
      try {
        const prompt = typeof input.prompt === 'string' ? input.prompt : ''
        if (input.agent !== undefined && !isExternalAcpAgentName(input.agent)) {
          return {
            output: `[external_acp error] unsupported external ACP agent: ${String(input.agent)}`,
            status: 'error',
            durationMs: Date.now() - startedAt,
            code: 'EXTERNAL_ACP_UNKNOWN_AGENT',
          }
        }
        const result = await dispatcher.dispatch({
          prompt,
          agent: isExternalAcpAgentName(input.agent) ? input.agent : undefined,
          cwd: typeof input.cwd === 'string' ? input.cwd : context?.cwd,
          timeoutMs: typeof input.timeoutMs === 'number' ? input.timeoutMs : undefined,
          parentSessionId: context?.sessionId,
          scopeTags: context?.scopeTags,
          runContract: context?.runContract,
        })
        const head = `[external_acp ${result.agent} session=${result.sessionId} external=${result.externalSessionId || '-'} status=${result.status} stop=${result.stopReason}]`
        return {
          output: result.output ? `${head}\n\n${result.output}` : head,
          status: result.status === 'failed' ? 'error' : 'success',
          durationMs: Date.now() - startedAt,
          code: result.status === 'failed' ? 'EXTERNAL_ACP_FAILED' : undefined,
        }
      } catch (error) {
        return {
          output: `[external_acp error] ${error instanceof Error ? error.message : String(error)}`,
          status: 'error',
          durationMs: Date.now() - startedAt,
          code: 'EXTERNAL_ACP_ERROR',
        }
      }
    },
  }
}
