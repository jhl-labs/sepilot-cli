import type { ToolDefinitionRuntime, ToolExecutionContext, ToolResult } from './registry.js'
import type { SubagentDispatcher } from '../agent/subagent-dispatcher.js'
import { listSubagentDelegationCategories } from '../agent/subagent-categories.js'

/**
 * LLM-callable wrapper around {@link SubagentDispatcher}. Exposes the
 * `subagent.dispatch` tool — when invoked the parent agent spawns an
 * isolated subagent run and receives its final output back as a single
 * tool result. Tool access for the subagent may be restricted to a
 * subset of the parent's allowed tool list; depth 1 enforcement
 * (stripping `subagent.dispatch` itself) lives inside the dispatcher.
 */
export function createSubagentDispatchTool(
  dispatcher: SubagentDispatcher,
  // Fallback for direct/manual tool execution. Normal agent execution passes
  // the concrete turn registry through ToolExecutionContext instead.
  _parentAllowedTools: () => string[],
): ToolDefinitionRuntime {
  const categoryIds = listSubagentDelegationCategories().map((category) => category.id)
  return {
    name: 'subagent.dispatch',
    description:
      'Dispatch an isolated subagent to handle a sub-task. Returns its final output without leaking internal turns into the parent session. Optional category presets tune scope, tool defaults, and iteration budget.',
    // Each dispatch spawns its own isolated session/engine and does not
    // touch parent state, so multiple dispatches in one turn are safe to
    // run concurrently. Without this, fan-out (e.g. autoDecompose's
    // "parallel subagent dispatch") was awaited serially — the batch
    // collector keys by the distinct task prompt so independent tasks
    // run in parallel while an identical re-dispatch stays serialized.
    scheduling: {
      mode: 'parallel-safe',
      resource: 'subagent',
      key: (input) => (typeof input.prompt === 'string' ? input.prompt : null),
    },
    inputSchema: {
      type: 'object',
      properties: {
        prompt: { type: 'string', description: 'Task instruction' },
        system: { type: 'string', description: 'Override system prompt' },
        category: {
          type: 'string',
          enum: categoryIds,
          description:
            `Delegation preset: ${categoryIds.join(', ')}`,
        },
        agentId: { type: 'string', description: 'Use registered user-defined agent' },
        maxIterations: {
          type: 'integer',
          minimum: 1,
          description: 'Max turns (default 20, hard cap 50)',
        },
        tools: {
          type: 'array',
          items: { type: 'string' },
          description: 'Restrict to subset of parent allowed tools',
        },
        model: { type: 'string', description: 'Provider model override' },
        contextPacket: {
          type: 'string',
          description: 'Compact parent context to avoid duplicate exploration',
        },
      },
      required: ['prompt'],
    },
    async execute(
      input: Record<string, unknown>,
      context?: ToolExecutionContext,
    ): Promise<ToolResult> {
      const inputTyped = input as {
        prompt: string
        system?: string
        category?: string
        agentId?: string
        maxIterations?: number
        tools?: string[]
        model?: string
        contextPacket?: string
      }
      const startedAt = Date.now()
      try {
        const result = await dispatcher.dispatch({
          prompt: inputTyped.prompt,
          system: inputTyped.system,
          category: inputTyped.category,
          agentId: inputTyped.agentId,
          maxIterations: inputTyped.maxIterations,
          tools: inputTyped.tools,
          model: inputTyped.model,
          contextPacket: inputTyped.contextPacket,
          parentSessionId: context?.sessionId,
          cwd: context?.cwd,
          workspaceRoot: context?.workspaceRoot,
          parentExecutionPolicy: context?.delegatedAgentPolicy
            ? {
                autonomy: context.delegatedAgentPolicy.autonomy,
                requireToolApproval: context.delegatedAgentPolicy.requireToolApproval,
                allowedToolNames: [...context.delegatedAgentPolicy.allowedToolNames],
              }
            : undefined,
          runContract: context?.runContract,
          scopeTags: context?.scopeTags ? [...context.scopeTags] : undefined,
          ...(context?.emitEvent ? { onEvent: context.emitEvent } : {}),
        })
        const summary = [
          `[subagent ${result.sessionId} status=${result.status}`,
          `iterations=${result.iterations}`,
          `tokens=${result.usage.inputTokens}+${result.usage.outputTokens}]`,
        ].join(' ')
        return {
          output: `${summary}\n\n${result.output}`,
          status: result.status === 'failed' ? 'error' : 'success',
          durationMs: Date.now() - startedAt,
          metadata: {
            delegatedUsage: {
              inputTokens: result.usage.inputTokens,
              outputTokens: result.usage.outputTokens,
            },
            // Structured board findings for the parent graph to roll up
            // (PLAN_065 T2). The parent's tool_result handler reads this and
            // merges evidence/failed-attempts/open-questions with provenance.
            ...(result.findings ? { subagentFindings: result.findings } : {}),
          },
          code: result.status === 'failed'
            ? result.error === 'subagent returned empty response'
              ? 'SUBAGENT_EMPTY_RESPONSE'
              : 'SUBAGENT_FAILED'
            : undefined,
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err)
        const code = (err as Error & { code?: string }).code
        return {
          output: `[subagent error] ${msg}`,
          status: 'error',
          durationMs: Date.now() - startedAt,
          code: code === 'SUBAGENT_TOOL_ESCALATION' ? 'SUBAGENT_TOOL_ESCALATION' : 'SUBAGENT_ERROR',
        }
      }
    },
  }
}
