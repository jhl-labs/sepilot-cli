import type { ToolDefinitionRuntime, ToolExecutionContext, ToolResult } from './registry.js'
import type { SubagentDispatcher } from '../agent/subagent-dispatcher.js'
import { listSubagentDelegationCategories } from '../agent/subagent-categories.js'
import type { BackgroundSubagents } from '../jobs/subagent.js'
import { setTimeout as delay } from 'node:timers/promises'

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
  background?: () => BackgroundSubagents | undefined,
): ToolDefinitionRuntime {
  const categoryIds = listSubagentDelegationCategories().map((category) => category.id)
  return {
    name: 'subagent.dispatch',
    description:
      'Dispatch an isolated subagent to handle a sub-task. Returns its final output without leaking internal turns into the parent session. Set background=true only for work the user wants to continue independently: returns a durable jobId immediately; inspect or cancel it with subagent.job. Parent turn completion does not cancel detached work. Optional category presets tune scope, tool defaults, and iteration budget.',
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
        isolation: {
          type: 'string', enum: ['worktree'],
          description: 'Run in a separate git checkout at committed HEAD. Parent uncommitted edits are not copied. Changed worktrees are retained and returned for explicit merge.',
        },
        background: {
          type: 'boolean',
          description: 'Detach this delegation and return a durable jobId. Defaults to false.',
        },
        requestKey: {
          type: 'string',
          maxLength: 128,
          description:
            'Stable key for a background submission. Reuse it on uncertain retries; do not reuse it for a different task.',
        },
        prompt: { type: 'string', description: 'Task instruction' },
        system: { type: 'string', description: 'Override system prompt' },
        category: {
          type: 'string',
          enum: categoryIds,
          description: `Delegation preset: ${categoryIds.join(', ')}`,
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
        isolation?: 'worktree'
      }
      const startedAt = Date.now()
      try {
        const request = {
          prompt: inputTyped.prompt,
          system: inputTyped.system,
          category: inputTyped.category,
          agentId: inputTyped.agentId,
          maxIterations: inputTyped.maxIterations,
          tools: inputTyped.tools,
          model: inputTyped.model,
          contextPacket: inputTyped.contextPacket,
          isolation: inputTyped.isolation,
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
          signal: context?.signal,
          ...(context?.emitEvent ? { onEvent: context.emitEvent } : {}),
        }
        if (input.background === true) {
          context?.signal?.throwIfAborted()
          const service = background?.()
          if (!service) throw new Error('Background subagent service is unavailable')
          const job = service.start({
            ...request,
            requestKey:
              typeof input.requestKey === 'string' ? input.requestKey : context?.executionId,
          })
          return {
            output: JSON.stringify({
              ...job,
              next: 'Use subagent.job with this jobId to inspect, wait, or cancel. Completion is retained even if this conversation disconnects.',
            }),
            status: 'success',
            durationMs: Date.now() - startedAt,
          }
        }
        const result = await dispatcher.dispatch(request)
        const summary = [
          `[subagent ${result.sessionId} status=${result.status}`,
          `iterations=${result.iterations}`,
          `tokens=${result.usage.inputTokens}+${result.usage.outputTokens}]`,
        ].join(' ')
        return {
          output: `${summary}\n\n${result.output}${result.worktree ? `\n\nWorktree: ${JSON.stringify(result.worktree)}` : ''}`,
          status: result.status === 'completed' ? 'success' : 'error',
          durationMs: Date.now() - startedAt,
          metadata: {
            ...(result.worktree ? { worktree: result.worktree } : {}),
            delegatedUsage: {
              inputTokens: result.usage.inputTokens,
              outputTokens: result.usage.outputTokens,
            },
            // Structured board findings for the parent graph to roll up
            // (PLAN_065 T2). The parent's tool_result handler reads this and
            // merges evidence/failed-attempts/open-questions with provenance.
            ...(result.findings ? { subagentFindings: result.findings } : {}),
          },
          code:
            result.status === 'failed'
              ? result.error === 'subagent returned empty response'
                ? 'SUBAGENT_EMPTY_RESPONSE'
                : 'SUBAGENT_FAILED'
              : result.status === 'truncated'
                ? 'SUBAGENT_TRUNCATED'
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

/** Scope checks live in the durable service, not model-supplied arguments. */
export function createSubagentJobTool(
  background: () => BackgroundSubagents | undefined,
): ToolDefinitionRuntime {
  return {
    name: 'subagent.job',
    description:
      'Inspect, wait for, or cancel an explicitly detached subagent belonging to this conversation. Returns bounded progress and the retained result. Waiting does not cancel work when the observation timeout expires. Never poll in a tight loop.',
    inputSchema: {
      type: 'object',
      properties: {
        jobId: { type: 'string' },
        action: { type: 'string', enum: ['status', 'wait', 'cancel'] },
        timeoutMs: {
          type: 'integer',
          minimum: 0,
          maximum: 30000,
          description: 'Wait window, default 10000ms. Only applies to wait.',
        },
      },
      required: ['jobId'],
      additionalProperties: false,
    },
    async execute(input, context) {
      const startedAt = Date.now()
      try {
        context?.signal?.throwIfAborted()
        const service = background()
        if (!service || !context?.sessionId)
          throw new Error('Background subagent service and parent session are required')
        const jobId = typeof input.jobId === 'string' ? input.jobId : ''
        const action = input.action ?? 'status'
        if (!['status', 'wait', 'cancel'].includes(String(action)))
          throw new Error('Invalid subagent job action')
        const snapshot = () => {
          const result = service.inspect(jobId, context.sessionId)
          if (!result) throw new Error('Subagent job not found in this conversation')
          return result
        }
        let result = snapshot()
        if (action === 'cancel') {
          service.cancel(jobId, context.sessionId)
          result = snapshot()
        }
        const timeout =
          typeof input.timeoutMs === 'number' && Number.isFinite(input.timeoutMs)
            ? Math.max(0, Math.min(30000, input.timeoutMs))
            : 10000
        while (
          action === 'wait' &&
          ['pending', 'running'].includes(result.job.status) &&
          Date.now() - startedAt < timeout
        ) {
          await delay(Math.min(250, timeout - (Date.now() - startedAt)), undefined, {
            signal: context.signal,
          })
          result = snapshot()
          if (result.activity.some((item) => item.status === 'running' && item.approvalRequestId))
            break
        }
        // Large final reports stay in the child session/job; keep parent context bounded.
        const serialized = JSON.stringify({
          ...result,
          items: result.items.map((item) => ({
            idx: item.idx,
            status: item.status,
            result: item.result,
            error: item.error,
          })),
        })
        return {
          output:
            serialized.length <= 16000
              ? serialized
              : JSON.stringify({
                  job: result.job,
                  activity: result.activity,
                  outputPreview: serialized.slice(0, 12000),
                  truncated: true,
                  next: 'Read the child session or use sepilot jobs resume to collect the full result.',
                }),
          status: 'success',
          durationMs: Date.now() - startedAt,
        }
      } catch (error) {
        return {
          output: error instanceof Error ? error.message : String(error),
          status: 'error',
          durationMs: Date.now() - startedAt,
        }
      }
    },
  }
}
