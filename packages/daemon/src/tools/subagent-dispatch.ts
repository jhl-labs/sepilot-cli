import type { ToolSecurityDescriptor } from '@sepilotd/core'
import type {
  ToolDefinitionRuntime,
  ToolExecutionContext,
  ToolResult,
  ToolResumeSafety,
} from './registry.js'
import type { SubagentDispatcher, SubagentDispatchResult } from '../agent/subagent-dispatcher.js'
import {
  listSubagentDelegationCategories,
  resolveSubagentDelegationCategory,
} from '../agent/subagent-categories.js'
import type { BackgroundSubagents } from '../jobs/subagent.js'
import { builtinToolSecurityDescriptor } from './security.js'
import { setTimeout as delay } from 'node:timers/promises'

export type SubagentToolEffectResolver = (
  name: string,
) => ToolSecurityDescriptor['effect'] | undefined

const builtinEffect: SubagentToolEffectResolver = (name) =>
  builtinToolSecurityDescriptor(name)?.effect

export function subagentDispatchResumeSafety(
  input: Record<string, unknown>,
  effectOf: SubagentToolEffectResolver = builtinEffect,
): ToolResumeSafety {
  if (input.background === true) return 'replay-risky'
  if (input.isolation !== undefined || input.agentId !== undefined) return 'replay-risky'
  if (input.tools !== undefined) {
    if (!Array.isArray(input.tools)) return 'replay-risky'
    return input.tools.every((name) => typeof name === 'string' && effectOf(name) === 'observe')
      ? 'replay-safe'
      : 'replay-risky'
  }
  const category = typeof input.category === 'string'
    ? resolveSubagentDelegationCategory(input.category)
    : undefined
  if (!category?.toolHints) return 'replay-risky'
  return category.toolHints.every((name) => {
    const effect = effectOf(name)
    return effect === undefined || effect === 'observe'
  })
    ? 'replay-safe'
    : 'replay-risky'
}

function formatSubagentResult(result: SubagentDispatchResult, startedAt: number): ToolResult {
  const summary = [
    `[subagent ${result.sessionId} status=${result.status}`,
    `iterations=${result.iterations}`,
    ...(result.budget ? [`budget=${result.budget.appliedIterations}/${result.budget.requestedIterations} hardCap=${result.budget.hardCap}`] : []),
    `tokens=${result.usage.inputTokens}+${result.usage.outputTokens}]`,
  ].join(' ')
  return {
    output: `${summary}\n\n${result.output}${result.worktree ? `\n\nWorktree: ${JSON.stringify(result.worktree)}` : ''}`,
    status: result.status === 'completed' ? 'success' : 'error',
    durationMs: Date.now() - startedAt,
    metadata: {
      ...(result.budget ? { subagentBudget: result.budget } : {}),
      ...(result.worktree ? { worktree: result.worktree } : {}),
      delegatedUsage: {
        inputTokens: result.usage.inputTokens,
        outputTokens: result.usage.outputTokens,
      },
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
}

function inheritedPolicy(context: ToolExecutionContext | undefined) {
  return context?.delegatedAgentPolicy
    ? {
        autonomy: context.delegatedAgentPolicy.autonomy,
        requireToolApproval: context.delegatedAgentPolicy.requireToolApproval,
        allowedToolNames: [...context.delegatedAgentPolicy.allowedToolNames],
      }
    : undefined
}

export function createSubagentDispatchTool(
  dispatcher: SubagentDispatcher,
  _parentAllowedTools: () => string[],
  background?: () => BackgroundSubagents | undefined,
  options: {
    toolEffect?: SubagentToolEffectResolver
  } = {},
): ToolDefinitionRuntime {
  const toolEffect = options.toolEffect ?? builtinEffect
  const categoryIds = listSubagentDelegationCategories().map((category) => category.id)
  return {
    name: 'subagent.dispatch',
    timeoutPolicy: 'inactivity',
    description:
      'Dispatch an isolated subagent to handle a sub-task. Returns its final output without leaking internal turns into the parent session. Independent read-only dispatches (read-only category or observe-only tools) in one turn run in parallel. Set background=true only for work the user wants to continue independently: returns a durable jobId immediately; inspect or cancel it with subagent.job. Parent turn completion does not cancel detached work. Send a follow-up to a finished subagent with subagent.job action=message. Optional category presets tune scope, tool defaults, and iteration budget.',
    scheduling: {
      mode: 'parallel-safe',
      resource: 'subagent',
      key: (input) => (typeof input.prompt === 'string' ? input.prompt : null),
    },
    resumeSafetyForInput: (input) => subagentDispatchResumeSafety(input, toolEffect),
    inputSchema: {
      type: 'object',
      properties: {
        isolation: {
          type: 'string', enum: ['worktree'],
          description: 'Run in a separate git checkout. Changed worktrees are retained; the result names the branch and a mergeCommand to bring changes back.',
        },
        worktreeBase: {
          type: 'string', enum: ['head', 'working-tree'],
          description: 'With isolation=worktree: head (default) starts at committed HEAD; working-tree also copies the parent uncommitted tracked edits and untracked non-ignored files.',
        },
        contextMode: {
          type: 'string', enum: ['packet', 'fork'],
          description: 'packet (default): child sees prompt + contextPacket only. fork: child also receives this conversation recent history (bounded). Tool restrictions are unchanged.',
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
        agentId: { type: 'string', description: 'Use a registered user-defined agent' },
        bot: {
          type: 'string',
          description: 'Run the child as a sepilot role bot (see `sepilot bots`): a specialist such as code-reviewer or qa-engineer, or an external coding CLI bot (claude, codex, opencode) that does the hands-on work under this run\'s autonomy. Alias of agentId for bots.',
        },
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
        bot?: string
        maxIterations?: number
        tools?: string[]
        model?: string
        contextPacket?: string
        isolation?: 'worktree'
        worktreeBase?: 'head' | 'working-tree'
        contextMode?: 'packet' | 'fork'
      }
      const startedAt = Date.now()
      try {
        const request = {
          prompt: inputTyped.prompt,
          system: inputTyped.system,
          category: inputTyped.category,
          agentId: inputTyped.bot?.trim() || inputTyped.agentId,
          maxIterations: inputTyped.maxIterations,
          tools: inputTyped.tools,
          model: inputTyped.model,
          contextPacket: inputTyped.contextPacket,
          isolation: inputTyped.isolation,
          ...(inputTyped.worktreeBase ? { worktreeBase: inputTyped.worktreeBase } : {}),
          ...(inputTyped.contextMode ? { contextMode: inputTyped.contextMode } : {}),
          parentSessionId: context?.sessionId,
          parentProvider: context?.provider,
          parentModel: context?.model,
          cwd: context?.cwd,
          workspaceRoot: context?.workspaceRoot,
          parentExecutionPolicy: inheritedPolicy(context),
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
        return formatSubagentResult(await dispatcher.dispatch(request), startedAt)
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

const JOB_ACTIONS = ['status', 'wait', 'cancel', 'message', 'list'] as const
type JobAction = (typeof JOB_ACTIONS)[number]

export function createSubagentJobTool(
  background: () => BackgroundSubagents | undefined,
  dispatcher?: () => SubagentDispatcher | undefined,
): ToolDefinitionRuntime {
  return {
    name: 'subagent.job',
    timeoutPolicy: 'inactivity',
    description:
      'Manage subagents started by this conversation. status/wait/cancel: a detached background job (jobId); returns bounded progress and the retained result; waiting never cancels work; never poll in a tight loop. list: this conversation\'s subagents (foreground children and background jobs). message: send a follow-up prompt to a finished subagent (childSessionId, or jobId of a finished background job); it continues with its earlier context and the same tools, permissions and working directory, and returns its reply.',
    inputSchema: {
      type: 'object',
      properties: {
        action: { type: 'string', enum: [...JOB_ACTIONS], description: 'Defaults to status.' },
        jobId: { type: 'string', description: 'Background job id (status, wait, cancel, message).' },
        childSessionId: { type: 'string', description: 'Subagent session id from a dispatch result or list (message).' },
        prompt: { type: 'string', description: 'Follow-up instruction (message).' },
        timeoutMs: {
          type: 'integer',
          minimum: 0,
          maximum: 600000,
          description: 'Wait window, default 10000ms, up to 600000ms. Only applies to wait; returns early on completion or a pending approval.',
        },
      },
      additionalProperties: false,
    },
    async execute(input, context) {
      const startedAt = Date.now()
      try {
        context?.signal?.throwIfAborted()
        if (!context?.sessionId) throw new Error('A parent session is required to manage subagents')
        const action = (input.action ?? 'status') as JobAction
        if (!JOB_ACTIONS.includes(action)) throw new Error('Invalid subagent job action')
        const jobId = typeof input.jobId === 'string' ? input.jobId : ''
        if (action === 'list') {
          return {
            output: JSON.stringify(await listSubagents(context.sessionId, background(), dispatcher?.())),
            status: 'success',
            durationMs: Date.now() - startedAt,
          }
        }
        if (action === 'message') {
          const service = dispatcher?.()
          if (!service) throw new Error('Subagent dispatcher is unavailable')
          const prompt = typeof input.prompt === 'string' ? input.prompt.trim() : ''
          if (!prompt) throw new Error('message requires a prompt')
          let childSessionId = typeof input.childSessionId === 'string' ? input.childSessionId : ''
          if (!childSessionId && jobId) {
            const job = background()?.inspect(jobId, context.sessionId)
            if (!job) throw new Error('Subagent job not found in this conversation')
            if (['pending', 'running'].includes(job.job.status))
              throw new Error('Subagent job is still running; wait for it before sending a follow-up')
            childSessionId = job.activity[0]?.sessionId ?? ''
            if (!childSessionId) throw new Error('Subagent job never started a child session')
          }
          if (!childSessionId) throw new Error('message requires childSessionId or jobId')
          const result = await service.continue({
            childSessionId,
            prompt,
            parentSessionId: context.sessionId,
            parentExecutionPolicy: inheritedPolicy(context),
            signal: context.signal,
            ...(context.emitEvent ? { onEvent: context.emitEvent } : {}),
          })
          return formatSubagentResult(result, startedAt)
        }
        const service = background()
        if (!service) throw new Error('Background subagent service is unavailable')
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
            ? Math.max(0, Math.min(600000, input.timeoutMs))
            : 10000
        let lastBeat = ''
        let lastBeatAt = 0
        const HEARTBEAT_MS = 15_000
        const heartbeat = () => {
          if (!context.emitEvent) return
          const latest = result.activity.at(-1) as { phase?: string; toolName?: string } | undefined
          const summary = `waiting · ${result.job.status}${latest?.phase ? ` · ${latest.phase}` : ''}${latest?.toolName ? ` ${latest.toolName}` : ''}`
          const now = Date.now()
          if (summary === lastBeat && now - lastBeatAt < HEARTBEAT_MS) return
          lastBeat = summary
          lastBeatAt = now
          const job = result.job as { childSessionId?: string; agentId?: string; category?: string }
          context.emitEvent({
            type: 'subagent_progress',
            subagentId: job.childSessionId ?? jobId ?? 'job',
            ...(job.agentId || job.category ? { label: job.agentId ?? job.category } : {}),
            inner: { type: 'action_progress', summary, nextStep: '', toolNames: [] },
          })
        }
        while (
          action === 'wait' &&
          ['pending', 'running'].includes(result.job.status) &&
          Date.now() - startedAt < timeout
        ) {
          heartbeat()
          await delay(Math.min(250, timeout - (Date.now() - startedAt)), undefined, {
            signal: context.signal,
          })
          result = snapshot()
          if (result.activity.some((item) => item.status === 'running' && item.approvalRequestId))
            break
        }
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
        const code = (error as { code?: unknown }).code
        return {
          output: error instanceof Error ? error.message : String(error),
          status: 'error',
          durationMs: Date.now() - startedAt,
          ...(typeof code === 'string' && code.startsWith('SUBAGENT_') ? { code } : {}),
        }
      }
    },
  }
}

async function listSubagents(
  parentSessionId: string,
  background: BackgroundSubagents | undefined,
  dispatcher: SubagentDispatcher | undefined,
) {
  const children = (await dispatcher?.listChildren(parentSessionId)) ?? []
  const jobs = background?.list(parentSessionId) ?? []
  const jobByChild = new Map(
    jobs.filter((job) => job.childSessionId).map((job) => [job.childSessionId!, job]),
  )
  const listed = children.map((child) => {
    const job = jobByChild.get(child.sessionId)
    return {
      id: child.sessionId,
      childSessionId: child.sessionId,
      ...(job ? { jobId: job.jobId, background: true } : { background: false }),
      ...(child.category ? { category: child.category } : {}),
      ...(child.agentId ? { agentId: child.agentId } : {}),
      status: job ? job.status : child.status,
      ...(child.summary ?? job?.summary ? { summary: child.summary ?? job?.summary } : {}),
      ...(child.worktree ? { worktree: child.worktree } : {}),
    }
  })
  const listedChildren = new Set(children.map((child) => child.sessionId))
  for (const job of jobs) {
    if (job.childSessionId && listedChildren.has(job.childSessionId)) continue
    listed.push({
      id: job.jobId,
      jobId: job.jobId,
      background: true,
      ...(job.childSessionId ? { childSessionId: job.childSessionId } : {}),
      ...(job.category ? { category: job.category } : {}),
      ...(job.agentId ? { agentId: job.agentId } : {}),
      status: job.status,
      ...(job.summary ? { summary: job.summary } : {}),
    } as (typeof listed)[number])
  }
  return {
    subagents: listed,
    next: 'Send a follow-up with subagent.job action=message and childSessionId (or jobId); inspect background jobs with action=status.',
  }
}
