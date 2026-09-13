import { randomUUID } from 'node:crypto'
import { createSubagentWorktree, type SubagentWorktreeReceipt } from './subagent-worktree.js'
import type {
  AgentContext,
  AgentEvent,
  AgentRunContract,
  ISessionStore,
  TokenUsage,
  ToolCall,
} from '@sepilotd/core'
import type { AgentEngine } from './engine.js'
import { ToolRegistry } from '../tools/registry.js'
import { persistAgentSessionEvent } from '../server/session-events.js'
import type { DelegatedAgentExecutionPolicy } from '../tools/registry.js'
import { isPolicyReadOnlyTool } from '../security/policy-engine.js'
import type { AgentState } from './graph/types.js'
import { emptyEvidenceLedger, updateEvidenceLedgerFromToolResult } from './graph/evidence-ledger.js'
import { extractSubagentFindings, type SubagentFindings } from './subagent-findings.js'
import { isBudgetExhaustedMessage } from './task-contract.js'
import { isTruncatedStopReason } from './stop-reason.js'
import type { RunStopReason } from '@sepilotd/core'
import { hasIncompleteAnswerStem } from './interim-progress.js'
import { TOOL_RESULT_EXECUTION_OBSERVED_METADATA_KEY } from './policy-failure.js'
import {
  resolveSubagentCategoryToolNames,
  resolveSubagentDelegationCategory,
  unknownSubagentCategoryError,
} from './subagent-categories.js'

const HARD_CAP_ENV = 'SEPILOTD_SUBAGENT_MAX_ITERATIONS_HARD_CAP'
const DEFAULT_HARD_CAP = 50

import { resolveRunIterationBudget } from './iteration-budget.js'

function normalizeIterationBudget(value: number, fallback: number): number {
  const finiteValue = Number.isFinite(value) ? value : fallback
  return Math.max(1, Math.floor(finiteValue))
}

export const SUBAGENT_EMPTY_RESPONSE_OUTPUT = [
  '[subagent empty response]',
  'The isolated subagent completed without a final assistant message.',
  'Treat this as a failed delegation: inspect the subagent session or retry with a narrower prompt.',
].join('\n')

function isEmptySubagentFinalOutput(output: string): boolean {
  const normalized = output.trim()
  return (
    normalized.length === 0 ||
    normalized === 'The run ended with an empty final reply.' ||
    normalized.startsWith(
      'The run completed some tool work but the model returned an empty final reply.',
    )
  )
}

function isIncompleteSubagentFinalOutput(output: string): boolean {
  const normalized = output.trim()
  return (
    hasIncompleteAnswerStem(normalized) ||
    normalized.startsWith(
      'The run ended with a progress-only reply instead of a finished answer.',
    ) ||
    normalized.startsWith(
      'The run completed some tool work but ended with a progress-only reply instead of a finished answer.',
    )
  )
}

export interface SubagentDispatchInput {
  prompt: string
  system?: string
  category?: string
  agentId?: string
  maxIterations?: number
  /**
   * Parent run budget. When no explicit budget is requested the child claims
   * `max(24, ceil(parent * 0.35))`, and never more than the parent itself.
   */
  parentMaxIterations?: number
  tools?: string[]
  model?: string
  contextPacket?: string
  parentSessionId?: string
  /** Parent working directory to preserve path-relative tool behavior. */
  cwd?: string
  /** Immutable strict workspace capability inherited from the parent turn. */
  workspaceRoot?: string
  /** Effective parent-turn authority; never replace this with daemon defaults. */
  parentExecutionPolicy?: DelegatedAgentExecutionPolicy
  /** Set only by the owned background service: approvals are inspectable in jobs. Not an authority grant. */
  detached?: boolean
  /** Explicit clean-HEAD checkout; never copies uncommitted parent changes. */
  isolation?: 'worktree'
  /** Parent run contract to preserve hard boundaries in the child agent. */
  runContract?: AgentRunContract
  /** Authoritative parent memory scope, copied into the isolated child context. */
  scopeTags?: string[]
  signal?: AbortSignal
  onSession?: (sessionId: string) => void
  /**
   * Optional sink for the subagent's intermediate progress events
   * (llm_request/tool_call/tool_result/reasoning_step/thinking/message), wrapped as
   * `subagent_progress`. The parent agent loop wires this to its own
   * event stream so surfaces can show nested activity. Terminal
   * done/error are not forwarded — the dispatch result reports those.
   */
  onEvent?: (event: AgentEvent) => void
}

const FORWARDED_SUBAGENT_EVENT_TYPES = new Set<AgentEvent['type']>([
  'approval_request',
  'approval_response',
  'auto_approval',
  'action_progress',
  'llm_request',
  'tool_call',
  'tool_result',
  'reasoning_step',
  'thinking',
  'message',
])

export interface SubagentDispatchResult {
  worktree?: SubagentWorktreeReceipt
  output: string
  sessionId: string
  category?: string
  iterations: number
  usage: TokenUsage
  truncated: boolean
  status: 'completed' | 'failed' | 'truncated'
  error?: string
  /**
   * Structured board findings reconstructed from the subagent's tool activity
   * (evidence ledger, failed-attempts, open-questions) so the parent can roll
   * them up instead of collapsing the whole run to text. PLAN_065 T1.
   */
  findings?: SubagentFindings
}

export interface SubagentEngineFactoryOptions {
  sessionId: string
  /** Already-journaled runtime events (such as a resolved approval). */
  onEvent?: (event: AgentEvent) => void
  /** Filtered ToolRegistry — already stripped of `subagent.dispatch` and limited to the requested subset. */
  tools: ToolRegistry
  /** Hard-clamped maxIterations for this dispatch. */
  maxIterations: number
  /** Optional model override forwarded by the caller. */
  model?: string
  /** Effective parent-turn authority inherited by this isolated engine. */
  executionPolicy?: DelegatedAgentExecutionPolicy
}

export interface SubagentUserAgentProfile {
  id: string
  name: string
  systemPrompt: string
  model?: string
  maxIterations?: number
  allowedTools?: string[]
  deniedTools?: string[]
  isolation?: 'worktree'
}

export interface SubagentDispatcherDeps {
  onLifecycle?: (
    event: 'started' | 'stopped',
    data: { sessionId: string; parentSessionId?: string; agentId?: string; status?: string },
  ) => Promise<void>
  onRunFinished?: (sessionId: string) => void
  sessions: ISessionStore
  /** Construct a fresh AgentEngine for each dispatch. The factory is responsible for
   * wiring provider/policy/etc; the dispatcher only supplies the per-run overrides
   * (tools subset and clamped maxIterations). */
  engineFactory: (options: SubagentEngineFactoryOptions) => AgentEngine
  /** Parent ToolRegistry — used to resolve concrete ToolDefinitionRuntime entries
   * for the subagent's allowed-tool subset. */
  toolRegistry: ToolRegistry
  /** Returns the parent session's currently allowed tool names (closure for runtime
   * filtering — must reflect current state at dispatch time). */
  parentAllowedTools: () => string[]
  /** Returns the daemon's default provider id and model. Used to populate the new
   * session's metadata when the caller does not pin a provider. */
  defaultProvider: () => { id: string; defaultModel: string } | null
  /** Resolves user-defined agent prompts for explicit `agentId` dispatches. */
  resolveUserAgent?: (
    id: string,
    cwd?: string,
    workspaceRoot?: string,
  ) =>
    | SubagentUserAgentProfile
    | null
    | undefined
    | Promise<SubagentUserAgentProfile | null | undefined>
  /** Optional hard-cap override (env or test). Defaults to env / DEFAULT_HARD_CAP. */
  hardCap?: number
}

function unknownSubagentAgentError(agentId: string): Error & { code: string } {
  const err = new Error(`SUBAGENT_AGENT_UNKNOWN: ${agentId}`) as Error & { code: string }
  err.code = 'SUBAGENT_AGENT_UNKNOWN'
  return err
}

/**
 * Dispatches an isolated subagent run.
 *
 * Each dispatch creates its own session and a fresh AgentEngine via the factory,
 * so internal turns never leak into the parent session's event log. Tool access
 * is restricted to a subset of the parent's allowed tools, and `subagent.dispatch`
 * itself is always stripped from the subagent's whitelist (depth 1 enforcement).
 */
export class SubagentDispatcher {
  constructor(private readonly deps: SubagentDispatcherDeps) {}

  async dispatch(input: SubagentDispatchInput): Promise<SubagentDispatchResult> {
    input.signal?.throwIfAborted()
    const category = input.category ? resolveSubagentDelegationCategory(input.category) : undefined
    if (input.category && !category) {
      throw unknownSubagentCategoryError(input.category)
    }
    const userAgent =
      input.agentId && this.deps.resolveUserAgent
        ? await this.deps.resolveUserAgent(input.agentId, input.cwd, input.workspaceRoot)
        : undefined
    if (input.agentId && this.deps.resolveUserAgent && !userAgent) {
      throw unknownSubagentAgentError(input.agentId)
    }

    const hardCap = normalizeIterationBudget(
      this.deps.hardCap ?? Number(process.env[HARD_CAP_ENV] ?? DEFAULT_HARD_CAP),
      DEFAULT_HARD_CAP,
    )
    const subagentDefault = resolveRunIterationBudget({
      surface: 'subagent',
      parentBudget: input.parentMaxIterations,
    }).maxIterations
    const requested =
      input.maxIterations
      ?? userAgent?.maxIterations
      ?? category?.defaultMaxIterations
      ?? subagentDefault
    const maxIterations = Math.min(
      normalizeIterationBudget(requested, subagentDefault),
      hardCap,
    )

    const parentExecutionPolicy = input.parentExecutionPolicy
      ? {
          autonomy: input.parentExecutionPolicy.autonomy,
          requireToolApproval: input.parentExecutionPolicy.requireToolApproval,
          allowedToolNames: [...input.parentExecutionPolicy.allowedToolNames],
        }
      : undefined
    const parentAllowedNames =
      parentExecutionPolicy?.allowedToolNames ?? this.deps.parentAllowedTools()
    const parentAllowed = new Set(parentAllowedNames)
    for (const name of [...(userAgent?.allowedTools ?? []), ...(userAgent?.deniedTools ?? [])]) {
      if (!this.deps.toolRegistry.get(name))
        throw new Error(
          `Agent "${input.agentId}" refers to unavailable tool "${name}"; its policy was not applied`,
        )
    }

    if (input.tools && input.tools.length > 0) {
      const escalated = input.tools.filter((t) => !parentAllowed.has(t))
      if (escalated.length > 0) {
        const err = new Error(`SUBAGENT_TOOL_ESCALATION: ${escalated.join(', ')}`)
        ;(err as Error & { code?: string }).code = 'SUBAGENT_TOOL_ESCALATION'
        throw err
      }
    }

    // Depth 1 enforcement — strip subagent.dispatch from the subagent's allowed tools
    // so it cannot recursively spawn further subagents. Also enforce the subset
    // restriction here even when no explicit `tools` is supplied (default = parent
    // allowed minus subagent.dispatch).
    const requestedNames =
      input.tools ?? resolveSubagentCategoryToolNames(category, parentAllowedNames)
    // Live progress transports carry nested approvals while the parent awaits
    // the tool; detached jobs have their own durable approval surface. Direct
    // calls without either transport still fail closed to read-only. Merely
    // having a transport never grants approval: retain the inherited policy.
    const hasApprovalTransport = input.detached === true || Boolean(input.onEvent)
    const sanitizedNames = requestedNames.filter(
      (toolName) =>
        toolName !== 'subagent.dispatch' &&
        (userAgent?.allowedTools === undefined || userAgent.allowedTools.includes(toolName)) &&
        !userAgent?.deniedTools?.includes(toolName) &&
        (!parentExecutionPolicy?.requireToolApproval ||
          hasApprovalTransport ||
          isPolicyReadOnlyTool(toolName)),
    )
    const subagentTools = new ToolRegistry()
    for (const name of sanitizedNames) {
      const tool = this.deps.toolRegistry.get(name)
      if (tool) {
        const source = this.deps.toolRegistry.registrationSource(name)
        subagentTools.register(tool, source ? { source } : undefined)
      }
    }

    const provider = this.deps.defaultProvider()
    const model = input.model ?? userAgent?.model ?? provider?.defaultModel ?? 'default'
    const sessionId = randomUUID()
    const createdAt = new Date().toISOString()
    let cwd = input.cwd?.trim() || undefined
    let workspaceRoot = input.workspaceRoot
    let worktree: Awaited<ReturnType<typeof createSubagentWorktree>> | undefined
    const titleSeed = input.prompt.slice(0, 40)
    const tags = [
      'subagent',
      ...(input.parentSessionId ? [`parent:${input.parentSessionId}`] : []),
      ...(category ? [`category:${category.id}`] : []),
      ...(input.agentId ? [`agent:${input.agentId}`] : []),
    ]
    input.signal?.throwIfAborted()
    await this.deps.sessions.create({
      id: sessionId,
      title: category ? `subagent:${category.id}: ${titleSeed}` : `subagent: ${titleSeed}`,
      createdAt,
      updatedAt: createdAt,
      provider: provider?.id ?? 'unknown',
      model,
      device: 'subagent',
      status: 'active',
      tags,
      ...(cwd ? { cwd } : {}),
    })
    let lifecycleStatus = 'failed'
    try {
      if ((input.isolation ?? userAgent?.isolation) === 'worktree') {
        if (!cwd) throw new Error('Worktree isolation requires an explicit parent working directory')
        worktree = await createSubagentWorktree({ cwd, workspaceRoot, sessionId })
        cwd = worktree.cwd
        workspaceRoot = worktree.receipt.path
        await this.deps.sessions.updateMeta?.(sessionId, { cwd, workspaceIsolation: 'strict' })
      }
      await this.deps.onLifecycle?.('started', {
        sessionId,
        parentSessionId: input.parentSessionId,
        agentId: input.agentId,
      })
      input.signal?.throwIfAborted()
      input.onSession?.(sessionId)
      const engine = this.deps.engineFactory({
        sessionId,
        onEvent: (event) =>
          input.onEvent?.({ type: 'subagent_progress', subagentId: sessionId, inner: event }),
        tools: subagentTools,
        maxIterations,
        model,
        executionPolicy: parentExecutionPolicy,
      })

      const contextPacket = input.contextPacket?.trim()
      const categorySystemPrompt = category?.systemPrompt
      const userAgentSystemPrompt = userAgent?.systemPrompt.trim()
        ? [`[User agent: ${userAgent.name || input.agentId}]`, userAgent.systemPrompt.trim()].join(
            '\n',
          )
        : ''
      const callerSystemPrompt = input.system?.trim()
        ? ['Caller system override:', input.system.trim()].join('\n')
        : ''
      const systemPrompt = [
        parentExecutionPolicy?.requireToolApproval && !hasApprovalTransport
          ? '[Delegated execution policy] This child run is read-only because the parent turn requires fresh human approval for every side effect. Return the requested change as guidance if a mutation is needed; do not claim it was applied.'
          : '',
        categorySystemPrompt,
        userAgentSystemPrompt,
        callerSystemPrompt,
        contextPacket
          ? [
              '[Parent context]',
              contextPacket,
              '(Use this to avoid re-exploring what the parent already knows; verify before relying on specifics.)',
            ].join('\n')
          : '',
      ]
        .filter(Boolean)
        .join('\n\n')

      const context: AgentContext = {
        sessionId,
        provider: provider?.id ?? 'unknown',
        model,
        cwd,
        workspaceRoot,
        workspaceIsolation: workspaceRoot ? 'strict' : undefined,
        toolAllowlist: subagentTools.list().map(tool => tool.name),
        requireToolApproval: parentExecutionPolicy?.requireToolApproval,
        systemPrompt,
        // Completion belongs to this task, not the parent's delegation receipt
        // or larger deliverable. Keep every inherited restriction unchanged.
        runContract: input.runContract ? {
          ...structuredClone(input.runContract),
          summary: input.prompt,
          acceptanceCriteria: [{ id: 'delegated-task', text: input.prompt }],
        } : undefined,
        primaryAgentId: input.agentId,
        scopeTags: input.scopeTags ? [...input.scopeTags] : undefined,
      }

      let lastAssistantText = ''
      let messageCount = 0
      let observedIterations = 0
      // Reconstruct board-relevant findings from the subagent's tool activity.
      // The isolated subagent runs a plain ReAct engine (no graph AgentState), so
      // there is no evidence ledger to read at the end — we rebuild it here from
      // the structured tool_call/tool_result event stream the dispatcher already
      // observes, using the same structural classifier the graph uses. No engine
      // accessor is added and depth-1 isolation is untouched.
      const findingsState: Pick<AgentState, 'evidenceLedger'> = {
        evidenceLedger: emptyEvidenceLedger(),
      }
      const toolCallsById = new Map<string, ToolCall>()
      const totalUsage: TokenUsage = { inputTokens: 0, outputTokens: 0 }
      let truncated = false
      let stopReason: RunStopReason | undefined
      let status: 'completed' | 'failed' | 'truncated' = 'completed'
      let error: string | undefined
      let sawDone = false
      const stopOnAbort = () => {
        // stop() may be async; an abort listener must never leak a rejection.
        void Promise.resolve()
          .then(() => engine.stop())
          .catch(() => {})
      }
      input.signal?.addEventListener('abort', stopOnAbort, { once: true })

      try {
        input.signal?.throwIfAborted()
        await this.deps.sessions.appendEvent(sessionId, {
          type: 'user_message',
          id: randomUUID(),
          timestamp: createdAt,
          content: input.prompt,
        })
        input.signal?.throwIfAborted()
        for await (const event of engine.run(input.prompt, context) as AsyncIterable<AgentEvent>) {
          if (event.type !== 'done')
            await persistAgentSessionEvent(this.deps.sessions, sessionId, event)
          if (event.type === 'llm_request')
            observedIterations = Math.max(observedIterations, event.iteration)
          if (input.onEvent && FORWARDED_SUBAGENT_EVENT_TYPES.has(event.type)) {
            input.onEvent({
              type: 'subagent_progress',
              subagentId: sessionId,
              ...(category ? { label: category.id } : {}),
              inner: event,
            })
          }
          if (event.type === 'tool_call') {
            toolCallsById.set(event.toolCall.id, event.toolCall)
          } else if (event.type === 'tool_result') {
            const toolCall = toolCallsById.get(event.toolCallId)
            if (toolCall) {
              updateEvidenceLedgerFromToolResult(
                findingsState as AgentState,
                toolCall,
                event.status,
                event.output,
                undefined,
                event.executionPosture,
                {
                  executionObserved:
                    event.metadata?.[TOOL_RESULT_EXECUTION_OBSERVED_METADATA_KEY] === true,
                  securityEffect: this.deps.toolRegistry.securityDescriptor(toolCall.name).effect,
                },
              )
            }
          }
          if (event.type === 'message') {
            lastAssistantText = event.content
            messageCount += 1
          } else if (event.type === 'done') {
            sawDone = true
            totalUsage.inputTokens += event.usage.inputTokens
            totalUsage.outputTokens += event.usage.outputTokens
            if (event.stopReason) stopReason = event.stopReason
          } else if (event.type === 'error') {
            status = 'failed'
            error = event.error?.message ?? 'subagent error'
          }
        }
      } catch (err) {
        status = 'failed'
        error = err instanceof Error ? err.message : String(err)
      } finally {
        input.signal?.removeEventListener('abort', stopOnAbort)
      }

      if (input.signal?.aborted || stopReason?.kind === 'cancelled') {
        status = 'failed'
        error = 'subagent canceled'
      } else if (
        stopReason?.kind === 'error' ||
        (stopReason?.kind === 'blocked' && !isTruncatedStopReason(stopReason))
      ) {
        status = 'failed'
        error = stopReason.summary ?? `subagent stopped: ${stopReason.code}`
      } else if (status !== 'failed' && !sawDone) {
        status = 'failed'
        error = 'subagent stream ended without completion evidence'
      }

      // Surface a resumable budget/loop-control stop as `status: 'truncated'` so
      // callers can distinguish a clean completion from a resumable stop. The
      // structured stop reason is authoritative; the message sentinel remains a
      // fallback for engines that predate `done.stopReason`.
      if (
        stopReason ? isTruncatedStopReason(stopReason) : isBudgetExhaustedMessage(lastAssistantText)
      ) {
        truncated = true
        if (status !== 'failed') status = 'truncated'
      }

      if (status === 'completed' && isEmptySubagentFinalOutput(lastAssistantText)) {
        status = 'failed'
        error = 'subagent returned empty response'
        lastAssistantText = SUBAGENT_EMPTY_RESPONSE_OUTPUT
      } else if (status === 'completed' && isIncompleteSubagentFinalOutput(lastAssistantText)) {
        status = 'failed'
        error = 'subagent returned incomplete response'
      }

      const findings = extractSubagentFindings(findingsState, {
        sessionId,
        category: category?.id ?? 'general',
      })

      // Use the same journal as foreground runs so a detached client's session
      // link contains real tool evidence, usage and a terminal state.
      if (lastAssistantText) {
        await this.deps.sessions.appendEvent(sessionId, {
          type: 'assistant_message',
          id: randomUUID(),
          timestamp: new Date().toISOString(),
          content: lastAssistantText,
        })
      }
      await this.deps.sessions.appendEvent(sessionId, {
        type: 'session_end',
        id: randomUUID(),
        timestamp: new Date().toISOString(),
        totalTokens: { input: totalUsage.inputTokens, output: totalUsage.outputTokens },
        totalCost: 0,
        duration_ms: Date.now() - Date.parse(createdAt),
        stopReason:
          status === 'failed'
            ? {
                kind:
                  input.signal?.aborted || stopReason?.kind === 'cancelled' ? 'cancelled' : 'error',
                code:
                  input.signal?.aborted || stopReason?.kind === 'cancelled'
                    ? 'user_abort'
                    : 'provider_error',
                summary: error,
                resumable: false,
                nextActions: ['retry'],
              }
            : (stopReason ?? {
                kind: truncated ? 'incomplete' : 'completed',
                code: truncated ? 'iteration_budget' : 'completed',
                resumable: truncated,
                nextActions: truncated ? ['resume'] : [],
              }),
      })
      if (status === 'failed')
        await this.deps.sessions.updateMeta?.(sessionId, { status: 'abandoned' })

      lifecycleStatus = status
      return {
        ...(worktree ? { worktree: worktree.receipt } : {}),
        output: lastAssistantText,
        sessionId,
        category: category?.id,
        iterations: observedIterations || messageCount,
        usage: totalUsage,
        truncated,
        status,
        error,
        findings,
      }
    } finally {
      await worktree?.finish(lifecycleStatus !== 'completed')
      this.deps.onRunFinished?.(sessionId)
      if (lifecycleStatus === 'failed') {
        await this.deps.sessions.updateMeta?.(sessionId, { status: 'abandoned' })
      }
      await this.deps.onLifecycle?.('stopped', {
        sessionId,
        parentSessionId: input.parentSessionId,
        agentId: input.agentId,
        status: lifecycleStatus,
      })
    }
  }
}
