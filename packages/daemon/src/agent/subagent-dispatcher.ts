import { randomUUID } from 'node:crypto'
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

const DEFAULT_MAX_ITERATIONS = 20
const HARD_CAP_ENV = 'SEPILOTD_SUBAGENT_MAX_ITERATIONS_HARD_CAP'
const DEFAULT_HARD_CAP = 50

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
    normalized.length === 0
    || normalized === 'The run ended with an empty final reply.'
    || normalized.startsWith('The run completed some tool work but the model returned an empty final reply.')
  )
}

function isIncompleteSubagentFinalOutput(output: string): boolean {
  const normalized = output.trim()
  return (
    hasIncompleteAnswerStem(normalized)
    || normalized.startsWith(
      'The run ended with a progress-only reply instead of a finished answer.',
    )
    || normalized.startsWith(
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
  /** Parent run contract to preserve hard boundaries in the child agent. */
  runContract?: AgentRunContract
  /** Authoritative parent memory scope, copied into the isolated child context. */
  scopeTags?: string[]
  signal?: AbortSignal
  /**
   * Optional sink for the subagent's intermediate progress events
   * (tool_call/tool_result/reasoning_step/thinking/message), wrapped as
   * `subagent_progress`. The parent agent loop wires this to its own
   * event stream so surfaces can show nested activity. Terminal
   * done/error are not forwarded — the dispatch result reports those.
   */
  onEvent?: (event: AgentEvent) => void
}

const FORWARDED_SUBAGENT_EVENT_TYPES = new Set<AgentEvent['type']>([
  'tool_call',
  'tool_result',
  'reasoning_step',
  'thinking',
  'message',
])

export interface SubagentDispatchResult {
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
}

export interface SubagentDispatcherDeps {
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
  resolveUserAgent?: (id: string) => SubagentUserAgentProfile | null | undefined
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
    const category = input.category ? resolveSubagentDelegationCategory(input.category) : undefined
    if (input.category && !category) {
      throw unknownSubagentCategoryError(input.category)
    }
    const userAgent = input.agentId && this.deps.resolveUserAgent
      ? this.deps.resolveUserAgent(input.agentId)
      : undefined
    if (input.agentId && this.deps.resolveUserAgent && !userAgent) {
      throw unknownSubagentAgentError(input.agentId)
    }

    const hardCap = normalizeIterationBudget(
      this.deps.hardCap ?? Number(process.env[HARD_CAP_ENV] ?? DEFAULT_HARD_CAP),
      DEFAULT_HARD_CAP,
    )
    const requested =
      input.maxIterations
      ?? userAgent?.maxIterations
      ?? category?.defaultMaxIterations
      ?? DEFAULT_MAX_ITERATIONS
    const maxIterations = Math.min(
      normalizeIterationBudget(requested, DEFAULT_MAX_ITERATIONS),
      hardCap,
    )

    const parentExecutionPolicy = input.parentExecutionPolicy
      ? {
          autonomy: input.parentExecutionPolicy.autonomy,
          requireToolApproval: input.parentExecutionPolicy.requireToolApproval,
          allowedToolNames: [...input.parentExecutionPolicy.allowedToolNames],
        }
      : undefined
    const parentAllowedNames = parentExecutionPolicy?.allowedToolNames
      ?? this.deps.parentAllowedTools()
    const parentAllowed = new Set(parentAllowedNames)

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
    // Nested approval requests cannot currently be streamed while the parent
    // tool is awaiting completion. Letting a child retain mutation tools would
    // either deadlock on approval or bypass the parent's explicit HITL guard.
    // Fail closed to the canonical read-only classification for strict turns.
    const sanitizedNames = requestedNames.filter((toolName) =>
      toolName !== 'subagent.dispatch'
      && (!parentExecutionPolicy?.requireToolApproval || isPolicyReadOnlyTool(toolName)))
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
    const cwd = input.cwd?.trim() || undefined
    const titleSeed = input.prompt.slice(0, 40)
    const tags = [
      'subagent',
      ...(input.parentSessionId ? [`parent:${input.parentSessionId}`] : []),
      ...(category ? [`category:${category.id}`] : []),
      ...(input.agentId ? [`agent:${input.agentId}`] : []),
    ]
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

    const engine = this.deps.engineFactory({
      tools: subagentTools,
      maxIterations,
      model,
      executionPolicy: parentExecutionPolicy,
    })

    const contextPacket = input.contextPacket?.trim()
    const categorySystemPrompt = category?.systemPrompt
    const userAgentSystemPrompt = userAgent?.systemPrompt.trim()
      ? [
          `[User agent: ${userAgent.name || input.agentId}]`,
          userAgent.systemPrompt.trim(),
        ].join('\n')
      : ''
    const callerSystemPrompt = input.system?.trim()
      ? ['Caller system override:', input.system.trim()].join('\n')
      : ''
    const systemPrompt = [
      parentExecutionPolicy?.requireToolApproval
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
    ].filter(Boolean).join('\n\n')

    const context: AgentContext = {
      sessionId,
      provider: provider?.id ?? 'unknown',
      model,
      cwd,
      workspaceRoot: input.workspaceRoot,
      requireToolApproval: parentExecutionPolicy?.requireToolApproval,
      systemPrompt,
      runContract: input.runContract,
      primaryAgentId: input.agentId,
      scopeTags: input.scopeTags ? [...input.scopeTags] : undefined,
    }

    let lastAssistantText = ''
    let messageCount = 0
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
    const stopOnAbort = () => {
      void engine.stop()
    }
    if (input.signal?.aborted) {
      stopOnAbort()
    } else {
      input.signal?.addEventListener('abort', stopOnAbort, { once: true })
    }

    try {
      for await (const event of engine.run(input.prompt, context) as AsyncIterable<AgentEvent>) {
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

    // Surface a resumable budget/loop-control stop as `status: 'truncated'` so
    // callers can distinguish a clean completion from a resumable stop. The
    // structured stop reason is authoritative; the message sentinel remains a
    // fallback for engines that predate `done.stopReason`.
    if (
      stopReason
        ? isTruncatedStopReason(stopReason)
        : isBudgetExhaustedMessage(lastAssistantText)
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

    return {
      output: lastAssistantText,
      sessionId,
      category: category?.id,
      iterations: messageCount,
      usage: totalUsage,
      truncated,
      status,
      error,
      findings,
    }
  }
}
