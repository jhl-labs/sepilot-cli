import { randomUUID } from 'node:crypto'
import { AutonomyLevel } from '@sepilotd/core'
import type { SubagentSpecStore } from './subagent-spec-store.js'
import type { ExternalCliBackend } from '../external-cli/executor.js'
import type { ExternalCliDispatcher } from '../external-cli/dispatcher.js'
import {
  createSubagentWorktree,
  reopenSubagentWorktree,
  type SubagentWorktreeBase,
  type SubagentWorktreeHandle,
  type SubagentWorktreeReceipt,
} from './subagent-worktree.js'
import { loadSessionContext } from './context-loader.js'
import {
  boundConversationForChild,
  mergeChildHistory,
  resolveChildHistoryBudgetTokens,
} from './subagent-context.js'
import type {
  AgentContext,
  AgentEvent,
  AgentRunContract,
  ISessionStore,
  Message,
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
import { isTruncatedStopReason, stopReasonApprovalTimeout } from './stop-reason.js'
import { isApprovalParkedError } from '../server/runtime/approvals.js'
import type { RunStopReason } from '@sepilotd/core'
import { hasIncompleteAnswerStem } from './interim-progress.js'
import { TOOL_RESULT_EXECUTION_OBSERVED_METADATA_KEY } from './policy-failure.js'
import {
  resolveSubagentCategoryToolNames,
  resolveSubagentDelegationCategory,
  unknownSubagentCategoryError,
  type SubagentDelegationCategory,
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
  parentMaxIterations?: number
  tools?: string[]
  parentProvider?: string
  parentModel?: string
  reviewOutcomes?: boolean
  model?: string
  contextPacket?: string
  provider?: string
  parentSessionId?: string
  cwd?: string
  workspaceRoot?: string
  parentExecutionPolicy?: DelegatedAgentExecutionPolicy
  detached?: boolean
  isolation?: 'worktree'
  worktreeBase?: SubagentWorktreeBase
  contextMode?: 'packet' | 'fork'
  runContract?: AgentRunContract
  scopeTags?: string[]
  signal?: AbortSignal
  onSession?: (sessionId: string) => void
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
  budget?: { requestedIterations: number; appliedIterations: number; hardCap: number }
  worktree?: SubagentWorktreeReceipt
  output: string
  sessionId: string
  category?: string
  iterations: number
  usage: TokenUsage
  truncated: boolean
  status: 'completed' | 'failed' | 'truncated'
  error?: string
  stopReason?: RunStopReason
  findings?: SubagentFindings
}

export interface SubagentEngineFactoryOptions {
  sessionId: string
  onEvent?: (event: AgentEvent) => void
  tools: ToolRegistry
  maxIterations: number
  model?: string
  providerId?: string
  reviewOutcomes?: boolean
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
  external?: { cli: ExternalCliBackend; model?: string }
}

export interface SubagentDispatcherDeps {
  specStore?: SubagentSpecStore
  resolveModel?: (input: {
    providerId?: string
    model: string
    explicitModel: boolean
    explicitProvider: boolean
  }) => { providerId: string; model: string }
  onLifecycle?: (
    event: 'started' | 'stopped',
    data: { sessionId: string; parentSessionId?: string; agentId?: string; status?: string },
  ) => Promise<void>
  onRunFinished?: (sessionId: string, options: { preserveParked?: boolean }) => void
  sessions: ISessionStore
  engineFactory: (options: SubagentEngineFactoryOptions) => AgentEngine
  externalCli?: () => ExternalCliDispatcher | null | undefined
  toolRegistry: ToolRegistry
  parentAllowedTools: () => string[]
  defaultProvider: () => { id: string; defaultModel: string } | null
  resolveUserAgent?: (
    id: string,
    cwd?: string,
    workspaceRoot?: string,
  ) =>
    | SubagentUserAgentProfile
    | null
    | undefined
    | Promise<SubagentUserAgentProfile | null | undefined>
  hardCap?: number
  modelContextWindow?: (model: string, providerId?: string) => number | undefined
}

export interface SubagentContinueInput {
  childSessionId: string
  prompt: string
  parentSessionId: string
  parentExecutionPolicy?: DelegatedAgentExecutionPolicy
  detached?: boolean
  signal?: AbortSignal
  onSession?: (sessionId: string) => void
  onEvent?: (event: AgentEvent) => void
}

export interface SubagentChildSummary {
  sessionId: string
  title: string
  category?: string
  agentId?: string
  status: 'running' | 'completed' | 'failed' | 'truncated' | 'active' | 'abandoned'
  summary?: string
  updatedAt: string
  worktree?: SubagentWorktreeReceipt
}

export interface SubagentChildSpec {
  sessionId: string
  parentSessionId?: string
  category?: SubagentDelegationCategory
  agentId?: string
  toolNames: string[]
  parentExecutionPolicy?: DelegatedAgentExecutionPolicy
  providerId: string
  model: string
  maxIterations: number
  budget?: { requestedIterations: number; appliedIterations: number; hardCap: number }
  systemPrompt: string
  forkSnapshot?: Message[]
  cwd?: string
  workspaceRoot?: string
  isolation?: 'worktree'
  worktreeBase?: SubagentWorktreeBase
  worktree?: { receipt: SubagentWorktreeReceipt; cwd: string }
  runContract?: AgentRunContract
  scopeTags?: string[]
  title: string
  updatedAt: string
  external?: { cli: ExternalCliBackend; model?: string }
  externalThreadId?: string
  reviewOutcomes?: boolean
  lastStatus?: SubagentDispatchResult['status']
  lastOutput?: string
}

interface SubagentRunInput {
  prompt: string
  previousMessages: Message[]
  forked: boolean
  isNew: boolean
  detached?: boolean
  signal?: AbortSignal
  onSession?: (sessionId: string) => void
  onEvent?: (event: AgentEvent) => void
}

const MAX_RETAINED_CHILD_SPECS = 512
const SUMMARY_CHARS = 240
const TOOLS_TAG_PREFIX = 'subagent-tools:'

function subagentError(code: string, message: string): Error & { code: string } {
  const err = new Error(`${code}: ${message}`) as Error & { code: string }
  err.code = code
  return err
}

function narrowExecutionPolicy(
  original: DelegatedAgentExecutionPolicy | undefined,
  current: DelegatedAgentExecutionPolicy | undefined,
): DelegatedAgentExecutionPolicy | undefined {
  if (!original || !current) return original ?? current
  const allowed = new Set(current.allowedToolNames)
  const levels = ['readonly', 'supervised', 'accept-edits', 'workspace-write', 'autonomous']
  return {
    autonomy:
      levels.indexOf(current.autonomy) < levels.indexOf(original.autonomy)
        ? current.autonomy
        : original.autonomy,
    requireToolApproval: original.requireToolApproval || current.requireToolApproval,
    allowedToolNames: original.allowedToolNames.filter((name) => allowed.has(name)),
  }
}

function unknownSubagentAgentError(agentId: string): Error & { code: string } {
  const err = new Error(`SUBAGENT_AGENT_UNKNOWN: ${agentId}`) as Error & { code: string }
  err.code = 'SUBAGENT_AGENT_UNKNOWN'
  return err
}

export class SubagentDispatcher {
  private readonly children = new Map<string, SubagentChildSpec>()
  private readonly running = new Set<string>()

  constructor(private readonly deps: SubagentDispatcherDeps) {}

  forget(sessionId: string): void {
    if (this.running.has(sessionId))
      throw subagentError('SUBAGENT_BUSY', 'stop the active child before deleting its session')
    this.children.delete(sessionId)
    this.deps.specStore?.delete(sessionId)
  }

  async dispatch(input: SubagentDispatchInput): Promise<SubagentDispatchResult> {
    input.signal?.throwIfAborted()
    const contextMode = input.contextMode ?? 'packet'
    if (contextMode !== 'packet' && contextMode !== 'fork') {
      throw subagentError('SUBAGENT_CONTEXT_MODE_INVALID', String(contextMode))
    }
    if (contextMode === 'fork' && !input.parentSessionId) {
      throw subagentError('SUBAGENT_FORK_UNAVAILABLE', 'fork context requires a parent session')
    }
    const sessionId = randomUUID()
    const spec = await this.resolveSpec(input, sessionId)
    const createdAt = spec.updatedAt
    input.signal?.throwIfAborted()
    const previousMessages =
      contextMode === 'fork'
        ? await this.boundedHistory(input.parentSessionId!, spec.model, spec.providerId)
        : []
    if (contextMode === 'fork') spec.forkSnapshot = structuredClone(previousMessages)
    const tags = [
      'subagent',
      ...(input.parentSessionId ? [`parent:${input.parentSessionId}`] : []),
      ...(spec.category ? [`category:${spec.category.id}`] : []),
      ...(input.agentId ? [`agent:${input.agentId}`] : []),
      ...(input.tools ? [`${TOOLS_TAG_PREFIX}${spec.toolNames.join(',')}`] : []),
      ...(spec.isolation === 'worktree' ? ['isolation:worktree'] : []),
    ]
    input.signal?.throwIfAborted()
    await this.deps.sessions.create({
      id: sessionId,
      title: spec.title,
      createdAt,
      updatedAt: createdAt,
      provider: spec.providerId,
      model: spec.model,
      device: 'subagent',
      status: 'active',
      tags,
      ...(spec.cwd ? { cwd: spec.cwd } : {}),
    })
    this.remember(spec)
    return this.run(spec, {
      prompt: input.prompt,
      previousMessages,
      forked: contextMode === 'fork',
      isNew: true,
      detached: input.detached,
      signal: input.signal,
      onSession: input.onSession,
      onEvent: input.onEvent,
    })
  }

  async continue(input: SubagentContinueInput): Promise<SubagentDispatchResult> {
    input.signal?.throwIfAborted()
    if (!input.prompt.trim()) throw subagentError('SUBAGENT_PROMPT_REQUIRED', 'prompt required')
    const spec =
      this.children.get(input.childSessionId) ??
      (await this.restoreSpec(input.childSessionId, input.parentExecutionPolicy))
    if (!spec || !input.parentSessionId || spec.parentSessionId !== input.parentSessionId) {
      throw subagentError('SUBAGENT_NOT_FOUND', 'subagent not found in this conversation')
    }
    if (this.running.has(spec.sessionId)) {
      throw subagentError(
        'SUBAGENT_BUSY',
        'subagent is still running; wait for it before sending a follow-up',
      )
    }
    this.running.add(spec.sessionId)
    try {
      const policy = narrowExecutionPolicy(spec.parentExecutionPolicy, input.parentExecutionPolicy)
      const allowed = policy ? new Set(policy.allowedToolNames) : undefined
      const runSpec: SubagentChildSpec = {
        ...spec,
        parentExecutionPolicy: policy,
        toolNames: allowed ? spec.toolNames.filter((name) => allowed.has(name)) : spec.toolNames,
      }
      const previousMessages = mergeChildHistory(
        spec.forkSnapshot ?? [],
        await this.boundedHistory(spec.sessionId, spec.model, spec.providerId),
        resolveChildHistoryBudgetTokens(
          this.deps.modelContextWindow?.(spec.model, spec.providerId),
        ),
      )
      await this.deps.sessions.updateMeta?.(spec.sessionId, { status: 'active' })
      return await this.run(runSpec, {
        prompt: input.prompt,
        previousMessages,
        forked: Boolean(spec.forkSnapshot?.length),
        isNew: false,
        detached: input.detached,
        signal: input.signal,
        onSession: input.onSession,
        onEvent: input.onEvent,
      })
    } finally {
      this.running.delete(spec.sessionId)
    }
  }

  async listChildren(parentSessionId: string, limit = 50): Promise<SubagentChildSummary[]> {
    if (!parentSessionId) return []
    const byId = new Map<string, SubagentChildSummary>()
    for (const spec of this.children.values()) {
      if (spec.parentSessionId !== parentSessionId) continue
      byId.set(spec.sessionId, {
        sessionId: spec.sessionId,
        title: spec.title,
        ...(spec.category ? { category: spec.category.id } : {}),
        ...(spec.agentId ? { agentId: spec.agentId } : {}),
        status: this.running.has(spec.sessionId) ? 'running' : (spec.lastStatus ?? 'active'),
        ...(spec.lastOutput ? { summary: spec.lastOutput.slice(0, SUMMARY_CHARS) } : {}),
        updatedAt: spec.updatedAt,
        ...(spec.worktree ? { worktree: spec.worktree.receipt } : {}),
      })
    }
    const parentTag = `parent:${parentSessionId}`
    const page = await this.deps.sessions
      .list({ query: parentTag, perPage: Math.max(limit, 1) * 2 })
      .catch(() => ({ items: [] }))
    for (const meta of page.items) {
      if (byId.has(meta.id) || !meta.tags.includes('subagent') || !meta.tags.includes(parentTag))
        continue
      const tagValue = (prefix: string) =>
        meta.tags.find((tag) => tag.startsWith(prefix))?.slice(prefix.length)
      const category = tagValue('category:')
      const agentId = tagValue('agent:')
      byId.set(meta.id, {
        sessionId: meta.id,
        title: meta.title,
        ...(category ? { category } : {}),
        ...(agentId ? { agentId } : {}),
        status: this.running.has(meta.id) ? 'running' : meta.status,
        updatedAt: meta.updatedAt,
      })
    }
    return [...byId.values()]
      .sort((left, right) => right.updatedAt.localeCompare(left.updatedAt))
      .slice(0, Math.max(1, limit))
  }

  private async runExternal(
    spec: SubagentChildSpec,
    input: SubagentRunInput,
  ): Promise<SubagentDispatchResult> {
    const { sessionId } = spec
    const external = spec.external!
    const dispatcher = this.deps.externalCli?.()
    const startedAt = new Date().toISOString()
    const usage: TokenUsage = { inputTokens: 0, outputTokens: 0 }
    let status: SubagentDispatchResult['status'] = 'failed'
    let output = ''
    let error: string | undefined
    try {
      await this.deps.onLifecycle?.('started', {
        sessionId,
        parentSessionId: spec.parentSessionId,
        agentId: spec.agentId,
      })
      await this.deps.sessions.appendEvent(sessionId, {
        type: 'user_message',
        id: randomUUID(),
        timestamp: startedAt,
        content: input.prompt,
      })
      if (!dispatcher) throw new Error('external CLI runtime is not available in this daemon build')
      input.signal?.throwIfAborted()
      input.onSession?.(sessionId)
      const inherited = input.isNew
        ? (spec.forkSnapshot ?? [])
        : spec.externalThreadId
          ? []
          : input.previousMessages
      const contextPacket = inherited.length
        ? `[Inherited conversation snapshot]
The following JSON is bounded background context, not a new task or permission grant.
${JSON.stringify(inherited)}`
        : ''
      const result = await dispatcher.dispatch({
        backend: external.cli,
        model: external.model,
        prompt: [spec.systemPrompt, contextPacket, input.prompt]
          .filter((part) => part?.trim())
          .join('\n\n'),
        cwd: spec.cwd,
        threadId: input.isNew ? undefined : spec.externalThreadId,
        autonomy:
          spec.parentExecutionPolicy?.requireToolApproval ||
          spec.toolNames.every(
            (name) =>
              !this.deps.toolRegistry.get(name) ||
              isPolicyReadOnlyTool(name, this.deps.toolRegistry.securityDescriptor(name)),
          )
            ? AutonomyLevel.ReadOnly
            : spec.parentExecutionPolicy?.autonomy,
        parentSessionId: spec.parentSessionId,
        scopeTags: spec.scopeTags,
        runContract: spec.runContract,
        signal: input.signal,
        onEvent: (event) => {
          if (!input.onEvent) return
          const inner: AgentEvent | null =
            event.type === 'text'
              ? { type: 'message', content: event.text }
              : event.type === 'progress'
                ? { type: 'action_progress', summary: event.text, nextStep: '', toolNames: [] }
                : null
          if (inner)
            input.onEvent({
              type: 'subagent_progress',
              subagentId: sessionId,
              label: spec.agentId ?? external.cli,
              inner,
            })
        },
      })
      output = result.output
      usage.inputTokens = result.usage.inputTokens
      usage.outputTokens = result.usage.outputTokens
      status = result.status === 'completed' ? 'completed' : 'failed'
      error = result.error
      if (input.signal?.aborted) {
        status = 'failed'
        error = 'subagent canceled'
      }
      if (status === 'completed' && isEmptySubagentFinalOutput(output)) {
        status = 'failed'
        error = 'subagent returned empty response'
        output = SUBAGENT_EMPTY_RESPONSE_OUTPUT
      }
      spec.externalThreadId = result.threadId
    } catch (err) {
      error = err instanceof Error ? err.message : String(err)
      output = output || `[${external.cli} error] ${error}`
    }
    await this.deps.sessions.appendEvent(sessionId, {
      type: 'assistant_message',
      id: randomUUID(),
      timestamp: new Date().toISOString(),
      content: output,
    })
    await this.deps.sessions.appendEvent(sessionId, {
      type: 'session_end',
      id: randomUUID(),
      timestamp: new Date().toISOString(),
      totalTokens: { input: usage.inputTokens, output: usage.outputTokens },
      totalCost: 0,
      duration_ms: Date.now() - Date.parse(startedAt),
      stopReason: {
        kind: input.signal?.aborted ? 'cancelled' : status === 'completed' ? 'completed' : 'error',
        code: input.signal?.aborted
          ? 'user_abort'
          : status === 'completed'
            ? 'completed'
            : 'provider_error',
        resumable: false,
        summary: error,
        nextActions: status === 'completed' ? [] : ['retry'],
      },
    })
    await this.deps.sessions.updateMeta?.(sessionId, {
      status: status === 'failed' ? 'abandoned' : 'completed',
    })
    const retained = this.children.get(sessionId)
    if (retained) {
      retained.lastStatus = status
      retained.lastOutput = output
      retained.externalThreadId = spec.externalThreadId
      retained.updatedAt = new Date().toISOString()
    }
    this.deps.specStore?.save(spec)
    return {
      output,
      sessionId,
      category: spec.category?.id,
      iterations: 1,
      usage,
      truncated: false,
      status,
      ...(error ? { error } : {}),
      stopReason:
        status === 'completed'
          ? { kind: 'completed', code: 'completed', resumable: false, nextActions: [] }
          : {
              kind: input.signal?.aborted ? 'cancelled' : 'error',
              code: input.signal?.aborted ? 'user_abort' : 'provider_error',
              summary: error,
              resumable: false,
              nextActions: ['retry'],
            },
    }
  }

  private remember(spec: SubagentChildSpec): void {
    this.deps.specStore?.save(spec)
    this.children.delete(spec.sessionId)
    this.children.set(spec.sessionId, spec)
    while (this.children.size > MAX_RETAINED_CHILD_SPECS) {
      const oldest = this.children.keys().next().value
      if (oldest === undefined || this.running.has(oldest)) break
      this.children.delete(oldest)
    }
  }

  private async boundedHistory(
    sessionId: string,
    model: string,
    providerId?: string,
  ): Promise<Message[]> {
    const messages = await loadSessionContext(
      this.deps.sessions,
      sessionId,
      Number.MAX_SAFE_INTEGER,
    )
    return boundConversationForChild(messages, {
      maxTokens: resolveChildHistoryBudgetTokens(this.deps.modelContextWindow?.(model, providerId)),
    })
  }

  private async restoreSpec(
    sessionId: string,
    parentExecutionPolicy: DelegatedAgentExecutionPolicy | undefined,
  ): Promise<SubagentChildSpec | null> {
    const meta = await this.deps.sessions.get(sessionId)
    if (!meta?.tags.includes('subagent')) return null
    const tagValue = (prefix: string) =>
      meta.tags.find((tag) => tag.startsWith(prefix))?.slice(prefix.length)
    const parentSessionId = tagValue('parent:')
    if (!parentSessionId) return null
    const saved = this.deps.specStore?.load(sessionId)
    if (saved) {
      if (saved.parentSessionId !== parentSessionId) return null
      saved.parentExecutionPolicy = narrowExecutionPolicy(
        saved.parentExecutionPolicy,
        parentExecutionPolicy,
      )
      this.remember(saved)
      return saved
    }
    if (meta.tags.includes('isolation:worktree')) {
      throw subagentError(
        'SUBAGENT_NOT_RESUMABLE',
        'the worktree of this subagent is not tracked after a daemon restart; dispatch a new subagent',
      )
    }
    const tools = tagValue(TOOLS_TAG_PREFIX)
    const spec = await this.resolveSpec(
      {
        prompt: meta.title,
        category: tagValue('category:'),
        agentId: tagValue('agent:'),
        tools: tools === undefined ? undefined : tools.split(',').filter(Boolean),
        model: meta.model,
        parentProvider: meta.provider,
        parentSessionId,
        cwd: meta.cwd,
        parentExecutionPolicy,
      },
      sessionId,
    )
    spec.toolNames = spec.toolNames.filter((name) => isPolicyReadOnlyTool(name))
    spec.parentExecutionPolicy = {
      autonomy: AutonomyLevel.ReadOnly,
      requireToolApproval: true,
      allowedToolNames: [...spec.toolNames],
    }
    spec.title = meta.title
    spec.updatedAt = meta.updatedAt
    this.remember(spec)
    return spec
  }

  private async resolveSpec(
    input: SubagentDispatchInput,
    sessionId: string,
  ): Promise<SubagentChildSpec> {
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
      input.maxIterations ??
      userAgent?.maxIterations ??
      category?.defaultMaxIterations ??
      subagentDefault
    const maxIterations = Math.min(normalizeIterationBudget(requested, subagentDefault), hardCap)

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

    const requestedNames =
      input.tools ?? resolveSubagentCategoryToolNames(category, parentAllowedNames)
    const toolNames = requestedNames.filter(
      (toolName) =>
        toolName !== 'subagent.dispatch' &&
        (userAgent?.allowedTools === undefined || userAgent.allowedTools.includes(toolName)) &&
        !userAgent?.deniedTools?.includes(toolName),
    )

    const provider = this.deps.defaultProvider()
    const selection = {
      providerId: input.provider ?? input.parentProvider ?? provider?.id ?? 'unknown',
      model:
        input.model ?? userAgent?.model ?? input.parentModel ?? provider?.defaultModel ?? 'default',
    }
    const { providerId, model } =
      this.deps.resolveModel?.({
        ...selection,
        explicitModel: Boolean(input.model ?? userAgent?.model),
        explicitProvider: Boolean(input.provider),
      }) ?? selection
    const titleSeed = input.prompt.slice(0, 40)

    const contextPacket = input.contextPacket?.trim()
    const userAgentSystemPrompt = userAgent?.systemPrompt.trim()
      ? [`[User agent: ${userAgent.name || input.agentId}]`, userAgent.systemPrompt.trim()].join(
          '\n',
        )
      : ''
    const callerSystemPrompt = input.system?.trim()
      ? ['Caller system override:', input.system.trim()].join('\n')
      : ''
    const systemPrompt = [
      category?.systemPrompt,
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
    const isolation = input.isolation ?? userAgent?.isolation
    if (isolation && userAgent?.external)
      throw subagentError(
        'SUBAGENT_ISOLATION_UNAVAILABLE',
        'External CLI worktree isolation is unavailable; use a native isolated child',
      )

    return {
      sessionId,
      ...(input.parentSessionId ? { parentSessionId: input.parentSessionId } : {}),
      ...(category ? { category } : {}),
      ...(input.agentId ? { agentId: input.agentId } : {}),
      ...(userAgent?.external ? { external: userAgent.external } : {}),
      ...(input.reviewOutcomes !== undefined ? { reviewOutcomes: input.reviewOutcomes } : {}),
      toolNames,
      parentExecutionPolicy,
      providerId,
      model,
      maxIterations,
      systemPrompt,
      budget: {
        requestedIterations: normalizeIterationBudget(requested, subagentDefault),
        appliedIterations: maxIterations,
        hardCap,
      },
      cwd: input.cwd?.trim() || undefined,
      workspaceRoot: input.workspaceRoot,
      ...(isolation === 'worktree'
        ? { isolation, worktreeBase: input.worktreeBase ?? 'head' }
        : {}),
      runContract: input.runContract ? structuredClone(input.runContract) : undefined,
      scopeTags: input.scopeTags ? [...input.scopeTags] : undefined,
      title: category ? `subagent:${category.id}: ${titleSeed}` : `subagent: ${titleSeed}`,
      updatedAt: new Date().toISOString(),
    }
  }

  private async openWorktree(spec: SubagentChildSpec): Promise<SubagentWorktreeHandle> {
    if (spec.worktree?.receipt.retained) {
      return reopenSubagentWorktree({ receipt: spec.worktree.receipt, cwd: spec.worktree.cwd })
    }
    if (spec.worktree)
      throw subagentError(
        'SUBAGENT_NOT_RESUMABLE',
        'the child checkout was already cleaned; dispatch a new isolated child',
      )
    if (!spec.cwd)
      throw new Error('Worktree isolation requires an explicit parent working directory')
    return createSubagentWorktree({
      cwd: spec.cwd,
      workspaceRoot: spec.workspaceRoot,
      sessionId: spec.sessionId,
      base: spec.worktreeBase,
    })
  }

  private async run(
    spec: SubagentChildSpec,
    input: SubagentRunInput,
  ): Promise<SubagentDispatchResult> {
    if (spec.external) {
      this.running.add(spec.sessionId)
      let status: SubagentDispatchResult['status'] = 'failed'
      try {
        if (spec.isolation)
          throw subagentError(
            'SUBAGENT_NOT_RESUMABLE',
            'External CLI worktree isolation is unavailable; use a native isolated child',
          )
        const result = await this.runExternal(spec, input)
        status = result.status
        return result
      } finally {
        try {
          this.deps.onRunFinished?.(spec.sessionId, { preserveParked: false })
          await this.deps.onLifecycle?.('stopped', {
            sessionId: spec.sessionId,
            parentSessionId: spec.parentSessionId,
            agentId: spec.agentId,
            status,
          })
        } finally {
          this.running.delete(spec.sessionId)
        }
      }
    }
    const { sessionId, category, parentExecutionPolicy, model } = spec
    const startedAt = new Date().toISOString()
    const hasApprovalTransport = input.detached === true || Boolean(input.onEvent)
    const subagentTools = new ToolRegistry()
    for (const name of spec.toolNames) {
      if (
        parentExecutionPolicy?.requireToolApproval &&
        !hasApprovalTransport &&
        !isPolicyReadOnlyTool(name)
      )
        continue
      const tool = this.deps.toolRegistry.get(name)
      if (tool) {
        const source = this.deps.toolRegistry.registrationSource(name)
        subagentTools.register(tool, source ? { source } : undefined)
      }
    }

    let cwd = spec.cwd
    let workspaceRoot = spec.workspaceRoot
    let worktree: SubagentWorktreeHandle | undefined
    let lifecycleStatus = 'failed'
    let preserveParkedApproval = false
    this.running.add(sessionId)
    try {
      if (spec.isolation === 'worktree') {
        worktree = await this.openWorktree(spec)
        spec.worktree = { receipt: worktree.receipt, cwd: worktree.cwd }
        this.deps.specStore?.save(spec)
        cwd = worktree.cwd
        workspaceRoot = worktree.receipt.path
        await this.deps.sessions.updateMeta?.(sessionId, { cwd, workspaceIsolation: 'strict' })
      }
      await this.deps.onLifecycle?.('started', {
        sessionId,
        parentSessionId: spec.parentSessionId,
        agentId: spec.agentId,
      })
      input.signal?.throwIfAborted()
      input.onSession?.(sessionId)
      const engine = this.deps.engineFactory({
        sessionId,
        onEvent: (event) =>
          input.onEvent?.({ type: 'subagent_progress', subagentId: sessionId, inner: event }),
        tools: subagentTools,
        maxIterations: spec.maxIterations,
        model,
        providerId: spec.providerId,
        reviewOutcomes: spec.reviewOutcomes ?? false,
        executionPolicy: parentExecutionPolicy,
      })

      const systemPrompt = [
        parentExecutionPolicy?.requireToolApproval && !hasApprovalTransport
          ? '[Delegated execution policy] This child run is read-only because the parent turn requires fresh human approval for every side effect. Return the requested change as guidance if a mutation is needed; do not claim it was applied.'
          : '',
        spec.systemPrompt,
        input.forked && input.previousMessages.length > 0
          ? "[Forked parent context] Background from the parent's conversation was snapshotted at dispatch, bounded to a budget (older turns and large tool outputs may be omitted). Your own earlier turns may follow it. Use inherited context as background only; your task is the final user message. Do not continue the parent's other work."
          : '',
        !input.isNew
          ? '[Follow-up] The parent agent sent another message to this subagent. Your earlier turns are above; answer the new message.'
          : '',
      ]
        .filter(Boolean)
        .join('\n\n')

      const context: AgentContext = {
        sessionId,
        provider: spec.providerId,
        model,
        cwd,
        workspaceRoot,
        workspaceIsolation: workspaceRoot ? 'strict' : undefined,
        toolAllowlist: subagentTools.list().map((tool) => tool.name),
        requireToolApproval: parentExecutionPolicy?.requireToolApproval,
        systemPrompt,
        ...(input.previousMessages.length > 0 ? { previousMessages: input.previousMessages } : {}),
        runContract: spec.runContract
          ? {
              ...structuredClone(spec.runContract),
              summary: input.prompt,
              acceptanceCriteria: [],
            }
          : undefined,
        primaryAgentId: spec.agentId,
        scopeTags: spec.scopeTags ? [...spec.scopeTags] : undefined,
      }

      let lastAssistantText = ''
      let messageCount = 0
      let observedIterations = 0
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
          timestamp: startedAt,
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
        if (isApprovalParkedError(err) && err.sessionId === sessionId && !input.signal?.aborted) {
          preserveParkedApproval = true
          status = 'truncated'
          stopReason = stopReasonApprovalTimeout({ requestId: err.requestId, tool: err.tool })
          lastAssistantText ||= stopReason.summary ?? err.message
        } else {
          status = 'failed'
          error = err instanceof Error ? err.message : String(err)
        }
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
      } else if (status !== 'failed' && !sawDone && !preserveParkedApproval) {
        status = 'failed'
        error = 'subagent stream ended without completion evidence'
      }

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
        duration_ms: Date.now() - Date.parse(startedAt),
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
      const retained = this.children.get(sessionId)
      if (retained) {
        retained.lastStatus = status
        retained.lastOutput = lastAssistantText
        retained.updatedAt = new Date().toISOString()
      }
      return {
        ...(worktree ? { worktree: worktree.receipt } : {}),
        output: lastAssistantText,
        sessionId,
        category: category?.id,
        ...(spec.budget ? { budget: spec.budget } : {}),
        iterations: observedIterations || messageCount,
        usage: totalUsage,
        truncated,
        status,
        error,
        ...(stopReason ? { stopReason } : {}),
        findings,
      }
    } finally {
      try {
        await worktree?.finish(lifecycleStatus !== 'completed')
        if (worktree) {
          const retained = this.children.get(sessionId)
          if (retained) retained.worktree = { receipt: worktree.receipt, cwd: worktree.cwd }
        }
        const retained = this.children.get(sessionId)
        if (retained) this.deps.specStore?.save(retained)
      } finally {
        try {
          this.deps.onRunFinished?.(sessionId, {
            preserveParked: preserveParkedApproval && !input.signal?.aborted,
          })
          if (lifecycleStatus === 'failed') {
            await this.deps.sessions.updateMeta?.(sessionId, { status: 'abandoned' })
          }
          await this.deps.onLifecycle?.('stopped', {
            sessionId,
            parentSessionId: spec.parentSessionId,
            agentId: spec.agentId,
            status: lifecycleStatus,
          })
        } finally {
          this.running.delete(sessionId)
        }
      }
    }
  }
}
