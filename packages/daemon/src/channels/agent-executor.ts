import { randomUUID } from 'node:crypto'
import type { AgentEvent, TokenUsage } from '@sepilotd/core'
import type { ScheduledAgentSkillRef } from '@sepilotd/api-client'
import { AgentModeRouter } from '../agent/mode-router.js'
import { resolveInstantModeToolNames } from '../agent/instant-mode-tool-intent.js'
import { withAuthorizedToolExposure, withContextualToolExposure } from '../tools/role-filter.js'
import {
  createJournalStateBoard,
  createJournalSteeringConsumed,
} from '../server/runtime/mode-router-options.js'
import { resolvePersona } from '../agent/custom/persona-resolver.js'
import { createAgentOutputTracker } from '../agent/event-output.js'
import { resolveAgentMaxIterations } from '../agent/iteration-budget.js'
import { buildRetrievalQuery } from '../agent/query-context.js'
import { buildSystemPrompt } from '../agent/system-prompt.js'
import {
  isAutoRetrieveDisabledByEnv,
  retrieveRelevantMemory,
} from '../memory/auto-retrieve.js'
import { createLogger } from '../logger.js'
import { resolveSkillRefsContent } from '../server/routes/chat-skills.js'
import { resolveSkillExecutionContext } from '../skills/execution-policy.js'

const log = createLogger('channel-agent-executor')

function truncate(value: string, max: number): string {
  const trimmed = value.replace(/\s+/g, ' ').trim()
  if (trimmed.length <= max) return trimmed
  return `${trimmed.slice(0, max - 1).trimEnd()}…`
}

function stringConfigValue(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim().length > 0
    ? value.trim()
    : undefined
}
import { persistAgentSessionEvent } from '../server/session-events.js'
import type { ChannelPipelineCapabilities } from '../server/runtime/capabilities.js'
import { resolveAgentInactivityMs } from '../server/sse-response.js'
import type { PendingQuestion } from '../tools/question.js'
import type { ScheduledJob } from '../scheduler/job-store.js'
import type {
  ChannelSessionContextMessages,
  DefaultProvider,
} from './session-resolver.js'

/**
 * Render the chat's pending scheduled tasks as a context block for the agent.
 * This is the "assistant glances at their own notes before writing a new one"
 * mechanism — the agent reconciles against this list rather than blindly
 * re-creating a job the user already asked for. Pure function; the store query
 * lives in ChannelAgentExecutor.collectPendingScheduledTasks.
 */
export function formatPendingScheduledTasksBlock(
  jobs: ScheduledJob[],
  now: number,
): string | undefined {
  const pending = jobs
    .filter((job) => job.status === 'pending' && job.enabled)
    .sort((a, b) => a.nextRunAt - b.nextRunAt)
    .slice(0, 8)
  if (pending.length === 0) return undefined
  const lines = ['Your scheduled tasks for this chat (the runtime re-invokes you at the listed time):']
  for (const job of pending) {
    const minutesAway = Math.max(0, Math.round((job.nextRunAt - now) / 60_000))
    const whenLabel = job.kind === 'recurring' && job.cron
      ? `recurring (cron \`${job.cron}\`)`
      : minutesAway < 60
        ? `in ~${minutesAway}m`
        : minutesAway < 24 * 60
          ? `in ~${Math.round(minutesAway / 60)}h`
          : new Date(job.nextRunAt).toISOString()
    lines.push(`- sched:${job.id} ${whenLabel} — ${truncate(job.instruction, 160)}`)
  }
  lines.push(
    'Before calling schedule_create, reconcile against this list: if the user is re-asking for something already scheduled here, just confirm the existing task — do NOT create a duplicate. If they want it changed, use schedule_update so the job identity and run history survive; cancel and recreate only when changing between one-shot and recurring. Only create a fresh task when it is genuinely new.',
  )
  return lines.join('\n')
}

export type ChannelApprovalRequestEvent = Extract<
  AgentEvent,
  { type: 'approval_request' }
>

export type ChannelAgentUsageObserver = (usage: TokenUsage) => void

interface ChannelRelevantMemory {
  relevantMemories: string[]
}

/**
 * Diagnostic context attached to a {@link ChannelAgentInactivityError} so the
 * channel reply can tell the user *what* stalled rather than guessing. The
 * common case is `sawModelOutput === false`: the run produced no tokens and no
 * tool activity at all, which almost always means the provider/model call
 * itself never came back (overloaded endpoint, rate limit, dead model) — quite
 * different from a tool that hung mid-run.
 */
export interface ChannelAgentInactivityContext {
  readonly provider?: string
  readonly model?: string
  /** True once the run emitted any model-produced output (text/tool/message/error). */
  readonly sawModelOutput?: boolean
  /** Type of the last agent event seen before the run went silent. */
  readonly lastEventType?: string
}

export class ChannelAgentInactivityError extends Error {
  readonly code = 'AGENT_INACTIVITY'
  readonly context: ChannelAgentInactivityContext

  constructor(readonly inactivityMs: number, context: ChannelAgentInactivityContext = {}) {
    super(`Channel agent produced no output for ${inactivityMs}ms`)
    this.name = 'ChannelAgentInactivityError'
    this.context = context
  }
}

export function isChannelAgentInactivityError(
  error: unknown,
): error is ChannelAgentInactivityError {
  return error instanceof ChannelAgentInactivityError
    || (
      typeof error === 'object'
      && error !== null
      && 'code' in error
      && (error as { code?: unknown }).code === 'AGENT_INACTIVITY'
    )
}

export class ChannelAgentAbortedError extends Error {
  readonly code = 'AGENT_ABORTED'
  readonly reason: unknown

  constructor(reason: unknown) {
    super('Channel agent run was aborted')
    this.name = 'ChannelAgentAbortedError'
    this.reason = reason
  }
}

export function isChannelAgentAbortedError(
  error: unknown,
): error is ChannelAgentAbortedError {
  return error instanceof ChannelAgentAbortedError
    || (
      typeof error === 'object'
      && error !== null
      && 'code' in error
      && (error as { code?: unknown }).code === 'AGENT_ABORTED'
    )
}

function createChannelAbortError(signal: AbortSignal): ChannelAgentAbortedError {
  return new ChannelAgentAbortedError(signal.reason)
}

class ChannelAgentInactivityWatchdog {
  private readonly timeoutMs: number
  private readonly timer: ReturnType<typeof setInterval>
  private lastActivityAt = Date.now()
  private suspended = 0
  private tripped = false
  private reject!: (error: ChannelAgentInactivityError) => void

  readonly promise: Promise<never>

  constructor(
    timeoutMs: number,
    private readonly onTrip: () => void | Promise<void>,
  ) {
    this.timeoutMs = timeoutMs
    this.promise = new Promise<never>((_, reject) => {
      this.reject = reject
    })

    const tickMs = Math.max(100, Math.min(10_000, Math.floor(timeoutMs / 6)))
    this.timer = setInterval(() => {
      if (this.tripped || this.suspended > 0) return
      if (Date.now() - this.lastActivityAt <= this.timeoutMs) return
      this.tripped = true
      try {
        void this.onTrip()
      } catch {
        // The rejection below is the authoritative failure signal.
      }
      this.reject(new ChannelAgentInactivityError(this.timeoutMs))
    }, tickMs)
  }

  recordActivity(): void {
    this.lastActivityAt = Date.now()
  }

  suspend(): () => void {
    this.suspended += 1
    let resumed = false
    return () => {
      if (resumed) return
      resumed = true
      this.suspended = Math.max(0, this.suspended - 1)
      this.recordActivity()
    }
  }

  dispose(): void {
    clearInterval(this.timer)
  }
}

export class ChannelAgentExecutor {
  constructor(private readonly runtime: ChannelPipelineCapabilities) {}

  private resolveConfiguredPersonaId(
    channelContext?: { channel: string; chatKey: string; triggerMessageId?: string },
  ): string {
    const channelPersona = channelContext?.channel
      ? (this.runtime.config.channels ?? []).find((channel) => channel.type === channelContext.channel)
        ?.config?.persona
      : undefined
    return stringConfigValue(channelPersona)
      ?? stringConfigValue(this.runtime.config.agent?.persona)
      ?? 'default'
  }

  private async collectUpcomingReminders(
    scopeTags?: string[],
  ): Promise<string | undefined> {
    const store = this.runtime.remindersStore
    if (!store) return undefined
    try {
      const list = scopeTags && scopeTags.length > 0
        ? await store.listForScope(scopeTags)
        : await store.list()
      const horizonMs = Date.now() + 24 * 60 * 60 * 1000
      const due = list
        .filter((reminder) => !reminder.firedAt && !reminder.cancelledAt)
        .filter((reminder) => Date.parse(reminder.dueAt) <= horizonMs)
        .sort((a, b) => Date.parse(a.dueAt) - Date.parse(b.dueAt))
        .slice(0, 5)
      if (due.length === 0) return undefined

      const lines = ['Upcoming reminders for this user (next 24h):']
      for (const reminder of due) {
        const when = new Date(reminder.dueAt)
        const minutesAway = Math.max(0, Math.round((when.getTime() - Date.now()) / 60_000))
        const whenLabel = minutesAway < 60
          ? `in ~${minutesAway}m`
          : minutesAway < 24 * 60
            ? `in ~${Math.round(minutesAway / 60)}h`
            : when.toISOString()
        lines.push(`- mem:${reminder.id} ${whenLabel} — ${truncate(reminder.content, 160)}`)
      }
      lines.push('Mention these naturally when relevant; the user can cancel one with "잊어 mem:<id>" / "cancel mem:<id>".')
      return lines.join('\n')
    } catch (error) {
      log.warn('Failed to load upcoming reminders', {
        error: error instanceof Error ? error.message : String(error),
      })
      return undefined
    }
  }

  private async collectPendingScheduledTasks(
    channelContext?: { channel: string; chatKey: string; triggerMessageId?: string },
  ): Promise<string | undefined> {
    const store = this.runtime.jobStore
    if (!store || !channelContext?.chatKey) return undefined
    try {
      const jobs = store.list({ status: ['pending'], channelTarget: channelContext.chatKey })
      return formatPendingScheduledTasksBlock(jobs, Date.now())
    } catch (error) {
      log.warn('Failed to load pending scheduled tasks', {
        error: error instanceof Error ? error.message : String(error),
      })
      return undefined
    }
  }

  private async collectRelevantMemoryContext(
    query: string,
    scopeTags?: string[],
  ): Promise<ChannelRelevantMemory> {
    if (!this.runtime.semanticIndex) return { relevantMemories: [] }
    try {
      const result = await retrieveRelevantMemory({
        query,
        semanticIndex: this.runtime.semanticIndex as Parameters<typeof retrieveRelevantMemory>[0]['semanticIndex'],
        disabled: isAutoRetrieveDisabledByEnv(),
        scopeTags,
      })
      if (result.block.length > 0) {
        log.debug('auto-retrieve hit', {
          hits: result.hits.length,
          chars: result.block.length,
        })
      } else if (result.reason && result.reason !== 'no-hits') {
        log.debug('auto-retrieve skipped', { reason: result.reason })
      }
      return {
        relevantMemories: result.block ? [result.block] : [],
      }
    } catch (error) {
      log.warn('auto-retrieve failed', {
        error: error instanceof Error ? error.message : String(error),
      })
      return { relevantMemories: [] }
    }
  }

  async run(
    message: { text: string; messageId?: string; sender?: { id?: string; name?: string } },
    provider: DefaultProvider,
    autonomy: ChannelPipelineCapabilities['autonomy'],
    sessionId: string,
    previousMessages: ChannelSessionContextMessages,
    cwd?: string,
    onApprovalRequest?: (event: ChannelApprovalRequestEvent) => Promise<void> | void,
    onQuestionRequest?: (question: PendingQuestion) => Promise<void> | void,
    signal?: AbortSignal,
    scopeTags?: string[],
    channelContext?: { channel: string; chatKey: string; triggerMessageId?: string },
    onUsage?: ChannelAgentUsageObserver,
    onEvent?: (event: AgentEvent) => Promise<void> | void,
    executionOptions?: { skillRefs?: ScheduledAgentSkillRef[] },
  ): Promise<string> {
    const modeRouterRef: { current?: AgentModeRouter } = {}
    // A channel session's selected cwd is also its filesystem boundary. Direct
    // API clients already send both values; keeping them identical here makes
    // Mattermost/Telegram development turns honor the same strict workspace
    // policy instead of treating cwd as a display-only working directory.
    const workspaceRoot = cwd

    const requestQuestion = async (input: {
      sessionId: string
      prompt: string
      choices?: string[]
    }): Promise<string> => {
      let resolveAnswer!: (answer: string) => void
      const answerPromise = new Promise<string>((resolve) => {
        resolveAnswer = resolve
      })
      const resume = watchdog.suspend()
      const question: PendingQuestion = {
        id: randomUUID(),
        sessionId: input.sessionId,
        prompt: input.prompt,
        choices: input.choices,
        resolve: resolveAnswer,
      }
      this.runtime.questions.enqueue(question)
      try {
        await onQuestionRequest?.(question)
      } catch {
        // The pending question can still be answered from the web UI.
      }
      try {
        return await answerPromise
      } finally {
        resume()
      }
    }
    const memoryQuery = buildRetrievalQuery(message.text, previousMessages)
    const relevantMemoryContext = await this.collectRelevantMemoryContext(memoryQuery, scopeTags)
    const upcomingReminders = await this.collectUpcomingReminders(scopeTags)
    const pendingScheduledTasks = await this.collectPendingScheduledTasks(channelContext)
    const personaId = this.resolveConfiguredPersonaId(channelContext)
    const customAgents = (
      await this.runtime.customDefs?.agentsForCwd(cwd, workspaceRoot).catch(() => [])
    ) ?? []
    const persona = resolvePersona(personaId, customAgents)
      ?? resolvePersona('default', customAgents)
    const combinedMemoryContext = [upcomingReminders, pendingScheduledTasks]
      .filter((block): block is string => Boolean(block && block.length > 0))
      .join('\n\n') || undefined
    const scopedFileMemory = this.runtime.fileMemoryRegistry?.get(scopeTags)
      ?? this.runtime.fileMemory
    const senderContext = message.sender && (message.sender.name || message.sender.id)
      ? `Active user on this channel: ${message.sender.name ?? message.sender.id}${
          message.sender.name && message.sender.id ? ` (id: ${message.sender.id})` : ''
        }${
          channelContext?.channel ? ` via ${channelContext.channel}` : ''
        }. Address them by name when natural; do not reveal their id unless they ask.`
      : undefined
    const declaredSkillToolNames = new Set<string>()
    const loadedExecutionSkillIds = new Set<string>()
    const skillPrefix = executionOptions?.skillRefs?.length
      ? await resolveSkillRefsContent(
          executionOptions.skillRefs,
          this.runtime.skillRegistry,
          this.runtime.toolRegistry,
          cwd,
          autonomy,
          declaredSkillToolNames,
          workspaceRoot,
          loadedExecutionSkillIds,
        )
      : ''
    const defaultMode = this.runtime.config.agent.mode
    const initialMode = !defaultMode || defaultMode === 'auto' ? 'instant' : defaultMode
    const authorizedTools = withAuthorizedToolExposure(this.runtime.toolRegistry, {
      declaredSkillToolNames,
      personaAllowedTools: persona?.allowedTools,
      personaDeniedTools: persona?.deniedTools,
    })
    const visibleTools = withContextualToolExposure(authorizedTools, {
      semanticRouting: true,
      routedMode: initialMode,
      explicitToolNames: resolveInstantModeToolNames(message.text, initialMode, undefined),
    })
    const baseSystemPrompt = await buildSystemPrompt({
      config: this.runtime.config,
      tools: visibleTools,
      profile: initialMode === 'instant' ? 'instant' : 'agent',
      includeDailyNotes: initialMode !== 'instant',
      skills: this.runtime.skillRegistry,
      fileMemory: scopedFileMemory,
      sessionId,
      cwd,
      workspaceRoot,
      relevantMemoryContext: combinedMemoryContext,
      senderContext,
      customInstructions: persona?.systemPromptAddition,
    })
    const systemPrompt = skillPrefix + baseSystemPrompt
    const inactivityMs = resolveAgentInactivityMs()
    const watchdog = new ChannelAgentInactivityWatchdog(inactivityMs, () => {
      void modeRouterRef.current?.stop().catch(() => {})
    })
    const modeRouter = new AgentModeRouter({
      provider,
      tools: visibleTools,
      authorizedTools,
      policy: this.runtime.policyEngine,
      autonomy,
      semanticIndex: this.runtime.semanticIndex,
      auditLogger: this.runtime.auditLogger,
      usageTracker: this.runtime.usageTracker,
      spendBudget: this.runtime.config.limits,
      hookRegistry: this.runtime.hookRegistry,
      systemPrompt,
      previousMessages,
      maxIterations: resolveAgentMaxIterations(),
      deviceName: this.runtime.config.device.name,
      providerCircuitBreaker: this.runtime.providerCircuitBreaker,
      defaultMode: defaultMode ?? 'instant',
      graphRegistry: this.runtime.graphRegistry,
      requestQuestion,
      reviewToollessFinals: true,
      strictFinalAnswerProtocol:
        process.env.SEPILOTD_STRICT_ANSWER_PROTOCOL === '1' || skillPrefix.trim().length > 0,
      journalStateBoard: createJournalStateBoard(this.runtime.sessions),
      journalSteeringConsumed: createJournalSteeringConsumed(this.runtime.sessions, this.runtime.sessionWatchBroker),
      evaluateAutoApproval: this.runtime.approvalRegistry
        ? (toolCall) =>
            this.runtime.approvalRegistry?.tryAutoApproval({
              sessionId,
              toolCall,
              runId: message.messageId,
            }) ?? null
        : undefined,
      approvalCallback: this.runtime.approvalRegistry
        ? (toolCall, requestId) => {
            const resume = watchdog.suspend()
            let result: ReturnType<ChannelPipelineCapabilities['approvalRegistry']['waitForApproval']>
            try {
              result = this.runtime.approvalRegistry.waitForApproval({
                sessionId,
                toolCall,
                requestId,
                runId: message.messageId,
              })
            } catch (error) {
              resume()
              throw error
            }
            if (result instanceof Promise) {
              return result.finally(resume)
            }
            resume()
            return result
          }
        : undefined,
    })
    modeRouterRef.current = modeRouter

    const abortPromise = signal
      ? new Promise<never>((_, reject) => {
          if (signal.aborted) {
            reject(createChannelAbortError(signal))
            return
          }
          signal.addEventListener(
            'abort',
            () => {
              void modeRouter.stop().catch(() => undefined)
              reject(createChannelAbortError(signal))
            },
            { once: true },
          )
        })
      : undefined

    const outputTracker = createAgentOutputTracker()
    let errorText = ''
    const providerId = provider.id
    const modelId = this.runtime.config.agent?.defaultModel ?? provider.models[0]?.id ?? 'default'
    // Track whether the run ever produced *model* output (tokens, a tool call,
    // an explicit message, or an error). If it did not, an inactivity trip means
    // the provider/model call itself never came back — surface that distinctly.
    let sawModelOutput = false
    let lastEventType: string | undefined
    const iterator = modeRouter.run(message.text, {
      sessionId,
      provider: providerId,
      model: modelId,
      cwd,
      workspaceRoot,
      systemPrompt,
      memoryQuery,
      relevantMemories: relevantMemoryContext.relevantMemories,
      scopeTags,
      channelContext,
      toolAllowlist: authorizedTools.list().map((tool) => tool.name),
      ...resolveSkillExecutionContext(loadedExecutionSkillIds, declaredSkillToolNames),
    })[Symbol.asyncIterator]()
    try {
      while (true) {
        const next = abortPromise
          ? await Promise.race([iterator.next(), watchdog.promise, abortPromise])
          : await Promise.race([iterator.next(), watchdog.promise])
        if (next.done) break

        const event = next.value
        watchdog.recordActivity()
        try {
          await onEvent?.(event)
        } catch (error) {
          log.warn('Channel agent event observer failed', {
            error: error instanceof Error ? error.message : String(error),
          })
        }
        lastEventType = event.type
        if (
          event.type === 'text_delta'
          || event.type === 'message'
          || event.type === 'tool_call'
          || event.type === 'tool_result'
          || event.type === 'approval_request'
          || event.type === 'error'
        ) {
          sawModelOutput = true
        }
        outputTracker.consume(event)
        if (event.type === 'done') {
          try {
            onUsage?.(event.usage)
          } catch (error) {
            log.warn('Channel usage observer failed', {
              error: error instanceof Error ? error.message : String(error),
            })
          }
        }
        // Channel bindings are long-lived conversations, so keep the session
        // active while still journaling tool, approval, memory, and cowork events.
        if (event.type !== 'done') {
          await persistAgentSessionEvent(this.runtime.sessions, sessionId, event)
        }
        if (event.type === 'approval_request') {
          await onApprovalRequest?.(event)
        }
        if (event.type === 'error') errorText = `Error: ${event.error.message}`
      }
    } catch (error) {
      if (isChannelAgentInactivityError(error)) {
        throw new ChannelAgentInactivityError(inactivityMs, {
          provider: providerId,
          model: modelId,
          sawModelOutput,
          lastEventType,
        })
      }
      throw error
    } finally {
      watchdog.dispose()
      void iterator.return?.().catch(() => undefined)
    }
    return errorText || outputTracker.finalContent()
  }
}
