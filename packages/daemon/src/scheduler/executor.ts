import type { AutonomyLevel, IChannel, RunStopReason, TokenUsage } from '@sepilotd/core'
import {
  scheduledAgentSkillRefsFromMetadata,
  type ScheduledAgentSkillRef,
} from '@sepilotd/api-client'
import { createLogger } from '../logger.js'
import type { JobStore, ScheduledJob } from './job-store.js'
import type { JobExecutionResult, JobExecutor } from './engine.js'
import type { ChannelAgentExecutor } from '../channels/agent-executor.js'
import type { ChannelSessionResolver } from '../channels/session-resolver.js'
import { channelSafeAgentResponse } from '../channels/safe-response.js'
import type { ProviderRegistry } from '../providers/registry.js'
import type { SchedulerDeliveryOutbox } from './delivery-outbox.js'
import { isGeekNewsInsightsJob, runGeekNewsInsightsJob } from './geeknews.js'
import { schedulerSessionIdForJob, schedulerSessionIdForRun } from './session-id.js'
import { ScheduledDeliveryTracker } from './delivery-control.js'
import { scheduledAgentExecutionResult } from './agent-outcome.js'
import { safeSchedulerDeliveryError } from './delivery-error.js'
import { scheduledInstructionWithSources } from './app-sources.js'
import { scriptMonitorConfigFromMetadata, type ScriptMonitorRunResult } from './script-monitor.js'

const log = createLogger('scheduler:executor')

/**
 * `channelTarget` on a scheduled job is the chat session key (`<channelType>:<id>`,
 * e.g. `telegram:8435235236`) — that's what `schedule_create` records and what
 * `schedule_list` filters by. But `IChannel.sendMessage` wants the raw channel id
 * (`8435235236`); Telegram does `parseInt(target.id)`, so a prefixed key becomes
 * `NaN` → "chat not found" and the scheduled output never reaches the user. Strip
 * the channel-type prefix here, at the boundary where a session key becomes a
 * delivery target. Leaves anything that isn't `<channelType>:`-prefixed untouched.
 */
function channelIdFromSessionKey(channelType: string, channelTarget: string): string {
  const prefix = `${channelType}:`
  return channelTarget.startsWith(prefix) ? channelTarget.slice(prefix.length) : channelTarget
}

/**
 * Minimal channel lookup interface used by the scheduler executor.
 *
 * Returns the live IChannel instance so the executor can dispatch the
 * agent's response back to the originating chat (Telegram, Slack, …)
 * once the scheduled instruction has been processed.
 */
export interface ChannelLookup {
  get(channelType: string): IChannel | undefined
}

export interface SchedulerExecutorDeps {
  jobStore?: JobStore
  channelRegistry: ChannelLookup
  channelAgentExecutor: ChannelAgentExecutor
  sessionResolver: ChannelSessionResolver
  providerRegistry: Pick<ProviderRegistry, 'getDefault'>
  runHeadlessAgent: (
    instruction: string,
    opts: {
      sessionId: string
      /** Per-run durable evidence journal; distinct from stable approval authority. */
      agentSessionId?: string
      /** Link the persisted run only after the audit session exists, before agent execution. */
      onAgentSessionReady?: () => void
      parentSessionId: string | null
      channelContext?: { channel: string; chatKey: string; triggerMessageId?: string }
      skillRefs?: ScheduledAgentSkillRef[]
      signal?: AbortSignal
      suppressDelivery?: boolean
    },
  ) => Promise<string | void | JobExecutionResult | null>
  runInternalJob: (kind: string, job: ScheduledJob, context?: { runId?: string }) => Promise<void>
  /** Deterministic, non-LLM script monitor execution. */
  runScriptMonitor?: (
    job: ScheduledJob,
    context?: { runId?: string; signal?: AbortSignal; suppressDelivery?: boolean },
  ) => Promise<ScriptMonitorRunResult>
  onJobProgress?: (job: ScheduledJob, message: string) => Promise<void> | void
  deliveryOutbox?: SchedulerDeliveryOutbox
  autonomy: AutonomyLevel
  /**
   * Re-establish an unattended job's pre-approval before it runs. The grant
   * itself is in-memory, so without this a restart silently returns the job to
   * stalling on approval prompts nobody is there to answer.
   */
  grantUnattendedApproval?: (schedulerSessionId: string) => void
}

function normalizeExecutorResult(result: string | void | JobExecutionResult | null): {
  output: string | void
  usage: TokenUsage
  outcome: 'complete' | 'incomplete'
  failureReason?: string
} {
  if (typeof result === 'string' || result == null) {
    return {
      output: result ?? undefined,
      usage: { inputTokens: 0, outputTokens: 0 },
      outcome: 'complete',
    }
  }
  return {
    output: typeof result.output === 'string' ? result.output : undefined,
    usage: {
      inputTokens: Math.max(0, Math.floor(result.usage?.inputTokens ?? 0)),
      outputTokens: Math.max(0, Math.floor(result.usage?.outputTokens ?? 0)),
    },
    outcome: result.outcome === 'incomplete' ? 'incomplete' : 'complete',
    ...(result.outcome === 'incomplete' && result.failureReason
      ? { failureReason: result.failureReason }
      : {}),
  }
}

function hasUsage(usage: TokenUsage): boolean {
  return usage.inputTokens + usage.outputTokens > 0
}

function resultWithUsage(
  output: string | void,
  usage: TokenUsage,
  incomplete?: { failureReason?: string },
): string | void | JobExecutionResult {
  if (!hasUsage(usage) && !incomplete) return output
  return {
    ...(typeof output === 'string' ? { output } : {}),
    ...(hasUsage(usage) ? { usage } : {}),
    ...(incomplete
      ? {
          outcome: 'incomplete' as const,
          ...(incomplete.failureReason ? { failureReason: incomplete.failureReason } : {}),
        }
      : {}),
  }
}

function deliveryIdempotencyKey(job: ScheduledJob, runId: string | null): string {
  return [
    'scheduler-delivery',
    job.id,
    runId ?? String(job.nextRunAt),
    job.channelType ?? '',
    job.channelTarget ?? '',
    job.replyToMessageId ?? '',
  ].join(':')
}

function commitMetadata(
  deps: SchedulerExecutorDeps,
  job: ScheduledJob,
  metadata: Record<string, unknown> | null | undefined,
): void {
  if (metadata === undefined) return
  if (!deps.jobStore) throw new Error('scheduler job metadata commit requires a job store')
  deps.jobStore.updateMetadata(job.id, metadata)
}

async function deliverOrQueueChannelMessage(
  deps: SchedulerExecutorDeps,
  job: ScheduledJob,
  channel: IChannel | undefined,
  channelId: string,
  text: string,
  runId: string | null,
  metadata?: Record<string, unknown> | null,
): Promise<string> {
  try {
    if (!channel) throw new Error(`channel ${job.channelType ?? 'unknown'} unavailable`)
    await channel.sendMessage(
      { id: channelId, type: 'channel' },
      {
        text,
        format: 'markdown',
        ...(job.replyToMessageId ? { replyTo: job.replyToMessageId } : {}),
      },
    )
    commitMetadata(deps, job, metadata)
    log.info('scheduled job dispatched', {
      jobId: job.id,
      channelType: job.channelType,
      channelTarget: job.channelTarget,
      channelId,
      chars: text.length,
      replyTo: job.replyToMessageId ?? null,
    })
    return text
  } catch (err) {
    const message = safeSchedulerDeliveryError(err)
    if (deps.deliveryOutbox && job.channelType && job.channelTarget) {
      const queued = deps.deliveryOutbox.enqueue({
        jobId: job.id,
        runId,
        idempotencyKey: deliveryIdempotencyKey(job, runId),
        channelType: job.channelType,
        channelId,
        channelTarget: job.channelTarget,
        replyToMessageId: job.replyToMessageId,
        text,
        format: 'markdown',
        commitJobMetadata: metadata,
      })
      log.warn(
        `scheduled job ${job.id} dispatch to ${job.channelType} failed; queued delivery ${queued.id} for retry: ${message}`,
      )
      return `Delivery queued for retry after the scheduled agent produced output: ${message}\n\n${text}`
    }

    log.warn(
      `scheduled job ${job.id} dispatch to ${job.channelType} failed: ${message}`,
    )
    return `Delivery failed after the scheduled agent produced output: ${message}\n\n${text}`
  }
}

async function notifyJobProgress(
  deps: SchedulerExecutorDeps,
  job: ScheduledJob,
  message: string,
): Promise<void> {
  try {
    await deps.onJobProgress?.(job, message)
  } catch (err) {
    log.warn('scheduled job progress delivery failed', {
      jobId: job.id,
      error: err instanceof Error ? err.message : String(err),
    })
  }
}

export function createSchedulerExecutor(deps: SchedulerExecutorDeps): JobExecutor {
  return async (job: ScheduledJob, context) => {
    context?.signal?.throwIfAborted()
    const scriptMonitor = scriptMonitorConfigFromMetadata(job.metadata)
    if (scriptMonitor) {
      if (!deps.runScriptMonitor) throw new Error('script monitor runner is unavailable')
      const result = await deps.runScriptMonitor(job, {
        ...(context?.runId ? { runId: context.runId } : {}),
        ...(context?.signal ? { signal: context.signal } : {}),
        ...(context?.suppressDelivery ? { suppressDelivery: true } : {}),
      })
      return result.output
    }
    // Skill selection is a persisted execution contract, not a prompt hint.
    // Parse it before choosing the channel/headless path so malformed or
    // unsupported metadata can never degrade to an unskilled run.
    const skillRefs = job.instruction.startsWith('__internal:')
      ? []
      : scheduledAgentSkillRefsFromMetadata(job.metadata)
    const agentSessionId = context?.runId
      ? schedulerSessionIdForRun(context.runId)
      : undefined
    const linkRunAgentSession = (sessionId: string): void => {
      if (!context?.runId || !deps.jobStore) return
      if (!deps.jobStore.linkRunAgentSession(job.id, context.runId, sessionId)) {
        throw new Error(`SCHEDULER_AGENT_SESSION_LINK_FAILED: run ${context.runId} is unavailable or already linked`)
      }
    }
    const headlessOptions = {
      sessionId: schedulerSessionIdForJob(job.id),
      ...(agentSessionId
        ? {
            agentSessionId,
            onAgentSessionReady: () => linkRunAgentSession(agentSessionId),
          }
        : {}),
      parentSessionId: job.parentSessionId,
      ...(context?.signal ? { signal: context.signal } : {}),
      ...(context?.suppressDelivery ? { suppressDelivery: true } : {}),
      ...(skillRefs.length > 0 ? { skillRefs } : {}),
      ...(job.channelType && job.channelTarget
        ? {
            channelContext: {
              channel: job.channelType,
              chatKey: job.channelTarget,
              triggerMessageId: schedulerSessionIdForJob(job.id),
            },
          }
        : {}),
    }
    const runHeadlessAgent = async (instruction: string) => {
      return deps.runHeadlessAgent(instruction, headlessOptions)
    }
    // The `unattended` flag is durable; the approval grant it represents is
    // not. Re-establish it on every fire so a daemon restart does not quietly
    // return the job to waiting on a prompt nobody is present to answer.
    if (job.unattended) {
      deps.grantUnattendedApproval?.(schedulerSessionIdForJob(job.id))
    }
    // Path 0: built-in app-owned jobs. GeekNews insights are deterministic
    // at the collection layer (RSS + dedupe), then delegate synthesis to the
    // normal agent so the delivered message still feels like a chat turn.
    if (isGeekNewsInsightsJob(job)) {
      if (!deps.jobStore) throw new Error('GeekNews scheduler job requires a job store')
      const totalUsage: TokenUsage = { inputTokens: 0, outputTokens: 0 }
      let incomplete: { failureReason?: string } | undefined
      const result = await runGeekNewsInsightsJob(job, {
        signal: context?.signal,
        runAgent: async (instruction) => {
          const result = normalizeExecutorResult(await runHeadlessAgent(instruction))
          totalUsage.inputTokens += result.usage.inputTokens
          totalUsage.outputTokens += result.usage.outputTokens
          if (result.outcome === 'incomplete') {
            incomplete = result.failureReason
              ? { failureReason: result.failureReason }
              : {}
          }
          return result.output
        },
        ...(context?.suppressDelivery
          ? {}
          : { onStatus: (message: string) => notifyJobProgress(deps, job, message) }),
      })
      context?.signal?.throwIfAborted()
      if (!result) return resultWithUsage(undefined, totalUsage, incomplete)
      if (incomplete) {
        const output = result.output?.trim()
        if (!output) return resultWithUsage(undefined, totalUsage, incomplete)
        if (job.channelType && job.channelTarget && !context?.suppressDelivery) {
          const channel = deps.channelRegistry.get(job.channelType)
          const channelId = channelIdFromSessionKey(job.channelType, job.channelTarget)
          const safeResponse = channelSafeAgentResponse(output)
          const delivered = await deliverOrQueueChannelMessage(
            deps,
            job,
            channel,
            channelId,
            safeResponse.text,
            context?.runId ?? null,
          )
          return resultWithUsage(delivered, totalUsage, incomplete)
        }
        return resultWithUsage(output, totalUsage, incomplete)
      }
      if (!result.hasNewItems) {
        commitMetadata(deps, job, result.metadata)
        return resultWithUsage(undefined, totalUsage)
      }

      const output = result.output?.trim()
      if (!output) return resultWithUsage(undefined, totalUsage)

      if (job.channelType && job.channelTarget && !context?.suppressDelivery) {
        const channel = deps.channelRegistry.get(job.channelType)
        const channelId = channelIdFromSessionKey(job.channelType, job.channelTarget)
        const delivered = await deliverOrQueueChannelMessage(
          deps,
          job,
          channel,
          channelId,
          output,
          context?.runId ?? null,
          result.metadata,
        )
        return resultWithUsage(delivered, totalUsage, incomplete)
      }

      if (!context?.suppressDelivery) commitMetadata(deps, job, result.metadata)
      return resultWithUsage(output, totalUsage, incomplete)
    }

    // Path 1: internal job (prefixed with __internal:)
    if (job.instruction.startsWith('__internal:')) {
      const kind = job.instruction.slice('__internal:'.length)
      await deps.runInternalJob(kind, job, context?.runId ? { runId: context.runId } : undefined)
      context?.signal?.throwIfAborted()
      return
    }

    // Parse the reserved source contract before either agent path or delivery.
    // Malformed links must not silently run using stale instruction prose.
    const instruction = scheduledInstructionWithSources(job)

    // Path 2: channel-bound job — route via ChannelAgentExecutor
    if (job.channelType && job.channelTarget) {
      const channel = deps.channelRegistry.get(job.channelType)
      if (!channel) {
        log.warn(`channel ${job.channelType} unavailable for job ${job.id}; running headless and relying on surface notifications`)
        return runHeadlessAgent(instruction)
      }

      const provider = deps.providerRegistry.getDefault()
      if (!provider) throw new Error('no default provider configured')

      const channelId = channelIdFromSessionKey(job.channelType, job.channelTarget)

      const synthesizedNormalized = {
        message: {
          channelType: job.channelType,
          channelId,
          messageId: `scheduler-${job.id}`,
          text: instruction,
          timestamp: new Date().toISOString(),
          sender: { id: 'scheduler', name: 'Scheduler' },
        },
        sessionKey: job.channelTarget,
      } as never

      const { sessionId, previousMessages } = await deps.sessionResolver.resolveFork(
        synthesizedNormalized,
        provider,
        job.parentSessionId ?? undefined,
      )
      linkRunAgentSession(sessionId)

      const deliveryTracker = new ScheduledDeliveryTracker()
      // The agent states its own termination cause on the done event. Capturing
      // it here is what lets run history tell a deliberate, resumable budget
      // stop apart from a hard failure.
      let observedStopReason: RunStopReason | undefined
      context?.signal?.throwIfAborted()
      const responseText = await deps.channelAgentExecutor.run(
        { text: instruction, messageId: `scheduler-${job.id}` },
        provider,
        deps.autonomy,
        sessionId,
        previousMessages,
        undefined,
        undefined,
        undefined,
        context?.signal,
        undefined,
        { channel: job.channelType, chatKey: job.channelTarget, triggerMessageId: schedulerSessionIdForJob(job.id) },
        undefined,
        (event) => {
          if (event.type === 'done' && event.stopReason) observedStopReason = event.stopReason
          return deliveryTracker.consume(event)
        },
        skillRefs.length > 0 ? { skillRefs } : undefined,
      )

      if (deliveryTracker.shouldSuppress()) {
        log.info(`scheduled job ${job.id} suppressed delivery from structured tool evidence`)
        return
      }

      // Without this dispatch the agent runs silently and the user never
      // sees the scheduled output — schedule_create's contract ("the
      // result is sent back to the same chat automatically") relied on
      // this final hop, which had been left as a TODO.
      const safeResponse = channelSafeAgentResponse(responseText ?? '')
      context?.signal?.throwIfAborted()
      const trimmed = safeResponse.text.trim()
      if (trimmed.length === 0) {
        log.warn(`scheduled job ${job.id} produced no response text; nothing to dispatch`)
        return
      }

      if (context?.suppressDelivery) {
        log.info(`scheduled job ${job.id} completed with scheduler-owned delivery suppressed`)
        return safeResponse.internalFallback
          ? scheduledAgentExecutionResult({
              rawOutput: responseText ?? '',
              output: trimmed,
              forceIncomplete: true,
              stopReason: observedStopReason,
            })
          : scheduledAgentExecutionResult({
              rawOutput: responseText ?? '',
              output: trimmed,
              stopReason: observedStopReason,
            })
      }

      const delivered = await deliverOrQueueChannelMessage(
        deps,
        job,
        channel,
        channelId,
        trimmed,
        context?.runId ?? null,
      )
      return safeResponse.internalFallback
        ? scheduledAgentExecutionResult({
            rawOutput: responseText ?? '',
            output: delivered,
            forceIncomplete: true,
            stopReason: observedStopReason,
          })
        : scheduledAgentExecutionResult({
            rawOutput: responseText ?? '',
            output: delivered,
            stopReason: observedStopReason,
          })
    }

    // Path 3: headless — no channel binding
    log.info(`headless execution for job ${job.id} (no channel binding)`)
    if (!context?.suppressDelivery) {
      await notifyJobProgress(
        deps,
        job,
        `[예약 작업 진행] ${job.name}\n스케줄러가 백그라운드 실행을 시작했습니다.`,
      )
    }
    return runHeadlessAgent(instruction)
  }
}
