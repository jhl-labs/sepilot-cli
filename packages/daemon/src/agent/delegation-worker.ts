import { randomUUID } from 'node:crypto'
import type { TokenUsage, ToolCall } from '@sepilotd/core'
import type { DelegationWorkerCapabilities } from '../server/runtime/capabilities.js'
import type { DelegationTask } from './delegator.js'
import { AgentModeRouter } from './mode-router.js'
import {
  createJournalStateBoard,
  createJournalSteeringConsumed,
} from '../server/runtime/mode-router-options.js'
import { createAgentOutputTracker } from './event-output.js'
import type { AgentState } from './graph/types.js'
import { emptyEvidenceLedger, updateEvidenceLedgerFromToolResult } from './graph/evidence-ledger.js'
import { extractSubagentFindings, type SubagentFindings } from './subagent-findings.js'
import { triggerDreamingSessionEnd } from '../memory/dreaming.js'
import { buildSystemPrompt } from './system-prompt.js'
import { persistAgentSessionEvent } from '../server/session-events.js'
import { createLogger } from '../logger.js'
import type { QuestionRequestInput } from '../tools/question.js'
import { TOOL_RESULT_EXECUTION_OBSERVED_METADATA_KEY } from './policy-failure.js'

const log = createLogger('delegation-worker')

export interface DelegationSessionLeaseState {
  sessionId: string
  delegationId: string
  targetDevice: string
  claimId: string
  generation?: number
  claimHealth: 'healthy' | 'degraded' | 'lost'
  startedAt: string
  updatedAt: string
  degradedSince?: string
  lastHeartbeatAt?: string
  lastError?: string
  leaseLossSource?: 'gateway' | 'comments' | 'transport'
}

function truncateForComment(text: string, limit = 2_000): string {
  const trimmed = text.trim()
  if (trimmed.length <= limit) {
    return trimmed
  }
  return `${trimmed.slice(0, limit - 1)}…`
}

/**
 * Serialize a delegate's structured findings as a fenced-JSON protocol block so
 * the delegator can roll them into the parent board (structured, not free text).
 * PLAN_065 T5.
 */
export function formatDelegationFindingsBlock(findings: SubagentFindings): string {
  return ['```findings', JSON.stringify(findings), '```'].join('\n')
}

export function formatDelegationCompletionComment(input: {
  workerDevice: string
  sessionId: string
  summary: string
  findings?: SubagentFindings
}): string {
  const lines = [
    `Completed by ${input.workerDevice} (session ${input.sessionId})`,
    'Result summary:',
    truncateForComment(input.summary),
    'Artifact handles:',
    `- session:${input.sessionId}`,
  ]
  if (input.findings) {
    lines.push(formatDelegationFindingsBlock(input.findings))
  }
  return lines.join('\n')
}

export class DelegationWorker {
  private pollTimer: ReturnType<typeof setInterval> | null = null
  private stopped = false
  private readonly inFlight = new Set<string>()
  private readonly sessionLeaseStates = new Map<string, DelegationSessionLeaseState>()
  private readonly pollIntervalMs: number
  private readonly pollBackoffBaseMs: number
  private readonly pollBackoffMaxMs: number
  private readonly claimTtlMs: number
  private readonly claimHeartbeatMs: number
  private readonly claimHeartbeatFailureThreshold: number
  private readonly claimHeartbeatDegradedMaxMs: number
  private consecutivePollFailures = 0
  private lastPollFailureError: string | null = null
  private nextPollAttemptAt = 0

  constructor(
    private readonly runtime: DelegationWorkerCapabilities,
    options?: {
      pollIntervalMs?: number
      pollBackoffBaseMs?: number
      pollBackoffMaxMs?: number
      claimTtlMs?: number
      claimHeartbeatMs?: number
      claimHeartbeatFailureThreshold?: number
      claimHeartbeatDegradedMaxMs?: number
    },
  ) {
    this.pollIntervalMs = options?.pollIntervalMs ?? 15_000
    this.pollBackoffBaseMs = Math.max(
      this.pollIntervalMs,
      options?.pollBackoffBaseMs ?? Math.max(60_000, this.pollIntervalMs),
    )
    this.pollBackoffMaxMs = Math.max(
      this.pollBackoffBaseMs,
      options?.pollBackoffMaxMs ?? Math.max(5 * 60_000, this.pollBackoffBaseMs),
    )
    this.claimTtlMs = Math.max(1_000, options?.claimTtlMs ?? 120_000)
    const maxHeartbeatMs = Math.max(20, this.claimTtlMs - 1)
    this.claimHeartbeatMs = options?.claimHeartbeatMs != null
      ? Math.min(maxHeartbeatMs, Math.max(20, options.claimHeartbeatMs))
      : Math.min(maxHeartbeatMs, Math.max(5_000, Math.floor(this.claimTtlMs / 3)))
    this.claimHeartbeatFailureThreshold = Math.max(
      1,
      options?.claimHeartbeatFailureThreshold ?? 3,
    )
    const preTtlDegradedMaxMs = Math.max(
      this.claimHeartbeatMs,
      this.claimTtlMs - 2 * this.claimHeartbeatMs,
    )
    const configuredDegradedMaxMs =
      options?.claimHeartbeatDegradedMaxMs ?? preTtlDegradedMaxMs
    this.claimHeartbeatDegradedMaxMs = Math.min(
      Math.max(1, this.claimTtlMs - 1),
      preTtlDegradedMaxMs,
      Math.max(this.claimHeartbeatMs, configuredDegradedMaxMs),
    )
  }

  async start(): Promise<void> {
    this.stopped = false
    await this.runScheduledPoll('startup')
    if (this.stopped) return
    this.pollTimer = setInterval(() => {
      void this.runScheduledPoll('interval')
    }, this.pollIntervalMs)
    this.pollTimer.unref?.()
  }

  async stop(): Promise<void> {
    this.stopped = true
    if (this.pollTimer) {
      clearInterval(this.pollTimer)
      this.pollTimer = null
    }
  }

  async poll(): Promise<void> {
    const tasks = await this.runtime.delegator.listPendingDelegationsForDevice([
      this.runtime.config.device.id,
      this.runtime.config.device.name,
    ])

    for (const task of tasks) {
      if (this.inFlight.has(task.delegationId)) {
        continue
      }

      this.inFlight.add(task.delegationId)
      void this.process(task).finally(() => {
        this.inFlight.delete(task.delegationId)
      })
    }
  }

  private async runScheduledPoll(reason: 'startup' | 'interval'): Promise<void> {
    if (reason === 'interval' && Date.now() < this.nextPollAttemptAt) {
      return
    }

    try {
      await this.poll()
      this.markPollHealthy()
    } catch (error) {
      this.markPollFailed(error)
    }
  }

  private markPollHealthy(): void {
    this.nextPollAttemptAt = 0
    if (this.consecutivePollFailures > 0) {
      log.info('Delegation polling recovered', {
        gatewayUrl: this.runtime.config.gateway.url,
        consecutiveFailures: this.consecutivePollFailures,
        lastError: this.lastPollFailureError ?? undefined,
      })
    }
    this.consecutivePollFailures = 0
    this.lastPollFailureError = null
  }

  private markPollFailed(error: unknown): void {
    this.consecutivePollFailures += 1
    const errorText = String(error)
    const backoffMs = Math.min(
      this.pollBackoffMaxMs,
      this.pollBackoffBaseMs * (2 ** (this.consecutivePollFailures - 1)),
    )
    this.nextPollAttemptAt = Date.now() + backoffMs

    const payload = {
      error: errorText,
      gatewayUrl: this.runtime.config.gateway.url,
      consecutiveFailures: this.consecutivePollFailures,
      nextRetryInMs: backoffMs,
    }

    if (
      this.consecutivePollFailures === 1
      || this.lastPollFailureError !== errorText
    ) {
      log.warn('Delegation polling unavailable; backing off', payload)
    } else {
      log.debug('Delegation polling still unavailable', payload)
    }

    this.lastPollFailureError = errorText
  }

  getSessionLeaseState(
    sessionId: string,
    delegationId?: string,
  ): DelegationSessionLeaseState | null {
    const state = this.sessionLeaseStates.get(sessionId)
    if (!state) {
      return null
    }
    if (delegationId && state.delegationId !== delegationId) {
      return null
    }
    return { ...state }
  }

  private updateSessionLeaseState(
    sessionId: string,
    nextState: Omit<DelegationSessionLeaseState, 'sessionId' | 'updatedAt'>,
  ): void {
    this.sessionLeaseStates.set(sessionId, {
      sessionId,
      ...nextState,
      updatedAt: new Date().toISOString(),
    })
  }

  private clearSessionLeaseState(sessionId: string): void {
    this.sessionLeaseStates.delete(sessionId)
  }

  private async appendDelegationStateEvent(input: {
    sessionId: string
    delegationId: string
    targetDevice: string
    claimHealth: 'healthy' | 'degraded' | 'lost'
    startedAt: string
    detail: string
    degradedSince?: string
    lastHeartbeatAt?: string
    lastError?: string
    source?: 'gateway' | 'comments' | 'transport'
  }): Promise<void> {
    await this.runtime.sessions.appendEvent(input.sessionId, {
      type: 'delegation_state',
      id: randomUUID(),
      timestamp: new Date().toISOString(),
      delegationId: input.delegationId,
      targetDevice: input.targetDevice,
      claimHealth: input.claimHealth,
      startedAt: input.startedAt,
      detail: input.detail,
      degradedSince: input.degradedSince,
      lastHeartbeatAt: input.lastHeartbeatAt,
      lastError: input.lastError,
      source: input.source,
    })
  }

  private async process(task: DelegationTask): Promise<void> {
    const provider = this.runtime.providerRegistry.getDefault()
    const workerDevice = this.runtime.config.device.name
    const claim = await this.runtime.delegator.tryAcquireDelegationClaim({
      delegationId: task.delegationId,
      targetDevice: workerDevice,
      ttlMs: this.claimTtlMs,
    })
    if (!claim.claimed) {
      log.info('Delegation claim lost', {
        delegationId: task.delegationId,
        targetDevice: workerDevice,
        source: claim.source,
      })
      return
    }

    const sessionId = randomUUID()
    const startedAt = Date.now()
    const startedAtIso = new Date(startedAt).toISOString()
    this.updateSessionLeaseState(sessionId, {
      delegationId: task.delegationId,
      targetDevice: workerDevice,
      claimId: claim.claimId,
      generation: claim.generation,
      claimHealth: 'healthy',
      startedAt: startedAtIso,
      lastHeartbeatAt: startedAtIso,
    })
    const leaseState: {
      lost: boolean
      cancelled: boolean
      generation?: number
      source?: 'gateway' | 'comments' | 'transport'
    } = {
      lost: false,
      cancelled: false,
      generation: claim.generation,
    }
    const leaseFence = () => ({
      claimId: claim.claimId,
      generation: leaseState.generation,
    })
    let modeRouter: AgentModeRouter | null = null
    const stopClaimHeartbeat = this.startClaimHeartbeat({
      sessionId,
      delegationId: task.delegationId,
      claimId: claim.claimId,
      generation: claim.generation,
      targetDevice: workerDevice,
      onGenerationUpdated: (generation) => {
        leaseState.generation = generation
      },
      onLeaseLost: (source) => {
        leaseState.lost = true
        leaseState.source = source
        if (modeRouter) {
          void modeRouter.stop()
        }
      },
      onCancelRequested: () => {
        leaseState.lost = true
        leaseState.cancelled = true
        leaseState.source = 'comments'
        if (modeRouter) {
          void modeRouter.stop()
        }
      },
    })

    try {
      if (!provider) {
        await this.runtime.delegator.recordDelegationStatus({
          delegationId: task.delegationId,
          targetDevice: workerDevice,
          status: 'failed',
          ...leaseFence(),
          message: `No LLM provider configured on ${workerDevice}`,
        })
        return
      }

      const model = this.runtime.config.agent.defaultModel ?? provider.models[0]?.id ?? 'default'

      await this.runtime.sessions.create({
        id: sessionId,
        title: `[Delegation] ${task.instruction.slice(0, 48)}`,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        provider: provider.id,
        model,
        device: workerDevice,
        status: 'active',
        tags: [
          'delegated-task',
          `delegation:${task.delegationId}`,
        ],
      })

      await this.runtime.sessions.appendEvent(sessionId, {
        type: 'user_message',
        id: randomUUID(),
        timestamp: new Date().toISOString(),
        content: task.instruction,
      })
      await this.appendDelegationStateEvent({
        sessionId,
        delegationId: task.delegationId,
        targetDevice: workerDevice,
        claimHealth: 'healthy',
        startedAt: startedAtIso,
        detail: `Delegated run claimed by ${workerDevice}.`,
        lastHeartbeatAt: startedAtIso,
      })

      await this.runtime.delegator.recordDelegationStatus({
        delegationId: task.delegationId,
        targetDevice: workerDevice,
        status: 'picked_up',
        ...leaseFence(),
        message: `Picked up by ${workerDevice} (session ${sessionId})`,
      })

      const systemPrompt = await buildSystemPrompt({
        config: this.runtime.config,
        tools: this.runtime.toolRegistry,
        skills: this.runtime.skillRegistry,
        fileMemory: this.runtime.fileMemory,
        sessionId,
      })
      const requestQuestion = this.createDelegationQuestionRequester({
        delegationId: task.delegationId,
        targetDevice: workerDevice,
      })

      modeRouter = new AgentModeRouter({
        provider,
        tools: this.runtime.toolRegistry,
        policy: this.runtime.policyEngine,
        autonomy: this.runtime.autonomy,
        semanticIndex: this.runtime.semanticIndex,
        systemPrompt,
        previousMessages: [],
        maxIterations: 10,
        auditLogger: this.runtime.auditLogger,
        usageTracker: this.runtime.usageTracker,
        spendBudget: this.runtime.config.limits,
        hookRegistry: this.runtime.hookRegistry,
        deviceName: workerDevice,
        thinkingLevel: this.runtime.config.agent.thinkingLevel,
        llmCache: this.runtime.llmCache,
        providerCircuitBreaker: this.runtime.providerCircuitBreaker,
        defaultMode: this.runtime.config.agent.mode,
        graphRegistry: this.runtime.graphRegistry,
        requestQuestion,
        saveRunCheckpoint: (checkpoint) =>
          this.runtime.runCheckpoints.save(checkpoint),
        clearRunCheckpoint: (nextSessionId) =>
          this.runtime.runCheckpoints.delete(nextSessionId),
        loadToolExecution: (nextSessionId) =>
          this.runtime.toolExecutions.get(nextSessionId),
        saveToolExecution: (record) =>
          this.runtime.toolExecutions.save(record),
        clearToolExecution: (nextSessionId) =>
          this.runtime.toolExecutions.clearActive(nextSessionId),
        journalStateBoard: createJournalStateBoard(this.runtime.sessions),
        journalSteeringConsumed: createJournalSteeringConsumed(this.runtime.sessions, this.runtime.sessionWatchBroker),
        reviewToollessFinals: true,
      })

      const outputTracker = createAgentOutputTracker()
      let totalUsage: TokenUsage = { inputTokens: 0, outputTokens: 0 }
      // Reconstruct structured board findings from the delegate's tool activity
      // so they can be handed back up to the parent board (PLAN_065 T5), mirroring
      // the isolated subagent path.
      const findingsState: Pick<AgentState, 'evidenceLedger'> = {
        evidenceLedger: emptyEvidenceLedger(),
      }
      const toolCallsById = new Map<string, ToolCall>()

      for await (const event of modeRouter.run(task.instruction, {
        sessionId,
        provider: provider.id,
        model,
        systemPrompt,
        previousMessages: [],
        memoryQuery: task.instruction,
      }, this.runtime.config.agent.mode)) {
        await persistAgentSessionEvent(this.runtime.sessions, sessionId, event)
        outputTracker.consume(event)
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
                securityEffect: this.runtime.toolRegistry.securityDescriptor(toolCall.name).effect,
              },
            )
          }
        }
        if (event.type === 'done') {
          totalUsage = event.usage
        }
        if (event.type === 'error') {
          throw new Error(event.error.message)
        }
      }

      const delegationFindings = extractSubagentFindings(findingsState, {
        sessionId,
        category: 'general',
      })

      if (leaseState.lost) {
        const lossDetail = leaseState.source === 'transport'
          ? 'heartbeat transport failure'
          : `claim source ${leaseState.source ?? 'unknown'}`
        throw new Error(`Delegation claim lost on ${workerDevice} (${lossDetail}); fenced run`)
      }

      const syntheticMessage = outputTracker.syntheticMessageEvent()
      if (syntheticMessage) {
        await persistAgentSessionEvent(this.runtime.sessions, sessionId, syntheticMessage)
      }
      const finalContent = outputTracker.finalContent()

      const completionMessage = finalContent.trim()
        ? finalContent
        : 'Delegated task completed without a final assistant response.'

      if (leaseState.cancelled) {
        await this.runtime.sessions.appendEvent(sessionId, {
          type: 'assistant_message',
          id: randomUUID(),
          timestamp: new Date().toISOString(),
          content: 'Delegated task cancelled remotely.',
        })
        await this.runtime.sessions.appendEvent(sessionId, {
          type: 'session_end',
          id: randomUUID(),
          timestamp: new Date().toISOString(),
          totalTokens: {
            input: totalUsage.inputTokens,
            output: totalUsage.outputTokens,
          },
          totalCost: totalUsage.estimatedCost ?? 0,
          duration_ms: Date.now() - startedAt,
        })
        triggerDreamingSessionEnd(this.runtime.dreaming, sessionId, 'delegation')
        await this.runtime.delegator.recordDelegationStatus({
          delegationId: task.delegationId,
          targetDevice: workerDevice,
          status: 'cancelled',
          ...leaseFence(),
          message: `Cancelled on ${workerDevice} (session ${sessionId})`,
        })
        return
      }

      await this.runtime.sessions.appendEvent(sessionId, {
        type: 'assistant_message',
        id: randomUUID(),
        timestamp: new Date().toISOString(),
        content: completionMessage,
      })
      await this.runtime.sessions.appendEvent(sessionId, {
        type: 'session_end',
        id: randomUUID(),
        timestamp: new Date().toISOString(),
        totalTokens: {
          input: totalUsage.inputTokens,
          output: totalUsage.outputTokens,
        },
        totalCost: totalUsage.estimatedCost ?? 0,
        duration_ms: Date.now() - startedAt,
      })
      triggerDreamingSessionEnd(this.runtime.dreaming, sessionId, 'delegation')

      await this.runtime.delegator.recordDelegationStatus({
        delegationId: task.delegationId,
        targetDevice: workerDevice,
        status: 'completed',
        ...leaseFence(),
        message: formatDelegationCompletionComment({
          workerDevice,
          sessionId,
          summary: completionMessage,
          findings: delegationFindings,
        }),
      })
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      if (leaseState.cancelled) {
        await this.runtime.sessions.appendEvent(sessionId, {
          type: 'assistant_message',
          id: randomUUID(),
          timestamp: new Date().toISOString(),
          content: 'Delegated task cancelled remotely.',
        })
        await this.runtime.sessions.appendEvent(sessionId, {
          type: 'session_end',
          id: randomUUID(),
          timestamp: new Date().toISOString(),
          totalTokens: { input: 0, output: 0 },
          totalCost: 0,
          duration_ms: Date.now() - startedAt,
        })
        triggerDreamingSessionEnd(this.runtime.dreaming, sessionId, 'delegation')
        await this.runtime.delegator.recordDelegationStatus({
          delegationId: task.delegationId,
          targetDevice: workerDevice,
          status: 'cancelled',
          ...leaseFence(),
          message: `Cancelled on ${workerDevice} (session ${sessionId})`,
        })
        return
      }
      await this.runtime.sessions.appendEvent(sessionId, {
        type: 'assistant_message',
        id: randomUUID(),
        timestamp: new Date().toISOString(),
        content: `Delegated task failed: ${message}`,
      })
      await this.runtime.sessions.appendEvent(sessionId, {
        type: 'session_end',
        id: randomUUID(),
        timestamp: new Date().toISOString(),
        totalTokens: { input: 0, output: 0 },
        totalCost: 0,
        duration_ms: Date.now() - startedAt,
      })
      triggerDreamingSessionEnd(this.runtime.dreaming, sessionId, 'delegation')
      await this.runtime.delegator.recordDelegationStatus({
        delegationId: task.delegationId,
        targetDevice: workerDevice,
        status: 'failed',
        ...leaseFence(),
        message: `Failed on ${workerDevice} (session ${sessionId})\n${truncateForComment(message)}`,
      })
    } finally {
      stopClaimHeartbeat()
      await this.runtime.delegator.releaseDelegationClaim({
        delegationId: task.delegationId,
        claimId: claim.claimId,
        targetDevice: workerDevice,
      }).catch((error) => {
        log.warn('Delegation claim release failed', {
          delegationId: task.delegationId,
          targetDevice: workerDevice,
          error: String(error),
        })
      })
      this.clearSessionLeaseState(sessionId)
    }
  }

  private createDelegationQuestionRequester(input: {
    delegationId: string
    targetDevice: string
  }): (question: QuestionRequestInput) => Promise<string> {
    return async (question) => {
      const questionId = await this.runtime.delegator.postDelegationQuestion({
        delegationId: input.delegationId,
        targetDevice: input.targetDevice,
        sessionId: question.sessionId,
        prompt: question.prompt,
        choices: question.choices,
      })
      const policy = (process.env.SEPILOTD_DELEGATION_HEADLESS_APPROVAL ?? 'escalate')
        .trim()
        .toLowerCase()
      if (policy === 'deny') {
        return 'deny'
      }
      return this.waitForDelegationQuestionAnswer({
        delegationId: input.delegationId,
        questionId,
      })
    }
  }

  private waitForDelegationQuestionAnswer(input: {
    delegationId: string
    questionId: string
  }): Promise<string> {
    const configuredPollMs = Number(process.env.SEPILOTD_DELEGATION_QUESTION_POLL_MS)
    const pollMs = Number.isFinite(configuredPollMs) && configuredPollMs > 0
      ? Math.max(20, configuredPollMs)
      : 1_000

    return new Promise<string>((resolve) => {
      let settled = false
      let polling = false

      const finish = (answer: string) => {
        if (settled) {
          return
        }
        settled = true
        clearInterval(timer)
        resolve(answer)
      }

      const poll = async () => {
        if (settled || polling) {
          return
        }
        if (this.stopped) {
          finish('Delegation worker stopped before the delegated question was answered.')
          return
        }
        polling = true
        try {
          const answer = await this.runtime.delegator.getDelegationQuestionAnswer(
            input.delegationId,
            input.questionId,
          )
          if (answer !== null) {
            finish(answer)
          }
        } catch (error) {
          log.warn('Delegation question answer poll failed', {
            delegationId: input.delegationId,
            questionId: input.questionId,
            error: String(error),
          })
        } finally {
          polling = false
        }
      }

      const timer = setInterval(() => {
        void poll()
      }, pollMs)
      timer.unref?.()
      void poll()
    })
  }

  private startClaimHeartbeat(input: {
    sessionId: string
    delegationId: string
    claimId: string
    generation?: number
    targetDevice: string
    onGenerationUpdated?: (generation: number | undefined) => void
    onLeaseLost?: (source: 'gateway' | 'comments' | 'transport') => void
    onCancelRequested?: (message?: string) => void
  }): () => void {
    let stopped = false
    let renewing = false
    let lost = false
    let cancelled = false
    let consecutiveFailures = 0
    let degradedSinceMs: number | null = null
    let generation = input.generation

    const heartbeat = async () => {
      if (stopped || renewing || lost || cancelled) {
        return
      }
      renewing = true
      try {
        const heartbeatAt = new Date().toISOString()
        const cancel = await this.runtime.delegator.getDelegationCancel(
          input.delegationId,
        )
        if (cancel) {
          cancelled = true
          this.updateSessionLeaseState(input.sessionId, {
            delegationId: input.delegationId,
            targetDevice: input.targetDevice,
            claimId: input.claimId,
            generation,
            claimHealth: 'lost',
            startedAt:
              this.sessionLeaseStates.get(input.sessionId)?.startedAt
              ?? heartbeatAt,
            lastHeartbeatAt: heartbeatAt,
            lastError: cancel.message ?? 'Delegation cancelled remotely',
            leaseLossSource: 'comments',
          })
          void this.appendDelegationStateEvent({
            sessionId: input.sessionId,
            delegationId: input.delegationId,
            targetDevice: input.targetDevice,
            claimHealth: 'lost',
            startedAt:
              this.sessionLeaseStates.get(input.sessionId)?.startedAt
              ?? heartbeatAt,
            detail: cancel.message ?? 'Delegation cancelled remotely.',
            lastHeartbeatAt: heartbeatAt,
            lastError: cancel.message ?? 'Delegation cancelled remotely',
            source: 'comments',
          }).catch(() => {})
          input.onCancelRequested?.(cancel.message)
          return
        }
        const result = await this.runtime.delegator.renewDelegationClaim({
          delegationId: input.delegationId,
          claimId: input.claimId,
          targetDevice: input.targetDevice,
          ttlMs: this.claimTtlMs,
          generation,
        })
        const previousHealth = this.sessionLeaseStates.get(input.sessionId)?.claimHealth
        if (result.generation !== undefined) {
          generation = result.generation
          input.onGenerationUpdated?.(generation)
        }
        consecutiveFailures = 0
        degradedSinceMs = null
        this.updateSessionLeaseState(input.sessionId, {
          delegationId: input.delegationId,
          targetDevice: input.targetDevice,
          claimId: input.claimId,
          generation,
          claimHealth: 'healthy',
          startedAt:
            this.sessionLeaseStates.get(input.sessionId)?.startedAt
            ?? heartbeatAt,
          lastHeartbeatAt: heartbeatAt,
        })
        if (previousHealth === 'degraded') {
          void this.appendDelegationStateEvent({
            sessionId: input.sessionId,
            delegationId: input.delegationId,
            targetDevice: input.targetDevice,
            claimHealth: 'healthy',
            startedAt:
              this.sessionLeaseStates.get(input.sessionId)?.startedAt
              ?? heartbeatAt,
            detail: `Delegation lease heartbeat recovered on ${input.targetDevice}.`,
            lastHeartbeatAt: heartbeatAt,
          }).catch(() => {})
        }
        if (!result.renewed) {
          lost = true
          this.updateSessionLeaseState(input.sessionId, {
            delegationId: input.delegationId,
            targetDevice: input.targetDevice,
            claimId: input.claimId,
            generation,
            claimHealth: 'lost',
            startedAt:
              this.sessionLeaseStates.get(input.sessionId)?.startedAt
              ?? heartbeatAt,
            lastHeartbeatAt: heartbeatAt,
            lastError: `Delegation lease lost via ${result.source}`,
            leaseLossSource: result.source,
          })
          void this.appendDelegationStateEvent({
            sessionId: input.sessionId,
            delegationId: input.delegationId,
            targetDevice: input.targetDevice,
            claimHealth: 'lost',
            startedAt:
              this.sessionLeaseStates.get(input.sessionId)?.startedAt
              ?? heartbeatAt,
            detail: `Delegation lease was lost via ${result.source}.`,
            lastHeartbeatAt: heartbeatAt,
            lastError: `Delegation lease lost via ${result.source}`,
            source: result.source,
          }).catch(() => {})
          log.warn('Delegation claim heartbeat lost', {
            delegationId: input.delegationId,
            targetDevice: input.targetDevice,
            source: result.source,
          })
          input.onLeaseLost?.(result.source)
        }
      } catch (error) {
        consecutiveFailures += 1
        const errorMessage = error instanceof Error ? error.message : String(error)
        log.warn('Delegation claim heartbeat failed', {
          delegationId: input.delegationId,
          targetDevice: input.targetDevice,
          consecutiveFailures,
          failureThreshold: this.claimHeartbeatFailureThreshold,
          error: errorMessage,
        })
        if (consecutiveFailures >= this.claimHeartbeatFailureThreshold) {
          const now = Date.now()
          const nowIso = new Date(now).toISOString()
          try {
            const stillHeld = await this.runtime.delegator.verifyDelegationClaim(
              input.delegationId,
              input.claimId,
              {
                claimTtlMs: this.claimTtlMs,
                generation,
              },
            )
            if (stillHeld) {
              const wasHealthy = degradedSinceMs == null
              if (degradedSinceMs == null) {
                degradedSinceMs = now
              }
              const degradedSinceIso = new Date(degradedSinceMs).toISOString()
              const degradedMs = now - degradedSinceMs
              this.updateSessionLeaseState(input.sessionId, {
                delegationId: input.delegationId,
                targetDevice: input.targetDevice,
                claimId: input.claimId,
                generation,
                claimHealth: 'degraded',
                startedAt:
                  this.sessionLeaseStates.get(input.sessionId)?.startedAt
                  ?? nowIso,
                degradedSince: degradedSinceIso,
                lastHeartbeatAt: nowIso,
                lastError: errorMessage,
              })
              if (wasHealthy) {
                void this.appendDelegationStateEvent({
                  sessionId: input.sessionId,
                  delegationId: input.delegationId,
                  targetDevice: input.targetDevice,
                  claimHealth: 'degraded',
                  startedAt:
                    this.sessionLeaseStates.get(input.sessionId)?.startedAt
                    ?? nowIso,
                  detail: `Delegation lease renewals are failing on ${input.targetDevice}, but ownership is still verified.`,
                  degradedSince: degradedSinceIso,
                  lastHeartbeatAt: nowIso,
                  lastError: errorMessage,
                  source: 'transport',
                }).catch(() => {})
              }
              if (degradedMs >= this.claimHeartbeatDegradedMaxMs) {
                lost = true
                this.updateSessionLeaseState(input.sessionId, {
                  delegationId: input.delegationId,
                  targetDevice: input.targetDevice,
                  claimId: input.claimId,
                  generation,
                  claimHealth: 'lost',
                  startedAt:
                    this.sessionLeaseStates.get(input.sessionId)?.startedAt
                    ?? nowIso,
                  degradedSince: degradedSinceIso,
                  lastHeartbeatAt: nowIso,
                  lastError: errorMessage,
                  leaseLossSource: 'transport',
                })
                void this.appendDelegationStateEvent({
                  sessionId: input.sessionId,
                  delegationId: input.delegationId,
                  targetDevice: input.targetDevice,
                  claimHealth: 'lost',
                  startedAt:
                    this.sessionLeaseStates.get(input.sessionId)?.startedAt
                    ?? nowIso,
                  detail: `Delegation lease stayed degraded past the grace window and the worker fenced the run.`,
                  degradedSince: degradedSinceIso,
                  lastHeartbeatAt: nowIso,
                  lastError: errorMessage,
                  source: 'transport',
                }).catch(() => {})
                log.warn('Delegation claim remained degraded past grace window; fencing run', {
                  delegationId: input.delegationId,
                  targetDevice: input.targetDevice,
                  degradedMs,
                  degradedMaxMs: this.claimHeartbeatDegradedMaxMs,
                })
                input.onLeaseLost?.('transport')
                return
              }
              consecutiveFailures = Math.max(
                0,
                this.claimHeartbeatFailureThreshold - 1,
              )
              log.warn('Delegation claim heartbeat degraded but lease verification succeeded', {
                delegationId: input.delegationId,
                targetDevice: input.targetDevice,
                consecutiveFailures,
                degradedMs,
                degradedMaxMs: this.claimHeartbeatDegradedMaxMs,
              })
              return
            }
          } catch (verificationError) {
            log.warn('Delegation claim verification after heartbeat failures failed', {
              delegationId: input.delegationId,
              targetDevice: input.targetDevice,
              error: String(verificationError),
            })
          }

          lost = true
          this.updateSessionLeaseState(input.sessionId, {
            delegationId: input.delegationId,
            targetDevice: input.targetDevice,
            claimId: input.claimId,
            generation,
            claimHealth: 'lost',
            startedAt:
              this.sessionLeaseStates.get(input.sessionId)?.startedAt
              ?? nowIso,
            lastHeartbeatAt: nowIso,
            lastError: errorMessage,
            leaseLossSource: 'transport',
          })
          void this.appendDelegationStateEvent({
            sessionId: input.sessionId,
            delegationId: input.delegationId,
            targetDevice: input.targetDevice,
            claimHealth: 'lost',
            startedAt:
              this.sessionLeaseStates.get(input.sessionId)?.startedAt
              ?? nowIso,
            detail: `Delegation lease verification failed after repeated renew errors and the worker fenced the run.`,
            lastHeartbeatAt: nowIso,
            lastError: errorMessage,
            source: 'transport',
          }).catch(() => {})
          log.warn('Delegation claim heartbeat failed repeatedly; fencing run', {
            delegationId: input.delegationId,
            targetDevice: input.targetDevice,
            consecutiveFailures,
          })
          input.onLeaseLost?.('transport')
        }
      } finally {
        renewing = false
      }
    }

    const timer = setInterval(() => {
      void heartbeat()
    }, this.claimHeartbeatMs)

    return () => {
      stopped = true
      clearInterval(timer)
    }
  }
}
