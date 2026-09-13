import { randomUUID } from 'node:crypto'
import type { AgentEvent, ApprovalDecision, ILLMProvider } from '@sepilotd/core'
import { createLogger } from '../../logger.js'
import type { ToolRegistry } from '../../tools/registry.js'
import type { ApprovalRunCheckpoint } from '../runtime/checkpoints.js'
import type { PendingApproval } from '../runtime/approvals.js'
import { resolveResumeAutonomy } from '../runtime/resume-autonomy.js'
import type { RuntimeServices } from '../runtime/types.js'
import type { FastifyInstance } from 'fastify'
import '../fastify-types.js'
import { AgentModeRouter } from '../../agent/mode-router.js'
import type { ApprovalCallback } from '../../agent/engine.js'
import {
  createJournalStateBoard,
  createJournalSteeringConsumed,
} from '../runtime/mode-router-options.js'
import { findApprovalResolution, findPendingApproval } from '../approval-state.js'
import { persistAgentSessionEvent } from '../session-events.js'
import {
  toRunLimiterApiError,
  type RunLease,
} from '../runtime/run-limiter.js'
import { createSessionBusyLeaseLifecycle } from '../runtime/session-busy.js'
import { resolveApiActor } from '../request-actor.js'
import { buildSseResponseHeaders } from '../sse-response.js'
import { createQuestionRequester } from '../../tools/question.js'
import { createAgentStreamOutputTracker } from './agent-stream-output.js'
import { checkpointMatchesSessionWorkspace } from './request-cwd.js'
import {
  assertApprovalCheckpointIdentity,
  assertCheckpointPendingToolCalls,
  CheckpointToolScopeError,
  resolveCheckpointToolRegistry,
} from '../runtime/resume-tools.js'
import { zodRequestValidation } from './utils.js'
import {
  approvalRequestSchema,
  rememberedApprovalDecisionDescribeSchema,
  rememberedApprovalDecisionInputSchema,
  rememberedApprovalDecisionUpdateSchema,
  type ApprovalResponseBody,
  type RememberedApprovalDecisionDescribeBody,
  type RememberedApprovalDecisionInputBody,
  type RememberedApprovalDecisionUpdateBody,
} from './approvals-schema.js'

export {
  approvalOpenApiComponents,
  approvalOpenApiOverrides,
} from './approvals-schema.js'

const log = createLogger('approvals')

function validatedParkedApprovalTools(
  runtime: RuntimeServices,
  checkpoint: ApprovalRunCheckpoint,
  pending: PendingApproval,
): ToolRegistry {
  if (checkpoint.requestId !== pending.requestId || checkpoint.sessionId !== pending.sessionId) {
    throw new CheckpointToolScopeError('Approval checkpoint request/session identity changed')
  }
  const tools = resolveCheckpointToolRegistry(runtime.toolRegistry, checkpoint)
  assertApprovalCheckpointIdentity(checkpoint.toolCalls[checkpoint.currentToolIndex], pending)
  assertCheckpointPendingToolCalls(checkpoint, checkpoint.toolCalls, checkpoint.currentToolIndex)
  return tools
}

/**
 * A parked approval was just answered: its run already unwound and released
 * its lease, so continue it from the checkpoint the way `/approvals/resume`
 * does — without a client holding a stream. Events persist into the session
 * journal, where every surface watching it picks them up. Returns whether a
 * resume was actually started.
 */
async function resumeParkedApproval(input: {
  runtime: RuntimeServices
  requestId: string
  decision: ApprovalDecision
  requestActor: string
  pending: PendingApproval
}): Promise<{ resumed: boolean; reason?: string }> {
  const { runtime, requestId, decision, requestActor } = input
  const checkpoint = await runtime.approvalCheckpoints.get(requestId)
  if (!checkpoint) return { resumed: false, reason: 'no resumable checkpoint' }
  const session = await runtime.sessions.get(checkpoint.sessionId)
  if (!session || !checkpointMatchesSessionWorkspace(session, checkpoint)) {
    return { resumed: false, reason: 'session workspace changed' }
  }
  const provider = runtime.providerRegistry.get(checkpoint.provider)
  if (!provider) return { resumed: false, reason: `provider ${checkpoint.provider} unavailable` }

  let checkpointTools: ToolRegistry
  try {
    // respond() has already durably journaled the answer, so this request is
    // intentionally no longer "pending" in the journal. Bind the checkpoint
    // to the exact registry-owned request returned by that successful respond.
    checkpointTools = validatedParkedApprovalTools(runtime, checkpoint, input.pending)
  } catch (error) {
    return {
      resumed: false,
      reason: error instanceof Error ? error.message : 'checkpoint tool scope invalid',
    }
  }

  const sessionLease = runtime.sessionBusy
    ? await runtime.sessionBusy.acquireLeaseWithGrace(checkpoint.sessionId)
    : undefined
  if (runtime.sessionBusy && !sessionLease) {
    return { resumed: false, reason: 'session busy' }
  }
  let runLease: RunLease | undefined
  try {
    runLease = await runtime.runLimiter?.acquire()
  } catch (error) {
    sessionLease?.release()
    return { resumed: false, reason: error instanceof Error ? error.message : 'run limit' }
  }

  void (async () => {
    try {
      await executeApprovalResume({
        runtime,
        checkpoint,
        checkpointTools,
        provider,
        decision,
        requestId,
        requestActor,
        send: () => undefined,
        recordDecision: false,
      })
    } catch (error) {
      log.warn('background resume of a parked approval failed', {
        requestId,
        sessionId: checkpoint.sessionId,
        error: error instanceof Error ? error.message : String(error),
      })
    } finally {
      runLease?.release()
      sessionLease?.release()
    }
  })()
  return { resumed: true }
}

export async function approvalRoutes(app: FastifyInstance) {
  app.post<{ Body: ApprovalResponseBody }>('/approvals/respond', {
    preValidation: zodRequestValidation({
      body: {
        schema: approvalRequestSchema,
        message: 'Invalid approval response request body',
      },
    }),
  }, async (request, reply) => {
    const runtime = app.runtime
    if (!runtime?.approvalRegistry || typeof runtime.approvalRegistry.respond !== 'function') {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
      })
    }

    const body = request.body
    const { requestId, approvedBy, sessionId, scope, rule } = body
    const decision = normalizeApprovalInput(body)
    const requestActor = resolveApiActor(
      request.authContext,
      approvedBy ?? 'api',
    )

    const livePending = runtime.approvalRegistry.listAll()
      .find((pending) => pending.requestId === requestId)
    if (livePending) {
      const [session, checkpoint] = await Promise.all([
        runtime.sessions.get(livePending.sessionId),
        runtime.approvalCheckpoints?.get
          ? runtime.approvalCheckpoints.get(requestId)
          : Promise.resolve(null),
      ])
      if (
        session
        && checkpoint
        && !checkpointMatchesSessionWorkspace(session, checkpoint)
      ) {
        return reply.status(409).send({
          error: {
            code: 'WORKSPACE_BINDING_CHANGED',
            message: 'The session workspace changed; this approval is no longer valid',
          },
        })
      }
      if (checkpoint && livePending.state === 'parked') {
        try {
          // Reject a changed checkpoint before consuming/journaling consent.
          validatedParkedApprovalTools(runtime, checkpoint, livePending)
        } catch (error) {
          return reply.status(409).send({
            error: { code: 'RESUME_TOOL_SCOPE_INVALID', message: error instanceof Error ? error.message : 'Invalid checkpoint' },
          })
        }
      }
    }

    const respondResult = await runtime.approvalRegistry.respond(
      requestId,
      decision,
      { approvedBy: requestActor, scope, rule },
    )

    if (respondResult.resolved) {
      const parkedState = respondResult.parked ? 'parked' : 'live'
      // The prompt outlived its run: the decision is journaled, now continue
      // the parked run from its checkpoint. Nothing else is waiting on it.
      const resume = respondResult.parked
        ? await resumeParkedApproval({ runtime, requestId, decision, requestActor, pending: respondResult.parked })
        : undefined
      await runtime.auditLogger?.log?.({
        timestamp: new Date().toISOString(),
        event: 'approval.respond.live',
        device: runtime.config.device.name,
        session: sessionId,
        actor: requestActor,
        requestId,
        approved: decision.approved,
        decision: decision.decision,
        note: decision.note,
        state: parkedState,
        ...(resume ? { resumed: resume.resumed, resumeReason: resume.reason } : {}),
      })
      return {
        data: {
          requestId,
          decision: decision.decision,
          approved: decision.approved,
          note: decision.note,
          resolved: respondResult.resolved,
          state: parkedState,
          ...(resume ? { resumed: resume.resumed, ...(resume.reason ? { resumeReason: resume.reason } : {}) } : {}),
          // Surface the derived rule when this respond persisted a
          // session/always decision — the cli prints it as part of
          // the success line so the operator sees the exact pattern
          // future rounds will short-circuit on, without having to
          // grep `decisions list` afterwards.
          ...(respondResult.rule ? { rule: respondResult.rule } : {}),
        },
      }
    }

    if (!sessionId) {
      return reply.status(404).send({
        error: { code: 'NOT_FOUND', message: 'Approval request not found' },
      })
    }

    const session = await runtime.sessions.get(sessionId)
    if (!session) {
      return reply.status(404).send({
        error: { code: 'NOT_FOUND', message: 'Session not found' },
      })
    }

    const events = await runtime.sessions.getEvents(sessionId)
    const pendingApproval = findPendingApproval(sessionId, events, requestId)
    if (!pendingApproval) {
      const priorResolution = findApprovalResolution(events, requestId)
      if (priorResolution) {
        await runtime.auditLogger?.log?.({
          timestamp: new Date().toISOString(),
          event: 'approval.respond.replayed',
          device: runtime.config.device.name,
          session: sessionId,
          actor: requestActor,
          requestId,
          approved: priorResolution.approved,
          decision: priorResolution.decision,
          note: priorResolution.note,
          state: 'stale',
        })
        return {
          data: {
            requestId,
            decision: priorResolution.decision,
            approved: priorResolution.approved,
            note: priorResolution.note,
            resolved: true,
            state: 'stale',
          },
        }
      }
      return reply.status(404).send({
        error: { code: 'NOT_FOUND', message: 'Approval request not found' },
      })
    }

    await runtime.sessions.appendEvent(sessionId, {
      type: 'approval_response',
      id: randomUUID(),
      timestamp: new Date().toISOString(),
      requestId,
      decision: decision.decision,
      approved: decision.approved,
      note: decision.note,
      approvedBy: resolveApiActor(
        request.authContext,
        approvedBy ?? 'api-recovered',
      ),
    })
    await runtime.sessions.appendEvent(sessionId, {
      type: 'tool_result',
      id: randomUUID(),
      timestamp: new Date().toISOString(),
      toolCallId: pendingApproval.toolCallId,
      output: staleApprovalMessage(decision),
      status: 'cancelled',
      duration_ms: 0,
    })
    await runtime.auditLogger?.log?.({
      timestamp: new Date().toISOString(),
      event: 'approval.respond.recovered',
      device: runtime.config.device.name,
      session: sessionId,
      actor: resolveApiActor(
        request.authContext,
        approvedBy ?? 'api-recovered',
      ),
      requestId,
      approved: decision.approved,
      decision: decision.decision,
      note: decision.note,
      state: 'stale',
      toolCallId: pendingApproval.toolCallId,
    })

    return {
      data: {
        requestId,
        decision: decision.decision,
        approved: decision.approved,
        note: decision.note,
        resolved: true,
        state: 'stale',
      },
    }
  })

  app.post<{ Body: ApprovalResponseBody }>('/approvals/resume', {
    preValidation: zodRequestValidation({
      body: {
        schema: approvalRequestSchema,
        message: 'Invalid approval resume request body',
      },
    }),
  }, async (request, reply) => {
    const runtime = app.runtime
    if (!runtime) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
      })
    }

    const body = request.body
    const { requestId, approvedBy, sessionId } = body
    const decision = normalizeApprovalInput(body)
    const requestActor = resolveApiActor(
      request.authContext,
      approvedBy ?? 'api-resume',
    )

    const checkpoint = await runtime.approvalCheckpoints.get(requestId)
    if (!checkpoint) {
      return reply.status(404).send({
        error: {
          code: 'NOT_FOUND',
          message: 'No resumable run checkpoint found for this approval',
        },
      })
    }

    if (sessionId && checkpoint.sessionId !== sessionId) {
      return reply.status(400).send({
        error: {
          code: 'INVALID_REQUEST',
          message: 'sessionId does not match the resumable checkpoint',
        },
      })
    }

    // Register disconnect cleanup before waiting for the session lease. A
    // cancelled request must not acquire and orphan a lease after the grace
    // wait finishes. Once the resumed run starts, its finally owns cleanup.
    const sessionLeaseLifecycle = createSessionBusyLeaseLifecycle()
    request.raw.once('aborted', sessionLeaseLifecycle.releasePreflight)
    reply.raw.once('finish', sessionLeaseLifecycle.releasePreflight)
    reply.raw.once('close', sessionLeaseLifecycle.releasePreflight)
    if (request.raw.aborted || reply.raw.destroyed) {
      sessionLeaseLifecycle.releasePreflight()
      return reply
    }

    const sessionLease = runtime.sessionBusy
      ? await runtime.sessionBusy.acquireLeaseWithGrace(checkpoint.sessionId)
      : undefined
    if (!sessionLeaseLifecycle.attachLease(sessionLease)) return reply
    if (runtime.sessionBusy && !sessionLease) {
      return reply.status(409).send({
        error: {
          code: 'BUSY',
          message:
            'This session is already processing another turn; it did not free up within the grace window. Wait a moment and retry.',
        },
      })
    }

    const checkpointSession = await runtime.sessions.get(checkpoint.sessionId)
    if (
      !checkpointSession
      || !checkpointMatchesSessionWorkspace(checkpointSession, checkpoint)
    ) {
      return reply.status(409).send({
        error: {
          code: 'WORKSPACE_BINDING_CHANGED',
          message: 'The session workspace changed; this approval checkpoint cannot resume',
        },
      })
    }

    const provider = runtime.providerRegistry.get(checkpoint.provider)
    if (!provider) {
      return reply.status(503).send({
        error: {
          code: 'SERVICE_UNAVAILABLE',
          message: `Provider ${checkpoint.provider} is unavailable for resume`,
        },
      })
    }

    let checkpointTools
    try {
      checkpointTools = resolveCheckpointToolRegistry(runtime.toolRegistry, checkpoint)
      const events = await runtime.sessions.getEvents(checkpoint.sessionId)
      assertApprovalCheckpointIdentity(
        checkpoint.toolCalls[checkpoint.currentToolIndex],
        findPendingApproval(checkpoint.sessionId, events, requestId),
      )
      assertCheckpointPendingToolCalls(
        checkpoint,
        checkpoint.toolCalls,
        checkpoint.currentToolIndex,
      )
    } catch (error) {
      return reply.status(409).send({
        error: {
          code: 'RESUME_TOOL_SCOPE_INVALID',
          message: error instanceof CheckpointToolScopeError
            ? error.message
            : 'The checkpoint tool scope is invalid',
        },
      })
    }

    let runLease: RunLease | undefined
    try {
      runLease = await runtime.runLimiter?.acquire()
    } catch (error) {
      return reply.status(503).send({
        error: toRunLimiterApiError(error),
      })
    }

    if (!sessionLeaseLifecycle.transferToRun()) {
      runLease?.release()
      return reply
    }

    try {
      await runtime.auditLogger?.log?.({
        timestamp: new Date().toISOString(),
        event: 'approval.resume.started',
        device: runtime.config.device.name,
        session: checkpoint.sessionId,
        actor: requestActor,
        requestId,
        approved: decision.approved,
        decision: decision.decision,
        note: decision.note,
        checkpointCreatedAt: checkpoint.createdAt,
        toolCallId: checkpoint.toolCalls[checkpoint.currentToolIndex]?.id,
        tool: checkpoint.toolCalls[checkpoint.currentToolIndex]?.name,
      })

      reply.raw.writeHead(200, buildSseResponseHeaders(request, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        Connection: 'keep-alive',
        'X-Request-ID': request.requestId ?? randomUUID(),
      }))

      const send = (event: string, data: unknown) => {
        reply.raw.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`)
      }

      await executeApprovalResume({
        runtime,
        checkpoint,
        checkpointTools,
        provider,
        decision,
        requestId,
        requestActor,
        send,
        recordDecision: true,
      })
      reply.raw.end()
    } finally {
      runLease?.release()
      sessionLeaseLifecycle.releaseRun()
    }
  })

  app.get('/approvals/decisions', async (_request, reply) => {
    const runtime = app.runtime
    if (!runtime?.approvalDecisions) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
      })
    }
    return { data: { decisions: runtime.approvalDecisions.list() } }
  })

  app.post<{ Body: RememberedApprovalDecisionDescribeBody }>(
    '/approvals/decisions/describe',
    {
      preValidation: zodRequestValidation({
        body: {
          schema: rememberedApprovalDecisionDescribeSchema,
          message: 'Invalid remembered approval decision describe request body',
        },
      }),
    },
    async (request, reply) => {
      const runtime = app.runtime
      if (!runtime?.approvalDecisions) {
        return reply.status(503).send({
          error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
        })
      }

      const body = request.body
      const rule = runtime.approvalDecisions.describeRule(body.tool, body.input)
      return { data: { rule } }
    },
  )

  app.post<{ Body: RememberedApprovalDecisionInputBody }>('/approvals/decisions', {
    preValidation: zodRequestValidation({
      body: {
        schema: rememberedApprovalDecisionInputSchema,
        message: 'Invalid remembered approval decision request body',
      },
    }),
  }, async (request, reply) => {
    const runtime = app.runtime
    if (!runtime?.approvalDecisions) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
      })
    }

    try {
      const body = request.body
      const decision = runtime.approvalDecisions.upsert(body)
      await runtime.approvalDecisions.flush()
      return { data: { decision } }
    } catch (err) {
      return reply.status(400).send({
        error: {
          code: 'INVALID_REQUEST',
          message: err instanceof Error ? err.message : 'Invalid remembered approval decision',
        },
      })
    }
  })

  app.patch<{ Body: RememberedApprovalDecisionUpdateBody }>('/approvals/decisions', {
    preValidation: zodRequestValidation({
      body: {
        schema: rememberedApprovalDecisionUpdateSchema,
        message: 'Invalid remembered approval decision update request body',
      },
    }),
  }, async (request, reply) => {
    const runtime = app.runtime
    if (!runtime?.approvalDecisions) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
      })
    }

    try {
      const { match, decision: update } = request.body
      const decision = runtime.approvalDecisions.update(match, update)
      if (!decision) {
        return reply.status(404).send({
          error: { code: 'NOT_FOUND', message: 'Remembered approval decision not found' },
        })
      }
      await runtime.approvalDecisions.flush()
      return { data: { decision } }
    } catch (err) {
      return reply.status(400).send({
        error: {
          code: 'INVALID_REQUEST',
          message: err instanceof Error ? err.message : 'Invalid remembered approval decision',
        },
      })
    }
  })

  app.delete<{
    Querystring: {
      scope?: 'session' | 'always'
      sessionId?: string
      /** When 'true' (string from query), only remove entries that
       * match isStaleRememberedDecision. Lets cli `decisions clear
       * --stale` prune cleanup candidates without nuking active
       * rules. */
      stale?: string
      /** Exact tool name. Pairs with `decisions list --tool` so the
       * operator can preview-then-prune the same set. */
      tool?: string
      /** Exact derived rule pattern. When present with tool/scope,
       * delete exactly one remembered rule instead of bulk clearing. */
      pattern?: string
      /** Optional outcome filter for bulk clearing. */
      approved?: string
    }
  }>('/approvals/decisions', async (request, reply) => {
    const runtime = app.runtime
    if (!runtime?.approvalDecisions) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
      })
    }
    const query = request.query
    if (query.pattern) {
      if (!query.tool || !query.scope) {
        return reply.status(400).send({
          error: {
            code: 'INVALID_REQUEST',
            message: 'tool and scope are required when deleting a remembered approval pattern',
          },
        })
      }
      try {
        const removed = runtime.approvalDecisions.remove({
          tool: query.tool,
          pattern: query.pattern,
          scope: query.scope,
          sessionId: query.sessionId,
        })
        await runtime.approvalDecisions.flush()
        return { data: { removed } }
      } catch (err) {
        return reply.status(400).send({
          error: {
            code: 'INVALID_REQUEST',
            message: err instanceof Error ? err.message : 'Invalid remembered approval decision',
          },
        })
      }
    }

    let approved: boolean | undefined
    if (query.approved !== undefined) {
      if (query.approved !== 'true' && query.approved !== 'false') {
        return reply.status(400).send({
          error: {
            code: 'INVALID_REQUEST',
            message: 'approved must be true or false',
          },
        })
      }
      approved = query.approved === 'true'
    }

    runtime.approvalDecisions.clear({
      scope: query.scope,
      sessionId: query.sessionId,
      stale: query.stale === 'true',
      tool: query.tool || undefined,
      approved,
    })
    await runtime.approvalDecisions.flush()
    return { data: { cleared: true } }
  })
}

interface ApprovalResumeRuntime {
  runtime: RuntimeServices
  checkpoint: ApprovalRunCheckpoint
  checkpointTools: ToolRegistry
  provider: ILLMProvider
  decision: ApprovalDecision
  requestId: string
  requestActor: string
  /** Emit a stream event; a background resume passes a no-op. */
  send: (event: string, data: unknown) => void
  /**
   * False when the decision was already journaled (a parked prompt answered
   * through `/approvals/respond`), so the resume does not record it twice.
   */
  recordDecision: boolean
}

/**
 * Drive a run from its approval checkpoint to completion. Shared by the SSE
 * `/approvals/resume` route and the background resume that a parked approval
 * triggers when it is finally answered.
 */
async function executeApprovalResume(input: ApprovalResumeRuntime): Promise<void> {
  const { runtime, checkpoint, checkpointTools, provider, decision, requestId, requestActor, send } = input
  if (input.recordDecision) {
    await runtime.sessions.appendEvent(checkpoint.sessionId, {
      type: 'approval_response',
      id: randomUUID(),
      timestamp: new Date().toISOString(),
      requestId,
      decision: decision.decision,
      approved: decision.approved,
      note: decision.note,
      approvedBy: requestActor,
    })
    // Keep a parked request answerable if validation, admission, or durable
    // consent recording fails. Only the successful resume consumes it.
    runtime.approvalRegistry.releaseParked?.(requestId)
  }

  const approvalCallback: ApprovalCallback = (toolCall, nextRequestId, options) =>
    runtime.approvalRegistry.waitForApproval({
      sessionId: checkpoint.sessionId,
      toolCall,
      requestId: nextRequestId,
      forcePrompt: options?.forcePrompt,
        signal: options?.signal,
    })
  const requestQuestion = createQuestionRequester(runtime.questions)
  if (checkpoint.autonomy !== undefined && checkpoint.autonomy !== runtime.autonomy) checkpoint.requireToolApproval = true

  const modeRouter = new AgentModeRouter({
    provider,
    tools: checkpointTools,
    policy: runtime.policyEngine,
    autonomy: resolveResumeAutonomy(checkpoint.autonomy, runtime.autonomy),
    maxIterations: checkpoint.maxIterations,
    systemPrompt: checkpoint.systemPrompt,
    auditLogger: runtime.auditLogger,
    usageTracker: runtime.usageTracker,
    spendBudget: runtime.config.limits,
    hookRegistry: runtime.hookRegistry,
    deviceName: runtime.config.device.name,
    thinkingLevel: checkpoint.thinkingLevel,
    textDeltaMode: checkpoint.textDeltaMode,
    llmCache: runtime.llmCache,
    providerCircuitBreaker: runtime.providerCircuitBreaker,
    graphRegistry: runtime.graphRegistry,
    approvalCallback,
    evaluateAutoApproval: (toolCall) =>
      runtime.approvalRegistry.tryAutoApproval({
        sessionId: checkpoint.sessionId,
        toolCall,
      }),
    requestQuestion,
    saveApprovalCheckpoint: (nextCheckpoint) =>
      runtime.approvalCheckpoints.save(nextCheckpoint),
    clearApprovalCheckpoint: (nextRequestId) =>
      runtime.approvalCheckpoints.delete(nextRequestId),
    saveRunCheckpoint: (nextCheckpoint) =>
      runtime.runCheckpoints?.save(nextCheckpoint) ?? Promise.resolve(),
    clearRunCheckpoint: (nextSessionId) =>
      runtime.runCheckpoints?.delete(nextSessionId) ?? Promise.resolve(),
    loadToolExecution: (nextSessionId) =>
      runtime.toolExecutions?.get(nextSessionId) ?? Promise.resolve(null),
    saveToolExecution: (record) =>
      runtime.toolExecutions?.save(record) ?? Promise.resolve(),
    clearToolExecution: (nextSessionId) =>
      runtime.toolExecutions?.clearActive(nextSessionId) ?? Promise.resolve(),
    editSnapshotStore: runtime.editSnapshotStore,
    toolStatsStore: runtime.toolStatsStore,
    workspaceMutationTracker: runtime.workspaceMutationTracker,
    pluginEvents: runtime.pluginEvents,
    journalStateBoard: createJournalStateBoard(runtime.sessions),
    journalSteeringConsumed: createJournalSteeringConsumed(runtime.sessions, runtime.sessionWatchBroker),
    reviewToollessFinals: true,
  })

  send('session', { sessionId: checkpoint.sessionId })

  const outputTracker = createAgentStreamOutputTracker()
  let terminalDoneEvent: Extract<AgentEvent, { type: 'done' }> | null = null
  let terminalStateChangeEvent: Extract<AgentEvent, { type: 'state_change' }> | null = null
  for await (const event of modeRouter.resumeFromApprovalCheckpoint(checkpoint, decision)) {
    outputTracker.consume(event)
    if (event.type === 'done') {
      terminalDoneEvent = event
      continue
    }
    if (event.type === 'state_change' && event.state === 'done') {
      terminalStateChangeEvent = event
      continue
    }
    send(event.type, event)
    await persistAgentSessionEvent(runtime.sessions, checkpoint.sessionId, event)
  }

  let finalContent = outputTracker.finalContent()
  const syntheticMessage = outputTracker.syntheticMessageEvent()
  if (syntheticMessage) {
    send(syntheticMessage.type, syntheticMessage)
    await persistAgentSessionEvent(runtime.sessions, checkpoint.sessionId, syntheticMessage)
    finalContent = syntheticMessage.content
  }

  if (finalContent) {
    await runtime.sessions.appendEvent(checkpoint.sessionId, {
      type: 'assistant_message',
      id: randomUUID(),
      timestamp: new Date().toISOString(),
      content: finalContent,
    })
  }
  if (terminalDoneEvent && terminalStateChangeEvent) {
    send(terminalStateChangeEvent.type, terminalStateChangeEvent)
    await persistAgentSessionEvent(
      runtime.sessions,
      checkpoint.sessionId,
      terminalStateChangeEvent,
    )
  }
  if (terminalDoneEvent) {
    send(terminalDoneEvent.type, terminalDoneEvent)
    await persistAgentSessionEvent(runtime.sessions, checkpoint.sessionId, terminalDoneEvent)
  }

  await runtime.auditLogger?.log?.({
    timestamp: new Date().toISOString(),
    event: 'approval.resume.completed',
    device: runtime.config.device.name,
    session: checkpoint.sessionId,
    actor: requestActor,
    requestId,
    approved: decision.approved,
    decision: decision.decision,
    note: decision.note,
    checkpointCreatedAt: checkpoint.createdAt,
    toolCallId: checkpoint.toolCalls[checkpoint.currentToolIndex]?.id,
    tool: checkpoint.toolCalls[checkpoint.currentToolIndex]?.name,
    finalContentPresent: Boolean(finalContent),
  })

  send('close', {})
}

function normalizeApprovalInput(input: ApprovalResponseBody): ApprovalDecision {
  const decision = input.decision
    ?? (input.approved ? 'approved' : 'denied')
  return {
    decision,
    approved: decision === 'approved',
    note: input.note,
    ...(decision === 'denied' && input.stop !== undefined ? { stop: input.stop } : {}),
  }
}

function staleApprovalMessage(decision: ApprovalDecision): string {
  if (decision.decision === 'approved') {
    return 'Approval was recorded after the daemon restarted. The original run cannot resume; send a new message to continue from this point.'
  }
  if (decision.decision === 'feedback') {
    return decision.note
      ? `Approval feedback was recorded after the daemon restarted: ${decision.note}`
      : 'Approval feedback was recorded after the daemon restarted. The original run cannot resume; send a new message to continue from this point.'
  }
  return decision.note
    ? `Approval was denied after the daemon restarted: ${decision.note}`
    : 'Approval was denied after the daemon restarted. The original run cannot resume; send a new message if you want to retry this action.'
}
