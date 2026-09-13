import { recoveredReactSteering, withSteeringCheckpoint } from '../../agent/react-steering.js'
import { resolveResumeAutonomy } from '../runtime/resume-autonomy.js'
import { randomUUID } from 'node:crypto'
import type { AgentEvent } from '@sepilotd/core'
import type { FastifyInstance } from 'fastify'
import '../fastify-types.js'
import { AgentModeRouter } from '../../agent/mode-router.js'
import type { ApprovalCallback } from '../../agent/engine.js'
import { appendStateBoardEvent } from '../../agent/graph/state-board-journal.js'
import {
  recoverStateBoard,
  latestStateBoardAt,
  applyRecoveredBoard,
  reconcileRecoveredSteeringNotes,
} from '../../agent/graph/state-board-recover.js'
import { isStateBoardEnabled } from '../../agent/graph/state-board.js'
import { zodRequestValidation } from './utils.js'
import {
  assessRunResume,
  attemptInterruptedToolRecovery,
  describeRunResumeTarget,
} from '../runtime/resume.js'
import {
  toRunLimiterApiError,
  type RunLease,
} from '../runtime/run-limiter.js'
import { persistAgentSessionEvent } from '../session-events.js'
import { resolveApiActor } from '../request-actor.js'
import {
  buildSseResponseHeaders,
  createAgentInactivityProbe,
  createSseLifecycle,
  isSubstantiveAgentActivity,
  registerSseDisconnectHandler,
  resolveAgentInactivityMs,
  resolvePendingDecisionTimeoutMs,
  trackSseConnection,
} from '../sse-response.js'
import { createAgentStreamOutputTracker } from './agent-stream-output.js'
import { createQuestionRequester } from '../../tools/question.js'
import { checkpointMatchesSessionWorkspace } from './request-cwd.js'
import {
  assertCheckpointPendingToolCalls,
  CheckpointToolScopeError,
  resolveCheckpointToolRegistry,
} from '../runtime/resume-tools.js'
import {
  sessionIdParamsSchema,
  sessionResumeRequestInputSchema,
  type SessionIdParams,
  type SessionResumeBody,
} from './sessions-schema.js'

export function registerSessionResumeRoute(app: FastifyInstance): void {
  const runtime = app.runtime

  app.post<{ Params: SessionIdParams; Body: SessionResumeBody }>('/sessions/:id/resume', {
    preValidation: zodRequestValidation({
      params: {
        schema: sessionIdParamsSchema,
        message: 'Invalid session id',
      },
      body: {
        schema: sessionResumeRequestInputSchema,
        message: 'Invalid resume request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
      })
    }
    const params = request.params
    const body = request.body
    const requestActor = resolveApiActor(request.authContext, 'api:resume')

    const checkpointClaim = await runtime.runCheckpoints?.claim(params.id)
    if (checkpointClaim?.status === 'busy') {
      await runtime.auditLogger?.log?.({
        timestamp: new Date().toISOString(),
        event: 'session.resume.blocked',
        device: runtime.config.device.name,
        session: params.id,
        actor: requestActor,
        reason: 'already_running',
        force: Boolean(body.force),
      })
      return reply.status(409).send({
        error: {
          code: 'RESUME_ALREADY_RUNNING',
          message: 'A resume run is already active for this session',
        },
      })
    }
    if (!checkpointClaim || checkpointClaim.status === 'missing') {
      return reply.status(404).send({
        error: { code: 'NOT_FOUND', message: 'No resumable run checkpoint found' },
      })
    }
    if (checkpointClaim.status === 'unavailable') {
      await runtime.auditLogger?.log?.({
        timestamp: new Date().toISOString(),
        event: 'session.resume.blocked',
        device: runtime.config.device.name,
        session: params.id,
        actor: requestActor,
        reason: 'checkpoint_unavailable',
        checkpointStatus: checkpointClaim.issue.status,
        force: Boolean(body.force),
      })
      return reply.status(409).send({
        error: {
          code: 'RESUME_CHECKPOINT_UNAVAILABLE',
          message: checkpointClaim.issue.message,
        },
      })
    }

    // Resume can recover/replay tools, so it must own the same session lease
    // before reading the bound workspace or inspecting the checkpoint.
    const sessionLease = runtime.sessionBusy
      ? await runtime.sessionBusy.acquireLeaseWithGrace(params.id)
      : undefined
    if (runtime.sessionBusy && !sessionLease) {
      await checkpointClaim.release()
      return reply.status(409).send({
        error: {
          code: 'BUSY',
          message: 'This session is already processing another turn; it did not free up within the grace window. Wait a moment and retry.',
        },
      })
    }

    try {
    const session = await runtime.sessions.get(params.id)
    if (!session) {
      return reply.status(404).send({
        error: { code: 'NOT_FOUND', message: 'Session not found' },
      })
    }

    const checkpoint = checkpointClaim.checkpoint
    const resumedAutonomy = resolveResumeAutonomy(checkpoint.autonomy, runtime.autonomy)
    if (checkpoint.autonomy !== undefined && checkpoint.autonomy !== runtime.autonomy) checkpoint.requireToolApproval = true
    if (!checkpointMatchesSessionWorkspace(session, checkpoint)) {
      await checkpointClaim.release()
      return reply.status(409).send({
        error: {
          code: 'WORKSPACE_BINDING_CHANGED',
          message: 'The session workspace changed; this run checkpoint cannot resume',
        },
      })
    }
    let checkpointTools
    try {
      checkpointTools = resolveCheckpointToolRegistry(runtime.toolRegistry, checkpoint)
      if (checkpoint.pendingToolExecution) {
        assertCheckpointPendingToolCalls(
          checkpoint,
          checkpoint.pendingToolExecution.toolCalls,
          checkpoint.pendingToolExecution.startIndex,
        )
      }
    } catch (error) {
      await checkpointClaim.release()
      return reply.status(409).send({
        error: {
          code: 'RESUME_TOOL_SCOPE_INVALID',
          message: error instanceof CheckpointToolScopeError
            ? error.message
            : 'The checkpoint tool scope is invalid',
        },
      })
    }
    const recoveredResult = await attemptInterruptedToolRecovery(
      checkpoint,
      checkpointTools,
      runtime.toolExecutions,
      runtime.policyEngine,
      resumedAutonomy,
    )
    const currentTool =
      checkpoint.pendingToolExecution?.toolCalls[checkpoint.pendingToolExecution.startIndex]

    const resumableRun = recoveredResult
      ? {
          mode: 'exact' as const,
          forceRequired: false,
          currentTool: currentTool?.name,
          currentToolCount: currentTool ? 1 : undefined,
          currentTools: currentTool ? [currentTool.name] : undefined,
        }
      : await assessRunResume(
          checkpoint,
          checkpointTools,
          runtime.toolExecutions,
        )
    if (resumableRun.forceRequired && body.force !== true) {
      await runtime.auditLogger?.log?.({
        timestamp: new Date().toISOString(),
        event: 'session.resume.blocked',
        device: runtime.config.device.name,
        session: checkpoint.sessionId,
        actor: requestActor,
        reason: 'force_required',
        force: Boolean(body.force),
        checkpointedAt: checkpoint.checkpointedAt,
        mode: resumableRun.mode,
        currentTool: resumableRun.currentTool,
        currentToolCount: resumableRun.currentToolCount,
      })
      await checkpointClaim.release()
      return reply.status(409).send({
        error: {
          code: 'RESUME_REQUIRES_FORCE',
          message: `Resuming this checkpoint may replay ${describeRunResumeTarget(resumableRun)}. Retry with force: true to continue.`,
        },
      })
    }

    const provider = runtime.providerRegistry.get(checkpoint.provider)
    if (!provider) {
      await checkpointClaim.release()
      return reply.status(503).send({
        error: {
          code: 'SERVICE_UNAVAILABLE',
          message: `Provider ${checkpoint.provider} is unavailable for resume`,
        },
      })
    }

    let runLease: RunLease | undefined
    try {
      runLease = await runtime.runLimiter?.acquire()
    } catch (error) {
      await checkpointClaim.release()
      return reply.status(503).send({
        error: toRunLimiterApiError(error),
      })
    }

    let lifecycle: ReturnType<typeof createSseLifecycle> | null = null
    try {
      await runtime.auditLogger?.log?.({
        timestamp: new Date().toISOString(),
        event: 'session.resume.started',
        device: runtime.config.device.name,
        session: checkpoint.sessionId,
        actor: requestActor,
        force: Boolean(body.force),
        checkpointedAt: checkpoint.checkpointedAt,
        mode: resumableRun.mode,
        currentTool: resumableRun.currentTool,
        currentToolCount: resumableRun.currentToolCount,
      })

      let clientClosed = false
      const send = (event: string, data: unknown) => {
        if (clientClosed || reply.raw.writableEnded) {
          return
        }
        reply.raw.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`)
      }

      const approvalCallback: ApprovalCallback = (toolCall, requestId, options) =>
        runtime.approvalRegistry.waitForApproval({
          sessionId: checkpoint.sessionId,
          toolCall,
          requestId,
          forcePrompt: options?.forcePrompt,
            signal: options?.signal,
        })
      const requestQuestion = createQuestionRequester(runtime.questions)

      const steeringEvents = await runtime.sessions.getEvents(checkpoint.sessionId)
      if (!checkpoint.graphState) {
        const state = { steeringNotes: recoveredReactSteering(checkpoint.messages) }
        reconcileRecoveredSteeringNotes(state, steeringEvents)
        checkpoint.messages = withSteeringCheckpoint(checkpoint.messages, state.steeringNotes)
      }
      if (checkpoint.graphState) {
        const events = steeringEvents

        // Steering acknowledgements and terminal states are independently
        // journaled, so always reconcile them with the checkpoint. This
        // prevents a cancelled follow-up from reappearing after restart and
        // restores a pending note acknowledged after the last checkpoint.
        reconcileRecoveredSteeringNotes(checkpoint.graphState, events)

        // Restore board-derived state journaled after the last checkpoint
        // write. Board and checkpoint are normally written together (same
        // cadence), so this only fills gaps left by a crash between the two.
        if (isStateBoardEnabled()) {
          const boardAt = latestStateBoardAt(events)
          const checkpointAt = Date.parse(checkpoint.checkpointedAt)
          if (
            boardAt !== undefined
            && (!Number.isFinite(checkpointAt) || boardAt > checkpointAt)
          ) {
            applyRecoveredBoard(checkpoint.graphState, recoverStateBoard(events))
          }
        }
      }

      const modeRouter = new AgentModeRouter({
        provider,
        tools: checkpointTools,
        policy: runtime.policyEngine,
        autonomy: resumedAutonomy,
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
        activeRuns: runtime.activeRuns,
        approvalCallback,
        evaluateAutoApproval: (toolCall) =>
          runtime.approvalRegistry.tryAutoApproval({
            sessionId: checkpoint.sessionId,
            toolCall,
          }),
        requestQuestion,
        saveApprovalCheckpoint: (nextCheckpoint) =>
          runtime.approvalCheckpoints.save(nextCheckpoint),
        clearApprovalCheckpoint: (requestId) =>
          runtime.approvalCheckpoints.delete(requestId),
        saveRunCheckpoint: (nextCheckpoint) =>
          runtime.runCheckpoints.save(nextCheckpoint),
        clearRunCheckpoint: (nextSessionId) =>
          runtime.runCheckpoints.delete(nextSessionId),
        journalStateBoard: isStateBoardEnabled()
          ? (nextSessionId, board) =>
              appendStateBoardEvent(runtime.sessions, nextSessionId, board)
          : undefined,
        journalSteeringConsumed: (nextSessionId, noteId) =>
          runtime.sessions.appendEvent(nextSessionId, {
            type: 'steering_consumed',
            id: randomUUID(),
            timestamp: new Date().toISOString(),
            noteId,
          }),
        loadToolExecution: (nextSessionId) =>
          runtime.toolExecutions.get(nextSessionId),
        saveToolExecution: (record) =>
          runtime.toolExecutions.save(record),
        clearToolExecution: (nextSessionId) =>
          runtime.toolExecutions.clearActive(nextSessionId),
        editSnapshotStore: runtime.editSnapshotStore,
        toolStatsStore: runtime.toolStatsStore,
        workspaceMutationTracker: runtime.workspaceMutationTracker,
        pluginEvents: runtime.pluginEvents,
        reviewToollessFinals: true,
      })

      const stopRun = () => {
        clientClosed = true
        void modeRouter.stop().catch(() => {})
      }

      registerSseDisconnectHandler(request, reply, stopRun)
      trackSseConnection(app, request, reply, { label: 'session-resume' })

      reply.hijack()
      reply.raw.writeHead(200, buildSseResponseHeaders(request, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        Connection: 'keep-alive',
        'X-Request-ID': request.requestId ?? randomUUID(),
      }))

      const AGENT_INACTIVITY_MS = resolveAgentInactivityMs()
      const inactivityProbe = createAgentInactivityProbe()
      lifecycle = createSseLifecycle({
        reply,
        inactivityMs: AGENT_INACTIVITY_MS,
        // A run blocked on a pending approval/question is not stalled: the
        // stall watchdog is suppressed and the separate decision bound applies.
        pendingDecision: () => inactivityProbe.pendingDecision(),
        decisionTimeoutMs: resolvePendingDecisionTimeoutMs(),
        onInactivity: () => {
          void modeRouter.stop().catch(() => {})
        },
        onDecisionTimeout: () => {
          void modeRouter.stop().catch(() => {})
        },
      })

      send('session', { sessionId: checkpoint.sessionId })

      const outputTracker = createAgentStreamOutputTracker()
      let terminalDoneEvent: Extract<AgentEvent, { type: 'done' }> | null = null
      let terminalStateChangeEvent: Extract<AgentEvent, { type: 'state_change' }> | null = null
      try {
        for await (const event of modeRouter.resumeFromRunCheckpoint(checkpoint)) {
          if (clientClosed) {
            break
          }
          if (isSubstantiveAgentActivity(event)) {
            lifecycle.recordEvent()
            inactivityProbe.note(event)
          }

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

        if (lifecycle.isInactivityTripped()) {
          throw new Error('Resume stopped after an inactivity timeout.')
        }

        let finalContent = outputTracker.finalContent()
        if (!clientClosed) {
          const syntheticMessage = outputTracker.syntheticMessageEvent()
          if (syntheticMessage) {
            send(syntheticMessage.type, syntheticMessage)
            await persistAgentSessionEvent(runtime.sessions, checkpoint.sessionId, syntheticMessage)
            finalContent = syntheticMessage.content
          }
        }

        if (!clientClosed && finalContent) {
          await runtime.sessions.appendEvent(checkpoint.sessionId, {
            type: 'assistant_message',
            id: randomUUID(),
            timestamp: new Date().toISOString(),
            content: finalContent,
          })
        }
        if (!clientClosed && terminalStateChangeEvent) {
          send(terminalStateChangeEvent.type, terminalStateChangeEvent)
          await persistAgentSessionEvent(
            runtime.sessions,
            checkpoint.sessionId,
            terminalStateChangeEvent,
          )
        }
        if (!clientClosed && terminalDoneEvent) {
          send(terminalDoneEvent.type, terminalDoneEvent)
          await persistAgentSessionEvent(runtime.sessions, checkpoint.sessionId, terminalDoneEvent)
        }

        await runtime.auditLogger?.log?.({
          timestamp: new Date().toISOString(),
          event: 'session.resume.completed',
          device: runtime.config.device.name,
          session: checkpoint.sessionId,
          actor: requestActor,
          force: Boolean(body.force),
          checkpointedAt: checkpoint.checkpointedAt,
          mode: resumableRun.mode,
          recoveredResult: Boolean(recoveredResult),
          finalContentPresent: Boolean(finalContent),
        })

        if (!clientClosed) {
          send('close', {})
          reply.raw.end()
        }
      } catch (err) {
        if (lifecycle.isInactivityTripped()) {
          try {
            await runtime.sessions.updateMeta?.(checkpoint.sessionId, { status: 'abandoned' })
            if (!reply.raw.writableEnded) {
              reply.raw.write(
                `event: error\ndata: ${JSON.stringify({
                  type: 'error',
                  error: inactivityProbe.describe({
                    inactivityMs: AGENT_INACTIVITY_MS,
                    provider: provider.id,
                    model: provider.models[0]?.id,
                    activityLabel: 'Resume',
                  }),
                })}\n\n`,
              )
              reply.raw.end()
            }
          } catch { /* best-effort */ }
          return
        }
        throw err
      }
    } catch (err) {
      try {
        if (!reply.raw.writableEnded) {
          reply.raw.write(
            `event: error\ndata: ${JSON.stringify({
              type: 'error',
              error: {
                code: 'INTERNAL_ERROR',
                message: err instanceof Error ? err.message : String(err),
              },
            })}\n\n`,
          )
        }
      } catch { /* best-effort */ }
      throw err
    } finally {
      lifecycle?.dispose()
      try {
        if (!reply.raw.writableEnded) reply.raw.end()
      } catch { /* noop */ }
      runLease?.release()
    }
    } finally {
      sessionLease?.release()
      await checkpointClaim.release()
    }
  })
}
