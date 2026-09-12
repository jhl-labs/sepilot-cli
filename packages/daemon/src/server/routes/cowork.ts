import { randomUUID } from 'node:crypto'
import { getDocRegistry } from '../../agent/doc/session.js'
import { resolveRequestedAutonomy } from '../../security/autonomy.js'
import type { FastifyInstance } from 'fastify'
import type { AgentEvent, AutonomyLevel, ISkillRegistry, Message } from '@sepilotd/core'
import { z } from 'zod'
import '../fastify-types.js'
import { AgentModeRouter } from '../../agent/mode-router.js'
import type { ApprovalCallback } from '../../agent/engine.js'
import { IntentRouter, type IntentDecision, type IntentRouterOptions } from '../../agent/intent-router.js'
import { createAuxiliaryLlmTurnBudget } from '../../agent/auxiliary-llm.js'
import { resolveSkillExecutionContext } from '../../skills/execution-policy.js'
import {
  loadAutoCompactedSessionContext,
  loadLatestRunContract,
  resolveSessionContextMaxMessages,
} from '../../agent/auto-compaction.js'
import { applyPreUserPromptHook } from '../../agent/turn-hooks.js'
import { resolveApiActor } from '../request-actor.js'
import { buildSystemPrompt } from '../../agent/system-prompt.js'
import { collectAgentsMd, formatAgentsMdSection } from '../../memory/agents-md.js'
import { triggerDreamingTurn } from '../../memory/dreaming.js'
import { resolveScopedFileMemoryReadView } from '../../memory/scoped-file-memory-read-view.js'
import { getPersona } from '../../agent/personas.js'
import { extractAndStoreArtifacts } from './artifact-support.js'
import { chatRequestSchema, resolveChatMaxIterations, type ChatBody } from './chat-schema.js'
import {
  resolveOfficeAutoSkillContent,
  resolveSkillRefsContent,
  sendSkillNotFoundReply,
  sendSkillUnavailableReply,
  SkillNotFoundError,
  SkillUnavailableError,
} from './chat-skills.js'
import { loadProjectContext } from './project-context.js'
import { zodRequestValidation } from './utils.js'
import { openApiComponentsFromZod } from '../openapi-zod.js'
import {
  openApiJsonResponseRef,
  openApiSchemaRef,
  type OpenApiComponentOverrides,
  type OpenApiOverrideMap,
} from '../openapi.js'
import { toRunLimiterApiError, type RunLease } from '../runtime/run-limiter.js'
import { persistAgentSessionEvent } from '../session-events.js'
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
import { createRuntimeBackedModeRouterOptions } from '../runtime/mode-router-options.js'
import { createSessionBusyLeaseLifecycle } from '../runtime/session-busy.js'
import { InvalidCwdError, invalidCwdResponse, resolveRequestWorkspace } from './request-cwd.js'
import type { ToolRegistry } from '../../tools/registry.js'
import { resolveClosedExactToolNames } from '../../agent/tool-call-budget.js'
import {
  withContextualToolExposure,
  withDirectApiMemoryToolset,
  withSwarmToolsForSession,
  withWritingDocTools,
} from '../../tools/role-filter.js'
import {
  buildProviderSelectionFailure,
  chatRequestPrefersVision,
  selectChatProviderCandidates,
} from './provider-selection.js'
import { isExtensionSessionReuseDenied, readChatMemoryScope } from '../chat-memory-context.js'
import { buildChatKnowledgeContext } from '../chat-knowledge.js'
import {
  INTENT_ROUTER_CIRCUIT_BREAKER,
  resolveIntentRouterModel,
} from './chat-intent.js'

export async function resolveCoworkSkillPrefix(
  message: string,
  refs: ReadonlyArray<{ name: string }> | undefined,
  skillRegistry: ISkillRegistry,
  toolRegistry: ToolRegistry,
  cwd: string | undefined,
  autonomy: AutonomyLevel,
  workspaceRoot?: string,
  previousMessages?: readonly Message[],
  loadedSkillIds?: Set<string>,
  declaredToolNames?: Set<string>,
  autoLoadedToolNames?: Set<string>,
): Promise<string> {
  let skillPrefix = await resolveSkillRefsContent(
    refs,
    skillRegistry,
    toolRegistry,
    cwd,
    autonomy,
    declaredToolNames,
    workspaceRoot,
    loadedSkillIds,
  )
  skillPrefix += await resolveOfficeAutoSkillContent(
    message,
    refs,
    skillRegistry,
    toolRegistry,
    cwd,
    autonomy,
    workspaceRoot,
    autoLoadedToolNames,
    previousMessages,
    loadedSkillIds,
  )
  return skillPrefix
}

const coworkAgentSchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string(),
})

const coworkAgentsResponseSchema = z.object({
  data: z.array(coworkAgentSchema),
})

export const coworkOpenApiComponents: OpenApiComponentOverrides = openApiComponentsFromZod({
  schemas: {
    CoworkAgent: coworkAgentSchema,
    CoworkAgentsResponse: coworkAgentsResponseSchema,
  },
})

export const coworkOpenApiOverrides: OpenApiOverrideMap = {
  '/api/v1/cowork/run': {
    post: {
      summary: 'Run cowork team session',
      tags: ['Agents'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('ChatRequest'),
          },
        },
      },
      responses: {
        200: {
          description: 'SSE stream',
          content: { 'text/event-stream': {} },
        },
        409: { description: 'Session already has an active turn' },
        503: { description: 'Service unavailable or agent run capacity exhausted' },
      },
    },
  },
  '/api/v1/cowork/agents': {
    get: {
      summary: 'List cowork-capable agents',
      tags: ['Agents'],
      responses: { 200: openApiJsonResponseRef('CoworkAgentsResponse') },
    },
  },
}

export async function coworkRoutes(app: FastifyInstance) {
  app.post<{ Body: ChatBody }>(
    '/cowork/run',
    {
      preValidation: zodRequestValidation({
        body: {
          schema: chatRequestSchema,
          message: 'Invalid cowork request body',
        },
      }),
    },
    async (request, reply) => {
      const runtime = app.runtime
      if (!runtime) {
        return reply.status(503).send({
          error: {
            code: 'SERVICE_UNAVAILABLE',
            message: 'Runtime not initialized',
          },
        })
      }

      const body = request.body
      const scopeTags = readChatMemoryScope(request.headers, request.authContext)
      const autonomyResolution = resolveRequestedAutonomy(runtime.autonomy, body.autonomy)
      const effectiveAutonomy = autonomyResolution.effective

      const {
        message,
        sessionId: reqSessionId,
        model,
        provider: reqProvider,
        persona,
        thinkingLevel,
        projectId,
      } = body

      if (isExtensionSessionReuseDenied(request.authContext, reqSessionId)) {
        return reply.status(403).send({
          error: {
            code: 'EXTENSION_SESSION_REUSE_UNSUPPORTED',
            message: 'Extensions must omit sessionId until session ownership is persisted.',
          },
        })
      }

      const sessionId = reqSessionId ?? randomUUID()
      const sessionLeaseLifecycle = createSessionBusyLeaseLifecycle()
      request.raw.once('aborted', sessionLeaseLifecycle.releasePreflight)
      reply.raw.once('finish', sessionLeaseLifecycle.releasePreflight)
      reply.raw.once('close', sessionLeaseLifecycle.releasePreflight)
      if (request.raw.aborted || reply.raw.destroyed) {
        sessionLeaseLifecycle.releasePreflight()
        return reply
      }
      const sessionLease = runtime.sessionBusy
        ? await runtime.sessionBusy.acquireLeaseWithGrace(sessionId)
        : undefined
      if (!sessionLeaseLifecycle.attachLease(sessionLease)) {
        return reply
      }
      if (runtime.sessionBusy && !sessionLease) {
        return reply.status(409).send({
          error: {
            code: 'BUSY',
            message:
              'This session is already processing another turn; it did not free up within the grace window. Wait a moment and retry.',
          },
        })
      }

      const existingSession = reqSessionId
        ? await runtime.sessions.get(sessionId)
        : null
      if (existingSession?.memoryNamespace) return reply.status(400).send({ error: { code: 'MEMORY_SPACE_MISMATCH', message: 'Use the persona chat transport for this isolated conversation' } })
      const projectContext = await loadProjectContext(runtime, projectId)
      const explicitWorkspaceRoot =
        typeof body.workspaceRoot === 'string' && body.workspaceRoot.trim()
          ? body.workspaceRoot
          : undefined
      const rawWorkspaceRoot =
        explicitWorkspaceRoot ?? existingSession?.cwd ?? projectContext?.workingDirectory
      const rawRequestCwd =
        typeof body.cwd === 'string' && body.cwd.trim()
          ? body.cwd
          : rawWorkspaceRoot
      let requestCwd: string | undefined
      let requestWorkspaceRoot: string | undefined
      try {
        const resolved = await resolveRequestWorkspace(
          rawRequestCwd,
          rawWorkspaceRoot,
          existingSession?.cwd,
        )
        requestCwd = resolved.cwd
        requestWorkspaceRoot = resolved.workspaceRoot
      } catch (error) {
        if (error instanceof InvalidCwdError) {
          return reply.status(400).send(invalidCwdResponse(error))
        }
        throw error
      }

      const priorRunContractForSelection = reqSessionId
        ? await loadLatestRunContract(runtime.sessions, sessionId)
        : undefined
      const selection = selectChatProviderCandidates({
        runtime,
        message,
        requestedProvider: reqProvider,
        requestedModel: model,
        existingSession,
        preferVision: chatRequestPrefersVision(
          message,
          'cowork',
          priorRunContractForSelection,
        ),
      })[0]
      if (!selection) {
        const failure = buildProviderSelectionFailure(runtime, reqProvider, model)
        return reply.status(failure.statusCode).send({ error: failure.error })
      }
      const provider = selection.provider
      const selectedModel = selection.model

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

      let lifecycle: ReturnType<typeof createSseLifecycle> | null = null
      let clientClosed = false
      try {
        // pre:user:prompt — fires on the cowork path too (was HTTP /chat-only)
        // so moderation/redaction is not bypassed by using cowork. Runs before
        // the SSE stream starts, so an abort can still return a clean 400.
        const promptGate = await applyPreUserPromptHook(message, {
          hookRegistry: runtime.hookRegistry,
          sessionId,
          actor: resolveApiActor(request.authContext, 'api-user'),
        })
        if (promptGate.aborted) {
          return reply.status(400).send({
            error: {
              code: 'USER_PROMPT_REJECTED',
              message: promptGate.reason ?? 'User prompt rejected by hook',
            },
          })
        }
        const finalPrompt = promptGate.prompt

        const sessionWorkspace = requestWorkspaceRoot ?? requestCwd
        let session = existingSession
        if (!session) {
          session = await runtime.sessions.create({
            id: sessionId,
            title: `[cowork] ${finalPrompt.slice(0, 40)}`,
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
            provider: provider.id,
            model: selectedModel,
            device: runtime.config.device.name,
            status: 'active',
            cwd: sessionWorkspace,
            tags: ['cowork'],
          })
        } else if (sessionWorkspace && session.cwd !== sessionWorkspace) {
          session =
            (await runtime.sessions.updateMeta?.(session.id, { cwd: sessionWorkspace })) ?? session
        }
        const previousContext = await loadAutoCompactedSessionContext({
          sessionStore: runtime.sessions,
          sessionId,
          provider,
          model: selectedModel,
          hooks: runtime.hookRegistry,
          maxMessages: resolveSessionContextMaxMessages(),
        })
        const previousMessages = previousContext.messages
        const previousRunContract = previousContext.runContract
        const auxiliaryLlmBudget = createAuxiliaryLlmTurnBudget()
        let contractRoutingDecision: IntentDecision | undefined
        const routerCfg = runtime.config.agent.intentRouter
        if (previousRunContract && routerCfg?.enabled) {
          try {
            const routerProvider = routerCfg.provider
              ? runtime.providerRegistry.get(routerCfg.provider)
              : provider
            const routerModel = resolveIntentRouterModel({
              routerProvider,
              configuredModel: routerCfg.model,
              configuredProviderId: routerCfg.provider,
              activeModel: routerProvider?.id === provider.id ? selectedModel : undefined,
              logger: app.log,
            })
            if (routerProvider && routerModel) {
              const intentRouter = new IntentRouter({
                provider: routerProvider,
                model: routerModel,
                graphRegistry: runtime.graphRegistry,
                skillRegistry: runtime.skillRegistry,
                personaList: () => [],
                currentAutonomy: effectiveAutonomy,
                cwd: requestCwd,
                workspaceRoot: requestWorkspaceRoot,
                timeoutMs: routerCfg.timeoutMs,
                maxPreviousMessages: routerCfg.maxPreviousMessages,
                perMessageCharLimit: routerCfg.perMessageCharLimit,
                reasonMaxChars: routerCfg.reasonMaxChars,
                logger: app.log as unknown as IntentRouterOptions['logger'],
                circuitBreaker: INTENT_ROUTER_CIRCUIT_BREAKER,
                auxiliaryLlmBudget,
                activeRunContract: previousRunContract,
                availableToolNames: runtime.toolRegistry.list().map((tool) => tool.name),
              })
              contractRoutingDecision = await intentRouter.decide(
                finalPrompt,
                previousMessages,
                {},
              )
            }
          } catch (error) {
            app.log.warn({ error }, 'cowork contract relation routing failed')
          }
        }

        await runtime.sessions.appendEvent(sessionId, {
          type: 'user_message',
          id: randomUUID(),
          timestamp: new Date().toISOString(),
          content: finalPrompt,
        })
        const personaObj = persona ? getPersona(persona) : undefined
        const agentsFiles = await collectAgentsMd({
          cwd: requestCwd,
          boundaryRoot: requestWorkspaceRoot,
        })
        const agentsMemory = formatAgentsMdSection(agentsFiles) ?? undefined
        let skillPrefix = ''
        const loadedExecutionSkillIds = new Set<string>()
        const declaredSkillToolNames = new Set<string>()
        const autoLoadedSkillToolNames = new Set<string>()
        try {
          skillPrefix = await resolveCoworkSkillPrefix(
            message,
            body.skillRefs,
            runtime.skillRegistry,
            runtime.toolRegistry,
            requestCwd,
            effectiveAutonomy,
            requestWorkspaceRoot,
            previousMessages,
            loadedExecutionSkillIds,
            declaredSkillToolNames,
            autoLoadedSkillToolNames,
          )
        } catch (error) {
          if (error instanceof SkillNotFoundError) {
            return sendSkillNotFoundReply(reply, error)
          }
          if (error instanceof SkillUnavailableError) {
            return sendSkillUnavailableReply(reply, error)
          }
          throw error
        }
        const relevantKnowledgeContext = await buildChatKnowledgeContext({
          registry: app.chatKnowledgeProviders,
          message: finalPrompt,
          previousMessages,
          allowLocalKnowledge: request.authContext?.kind !== 'extension',
        })
        const directApiToolRegistry = withWritingDocTools(
          withSwarmToolsForSession(
            withDirectApiMemoryToolset(
              runtime.toolRegistry,
              runtime.config.memory?.directApiToolset ?? 'lean',
            ),
            sessionId,
          ),
          // The same expression the doc.* tools resolve with; cowork carries no
          // writingDocId of its own.
          getDocRegistry().getActiveId(),
        )
        const agentToolRegistry = withContextualToolExposure(directApiToolRegistry, {
          selectedGroups: ['files', 'code', 'process'],
          activeWritingDocument: Boolean(getDocRegistry().getActiveId()),
          swarmSession: sessionId.startsWith('swarm_'),
          declaredSkillToolNames,
          supplementalToolNames: autoLoadedSkillToolNames,
          requestInput: finalPrompt,
          explicitToolNames: body.toolNames,
        })
        const closedToolSurface = resolveClosedExactToolNames(
          finalPrompt,
          directApiToolRegistry.list(),
        ) !== null
        const baseSystemPrompt = await buildSystemPrompt({
          config: runtime.config,
          tools: agentToolRegistry,
          skills: runtime.skillRegistry,
          fileMemory: resolveScopedFileMemoryReadView(
            runtime.fileMemory,
            runtime.fileMemoryRegistry,
            scopeTags,
          ),
          agentsMemory,
          relevantMemoryContext: relevantKnowledgeContext,
          customInstructions: personaObj?.systemPromptAddition,
          projectContext,
          cwd: requestCwd,
          workspaceRoot: requestWorkspaceRoot,
          sessionId,
          closedToolSurface,
          includeDailyNotes: false,
        })
        const systemPrompt = skillPrefix + baseSystemPrompt

        const approvalCallback: ApprovalCallback = (toolCall, requestId, options) =>
          runtime.approvalRegistry.waitForApproval({
            sessionId,
            toolCall,
            requestId,
            forcePrompt: options?.forcePrompt,
            signal: options?.signal,
          })
        const requestQuestion = createQuestionRequester(runtime.questions)

        const modeRouter = new AgentModeRouter({
          provider,
          tools: agentToolRegistry,
          policy: runtime.policyEngine,
          autonomy: effectiveAutonomy,
          auxiliaryLlmBudget,
          semanticIndex: runtime.semanticIndex,
          systemPrompt,
          previousMessages,
          maxIterations: resolveChatMaxIterations(body),
          auditLogger: runtime.auditLogger,
          usageTracker: runtime.usageTracker,
          spendBudget: runtime.config.limits,
          hookRegistry: runtime.hookRegistry,
          deviceName: runtime.config.device.name,
          thinkingLevel,
          maxTokens: body.maxTokens,
          temperature: body.temperature,
          textDeltaMode: body.textDeltaMode,
          ragEnabled: body.ragEnabled,
          llmCache: runtime.llmCache,
          providerCircuitBreaker: runtime.providerCircuitBreaker,
          defaultMode: runtime.config.agent.mode,
          graphRegistry: runtime.graphRegistry,
          ...(contractRoutingDecision
            ? {
                intentModeHint: {
                  mode: 'cowork' as const,
                  confidence: contractRoutingDecision.confidence,
                  fallback: contractRoutingDecision.fallback,
                  executionIntent: contractRoutingDecision.executionIntent,
                  contractRelation: contractRoutingDecision.contractRelation,
                },
              }
            : {}),
          approvalCallback,
          evaluateAutoApproval: (toolCall) =>
            runtime.approvalRegistry.tryAutoApproval({ sessionId, toolCall }),
          requestQuestion,
          ...createRuntimeBackedModeRouterOptions(runtime),
          reviewToollessFinals: true,
        })

        const stopRun = () => {
          clientClosed = true
          void modeRouter.stop().catch(() => {})
        }

        registerSseDisconnectHandler(request, reply, stopRun)
        trackSseConnection(app, request, reply, { label: 'cowork' })

        reply.hijack()
        reply.raw.writeHead(
          200,
          buildSseResponseHeaders(request, {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            Connection: 'keep-alive',
            'X-Request-ID': request.requestId ?? randomUUID(),
          }),
        )

        const send = (event: string, data: unknown) => {
          if (clientClosed || reply.raw.writableEnded) {
            return
          }
          reply.raw.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`)
        }

        const AGENT_INACTIVITY_MS = resolveAgentInactivityMs()
        const inactivityProbe = createAgentInactivityProbe()
        lifecycle = createSseLifecycle({
          reply,
          inactivityMs: AGENT_INACTIVITY_MS,
          // A run blocked on a pending approval/question is not stalled.
          pendingDecision: () => inactivityProbe.pendingDecision(),
          decisionTimeoutMs: resolvePendingDecisionTimeoutMs(),
          onInactivity: () => {
            void modeRouter.stop().catch(() => {})
          },
          onDecisionTimeout: () => {
            void modeRouter.stop().catch(() => {})
          },
        })

        send('session', { sessionId, mode: 'cowork' })
        send('team', { agents: ['coder', 'reviewer', 'researcher'] })

        const outputTracker = createAgentStreamOutputTracker()
        let terminalDoneEvent: Extract<AgentEvent, { type: 'done' }> | null = null
        let terminalStateChangeEvent: Extract<AgentEvent, { type: 'state_change' }> | null = null
        try {
          for await (const event of modeRouter.run(
            finalPrompt,
            {
              sessionId,
              provider: provider.id,
              model: selectedModel,
              cwd: requestCwd,
              workspaceRoot: requestWorkspaceRoot,
              executionPolicy: {
                requestedAutonomy: autonomyResolution.requested,
                configuredAutonomy: autonomyResolution.configured,
                effectiveAutonomy: autonomyResolution.effective,
                clamped: autonomyResolution.clamped,
                clampReason: autonomyResolution.reason,
                agentMode: 'cowork',
                primaryAgentId: runtime.primaryAgents?.get(sessionId),
                workspaceBoundary: requestWorkspaceRoot ? 'strict' : 'unrestricted',
                freshApprovalRequired: body.requireToolApproval === true,
              },
              systemPrompt,
              previousMessages,
              toolAllowlist: agentToolRegistry.list().map((tool) => tool.name),
              ...resolveSkillExecutionContext(
                loadedExecutionSkillIds,
                new Set([...declaredSkillToolNames, ...autoLoadedSkillToolNames]),
              ),
              runContract: previousRunContract,
              runContractScope: 'previous-turn',
              memoryQuery: finalPrompt,
              scopeTags,
              primaryAgentId: runtime.primaryAgents?.get(sessionId),
              autoApprove: runtime.autoApprove,
              requireToolApproval: body.requireToolApproval,
            },
            'cowork',
          )) {
            if (clientClosed) break
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
            await persistAgentSessionEvent(runtime.sessions, sessionId, event)
          }

          if (lifecycle.isInactivityTripped()) {
            throw new Error('Cowork run stopped after an inactivity timeout.')
          }

          let finalContent = outputTracker.finalContent()
          if (!clientClosed) {
            const syntheticMessage = outputTracker.syntheticMessageEvent()
            if (syntheticMessage) {
              send(syntheticMessage.type, syntheticMessage)
              await persistAgentSessionEvent(runtime.sessions, sessionId, syntheticMessage)
              finalContent = syntheticMessage.content
            }
          }

          if (!clientClosed && finalContent) {
            const artifacts = await extractAndStoreArtifacts(runtime, sessionId, finalContent)
            if (artifacts.length > 0) {
              send('artifacts', { artifacts })
            }

            await runtime.sessions.appendEvent(sessionId, {
              type: 'assistant_message',
              id: randomUUID(),
              timestamp: new Date().toISOString(),
              content: finalContent,
            })
            triggerDreamingTurn(runtime.dreaming, sessionId, 'cowork', scopeTags)
          }

          if (!clientClosed && terminalStateChangeEvent) {
            send(terminalStateChangeEvent.type, terminalStateChangeEvent)
            await persistAgentSessionEvent(runtime.sessions, sessionId, terminalStateChangeEvent)
          }
          if (!clientClosed && terminalDoneEvent) {
            send(terminalDoneEvent.type, terminalDoneEvent)
            await persistAgentSessionEvent(runtime.sessions, sessionId, terminalDoneEvent)
          }

          if (!clientClosed) {
            send('close', {})
            reply.raw.end()
          }
        } catch (err) {
          if (lifecycle.isInactivityTripped()) {
            try {
              await runtime.sessions.updateMeta?.(sessionId, { status: 'abandoned' })
              if (!reply.raw.writableEnded) {
                reply.raw.write(
                  `event: error\ndata: ${JSON.stringify({
                    type: 'error',
                    error: inactivityProbe.describe({
                      inactivityMs: AGENT_INACTIVITY_MS,
                      provider: provider.id,
                      model: model ?? session.model,
                    }),
                  })}\n\n`,
                )
                reply.raw.end()
              }
            } catch {
              /* best-effort */
            }
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
        } catch {
          /* best-effort */
        }
        throw err
      } finally {
        lifecycle?.dispose()
        try {
          if (!reply.raw.writableEnded) reply.raw.end()
        } catch {
          /* noop */
        }
        runLease?.release()
        sessionLeaseLifecycle.releaseRun()
      }
    },
  )

  app.get('/cowork/agents', async (_request, _reply) => {
    const runtime = app.runtime
    if (!runtime?.graphRegistry) return { data: [] }

    return {
      data: runtime.graphRegistry.list().map((agent) => ({
        id: agent.id,
        name: agent.name,
        description: agent.description,
      })),
    }
  })
}
