import { resolveTurnMemoryPolicy, memoryToolAllowed } from '../../memory/turn-policy.js'
import { resolvePersonaMemoryScope } from '../../memory/persona-scope.js'
import { withSelectedBrowserTarget } from '../../tools/browser-target.js'
import { remoteBrowserBridge } from '../../tools/browser-remote.js'
import { randomUUID } from 'node:crypto'
import { resolveRequestedAutonomy } from '../../security/autonomy.js'
import type { AgentEvent, ApiError, ContentPart, SessionEvent } from '@sepilotd/core'
import type { FastifyInstance, FastifyReply, FastifyRequest } from 'fastify'
import '../fastify-types.js'
import { AgentModeRouter, type AgentMode } from '../../agent/mode-router.js'
import { requiresNativeMultimodalInput } from '../../agent/multimodal-input.js'
import type { ApprovalCallback } from '../../agent/engine.js'
import { resolveInstantModeToolNames } from '../../agent/instant-mode-tool-intent.js'
import { resolveSkillExecutionContext } from '../../skills/execution-policy.js'
import {
  formatContextCompactionNotice,
  loadAutoCompactedSessionContextWithinBudget,
  loadLatestRoutedPersona,
  loadLatestRunContract,
  resolveSessionContextMaxMessages,
} from '../../agent/auto-compaction.js'
import { fallbackSessionTitle } from '../../agent/session-title.js'
import { triggerDreamingTurn } from '../../memory/dreaming.js'
import { publishChatCompletionNotification } from '../../notifications/publish.js'
import { buildDeferredSystemPrompt } from '../../agent/system-prompt.js'
import {
  applyWritingDocFallback,
  snapshotWritingDocStart,
} from '../../agent/doc/writing-fallback.js'
import { buildShellModePrefix, buildWritingDocPrefix } from './chat-mode-prefixes.js'
import { getDocRegistry } from '../../agent/doc/session.js'

import { collectAgentsMd, formatAgentsMdSection } from '../../memory/agents-md.js'
import {
  resolvePersona,
  resolvePersonaCatalog,
  resolvePersonas,
} from '../../agent/custom/persona-resolver.js'
import { createPersonaRepo } from '../../persona/repo.js'
import {
  withContextualToolExposure, withTurnToolAuthority,
  withDirectApiMemoryToolset,
  withSwarmToolsForSession,
  withWritingDocTools,
} from '../../tools/role-filter.js'
import {
  chatRequestSchema,
  excludesLocalContext,
  isDesktopExternalAgentMode,
  resolveChatHardMaxIterations,
  resolveChatMaxIterations,
  type ChatBody,
} from './chat-schema.js'
import {
  resolveImageGeneratorAutoSkillContent,
  resolveOfficeAutoSkillContent,
  resolveSkillRefsContent,
  filterLoadableSkillRefs,
  buildSkillsLoaded,
  shouldDeferPptxAttachmentText,
  sendSkillNotFoundReply,
  sendSkillUnavailableReply,
  SkillNotFoundError,
  SkillUnavailableError,
} from './chat-skills.js'
import { extractAndStoreArtifacts, extractAndStoreUserImageArtifacts } from './artifact-support.js'
import { resolveFileIds, resolveSessionAttachmentRefs } from './file-registry.js'
import { loadProjectContext } from './project-context.js'
import {
  AttachmentLimitError,
  assertChatAttachmentLimits,
  type SkippedAttachment,
} from '../../media/pipeline.js'
import { zodRequestValidation } from './utils.js'
import { persistAgentSessionEvent } from '../session-events.js'
import { resolveApiActor } from '../request-actor.js'
import { createQuestionRequester } from '../../tools/question.js'
import { InvalidCwdError, invalidCwdResponse, resolveChatRequestWorkspace } from './request-cwd.js'
import { toRunLimiterApiError, type RunLease } from '../runtime/run-limiter.js'
import { createSessionBusyLeaseLifecycle } from '../runtime/session-busy.js'
import { createRuntimeBackedModeRouterOptions } from '../runtime/mode-router-options.js'
import { openApiSchemaRef, type OpenApiOverrideMap } from '../openapi.js'
import {
  buildSseResponseHeaders,
  createAgentInactivityProbe,
  createSseLifecycle,
  describePendingDecisionTimeout,
  isSubstantiveAgentActivity,
  registerSseDisconnectHandler,
  resolveAgentInactivityMs,
  resolvePendingDecisionTimeoutMs,
  trackSseConnection,
  type PendingDecision,
} from '../sse-response.js'
import { isStreamWriteStoppedError, writeSseFrame } from '../sse-write.js'
import { createAgentStreamOutputTracker } from './agent-stream-output.js'
import {
  appendChatProviderAttemptEvent,
  canRetryChatProviderAttempt,
  isAgentAttemptCommitted,
  providerRetryThinkingEvent,
  recordChatProviderFailure,
  buildProviderSelectionFailure,
  recordChatProviderSuccess,
  chatRequestPrefersVision,
  selectChatProviderCandidates,
  updateChatSessionProviderMeta,
} from './provider-selection.js'
import { IntentRouter, type IntentRouterOptions } from '../../agent/intent-router.js'
import { createAuxiliaryLlmTurnBudget } from '../../agent/auxiliary-llm.js'
import {
  applyRoleBotRequestDefaults,
  applyRoutedRoleBot,
  keepContinuingRoleBot,
  roleBotLongTermInjectKb,
  roleBotModeRouterOptions,
  withRoleBotPersonas,
  routerPersonaCandidates,
  roleBotSkillRefs,
} from '../chat-bots.js'
import {
  applyIntentRouting,
  resolveInitialExecutionMode,
  logIntentRouterDecision,
  INTENT_ROUTER_CIRCUIT_BREAKER,
  resolveIntentRouterModel,
  routingLeavesMemoryIntentUnresolved,
} from './chat-intent.js'
import {
  findUnknownChatPersonaIds,
  resolveChatPersonaSelection,
  unknownChatPersonaResponse,
} from '../chat-personas.js'
import { createLogger } from '../../logger.js'
import { buildChatKnowledgeContext } from '../chat-knowledge.js'
import { resolveRequestSurface } from '../request-surface.js'
import { isSafeId } from '../../utils/safe-id.js'
import {
  isExtensionSessionReuseDenied,
  isExtensionWritingModeDenied,
  readChatMemoryScope,
} from '../chat-memory-context.js'
import { resolveScopedFileMemoryReadView } from '../../memory/scoped-file-memory-read-view.js'
import { stopReasonProviderError, stopReasonUserAbort } from '../../agent/stop-reason.js'
import { DEFAULT_LIVE_VIEW_STILLS, createLiveViewTap, type LiveViewTap } from '../../views/live-view-event-tap.js'
import { liveViewRegistry } from '../../views/live-view-registry.js'

const logger = createLogger('route.chat-stream')

export const chatStreamOpenApiOverrides: OpenApiOverrideMap = {
  '/api/v1/chat/stream': {
    post: {
      summary: 'Send message (browser-safe SSE streaming)',
      tags: ['Chat'],
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
        503: { description: 'Service unavailable or agent run capacity exhausted' },
      },
    },
  },
}

type ChatReplayFrame = {
  event: string
  data: unknown
}

function encodeChatStreamCursor(sessionId: string, sequence: number): string {
  return `${sessionId}:${sequence}`
}

function parseChatStreamCursor(
  value: string | undefined,
  sessionId: string,
): { sequence: number } | null {
  if (!value) return null
  const prefix = `${sessionId}:`
  if (!value.startsWith(prefix)) return null
  const sequence = Number.parseInt(value.slice(prefix.length), 10)
  if (!Number.isFinite(sequence) || sequence < 0) return null
  return { sequence }
}

function resolveLastEventId(
  bodyValue: string | undefined,
  headerValue: string | string[] | undefined,
): string | undefined {
  if (bodyValue?.trim()) return bodyValue.trim()
  if (Array.isArray(headerValue)) return headerValue[0]?.trim() || undefined
  return headerValue?.trim() || undefined
}

function parseChatStreamCursorSessionId(value: string | undefined): string | undefined {
  if (!value) return undefined
  const separatorIndex = value.lastIndexOf(':')
  if (separatorIndex <= 0 || separatorIndex === value.length - 1) return undefined
  const sessionId = value.slice(0, separatorIndex)
  const sequence = Number.parseInt(value.slice(separatorIndex + 1), 10)
  if (!Number.isFinite(sequence) || sequence < 0 || !isSafeId(sessionId)) {
    return undefined
  }
  return sessionId
}

function sessionEventToReplayFrame(event: SessionEvent): ChatReplayFrame | null {
  switch (event.type) {
    case 'assistant_message':
      return {
        event: 'message',
        data: { type: 'message', content: event.content },
      }
    case 'session_end':
      return {
        event: 'done',
        data: {
          type: 'done',
          usage: {
            inputTokens: event.totalTokens?.input ?? 0,
            outputTokens: event.totalTokens?.output ?? 0,
          },
        },
      }
    case 'tool_call':
      return {
        event: 'tool_call',
        data: {
          type: 'tool_call',
          toolCall: {
            id: event.id,
            name: event.tool,
            arguments: event.input ?? {},
          },
        },
      }
    case 'tool_result':
      return {
        event: 'tool_result',
        data: {
          type: 'tool_result',
          toolCallId: event.toolCallId,
          output: event.output,
          status: event.status === 'success' ? 'success' : 'error',
        },
      }
    default:
      return null
  }
}

function countReplayableChatFrames(events: readonly SessionEvent[]): number {
  let count = 0
  for (const event of events) {
    if (sessionEventToReplayFrame(event)) {
      count += 1
    }
  }
  return count
}

function isReplayableLiveAgentEvent(
  event: AgentEvent,
): event is Extract<AgentEvent, { type: 'tool_call' | 'tool_result' }> {
  return event.type === 'tool_call' || event.type === 'tool_result'
}

async function replayPersistedChatStream(
  app: FastifyInstance,
  request: FastifyRequest,
  reply: FastifyReply,
  sessionId: string,
  lastEventId: string,
): Promise<void> {
  const cursor = parseChatStreamCursor(lastEventId, sessionId)
  if (!cursor) {
    await reply.status(400).send({
      error: {
        code: 'INVALID_CURSOR',
        message: 'lastEventId does not belong to the requested session.',
      },
    })
    return
  }

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
  const lifecycle = createSseLifecycle({ reply })
  trackSseConnection(app, request, reply, { label: 'chat-stream-resume' })

  try {
    const events = await app.runtime!.sessions.getEvents(sessionId)
    let replaySequence = 0
    for (const event of events) {
      const frame = sessionEventToReplayFrame(event)
      if (!frame) continue
      replaySequence += 1
      if (replaySequence <= cursor.sequence) continue
      await writeSseFrame(reply.raw, frame.event, frame.data, {
        id: encodeChatStreamCursor(sessionId, replaySequence),
      })
    }
    await writeSseFrame(
      reply.raw,
      'close',
      {},
      {
        id: encodeChatStreamCursor(sessionId, replaySequence + 1),
      },
    )
  } finally {
    lifecycle.dispose()
    try {
      if (!reply.raw.writableEnded) reply.raw.end()
    } catch {
    }
  }
}

export async function chatStreamRoutes(app: FastifyInstance) {
  app.post<{ Body: ChatBody }>(
    '/chat/stream',
    {
      preValidation: zodRequestValidation({
        body: {
          schema: chatRequestSchema,
          message: 'Invalid chat request body',
        },
      }),
    },
    async (request, reply) => {
      const runtime = app.runtime
      if (!runtime)
        return reply
          .status(503)
          .send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

      const roleBotRequest = applyRoleBotRequestDefaults(request.body, runtime.roleBots)
      if (roleBotRequest.error) {
        return reply.status(400).send({ error: { code: roleBotRequest.error.code, message: roleBotRequest.error.message, details: { unknownIds: roleBotRequest.error.unknownIds } } })
      }
      const body = roleBotRequest.body
      const explicitRoleBot = roleBotRequest.bot
      const baseScopeTags = readChatMemoryScope(request.headers, request.authContext)

      if (isExtensionWritingModeDenied(request.authContext, body.mode)) {
        return reply.status(403).send({
          error: {
            code: 'EXTENSION_WRITING_ISOLATION_UNSUPPORTED',
            message: 'Extension writing mode is disabled until document ownership is persisted.',
          },
        })
      }

      if (isDesktopExternalAgentMode(body.mode)) {
        return reply.status(400).send({
          error: {
            code: 'UNSUPPORTED_MODE',
            message:
              'External CLI agent modes are desktop-only. Use the Electron desktop app so it can bridge the local CLI through a PTY.',
          },
        })
      }

      const {
        message: rawMessage,
        sessionId: reqSessionId,
        model,
        provider: reqProvider,
        panelStrategy,
        thinkingLevel,
        maxTokens,
        temperature,
        ragEnabled,
        textDeltaMode,
        fileIds,
        projectId,
      } = body

      const attachments = [
        ...(body.attachments ?? []),
        ...(fileIds?.length ? resolveFileIds(fileIds) : []),
      ]
      const sessionAttachments = resolveSessionAttachmentRefs(fileIds ?? [])
      try {
        await assertChatAttachmentLimits(attachments)
      } catch (error) {
        if (error instanceof AttachmentLimitError) {
          return reply.status(error.statusCode).send({
            error: {
              code: error.code,
              message: error.message,
            },
          })
        }
        throw error
      }

      const lastEventId = resolveLastEventId(body.lastEventId, request.headers['last-event-id'])
      const cursorSessionId = parseChatStreamCursorSessionId(lastEventId)
      if (isExtensionSessionReuseDenied(
        request.authContext,
        reqSessionId ?? cursorSessionId,
      )) {
        return reply.status(403).send({
          error: {
            code: 'EXTENSION_SESSION_REUSE_UNSUPPORTED',
            message: 'Extensions must omit sessionId and replay cursors until session ownership is persisted.',
          },
        })
      }
      if (lastEventId && !reqSessionId && !cursorSessionId) {
        return reply.status(400).send({
          error: {
            code: 'INVALID_CURSOR',
            message: 'lastEventId must include a valid session cursor.',
          },
        })
      }
      const sessionId = reqSessionId ?? cursorSessionId ?? randomUUID()
      if (lastEventId) {
        await replayPersistedChatStream(app, request, reply, sessionId, lastEventId)
        return
      }

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
      const existingSession = runtime.sessions?.get ? await runtime.sessions.get(sessionId) : null
      const projectContext = await loadProjectContext(runtime, projectId)
      let requestCwd: string | undefined
      let requestWorkspaceRoot: string | undefined
      let requestWorkspaceIsolation: 'policy' | 'strict'
      try {
        const resolved = await resolveChatRequestWorkspace({
          cwd: body.cwd,
          workspaceRoot: body.workspaceRoot,
          session: existingSession,
          projectDirectory: projectContext?.workingDirectory,
          surface: resolveRequestSurface(request),
        })
        requestWorkspaceIsolation = resolved.workspaceIsolation
        requestCwd = resolved.cwd
        requestWorkspaceRoot = resolved.workspaceRoot
      } catch (error) {
        if (error instanceof InvalidCwdError) {
          return reply.status(400).send(invalidCwdResponse(error))
        }
        throw error
      }

      const autonomyResolution = resolveRequestedAutonomy(runtime.autonomy, body.autonomy)
      const effectiveAutonomy = autonomyResolution.effective
      const {
        SlashWorkspaceBoundaryError,
        tryExpandSlashInput,
      } = await import('../../agent/custom/slash-expand.js')
      const customCommands =
        (await runtime.customDefs
          ?.commandsForCwd(requestCwd, requestWorkspaceRoot)
          .catch(() => [])) ?? []
      const customAgents =
        (await runtime.customDefs
          ?.agentsForCwd(requestCwd, requestWorkspaceRoot)
          .catch(() => [])) ?? []
      const storedPersonas = createPersonaRepo().list()
      const userDefinedPersonaIds = new Set([...storedPersonas.map((persona) => persona.id), ...customAgents.map((agent) => agent.id)])
      const persistedPersonas = withRoleBotPersonas(storedPersonas, runtime.roleBots)
      const personaSelection = resolveChatPersonaSelection(body, existingSession)
      const unknownPersonaIds = findUnknownChatPersonaIds(
        personaSelection,
        (id) => Boolean(resolvePersona(id, customAgents, persistedPersonas)),
      )
      if (unknownPersonaIds.length > 0) {
        return reply.status(400).send(unknownChatPersonaResponse(unknownPersonaIds))
      }
      let memoryContext: ReturnType<typeof resolvePersonaMemoryScope>
      try { memoryContext = resolvePersonaMemoryScope(baseScopeTags, personaSelection, persistedPersonas, existingSession) }
      catch (error) { return reply.status(400).send({ error: { code: 'MEMORY_SPACE_MISMATCH', message: String(error) } }) }
      const { scopeTags, memoryNamespace } = memoryContext
      const resolvedPanelPersonas = resolvePersonas(
        personaSelection.personaIds,
        customAgents,
        persistedPersonas,
      )
      let slashMatch: Awaited<ReturnType<typeof tryExpandSlashInput>> = null
      try {
        slashMatch = await tryExpandSlashInput(rawMessage, {
          commands: customCommands,
          cwd: requestCwd,
          workspaceRoot: requestWorkspaceRoot,
        })
      } catch (error) {
        if (error instanceof SlashWorkspaceBoundaryError) {
          return reply.status(403).send({
            error: {
              code: error.code,
              message: error.message,
            },
          })
        }
      }
      const message = slashMatch?.expanded ?? rawMessage
      const priorRunContractForSelection = reqSessionId
        ? await loadLatestRunContract(runtime.sessions, sessionId)
        : undefined

      const selections = selectChatProviderCandidates({
        runtime,
        routing: body.modelRouting,
        message,
        requestedProvider: reqProvider,
        requestedModel: model,
        existingSession,
        requireVision: body.mode === 'computer-use',
        preferVision: chatRequestPrefersVision(message, body.mode, priorRunContractForSelection),
      })
      const selection = selections[0]
      if (!selection) {
        const failure = buildProviderSelectionFailure(runtime, reqProvider, model, body.modelRouting)
        return reply.status(failure.statusCode).send({ error: failure.error })
      }
      const selectedProvider = selection.provider
      const selectedModel = selection.model

      const declaredSkillToolNames = new Set<string>()
      const autoLoadedSkillToolNames = new Set<string>()
      const loadedExecutionSkillIds = new Set<string>()
      try {
        await resolveSkillRefsContent(
          body.skillRefs,
          runtime.skillRegistry,
          runtime.toolRegistry,
          requestCwd,
          effectiveAutonomy,
          declaredSkillToolNames,
          requestWorkspaceRoot,
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

      let clientClosed = false
      let terminalSessionPersisted = false
      let activeModeRouter: AgentModeRouter | null = null
      let parkedDecision: PendingDecision | null = null
      const preflightAbortController = new AbortController()
      const stopRun = async () => {
        clientClosed = true
        preflightAbortController.abort(new Error('Chat stream cancelled'))
        await activeModeRouter?.stop().catch(() => {})
      }
      registerSseDisconnectHandler(request, reply, stopRun)
      const unregisterRunCanceller =
        runtime.activeRuns?.registerCanceller(sessionId, stopRun) ?? (() => undefined)

      let lifecycle: ReturnType<typeof createSseLifecycle> | null = null
      try {
        const userPromptHook = await runtime.hookRegistry.trigger({
          event: 'pre:user:prompt',
          data: {
            sessionId,
            prompt: message,
            actor: resolveApiActor(request.authContext, 'api-user'),
          },
        })
        if (userPromptHook.action === 'abort') {
          return reply.status(400).send({
            error: {
              code: 'USER_PROMPT_REJECTED',
              message: 'User prompt rejected by hook',
            },
          })
        }
        const rewrittenPrompt = (
          userPromptHook.modifiedPayload?.data as { prompt?: unknown } | undefined
        )?.prompt
        const finalPrompt = typeof rewrittenPrompt === 'string' ? rewrittenPrompt : message
        const presentationAttachmentIntent = [
          finalPrompt,
          ...attachments.map((attachment) => attachment.filename ?? ''),
        ].join('\n')
        const deferPptxTextExtraction = shouldDeferPptxAttachmentText(
          presentationAttachmentIntent,
          body.skillRefs,
          finalPrompt,
        )
        const officeAutoIntent = deferPptxTextExtraction
          ? presentationAttachmentIntent
          : finalPrompt

        let agentInput = finalPrompt
        let requiresNativeMultimodal = false
        const skippedAttachments: SkippedAttachment[] = []
        if (attachments.length > 0) {
          const { buildAttachmentContentParts } = await import('../../media/pipeline.js')
          const { encodeMultimodalInput } = await import('../../agent/multimodal-input.js')
          const { attachmentAllowedRoots } = await import('./file-registry.js')
          const built = await buildAttachmentContentParts(attachments, {
            allowedRoots: attachmentAllowedRoots({ dataDir: runtime.dataDir, cwd: requestCwd }),
            deferPptxTextExtraction,
          })
          for (const skip of built.skipped) {
            skippedAttachments.push(skip)
            logger.warn('attachment_skipped', {
              sessionId,
              source: skip.source,
              error: skip.reason,
            })
          }
          const parts: ContentPart[] = [{ type: 'text', text: finalPrompt }, ...built.parts]
          requiresNativeMultimodal = requiresNativeMultimodalInput(parts)
          agentInput = encodeMultimodalInput(parts)
        }

        const sessionWorkspace = requestWorkspaceRoot ?? requestCwd
        let session = existingSession
        if (!session) {
          session = await runtime.sessions.create({
            id: sessionId,
            title: fallbackSessionTitle(message),
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
            provider: selectedProvider.id,
            model: selectedModel,
            device: runtime.config.device.name,
            status: 'active',
            cwd: sessionWorkspace,
            workspaceIsolation: requestWorkspaceIsolation,
            tags: body.tags ?? [],
            personaIds: personaSelection.personaIds ?? (personaSelection.persona ? [personaSelection.persona] : undefined),
            memoryNamespace,
          })
        } else {
          const patch: {
            cwd?: string
            workspaceIsolation?: 'policy' | 'strict'
            memoryNamespace?: string
            personaIds?: string[]
            status?: 'active'
          } = {}
          if (memoryNamespace) patch.memoryNamespace = memoryNamespace
          if (sessionWorkspace && session.cwd !== sessionWorkspace) patch.cwd = sessionWorkspace
          if (!session.cwd && sessionWorkspace) patch.workspaceIsolation = requestWorkspaceIsolation
          if (body.personaIds !== undefined || body.persona) patch.personaIds = body.personaIds ?? [body.persona!]
          if (session.status !== 'active') patch.status = 'active'
          if (Object.keys(patch).length > 0) {
            session = (await runtime.sessions.updateMeta?.(session.id, patch)) ?? session
          }
        }
        const previousContext = await loadAutoCompactedSessionContextWithinBudget({
          sessionStore: runtime.sessions,
          sessionId,
          provider: selectedProvider,
          model: selectedModel,
          hooks: runtime.hookRegistry,
          maxMessages: resolveSessionContextMaxMessages(),
        })
        const previousMessages = previousContext.messages
        const previousRunContract = previousContext.runContract
        const compactionNotice = formatContextCompactionNotice(previousContext)

        const messageId = randomUUID()
        await runtime.sessions.appendEvent(sessionId, {
          type: 'user_message',
          id: messageId,
          timestamp: new Date().toISOString(),
          content: finalPrompt,
          ...(sessionAttachments.length > 0 ? { attachments: sessionAttachments } : {}),
        })
        const userImageArtifacts = await extractAndStoreUserImageArtifacts(
          runtime,
          sessionId,
          finalPrompt,
        )

        const auxiliaryLlmBudget = createAuxiliaryLlmTurnBudget()
        const routerCfg = runtime.config.agent.intentRouter
        let intentRouter: IntentRouter | null = null
        if (routerCfg?.enabled) {
          try {
            const routerProvider = routerCfg.provider
              ? runtime.providerRegistry.get(routerCfg.provider)
              : selectedProvider
            const routerModel = resolveIntentRouterModel({
              routerProvider,
              configuredModel: routerCfg.model,
              configuredProviderId: routerCfg.provider,
              activeModel: routerProvider?.id === selectedProvider.id
                ? selectedModel
                : undefined,
              logger,
            })
            if (routerProvider && routerModel) {
              intentRouter = new IntentRouter({
                provider: routerProvider,
                model: routerModel,
                graphRegistry: runtime.graphRegistry,
                skillRegistry: runtime.skillRegistry,
                personaList: () => routerPersonaCandidates(
                  resolvePersonaCatalog(customAgents, persistedPersonas),
                  runtime.roleBots,
                  userDefinedPersonaIds,
                ),
                currentAutonomy: effectiveAutonomy,
                cwd: requestCwd,
                workspaceRoot: requestWorkspaceRoot,
                timeoutMs: routerCfg.timeoutMs,
                maxPreviousMessages: routerCfg.maxPreviousMessages,
                perMessageCharLimit: routerCfg.perMessageCharLimit,
                reasonMaxChars: routerCfg.reasonMaxChars,
                logger: logger as unknown as IntentRouterOptions['logger'],
                circuitBreaker: INTENT_ROUTER_CIRCUIT_BREAKER,
                auxiliaryLlmBudget,
                signal: preflightAbortController.signal,
                activeRunContract: previousContext.runContract,
                availableToolNames: runtime.toolRegistry.list().map((tool) => tool.name),
              })
            }
          } catch (err) {
            logger.warn('intent_router.init_failed', { err: (err as Error).message })
          }
        }

        const panelLocked =
          body.mode === 'persona-panel' && (personaSelection.personaIds?.length ?? 0) > 0
        const personaLocked = panelLocked || Boolean(personaSelection.persona)
        const effectiveIntentRouting = personaLocked ? { enabled: false } : body.intentRouting
        const routingMode = body.mode

        const initialRouting = await applyIntentRouting({
          defaultMode: runtime.config.agent.mode,
          router: intentRouter,
          graphRegistry: runtime.graphRegistry,
          message: finalPrompt,
          prevMessages: previousMessages,
          body: {
            mode: routingMode,
            persona: personaSelection.persona,
            skillRefs: body.skillRefs,
            intentRouting: effectiveIntentRouting,
          },
          isMultimodal: requiresNativeMultimodal,
        })
        const continuingRouting = initialRouting.decision.contractRelation === 'continue'
          ? keepContinuingRoleBot(
              initialRouting,
              await loadLatestRoutedPersona(runtime.sessions, sessionId).catch(() => undefined),
              runtime.roleBots,
            )
          : initialRouting
        const { routing, bot: routedRoleBot } = applyRoutedRoleBot(continuingRouting, runtime.roleBots)
        const turnRoleBot = explicitRoleBot ?? routedRoleBot

        if (clientClosed || reply.raw.destroyed) return reply

const memoryPolicy = resolveTurnMemoryPolicy(body.memoryPolicy, routing.decision.memoryPolicy, routingLeavesMemoryIntentUnresolved(routing.decision))
                await runtime.sessions.appendEvent(sessionId, {
          type: 'router_decision',
          id: randomUUID(),
          timestamp: new Date().toISOString(),
          decision: {
            memoryPolicy,
            mode: routing.decision.mode,
            persona: routing.decision.persona,
            skillIds: routing.decision.skillIds,
            toolGroups: routing.decision.toolGroups,
            reason: routing.decision.reason,
            confidence: routing.decision.confidence,
            fallback: routing.decision.fallback,
            ...(routing.decision.skipped ? { skipped: true } : {}),
          },
        })

        logIntentRouterDecision(
          logger,
          routing,
          {
            mode: body.mode,
            persona: personaSelection.persona,
            skillCount: body.skillRefs?.length ?? 0,
          },
          finalPrompt.length,
          sessionId,
          routerCfg?.model,
        )

        const botSkillRefs = await filterLoadableSkillRefs(
          roleBotSkillRefs(turnRoleBot, routing.effectiveSkillRefs),
          runtime.skillRegistry, requestCwd, effectiveAutonomy, requestWorkspaceRoot,
          routing.effectiveSkillRefs.length,
        )
        let skillPrefix = ''
        let botSkillPrefix = ''
        let officeAutoSkillContent = ''
        let imageGenSkillContent = ''
        try {
          skillPrefix = await resolveSkillRefsContent(
            routing.effectiveSkillRefs,
            runtime.skillRegistry,
            runtime.toolRegistry,
            requestCwd,
            effectiveAutonomy,
            declaredSkillToolNames,
            requestWorkspaceRoot,
            loadedExecutionSkillIds,
          )
          botSkillPrefix = await resolveSkillRefsContent(
            botSkillRefs,
            runtime.skillRegistry,
            runtime.toolRegistry,
            requestCwd,
            effectiveAutonomy,
            autoLoadedSkillToolNames,
            requestWorkspaceRoot,
            loadedExecutionSkillIds,
          )
          officeAutoSkillContent = resolveInitialExecutionMode(routing, runtime.config.agent.mode, false, resolveRequestSurface(request) ?? undefined) === 'instant' ? '' : await resolveOfficeAutoSkillContent(
            officeAutoIntent,
            routing.effectiveSkillRefs,
            runtime.skillRegistry,
            runtime.toolRegistry,
            requestCwd,
            effectiveAutonomy,
            requestWorkspaceRoot,
            autoLoadedSkillToolNames,
            previousMessages,
            loadedExecutionSkillIds,
            finalPrompt,
            turnRoleBot ? turnRoleBot.skills ?? [] : undefined,
          )
          imageGenSkillContent = await resolveImageGeneratorAutoSkillContent(
            message,
            body.imageGenEnabled,
            routing.effectiveSkillRefs,
            runtime.skillRegistry,
            runtime.toolRegistry,
            requestCwd,
            effectiveAutonomy,
            autoLoadedSkillToolNames,
            requestWorkspaceRoot,
            loadedExecutionSkillIds,
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

        const personaObj = resolvePersona(
          routing.effectivePersona,
          customAgents,
          persistedPersonas,
        )
        const agentsFiles = await collectAgentsMd({
          cwd: requestCwd,
          includeUserLevel: !excludesLocalContext(body),
          boundaryRoot: requestWorkspaceRoot,
        })
        const agentsMemory = formatAgentsMdSection(agentsFiles) ?? undefined
        const relevantKnowledgeContext = memoryPolicy.read && memoryPolicy.automaticRead !== false ? await buildChatKnowledgeContext({
          registry: app.chatKnowledgeProviders,
          message: finalPrompt,
          previousMessages,
          allowLocalKnowledge: request.authContext?.kind !== 'extension' && !excludesLocalContext(body),
        }) : undefined
        const directApiToolRegistry = withWritingDocTools(
          withSwarmToolsForSession(
            withDirectApiMemoryToolset(
              runtime.toolRegistry,
              runtime.config.memory.directApiToolset,
            ),
            sessionId,
          ),
          body.writingDocId ?? getDocRegistry().getActiveId(),
        )
        const activeRemoteBrowser = remoteBrowserBridge(runtime.toolRegistry).list().some(
          (connection) => connection.sessionId === sessionId,
        )
        const initialMode = resolveInitialExecutionMode(routing, runtime.config.agent.mode, activeRemoteBrowser, resolveRequestSurface(request) ?? undefined)
        const turnToolAuthority = withTurnToolAuthority(directApiToolRegistry, {
          explicitToolNames: (body.toolNames ?? directApiToolRegistry.list().map(tool => tool.name)).filter(name => memoryToolAllowed(name, memoryPolicy)), declaredSkillToolNames,
          personaAllowedTools: personaObj?.allowedTools, personaDeniedTools: personaObj?.deniedTools,
        })
        const delegationToolRegistry = withSelectedBrowserTarget(turnToolAuthority.delegation, activeRemoteBrowser)
        const authorizedToolRegistry = withSelectedBrowserTarget(turnToolAuthority.own, activeRemoteBrowser)
        const agentToolRegistry = withContextualToolExposure(authorizedToolRegistry, {
          previousMessages,
          surface: resolveRequestSurface(request),
          semanticRouting: true,
          activeRemoteBrowser,
          selectedGroups: routing.decision.toolGroups,
          botToolGroups: turnRoleBot?.toolGroups,
          routedMode: initialMode,
          routingConfidence: routing.decision.confidence,
          routingFallback: routing.decision.fallback,
          activeWritingDocument: Boolean(body.writingDocId ?? getDocRegistry().getActiveId()),
          swarmSession: sessionId.startsWith('swarm_'),
          supplementalToolNames: autoLoadedSkillToolNames,
          requestInput: finalPrompt,
          explicitToolNames: resolveInstantModeToolNames(
            finalPrompt,
            initialMode,
            body.toolNames,
            routing.decision.executionIntent,
            routing.decision.mode,
            activeRemoteBrowser,
          ),
        })
        const closedToolSurface = body.toolNames !== undefined || declaredSkillToolNames.size > 0
        const { prompt: baseSystemPrompt, systemToolGuidance } = await buildDeferredSystemPrompt({
          config: runtime.config,
          profile: initialMode === 'instant' ? 'instant' : 'agent',
          longTermInjectKb: roleBotLongTermInjectKb(turnRoleBot),
          skillInventory: turnRoleBot ? (turnRoleBot.skills ?? []) : undefined,
          tools: agentToolRegistry,
          skills: runtime.skillRegistry,
          fileMemory: !excludesLocalContext(body) && memoryPolicy.read && memoryPolicy.automaticRead !== false ? resolveScopedFileMemoryReadView(
            runtime.fileMemory,
            runtime.fileMemoryRegistry,
            scopeTags,
          ) : undefined,
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
        let strictFinalAnswerProtocol = process.env.SEPILOTD_STRICT_ANSWER_PROTOCOL === '1'
        const docPrefix = buildWritingDocPrefix(
          body.mode,
          body.writingDocId,
          requestWorkspaceRoot,
        )
        const shellPrefix = buildShellModePrefix(body.mode)
        const systemPrompt = () => shellPrefix + docPrefix + skillPrefix + botSkillPrefix + baseSystemPrompt
        const writingDocStart = snapshotWritingDocStart(
          body.mode,
          body.writingDocId,
          requestWorkspaceRoot,
        )

        const approvalCallback: ApprovalCallback = (toolCall, requestId, options) =>
          runtime.approvalRegistry.waitForApproval({
            sessionId,
            toolCall,
            requestId,
            runId: messageId,
            recheckPolicy: options?.recheckPolicy,
            forcePrompt: options?.forcePrompt,
            signal: options?.signal,
            onParked: (approval) => {
              parkedDecision = {
                kind: 'approval',
                id: approval.requestId,
                label: approval.tool,
                since: Date.parse(approval.requestedAt),
              }
              void activeModeRouter?.stop().catch(() => {})
            },
          })
        let emitQuestionRequestEvent: (event: AgentEvent) => Promise<void> = async () => undefined
        const requestQuestion = createQuestionRequester(runtime.questions, async (question) => {
          await emitQuestionRequestEvent({
            type: 'question_request',
            questionId: question.id,
            prompt: question.prompt,
            choices: question.choices,
          })
        })

        const createModeRouter = (nextSelection: typeof selection) =>
          new AgentModeRouter({
            systemToolGuidance,
            provider: nextSelection.provider,
            tools: agentToolRegistry,
            authorizedTools: authorizedToolRegistry,
            delegationTools: delegationToolRegistry,
            activeRemoteBrowser,
            policy: runtime.policyEngine,
            autonomy: effectiveAutonomy,
            auxiliaryLlmBudget,
            semanticIndex: excludesLocalContext(body) ? undefined : runtime.semanticIndex,
            systemPrompt: systemPrompt(),
            previousMessages,
            maxIterations: resolveChatMaxIterations({ ...body, maxIterations: body.maxIterations ?? turnRoleBot?.maxIterations }),
            ...roleBotModeRouterOptions(turnRoleBot),
            hardMaxIterations: resolveChatHardMaxIterations(body),
            auditLogger: runtime.auditLogger,
            usageTracker: runtime.usageTracker,
            spendBudget: runtime.config.limits,
            hookRegistry: runtime.hookRegistry,
            deviceName: runtime.config.device.name,
            thinkingLevel: thinkingLevel ?? runtime.config.agent.thinkingLevel,
            thinkingLevelSource: thinkingLevel === undefined ? 'default' : 'request',
            maxTokens,
            temperature,
            textDeltaMode,
            ragEnabled: ragEnabled ?? turnRoleBot?.documents,
            panelPersonas: resolvedPanelPersonas,
            panelStrategy,
            providerCircuitBreaker: runtime.providerCircuitBreaker,
            defaultMode: runtime.config.agent.mode,
            graphRegistry: runtime.graphRegistry,
            intentModeHint: {
              mode: routing.decision.mode,
              confidence: routing.decision.confidence,
              fallback: routing.decision.fallback,
              executionIntent: routing.decision.executionIntent,
              contractRelation: routing.decision.contractRelation,
            },
            approvalCallback,
            evaluateAutoApproval: (toolCall) =>
              runtime.approvalRegistry.tryAutoApproval({ sessionId, toolCall, runId: messageId }),
            requestQuestion,
            persistTransportPreference: async (sid, preferPromptReact) => {
              await runtime.sessions.updateMeta?.(sid, { preferPromptReact })
            },
            ...createRuntimeBackedModeRouterOptions(runtime),
            strictFinalAnswerProtocol,
            reviewToollessFinals: true,
          })

        if (clientClosed || reply.raw.destroyed) {
          return reply
        }
        trackSseConnection(app, request, reply, { label: 'chat-stream' })

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

        let streamEventSequence = countReplayableChatFrames(
          await runtime.sessions.getEvents(sessionId),
        )
        const nextReplayableCursor = () => {
          streamEventSequence += 1
          return encodeChatStreamCursor(sessionId, streamEventSequence)
        }
        const send = async (event: string, data: unknown, options: { id?: string } = {}) => {
          if (clientClosed || reply.raw.writableEnded) {
            return
          }
          try {
            await writeSseFrame(reply.raw, event, data, options.id ? { id: options.id } : undefined)
          } catch (error) {
            if (isStreamWriteStoppedError(error)) {
              clientClosed = true
              void activeModeRouter?.stop().catch(() => {})
              return
            }
            throw error
          }
        }

        const AGENT_INACTIVITY_MS = resolveAgentInactivityMs()
        const inactivityProbe = createAgentInactivityProbe()
        lifecycle = createSseLifecycle({
          reply,
          inactivityMs: AGENT_INACTIVITY_MS,
          pendingDecision: () => inactivityProbe.pendingDecision(),
          decisionTimeoutMs: resolvePendingDecisionTimeoutMs(),
          onInactivity: () => {
            void activeModeRouter?.stop().catch(() => {})
          },
          onDecisionTimeout: () => {
            void activeModeRouter?.stop().catch(() => {})
          },
        })

        await send('session', { sessionId })
        if (skippedAttachments.length > 0) {
          await send('warning', {
            type: 'warning',
            code: 'ATTACHMENT_SKIPPED',
            message: `${skippedAttachments.length} attachment(s) were skipped: ${skippedAttachments
              .map((skip) => skip.reason)
              .join('; ')}`,
            skipped: skippedAttachments,
          })
        }
        if (userImageArtifacts.length > 0) {
          await send('artifacts', { artifacts: userImageArtifacts })
        }
        await send('router_decision', {
          type: 'router_decision',
          id: randomUUID(),
          decision: routing.decision,
        })
        const skillsLoaded = buildSkillsLoaded({
          strictPrefix: skillPrefix,
          callerSkillNames: (body.skillRefs ?? []).map((ref) => ref.name),
          botPrefix: botSkillPrefix,
          autoPrefix: officeAutoSkillContent + imageGenSkillContent,
        })
        if (skillsLoaded) await send('skills_loaded', { type: 'skills_loaded', skills: skillsLoaded })

        skillPrefix += officeAutoSkillContent
        skillPrefix += imageGenSkillContent
        strictFinalAnswerProtocol =
          process.env.SEPILOTD_STRICT_ANSWER_PROTOCOL === '1' || skillPrefix.trim().length > 0

        const requestMode = routing.effectiveMode as AgentMode | undefined
        const outputTracker = createAgentStreamOutputTracker()
        let activeSelection = selection
        let activeAttempt = 1
        let terminalDoneEvent: Extract<AgentEvent, { type: 'done' }> | null = null
        let terminalStateChangeEvent: Extract<AgentEvent, { type: 'state_change' }> | null = null
        let providerError: ApiError | null = null

        const taskId = randomUUID()
        const taskStartedAt = Date.now()
        const requestSurface = resolveRequestSurface(request)
        const notifyChatCompletion = (
          outcome: 'completed' | 'failed',
          errorMessage?: string,
        ): void => {
          try {
            publishChatCompletionNotification(runtime.config, {
              outcome,
              sessionId,
              sessionTitle: session?.title,
              durationMs: Date.now() - taskStartedAt,
              surface: requestSurface,
              errorMessage,
            })
          } catch {
          }
        }
        const recordTaskEvent = (
          eventType: 'task.started' | 'task.completed' | 'task.failed',
          severity: 'info' | 'error',
          attributes?: Record<string, unknown>,
        ) => {
          try {
            runtime.observability?.recordEvents([
              {
                source: 'daemon',
                surface: 'chat-stream',
                eventType,
                severity,
                privacy: 'operational',
                sessionId,
                taskId,
                provider: activeSelection.provider.id,
                model: activeSelection.model,
                attributes: {
                  mode: requestMode ?? null,
                  ...attributes,
                },
              },
            ])
          } catch {
          }
        }
        recordTaskEvent('task.started', 'info')
        const noteEvent = (event: AgentEvent) => {
          if (isSubstantiveAgentActivity(event)) {
            lifecycle?.recordEvent()
            inactivityProbe.note(event)
          }
        }
        const emitEvent = async (event: AgentEvent) => {
          outputTracker.consume(event)
          if (event.type === 'done') {
            terminalDoneEvent = event
            return
          }
          if (event.type === 'state_change' && event.state === 'done') {
            terminalStateChangeEvent = event
            return
          }
          if (event.type === 'text_delta' && textDeltaMode !== 'live') {
            return
          }
          if (event.type === 'error') {
            providerError = event.error
            return
          }
          if (isReplayableLiveAgentEvent(event)) {
            await persistAgentSessionEvent(runtime.sessions, sessionId, event)
            await send(event.type, event, { id: nextReplayableCursor() })
            return
          }
          if (event.type !== 'message') {
            await send(event.type, event)
            await persistAgentSessionEvent(runtime.sessions, sessionId, event)
          }
        }
        emitQuestionRequestEvent = async (event: AgentEvent) => {
          noteEvent(event)
          await emitEvent(event)
        }
        if (compactionNotice) {
          await emitEvent({ type: 'thinking', content: compactionNotice })
        }
        let liveViewTap: LiveViewTap | undefined
        try {
          for (let attemptIndex = 0; attemptIndex < selections.length; attemptIndex += 1) {
            const attemptSelection = selections[attemptIndex]!
            const nextSelection = selections[attemptIndex + 1]
            const attempt = attemptIndex + 1
            const modeRouter = createModeRouter(attemptSelection)
            const buffer: AgentEvent[] = []
            let committed = false
            let retrying = false
            activeModeRouter = modeRouter
            activeAttempt = attempt
            if (clientClosed) {
              await modeRouter.stop().catch(() => {})
              break
            }

            await appendChatProviderAttemptEvent(
              runtime.sessions,
              sessionId,
              attemptSelection,
              attempt,
              'started',
            )

            const attemptViewTap = createLiveViewTap({
              views: liveViewRegistry(runtime.toolRegistry),
              sessionId,
              stills: runtime.config.views?.stills ?? DEFAULT_LIVE_VIEW_STILLS,
            })
            liveViewTap = attemptViewTap
            for await (const event of attemptViewTap.wrap(modeRouter.run(
              agentInput,
              {
                sessionId,
                provider: attemptSelection.provider.id,
                model: attemptSelection.model,
                cwd: requestCwd,
                workspaceRoot: requestWorkspaceRoot,
                workspaceIsolation: requestWorkspaceIsolation,
                executionPolicy: {
                  requestedAutonomy: autonomyResolution.requested,
                  configuredAutonomy: autonomyResolution.configured,
                  effectiveAutonomy: autonomyResolution.effective,
                  clamped: autonomyResolution.clamped,
                  clampReason: autonomyResolution.reason,
                  agentMode: requestMode ?? runtime.config.agent.mode ?? 'auto',
                  primaryAgentId: runtime.primaryAgents?.get(sessionId),
                  workspaceBoundary: requestWorkspaceRoot ? 'strict' : 'unrestricted',
                  freshApprovalRequired: body.requireToolApproval === true,
                },
                writingDocId: body.mode === 'writing' ? body.writingDocId : undefined,
                systemPrompt: systemPrompt(),
                previousMessages,
                toolAllowlist: authorizedToolRegistry.list().map((tool) => tool.name),
                delegationToolAllowlist: delegationToolRegistry.list().map((tool) => tool.name),
                ...resolveSkillExecutionContext(
                  loadedExecutionSkillIds,
                  new Set([...declaredSkillToolNames, ...autoLoadedSkillToolNames]),
                ),
                runContract: previousRunContract,
                runContractScope: 'previous-turn',
                deadlineAt: body.deadlineAt,
                  completionRequest: body.completionRequest,
                memoryPolicy,
                memoryQuery: finalPrompt,
                scopeTags,
                primaryAgentId: runtime.primaryAgents?.get(sessionId),
                autoApprove: runtime.autoApprove,
                requireToolApproval: body.requireToolApproval,
                preferPromptReact: existingSession?.preferPromptReact,
                ...(requestSurface ? { surface: requestSurface } : {}),
                providerAttempt: { attempt, isFallbackPending: () => retrying },
                ...(turnRoleBot?.id ? { roleBotId: turnRoleBot.id } : {}),
                ...(body.toolNames !== undefined ? { requestedToolNames: body.toolNames } : {}),
                requestAuthorityComplete: true,
              },
              requestMode,
            ))) {
              if (clientClosed) {
                break
              }
              noteEvent(event)

              if (
                event.type === 'error' &&
                nextSelection &&
                canRetryChatProviderAttempt(attemptSelection, event.error, committed, true)
              ) {
                recordChatProviderFailure(runtime, attemptSelection, event.error)
                await appendChatProviderAttemptEvent(
                  runtime.sessions,
                  sessionId,
                  attemptSelection,
                  attempt,
                  'failed',
                  {
                    error: event.error,
                    retryable: true,
                    nextSelection,
                  },
                )
                const retryEvent = providerRetryThinkingEvent(
                  attemptSelection,
                  nextSelection,
                  event.error,
                )
                noteEvent(retryEvent)
                await emitEvent(retryEvent)
                retrying = true
                break
              }

              if (!committed) {
                buffer.push(event)
                if (!isAgentAttemptCommitted(event)) {
                  continue
                }
                committed = true
                activeSelection = attemptSelection
                activeAttempt = attempt
                for (const bufferedEvent of buffer) {
                  await emitEvent(bufferedEvent)
                }
                buffer.length = 0
                continue
              }

              await emitEvent(event)
            }

            if (activeModeRouter === modeRouter) {
              activeModeRouter = null
            }
            if (clientClosed) {
              break
            }
            if (retrying) {
              continue
            }
            if (!committed) {
              activeSelection = attemptSelection
              activeAttempt = attempt
              for (const bufferedEvent of buffer) {
                await emitEvent(bufferedEvent)
              }
            }
            break
          }
          if (lifecycle.isInactivityTripped() || parkedDecision) {
            throw new Error(
              parkedDecision
                ? 'Agent run parked on a pending approval.'
                : 'Agent run stopped after an inactivity timeout.',
            )
          }
          if (providerError) {
            recordChatProviderFailure(runtime, activeSelection, providerError)
            await appendChatProviderAttemptEvent(
              runtime.sessions,
              sessionId,
              activeSelection,
              activeAttempt,
              'failed',
              { error: providerError, retryable: false },
            )
          } else if (terminalDoneEvent) {
            recordChatProviderSuccess(runtime, activeSelection)
            await appendChatProviderAttemptEvent(
              runtime.sessions,
              sessionId,
              activeSelection,
              activeAttempt,
              'succeeded',
            )
            await updateChatSessionProviderMeta(runtime.sessions, sessionId, activeSelection)
            recordTaskEvent('task.completed', 'info', {
              durationMs: Date.now() - taskStartedAt,
              attempts: activeAttempt,
            })
            notifyChatCompletion('completed')
          }

          const finalProviderError = providerError as ApiError | null
          const retainedEvidenceIncomplete = finalProviderError && !terminalDoneEvent
            ? outputTracker.providerFailureContent(finalProviderError)
            : null
          if (finalProviderError && !terminalDoneEvent) {
            recordTaskEvent('task.failed', 'error', {
              durationMs: Date.now() - taskStartedAt,
              attempts: activeAttempt,
              code: finalProviderError.code ?? null,
            })
            notifyChatCompletion('failed', finalProviderError.message ?? finalProviderError.code)
            if (retainedEvidenceIncomplete) {
              const recoveryEvent: Extract<AgentEvent, { type: 'recovery' }> = {
                type: 'recovery',
                scope: 'output_synthesis',
                kind: 'provider_failure_with_retained_evidence',
                action: 'synthesize_from_retained_evidence',
                message: 'The provider failed after successful tool execution; closing the stream with a deterministic incomplete response from retained evidence.',
                recoverable: true,
                details: {
                  provider: activeSelection.provider.id,
                  model: activeSelection.model,
                  errorCode: finalProviderError.code,
                },
              }
              await send(recoveryEvent.type, recoveryEvent)
              await persistAgentSessionEvent(runtime.sessions, sessionId, recoveryEvent)
            } else {
              await send('error', {
                type: 'error',
                error: finalProviderError,
              })
            }
          }

          const finalContent = retainedEvidenceIncomplete ?? (
            finalProviderError && !terminalDoneEvent
              ? ''
              : outputTracker.finalContent()
          )
          let assistantChatContent = finalContent
          let finalMessageEvent: Extract<AgentEvent, { type: 'message' }> | null = null
          const doneEvent = (
            terminalDoneEvent
            ?? (finalProviderError
              ? {
                  type: 'done' as const,
                  usage: { inputTokens: 0, outputTokens: 0 },
                  stopReason: stopReasonProviderError({
                    layer: finalProviderError.code,
                    summary: finalProviderError.message,
                  }),
                }
              : null)
          ) as Extract<AgentEvent, { type: 'done' }> | null
          const stateChangeEvent = terminalStateChangeEvent as Extract<
            AgentEvent,
            { type: 'state_change' }
          > | null
          let finalArtifacts: Awaited<ReturnType<typeof extractAndStoreArtifacts>> = []
          if (!clientClosed && finalContent) {
            const fallback = applyWritingDocFallback(body.mode, writingDocStart, finalContent)
            assistantChatContent = fallback.chatContent ?? finalContent

            finalArtifacts = await extractAndStoreArtifacts(
              runtime,
              sessionId,
              assistantChatContent,
            )

            const liveViewRecords = liveViewTap?.records() ?? []
            await runtime.sessions.appendEvent(sessionId, {
              type: 'assistant_message',
              id: randomUUID(),
              timestamp: new Date().toISOString(),
              content: assistantChatContent,
              ...(liveViewRecords.length > 0 ? { views: liveViewRecords } : {}),
            })
            if (!finalProviderError && doneEvent && (!doneEvent.stopReason || doneEvent.stopReason.kind === 'completed')) {
              app.sessionTitleUpdates?.schedule({
                sessions: runtime.sessions,
                session,
                provider: activeSelection.provider,
                model: activeSelection.model,
                firstMessage: finalPrompt,
                assistantReply: assistantChatContent,
              })
              if (!excludesLocalContext(body)) triggerDreamingTurn(runtime.dreaming, sessionId, 'chat-stream', scopeTags)
            }
            finalMessageEvent = {
              type: 'message',
              content: assistantChatContent,
            }
          }

          if (
            !clientClosed
            && finalMessageEvent
            && textDeltaMode !== 'live'
          ) {
            const finalTextDelta: AgentEvent = {
              type: 'text_delta',
              text: assistantChatContent,
            }
            outputTracker.consume(finalTextDelta)
            await send(finalTextDelta.type, finalTextDelta)
          }

          if (!clientClosed && finalArtifacts.length > 0) {
            await send('artifacts', { artifacts: finalArtifacts })
          }

          if (!clientClosed && finalMessageEvent) {
            await send(finalMessageEvent.type, finalMessageEvent, {
              id: nextReplayableCursor(),
            })
          }

          if (!clientClosed && doneEvent && stateChangeEvent) {
            await send(stateChangeEvent.type, stateChangeEvent)
            await persistAgentSessionEvent(runtime.sessions, sessionId, stateChangeEvent)
          }

          if (!clientClosed && doneEvent) {
            await persistAgentSessionEvent(runtime.sessions, sessionId, doneEvent)
            if (finalProviderError && !terminalDoneEvent && !retainedEvidenceIncomplete) {
              await runtime.sessions.updateMeta?.(sessionId, { status: 'abandoned' })
            }
            terminalSessionPersisted = true
            await send(doneEvent.type, doneEvent, {
              id: nextReplayableCursor(),
            })
          }

          if (!clientClosed) {
            await send(
              'close',
              {},
              {
                id: encodeChatStreamCursor(sessionId, streamEventSequence + 1),
              },
            )
            reply.raw.end()
          }
        } catch (err) {
          recordTaskEvent('task.failed', 'error', {
            durationMs: Date.now() - taskStartedAt,
            attempts: activeAttempt,
            code: lifecycle.isInactivityTripped() || parkedDecision ? 'TIMEOUT' : 'INTERNAL_ERROR',
            message: err instanceof Error ? err.message : String(err),
          })
          notifyChatCompletion('failed', err instanceof Error ? err.message : String(err))
          if (lifecycle.isInactivityTripped() || parkedDecision) {
            try {
              if (!reply.raw.writableEnded) {
                const parked = parkedDecision as PendingDecision | null
                const timeoutFrame = parked
                  ? describePendingDecisionTimeout({
                      decisionTimeoutMs: resolvePendingDecisionTimeoutMs(),
                      pending: parked,
                      provider: selectedProvider.id,
                      model: selectedModel,
                    })
                  : inactivityProbe.describe({
                      inactivityMs: AGENT_INACTIVITY_MS,
                      provider: selectedProvider.id,
                      model: selectedModel,
                    })
                await send('error', {
                  type: 'error',
                  error: timeoutFrame,
                })
                await appendChatProviderAttemptEvent(
                  runtime.sessions,
                  sessionId,
                  activeSelection,
                  activeAttempt,
                  'failed',
                  {
                    error: {
                      code: 'TIMEOUT',
                      message: parked
                        ? 'Agent run parked on a pending approval.'
                        : 'Agent run stopped after an inactivity timeout.',
                    },
                    retryable: false,
                  },
                )
                const failedDoneEvent: Extract<AgentEvent, { type: 'done' }> = {
                  type: 'done',
                  usage: { inputTokens: 0, outputTokens: 0 },
                  stopReason: timeoutFrame.stopReason,
                }
                await persistAgentSessionEvent(runtime.sessions, sessionId, failedDoneEvent)
                if (!parked) {
                  await runtime.sessions.updateMeta?.(sessionId, { status: 'abandoned' })
                }
                terminalSessionPersisted = true
                await send(failedDoneEvent.type, failedDoneEvent, {
                  id: nextReplayableCursor(),
                })
                reply.raw.end()
              }
            } catch {
            }
            return
          }
          throw err
        }
      } catch (err) {
        try {
          if (!reply.raw.writableEnded) {
            if (!reply.raw.headersSent) {
              try {
                reply.hijack()
              } catch {
              }
              reply.raw.writeHead(
                200,
                buildSseResponseHeaders(request, {
                  'Content-Type': 'text/event-stream',
                  'Cache-Control': 'no-cache',
                  Connection: 'keep-alive',
                  'X-Request-ID': request.requestId ?? randomUUID(),
                }),
              )
            }
            await writeSseFrame(reply.raw, 'error', {
              type: 'error',
              error: {
                code: 'INTERNAL_ERROR',
                message: err instanceof Error ? err.message : String(err),
              },
            })
          }
        } catch {
        }
        try {
          const failedSession = await runtime.sessions.get(sessionId)
          if (failedSession?.status === 'active') {
            await persistAgentSessionEvent(runtime.sessions, sessionId, {
              type: 'done',
              usage: { inputTokens: 0, outputTokens: 0 },
              stopReason: request.raw.aborted || clientClosed
                ? stopReasonUserAbort()
                : stopReasonProviderError({
                    layer: 'INTERNAL_ERROR',
                    summary: err instanceof Error ? err.message : String(err),
                  }),
            })
            await runtime.sessions.updateMeta?.(sessionId, { status: 'abandoned' })
            terminalSessionPersisted = true
          }
        } catch {
        }
        throw err
      } finally {
        lifecycle?.dispose()
        unregisterRunCanceller()
        if (clientClosed && !terminalSessionPersisted) {
          try {
            await runtime.sessions.updateMeta?.(sessionId, { status: 'abandoned' })
          } catch {
          }
        }
        try {
          if (!reply.raw.writableEnded) reply.raw.end()
        } catch {
        }
        runLease?.release()
        sessionLeaseLifecycle.releaseRun()
      }
    },
  )
}
