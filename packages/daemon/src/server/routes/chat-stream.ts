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
  loadAutoCompactedSessionContext,
  loadLatestRunContract,
  resolveSessionContextMaxMessages,
} from '../../agent/auto-compaction.js'
import { fallbackSessionTitle, updateSessionTitleFromFirstTurn } from '../../agent/session-title.js'
import { triggerDreamingTurn } from '../../memory/dreaming.js'
import { publishChatCompletionNotification } from '../../notifications/publish.js'
import { buildSystemPrompt } from '../../agent/system-prompt.js'
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
  withContextualToolExposure, withAuthorizedToolExposure,
  withDirectApiMemoryToolset,
  withSwarmToolsForSession,
  withWritingDocTools,
} from '../../tools/role-filter.js'
import {
  chatRequestSchema,
  isDesktopExternalAgentMode,
  resolveChatHardMaxIterations,
  resolveChatMaxIterations,
  type ChatBody,
} from './chat-schema.js'
import {
  resolveImageGeneratorAutoSkillContent,
  resolveOfficeAutoSkillContent,
  resolveSkillRefsContent,
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
  applyIntentRouting,
  resolveInitialExecutionMode,
  logIntentRouterDecision,
  INTENT_ROUTER_CIRCUIT_BREAKER,
  resolveIntentRouterModel,
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
      /* noop */
    }
  }
}

export async function chatStreamRoutes(app: FastifyInstance) {
  // POST /chat/stream — SSE streaming response
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

      const body = request.body
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

      // Claim before the first session/workspace read. The response listeners
      // release pre-flight failures; the run-level finally releases successful
      // streams. The lease itself is idempotent, so late socket events cannot
      // release a newer owner of the same session.
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

      // Explicit per-turn autonomy selection plus a structured trace. Real
      // ceilings are enforced by channel ACL and tool policy, not by treating
      // the configured local default as a security boundary.
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
      const persistedPersonas = createPersonaRepo().list()
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
        message,
        requestedProvider: reqProvider,
        requestedModel: model,
        existingSession,
        requireVision: body.mode === 'computer-use',
        preferVision: chatRequestPrefersVision(message, body.mode, priorRunContractForSelection),
      })
      const selection = selections[0]
      if (!selection) {
        const failure = buildProviderSelectionFailure(runtime, reqProvider, model)
        return reply.status(failure.statusCode).send({ error: failure.error })
      }
      const selectedProvider = selection.provider
      const selectedModel = selection.model

      // Pre-flight skill validation: user-requested skillRefs are validated
      // before the SSE stream opens so unknown/unavailable refs produce 4xx
      // responses (matching the original behavior). The effective post-routing
      // set is validated again below before its tool allowlist is constructed.
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

      // Register disconnect handling before hook/routing/context preflight.
      // Those steps can involve slow storage or auxiliary-model calls; if the
      // client presses Ctrl+C before SSE headers are opened, registering only
      // beside reply.hijack() permanently misses the closed socket and lets the
      // abandoned run retain its limiter lease until the inactivity watchdog.
      let clientClosed = false
      let terminalSessionPersisted = false
      let activeModeRouter: AgentModeRouter | null = null
      // Set when the approval registry parks this run: the prompt stays open
      // and its checkpoint stays on disk, so the stream ends with an
      // APPROVAL_TIMEOUT frame plus a resumable `approval_timeout` done.
      let parkedDecision: PendingDecision | null = null
      const preflightAbortController = new AbortController()
      const stopRun = async () => {
        clientClosed = true
        preflightAbortController.abort(new Error('Chat stream cancelled'))
        await activeModeRouter?.stop().catch(() => {})
      }
      registerSseDisconnectHandler(request, reply, stopRun)
      // Some embedders and lightweight test runtimes predate the active-run
      // registry. Disconnect cancellation must remain available through the
      // socket handler even when the explicit cancel endpoint is unavailable.
      const unregisterRunCanceller =
        runtime.activeRuns?.registerCanceller(sessionId, stopRun) ?? (() => undefined)

      let lifecycle: ReturnType<typeof createSseLifecycle> | null = null
      try {
        // pre:user:prompt — fires before any session/user_message side
        // effects so a handler can abort (4xx) or rewrite the prompt before
        // it reaches the agent and the session journal.
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
          // Tag the payload so engine.buildInitialMessages can recover the
          // ContentPart[] verbatim instead of stuffing JSON into a string
          // message (which is what was making vision models answer the
          // wrong question).
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
        const previousContext = await loadAutoCompactedSessionContext({
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

        // ── Intent routing ────────────────────────────────────────────────
        // Run BEFORE reply.hijack() so errors can still be sent as normal
        // JSON responses. Skill prefix resolution (which needs `send`) is
        // deferred to after the SSE stream opens.
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
                personaList: () => resolvePersonaCatalog(customAgents, persistedPersonas),
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

        // A session-bound solo persona and an explicit panel are deliberate
        // user selections. Do not let intent routing silently replace either.
        const panelLocked =
          body.mode === 'persona-panel' && (personaSelection.personaIds?.length ?? 0) > 0
        const personaLocked = panelLocked || Boolean(personaSelection.persona)
        const effectiveIntentRouting = personaLocked ? { enabled: false } : body.intentRouting
        const routingMode = body.mode

        const routing = await applyIntentRouting({
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

        if (clientClosed || reply.raw.destroyed) return reply

        await runtime.sessions.appendEvent(sessionId, {
          type: 'router_decision',
          id: randomUUID(),
          timestamp: new Date().toISOString(),
          decision: {
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

        // ── Structured log: per-decision telemetry ───────────────────────
        // Emitted for every turn so dashboards/log-aggregators can track
        // routing quality without instrumenting the database.
        // TODO: wire Prometheus-style counters (intentRouterCalls,
        // intentRouterLatency, intentRouterTokens) once the daemon exposes
        // registerCounter/registerHistogram helpers.
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

        // Resolve the post-routing skill set before the tool registry is scoped.
        // This keeps the streaming path in parity with /chat: a declared skill
        // can expose only its own tools, and a router-selected skill uses its
        // stable declaration rather than inheriting the full registry.
        let skillPrefix = ''
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
          officeAutoSkillContent = resolveInitialExecutionMode(routing, runtime.config.agent.mode) === 'instant' ? '' : await resolveOfficeAutoSkillContent(
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
          boundaryRoot: requestWorkspaceRoot,
        })
        const agentsMemory = formatAgentsMdSection(agentsFiles) ?? undefined
        const relevantKnowledgeContext = await buildChatKnowledgeContext({
          registry: app.chatKnowledgeProviders,
          message: finalPrompt,
          previousMessages,
          allowLocalKnowledge: request.authContext?.kind !== 'extension',
        })
        // Direct-API surface: trim the memory toolset (default `lean`). Channels
        // run through the pipeline and keep the full `runtime.toolRegistry`.
        const directApiToolRegistry = withWritingDocTools(
          withSwarmToolsForSession(
            withDirectApiMemoryToolset(
              runtime.toolRegistry,
              runtime.config.memory.directApiToolset,
            ),
            sessionId,
          ),
          // The same expression the doc.* tools resolve with.
          body.writingDocId ?? getDocRegistry().getActiveId(),
        )
        const activeRemoteBrowser = remoteBrowserBridge(runtime.toolRegistry).list().some(
          (connection) => connection.sessionId === sessionId,
        )
        const initialMode = resolveInitialExecutionMode(routing, runtime.config.agent.mode, activeRemoteBrowser)
        const authorizedToolRegistry = withSelectedBrowserTarget(withAuthorizedToolExposure(directApiToolRegistry, {
          explicitToolNames: body.toolNames, declaredSkillToolNames,
          personaAllowedTools: personaObj?.allowedTools, personaDeniedTools: personaObj?.deniedTools,
        }), activeRemoteBrowser)
        const agentToolRegistry = withContextualToolExposure(authorizedToolRegistry, {
          surface: resolveRequestSurface(request),
          semanticRouting: true,
          activeRemoteBrowser,
          selectedGroups: routing.decision.toolGroups,
          routedMode: initialMode,
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
        const baseSystemPrompt = await buildSystemPrompt({
          config: runtime.config,
          profile: initialMode === 'instant' ? 'instant' : 'agent',
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
          // Session history already supplies same-session context; journals
          // can aggregate unrelated conversations from the same day.
          includeDailyNotes: false,
        })
        let strictFinalAnswerProtocol = process.env.SEPILOTD_STRICT_ANSWER_PROTOCOL === '1'
        // 글쓰기 모드(mode='writing') 자동 컨텍스트:
        // - 활성 doc session이 있으면 system prompt에 outline + (작은 doc) 전체 본문
        //   + doc.* 도구 우선 사용 안내를 prepend.
        // - desktop이 chat 보낼 때 별도 컨텍스트 button 누를 필요 없이 LLM이
        //   바로 활성 문서를 알게 됨. canvas UX의 핵심.
        const docPrefix = buildWritingDocPrefix(
          body.mode,
          body.writingDocId,
          requestWorkspaceRoot,
        )
        const shellPrefix = buildShellModePrefix(body.mode)
        const systemPrompt = () => shellPrefix + docPrefix + skillPrefix + baseSystemPrompt
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
            forcePrompt: options?.forcePrompt,
            signal: options?.signal,
            onParked: (approval) => {
              parkedDecision = {
                kind: 'approval',
                id: approval.requestId,
                label: approval.tool,
                since: Date.parse(approval.requestedAt),
              }
              // Same abort-without-cancel path the decision watchdog uses:
              // the wait rejects through `signal`, the registry entry stays.
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
            provider: nextSelection.provider,
            tools: agentToolRegistry,
            authorizedTools: authorizedToolRegistry,
            activeRemoteBrowser,
            policy: runtime.policyEngine,
            autonomy: effectiveAutonomy,
            auxiliaryLlmBudget,
            semanticIndex: runtime.semanticIndex,
            systemPrompt: systemPrompt(),
            previousMessages,
            maxIterations: resolveChatMaxIterations(body),
            hardMaxIterations: resolveChatHardMaxIterations(body),
            auditLogger: runtime.auditLogger,
            usageTracker: runtime.usageTracker,
            spendBudget: runtime.config.limits,
            hookRegistry: runtime.hookRegistry,
            deviceName: runtime.config.device.name,
            thinkingLevel,
            maxTokens,
            temperature,
            textDeltaMode,
            ragEnabled,
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

        // SSE keepalive + agent inactivity watchdog.
        // Keepalive comment frames stop proxies/cli watchdogs from idling the
        // connection while the agent is waiting on a slow first token
        // (cold-start, big context). Inactivity timer fires once if no real
        // event arrives within SEPILOTD_AGENT_INACTIVITY_MS — without it, an
        // LLM/tool hang inside the for-await loop kept the run lease alive
        // forever because the cosmetic keepalive frames hid the lockup.
        const AGENT_INACTIVITY_MS = resolveAgentInactivityMs()
        const inactivityProbe = createAgentInactivityProbe()
        lifecycle = createSseLifecycle({
          reply,
          inactivityMs: AGENT_INACTIVITY_MS,
          // Pending human decision != stall. While an approval/question is
          // outstanding the stall watchdog stays quiet and only the separate,
          // explicitly named decision bound can end the run.
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
        // Surface skipped attachments (oversized/malformed/wrong-type/out-of-root)
        // instead of silently dropping them, so the user knows their file was
        // not included in the turn.
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
        // Emit the routing decision as the first typed event so consumers can
        // display what configuration is driving this turn before text arrives.
        await send('router_decision', {
          type: 'router_decision',
          id: randomUUID(),
          decision: routing.decision,
        })

        // Auto skills were resolved before the tool registry was scoped so
        // their declared tools participate in the same least-privilege set as
        // explicit/router-selected skills.
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

        // 한 번의 chat-stream 요청 = 한 개의 task. Settings → Agent Stats 의
        // productivity KPI(completion rate, throughput/day, channel resolution)
        // 가 task.started/task.completed 이벤트 카운트에 의존하는데, channel
        // pipeline 만 발화시키고 일반 chat 흐름은 발화 안 시켜서 desktop/cli/web
        // 사용자에겐 항상 0/n/a 로 보이는 문제가 있었음. 여기서 한 turn 단위로
        // 묶어 emit. taskId 는 새 UUID — sessionId/run 메타와 별개로 추적된다.
        const taskId = randomUUID()
        const taskStartedAt = Date.now()
        // Opt-in chat-completion notifications: when the user enables this in
        // config the daemon publishes an in-app notification at the end of
        // every chat run, so other surfaces (and the desktop tray toast when
        // the window is unfocused) can surface "your chat finished".
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
            // best-effort — never let a notification failure break the chat reply.
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
            // Observability is best-effort; never block the chat flow on it.
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
            // Delay the public error frame until the run has been classified.
            // A provider can fail after successful tool execution; that path
            // closes with a transparent retained-evidence INCOMPLETE instead
            // of making stream clients discard the later terminal message.
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
        // Tell the user their earlier conversation was replaced by a summary.
        // Compaction changes what the assistant can recall, so it must not be
        // something the user only infers from a thinner answer.
        if (compactionNotice) {
          await emitEvent({ type: 'thinking', content: compactionNotice })
        }
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

            await appendChatProviderAttemptEvent(
              runtime.sessions,
              sessionId,
              attemptSelection,
              attempt,
              'started',
            )

            for await (const event of modeRouter.run(
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
                preferPromptReact: existingSession?.preferPromptReact,
              },
              requestMode,
            )) {
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
          // stop() may make a router return normally. A watchdog trip is
          // nevertheless a terminal failure, not permission to synthesize a
          // successful done event and fallback final answer.
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

          // emitEvent closure 가 providerError 를 mutate 하기 때문에 TS의
          // control-flow narrowing이 여전히 null 로만 보고 .code 접근에서
          // 'never' 추정을 내림 — 명시적 alias 로 풀어준다.
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

            await runtime.sessions.appendEvent(sessionId, {
              type: 'assistant_message',
              id: randomUUID(),
              timestamp: new Date().toISOString(),
              content: assistantChatContent,
            })
            await updateSessionTitleFromFirstTurn({
              sessions: runtime.sessions,
              session,
              provider: activeSelection.provider,
              model: activeSelection.model,
              firstMessage: finalPrompt,
              assistantReply: assistantChatContent,
            })
            triggerDreamingTurn(runtime.dreaming, sessionId, 'chat-stream', scopeTags)
            finalMessageEvent = {
              type: 'message',
              content: assistantChatContent,
            }
          }

          // Outcome review, memory-write verification, and strict answer
          // protocol may intentionally suppress speculative provider chunks.
          // Preserve the public buffered-stream contract by emitting the
          // verified final body once, after every guard has accepted it.
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
            // Convert the abort/throw into a structured signal the cli can
            // recognise instead of a generic INTERNAL_ERROR, and say *what*
            // stalled (provider never responded vs. a stuck step). The run
            // lease and keepalive timer are released by the outer finally.
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
                // A parked run is resumable from its checkpoint; do not mark
                // the session abandoned the way a genuine stall does.
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
              /* best-effort */
            }
            return
          }
          throw err
        }
      } catch (err) {
        // SSE handlers are responsible for terminating the response so
        // the client doesn't hang. Surface the error frame, then close.
        try {
          if (!reply.raw.writableEnded) {
            // If we throw before `reply.hijack()` / `reply.raw.writeHead`,
            // the response goes out with only Fastify's defaults — no
            // Content-Type, no CORS headers — and the browser blocks the
            // chunked body with a generic "blocked by CORS policy" error
            // that hides the real cause. Emit a proper SSE error frame
            // with the right headers so the renderer sees it as a stream
            // event instead of a CORS failure.
            if (!reply.raw.headersSent) {
              try {
                reply.hijack()
              } catch {
                /* hijack is idempotent; ignore "already hijacked" */
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
          /* best-effort */
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
          /* best-effort terminal-state repair */
        }
        throw err
      } finally {
        // Always tear down lifecycle timers and force-close the stream so
        // the cli sees EOF, even on uncaught throws.
        lifecycle?.dispose()
        unregisterRunCanceller()
        if (clientClosed && !terminalSessionPersisted) {
          try {
            await runtime.sessions.updateMeta?.(sessionId, { status: 'abandoned' })
          } catch {
            /* best-effort disconnect-state repair */
          }
        }
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
}
