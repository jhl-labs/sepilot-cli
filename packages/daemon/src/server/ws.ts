import { resolveTurnMemoryPolicy, memoryToolAllowed } from '../memory/turn-policy.js'
import { remoteBrowserBridge } from '../tools/browser-remote.js'
import { withSelectedBrowserTarget } from '../tools/browser-target.js'
import { resolvePersonaMemoryScope } from '../memory/persona-scope.js'
import type { FastifyInstance } from 'fastify'
import { getDocRegistry } from '../agent/doc/session.js'
import type { AgentEvent, ApiError, ApprovalDecision } from '@sepilotd/core'
import type { ContentPart } from '@sepilotd/core'
import { randomUUID } from 'node:crypto'
import './fastify-types.js'
import { AgentModeRouter, type AgentMode } from '../agent/mode-router.js'
import { requiresNativeMultimodalInput } from '../agent/multimodal-input.js'
import type { ApprovalCallback } from '../agent/engine.js'
import { resolveInstantModeToolNames } from '../agent/instant-mode-tool-intent.js'
import { resolveSkillExecutionContext } from '../skills/execution-policy.js'
import { resolveRequestedAutonomy } from '../security/autonomy.js'
import { createAgentOutputTracker } from '../agent/event-output.js'
import {
  formatContextCompactionNotice,
  loadAutoCompactedSessionContextWithinBudget,
  loadLatestRoutedPersona,
  loadLatestRunContract,
  resolveSessionContextMaxMessages,
} from '../agent/auto-compaction.js'
import { applyPreUserPromptHook } from '../agent/turn-hooks.js'
import { fallbackSessionTitle } from '../agent/session-title.js'
import { triggerDreamingTurn } from '../memory/dreaming.js'
import { buildDeferredSystemPrompt } from '../agent/system-prompt.js'
import { applyWritingDocFallback, snapshotWritingDocStart } from '../agent/doc/writing-fallback.js'
import { collectAgentsMd, formatAgentsMdSection } from '../memory/agents-md.js'
import {
  resolvePersona,
  resolvePersonaCatalog,
  resolvePersonas,
} from '../agent/custom/persona-resolver.js'
import { createPersonaRepo } from '../persona/repo.js'
import { persistAgentSessionEvent } from './session-events.js'
import { toRunLimiterApiError, type RunLease } from './runtime/run-limiter.js'
import { createRuntimeBackedModeRouterOptions } from './runtime/mode-router-options.js'
import {
  extractAndStoreArtifacts,
  extractAndStoreUserImageArtifacts,
} from './routes/artifact-support.js'
import {
  attachmentAllowedRoots,
  resolveFileIds,
  resolveSessionAttachmentRefs,
} from './routes/file-registry.js'
import {
  chatAttachmentSchema,
  chatAutonomySchema,
  chatRequestSchema,
  excludesLocalContext,
  resolveChatMaxIterations,
  type ChatBody,
} from './routes/chat-schema.js'
import type { z } from 'zod'
import { loadProjectContext } from './routes/project-context.js'
import { normalizeHeaders } from './routes/utils.js'
import { InvalidCwdError, invalidCwdResponse, resolveChatRequestWorkspace } from './routes/request-cwd.js'
import { resolveWsClientIdentity } from './runtime/ws-client-identity.js'
import { createQuestionRequester } from '../tools/question.js'
import {
  withContextualToolExposure, withTurnToolAuthority,
  withDirectApiMemoryToolset,
  withSwarmToolsForSession,
  withWritingDocTools,
} from '../tools/role-filter.js'
import {
  createAgentInactivityProbe,
  isSubstantiveAgentActivity,
  resolveAgentInactivityMs,
  resolvePendingDecisionTimeoutMs,
} from './sse-response.js'
import { sendJsonWs, type WsSendOptions } from './sse-write.js'
import { normalizeSurfaceLabel, resolveClientLabel } from './request-surface.js'
import { buildShellModePrefix, buildWritingDocPrefix } from './routes/chat-mode-prefixes.js'
import {
  buildProviderSelectionFailure,
  chatRequestPrefersVision,
  selectChatProviderCandidates,
} from './routes/provider-selection.js'
import { ConnectionLimitExceededError } from './runtime/connection-registry.js'
import {
  isExtensionSessionReuseDenied,
  isExtensionWritingModeDenied,
  readChatMemoryScope,
} from './chat-memory-context.js'
import { resolveScopedFileMemoryReadView } from '../memory/scoped-file-memory-read-view.js'
import { buildChatKnowledgeContext } from './chat-knowledge.js'
import { IntentRouter, type IntentRouterOptions } from '../agent/intent-router.js'
import { createAuxiliaryLlmTurnBudget } from '../agent/auxiliary-llm.js'
import { AttachmentLimitError, assertChatAttachmentLimits } from '../media/pipeline.js'
import {
  applyIntentRouting,
  resolveInitialExecutionMode,
  logIntentRouterDecision,
  resolveIntentRouterModel,
  INTENT_ROUTER_CIRCUIT_BREAKER,
  routingLeavesMemoryIntentUnresolved,
} from './routes/chat-intent.js'
import {
  applyRoleBotRequestDefaults,
  applyRoutedRoleBot,
  keepContinuingRoleBot,
  roleBotLongTermInjectKb,
  roleBotModeRouterOptions,
  withRoleBotPersonas,
  routerPersonaCandidates,
  roleBotSkillRefs,
} from './chat-bots.js'
import {
  resolveImageGeneratorAutoSkillContent,
  resolveOfficeAutoSkillContent,
  resolveSkillRefsContent,
  filterLoadableSkillRefs,
  buildSkillsLoaded,
  shouldDeferPptxAttachmentText,
  SkillNotFoundError,
  SkillUnavailableError,
} from './routes/chat-skills.js'
import {
  findUnknownChatPersonaIds,
  resolveChatPersonaSelection,
  unknownChatPersonaResponse,
} from './chat-personas.js'
import { createLogger } from '../logger.js'
import { stopReasonProviderError, stopReasonUserAbort } from '../agent/stop-reason.js'
import { DEFAULT_LIVE_VIEW_STILLS, createLiveViewTap } from '../views/live-view-event-tap.js'
import { liveViewRegistry } from '../views/live-view-registry.js'

const DROPPABLE_AGENT_WS_EVENTS = new Set(['text_delta', 'thinking', 'view_still'])
const DEFAULT_WS_HEARTBEAT_MS = 30_000
const logger = createLogger('server.ws')

type HeartbeatSocket = {
  readyState: number
  isAlive?: boolean
  on(event: string, listener: (...args: unknown[]) => void): unknown
  off?(event: string, listener: (...args: unknown[]) => void): unknown
  ping(): void
  terminate(): void
}

function resolveWsHeartbeatMs(): number {
  const raw = process.env.SEPILOTD_WS_HEARTBEAT_MS
  if (!raw) return DEFAULT_WS_HEARTBEAT_MS
  const parsed = Number.parseInt(raw, 10)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_WS_HEARTBEAT_MS
}

export async function wsRoutes(app: FastifyInstance) {
  const heartbeatSockets = new Set<HeartbeatSocket>()
  const heartbeatTimer = setInterval(() => {
    for (const heartbeatSocket of [...heartbeatSockets]) {
      if (heartbeatSocket.readyState !== 1) {
        heartbeatSockets.delete(heartbeatSocket)
        continue
      }
      if (heartbeatSocket.isAlive === false) {
        heartbeatSockets.delete(heartbeatSocket)
        heartbeatSocket.terminate()
        continue
      }
      heartbeatSocket.isAlive = false
      try {
        heartbeatSocket.ping()
      } catch {
        heartbeatSockets.delete(heartbeatSocket)
        heartbeatSocket.terminate()
      }
    }
  }, resolveWsHeartbeatMs())
  heartbeatTimer.unref?.()

  app.addHook?.('onClose', async () => {
    clearInterval(heartbeatTimer)
    heartbeatSockets.clear()
  })

  app.get('/api/v1/ws', { websocket: true }, (socket, req) => {
    const runtime = app.runtime!
    let closed = false
    const activeRouters = new Set<AgentModeRouter>()
    const authContext = req.authContext
    const clientHeaders = normalizeHeaders(req.headers)
    const baseScopeTags = readChatMemoryScope(clientHeaders, authContext)

    const registry = app.connectionRegistry
    const surfaceLabel = normalizeSurfaceLabel(clientHeaders['x-sepilotd-surface'])
    const clientLabel = resolveClientLabel(authContext, surfaceLabel) ?? 'master'
    let connectionId: string | null = null
    try {
      connectionId =
        registry?.add({
          kind: 'ws',
          label: '/api/v1/ws',
          client: clientLabel,
        }) ?? null
    } catch (error) {
      if (error instanceof ConnectionLimitExceededError) {
        socket.close(1013, 'WebSocket connection limit exceeded')
        return
      }
      throw error
    }
    const heartbeatSocket = socket as unknown as HeartbeatSocket
    const markAlive = () => {
      heartbeatSocket.isAlive = true
    }
    markAlive()
    heartbeatSocket.on('pong', markAlive)
    heartbeatSockets.add(heartbeatSocket)

    function safeSend(data: string, options: WsSendOptions = {}) {
      try {
        if (closed) return false
        return sendJsonWs(socket, data, options)
      } catch {
        return false
      }
    }

    function extensionHasScope(scope: 'chat' | 'approvals' | 'ws'): boolean {
      if (authContext?.kind !== 'extension') {
        return true
      }

      return authContext.scopes.includes('all') || authContext.scopes.includes(scope)
    }

    socket.on('close', () => {
      closed = true
      heartbeatSockets.delete(heartbeatSocket)
      heartbeatSocket.off?.('pong', markAlive)
      if (connectionId) registry?.remove(connectionId)
      for (const router of [...activeRouters]) {
        activeRouters.delete(router)
        void router.stop().catch(() => {})
      }
    })

    const identityResolution = resolveWsClientIdentity(
      clientHeaders,
      authContext,
      runtime.devicePairingRegistry,
    )
    const requiresAuthenticatedClient =
      Boolean(app.authTokenRequired || app.authToken) || Boolean(runtime.extensionTokenStore)
    const unauthorizedMessage = !identityResolution.ok
      ? (identityResolution.errorMessage ?? 'Unauthorized websocket client')
      : requiresAuthenticatedClient && identityResolution.identity.kind === 'anonymous'
        ? 'WebSocket authentication required'
        : null

    if (unauthorizedMessage) {
      safeSend(
        JSON.stringify({
          type: 'error',
          error: {
            code: 'UNAUTHORIZED',
            message: unauthorizedMessage,
          },
        }),
      )
      void runtime.auditLogger?.log?.({
        timestamp: new Date().toISOString(),
        event: 'ws.client.rejected',
        device: runtime.config.device.name,
        surface: surfaceLabel,
        reason: unauthorizedMessage,
      })
      socket.close(1008, 'Unauthorized')
      return
    }

    const clientIdentity = identityResolution.identity
    if (clientIdentity.kind === 'paired-device' && clientIdentity.pairedDeviceId) {
      void runtime.devicePairingRegistry.touch(clientIdentity.pairedDeviceId)
    }
    void runtime.auditLogger?.log?.({
      timestamp: new Date().toISOString(),
      event: 'ws.client.connected',
      device: runtime.config.device.name,
      clientKind: clientIdentity.kind,
      surface: surfaceLabel,
      pairedDeviceId: clientIdentity.pairedDeviceId,
      extensionTokenId: clientIdentity.extensionTokenId,
    })

    socket.on('close', () => {
      void runtime.auditLogger?.log?.({
        timestamp: new Date().toISOString(),
        event: 'ws.client.disconnected',
        device: runtime.config.device.name,
        clientKind: clientIdentity.kind,
        surface: surfaceLabel,
        pairedDeviceId: clientIdentity.pairedDeviceId,
        extensionTokenId: clientIdentity.extensionTokenId,
      })
    })

    socket.on('message', async (raw: Buffer) => {
      try {
        const msg = JSON.parse(raw.toString())

        if (msg.type === 'approval.respond') {
          if (authContext?.kind === 'extension') {
            safeSend(
              JSON.stringify({
                type: 'error',
                error: {
                  code: 'EXTENSION_SESSION_ISOLATION_UNSUPPORTED',
                  message: 'Extension approval responses are disabled until session ownership is persisted.',
                },
              }),
            )
            return
          }
          if (!extensionHasScope('approvals')) {
            safeSend(
              JSON.stringify({
                type: 'error',
                error: {
                  code: 'FORBIDDEN',
                  message: 'Token does not permit approval responses',
                },
              }),
            )
            return
          }

          const { requestId, approved, scope, decision, note } = msg as {
            requestId: string
            approved?: boolean
            decision?: ApprovalDecision['decision']
            note?: string
            scope?: 'once' | 'session' | 'always' | 'run' | 'session-all'
          }
          const normalizedScope =
            scope === 'session' || scope === 'always' || scope === 'run' || scope === 'session-all'
              ? scope
              : 'once'
          const respondResult = await runtime.approvalRegistry.respond(
            requestId,
            {
              decision: decision ?? (approved === true ? 'approved' : 'denied'),
              approved: decision ? decision === 'approved' : approved === true,
              note,
            },
            { approvedBy: clientIdentity.approvedBy, scope: normalizedScope },
          )
          if (!respondResult.resolved) {
            safeSend(
              JSON.stringify({
                type: 'error',
                error: { code: 'NOT_FOUND', message: 'Approval request not found' },
              }),
            )
          }
          return
        }

        if (msg.type === 'chat.send') {
          if (!extensionHasScope('chat')) {
            safeSend(
              JSON.stringify({
                type: 'error',
                error: {
                  code: 'FORBIDDEN',
                  message: 'Token does not permit WebSocket chat',
                },
              }),
            )
            return
          }

          const {
            message,
            sessionId: reqSessionId,
            model,
            provider: reqProvider,
            persona: rawPersona,
            thinkingLevel,
            maxTokens,
            fileIds,
            projectId,
            cwd,
          } = msg
          const sessionTags = chatRequestSchema.shape.tags.safeParse(msg.tags)
          const completionRequest = chatRequestSchema.shape.completionRequest.safeParse(msg.completionRequest)
          if (!completionRequest.success) {
            safeSend(JSON.stringify({ type: 'error', error: {
              code: 'BAD_REQUEST', message: 'Invalid completion request',
            } }))
            return
          }
          if (!sessionTags.success) {
            safeSend(JSON.stringify({ type: 'error', error: {
              code: 'BAD_REQUEST', message: 'Invalid session tags',
            } }))
            return
          }
          let requestedPersona =
            typeof rawPersona === 'string' && rawPersona.trim() ? rawPersona.trim() : undefined
          let requestedPersonaIds = Array.isArray(msg.personaIds)
            ? msg.personaIds
                .filter((id: unknown): id is string => typeof id === 'string' && Boolean(id.trim()))
                .map((id: string) => id.trim())
                .slice(0, 6)
            : undefined
          let panelStrategy: ChatBody['panelStrategy'] =
            msg.panelStrategy === 'sequential' || msg.panelStrategy === 'moderated'
              ? msg.panelStrategy
              : undefined
          let skillRefs: ChatBody['skillRefs'] = Array.isArray(msg.skillRefs)
            ? msg.skillRefs
                .filter(
                  (ref: unknown): ref is { name: string } =>
                    typeof ref === 'object'
                    && ref !== null
                    && typeof (ref as { name?: unknown }).name === 'string'
                    && Boolean((ref as { name: string }).name.trim()),
                )
                .map((ref: { name: string }) => ({ name: ref.name.trim() }))
            : undefined
          const intentRouting: ChatBody['intentRouting'] =
            typeof msg.intentRouting === 'object'
              && msg.intentRouting !== null
              && typeof msg.intentRouting.enabled === 'boolean'
              ? { enabled: msg.intentRouting.enabled }
              : undefined
          let requestedMaxIterations =
            typeof msg.maxIterations === 'number'
            && Number.isInteger(msg.maxIterations)
            && msg.maxIterations >= 1
            && msg.maxIterations <= 500
              ? msg.maxIterations
              : undefined
          const requestedHardMaxIterations = msg.hardMaxIterations === true
          const temperature =
            typeof msg.temperature === 'number'
            && msg.temperature >= 0
            && msg.temperature <= 2
              ? msg.temperature
              : undefined
          let ragEnabled =
            typeof msg.ragEnabled === 'boolean' ? msg.ragEnabled : undefined
          const textDeltaMode =
            msg.textDeltaMode === 'live'
              ? 'live'
              : msg.textDeltaMode === 'buffered'
                ? 'buffered'
                : undefined
          let requestMode = typeof msg.mode === 'string' ? (msg.mode as AgentMode) : undefined
          const roleBotRequest = applyRoleBotRequestDefaults({
            bot: typeof msg.bot === 'string' ? msg.bot : undefined,
            bots: Array.isArray(msg.bots) ? msg.bots.filter((id: unknown): id is string => typeof id === 'string') : undefined,
            persona: requestedPersona,
            personaIds: requestedPersonaIds,
            panelStrategy,
            mode: requestMode,
            skillRefs,
            maxIterations: requestedMaxIterations,
            ragEnabled,
          }, runtime.roleBots)
          if (roleBotRequest.error) {
            safeSend(JSON.stringify({ type: 'error', error: { code: roleBotRequest.error.code, message: roleBotRequest.error.message } }))
            return
          }
          const explicitRoleBot = roleBotRequest.bot
          requestedPersona = roleBotRequest.body.persona
          requestedPersonaIds = roleBotRequest.body.personaIds ? [...roleBotRequest.body.personaIds] : undefined
          panelStrategy = roleBotRequest.body.panelStrategy as ChatBody['panelStrategy']
          requestMode = roleBotRequest.body.mode as AgentMode | undefined
          skillRefs = roleBotRequest.body.skillRefs ? [...roleBotRequest.body.skillRefs] : undefined
          requestedMaxIterations = roleBotRequest.body.maxIterations
          ragEnabled = roleBotRequest.body.ragEnabled
          if (isExtensionWritingModeDenied(authContext, requestMode)) {
            safeSend(
              JSON.stringify({
                type: 'error',
                error: {
                  code: 'EXTENSION_WRITING_ISOLATION_UNSUPPORTED',
                  message: 'Extension writing mode is disabled until document ownership is persisted.',
                },
              }),
            )
            return
          }
          const writingDocId =
            typeof msg.writingDocId === 'string' && msg.writingDocId.trim()
              ? msg.writingDocId.trim()
              : undefined
          const toolNames = Array.isArray(msg.toolNames)
            ? msg.toolNames.filter((name: unknown): name is string => typeof name === 'string')
            : undefined
          const requireToolApproval = msg.requireToolApproval === true
          const userMessage = typeof message === 'string' ? message : ''
          const wsSkippedAttachments: Array<{
            source: 'path' | 'url' | 'unknown'
            reason: string
          }> = []
          const clientAttachments: Array<z.infer<typeof chatAttachmentSchema>> = []
          if (Array.isArray(msg.attachments)) {
            for (const raw of msg.attachments) {
              const parsed = chatAttachmentSchema.safeParse(raw)
              if (parsed.success) {
                clientAttachments.push(parsed.data)
              } else {
                wsSkippedAttachments.push({
                  source: 'unknown',
                  reason: 'attachment failed schema validation',
                })
              }
            }
          }
          const requestedFileIds = Array.isArray(fileIds)
            ? fileIds.filter((id: unknown): id is string => typeof id === 'string')
            : []
          const attachments = [...clientAttachments, ...resolveFileIds(requestedFileIds)]
          const sessionAttachments = resolveSessionAttachmentRefs(requestedFileIds)
          try {
            await assertChatAttachmentLimits(attachments)
          } catch (error) {
            if (error instanceof AttachmentLimitError) {
              safeSend(
                JSON.stringify({
                  type: 'error',
                  error: {
                    code: error.code,
                    message: error.message,
                  },
                }),
              )
              return
            }
            throw error
          }
          if (!userMessage && attachments.length === 0) {
            safeSend(
              JSON.stringify({
                type: 'error',
                error: { code: 'INVALID_REQUEST', message: 'message required' },
              }),
            )
            return
          }

          if (!runtime) {
            safeSend(
              JSON.stringify({
                type: 'error',
                error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
              }),
            )
            return
          }

          const requestedAutonomy = chatAutonomySchema.safeParse(msg.autonomy)
          const autonomyResolution = resolveRequestedAutonomy(
            runtime.autonomy,
            requestedAutonomy.success ? requestedAutonomy.data : undefined,
          )
          const effectiveAutonomy = autonomyResolution.effective

          const sessionId = reqSessionId ?? randomUUID()
          if (isExtensionSessionReuseDenied(authContext, reqSessionId)) {
            safeSend(
              JSON.stringify({
                type: 'error',
                error: {
                  code: 'EXTENSION_SESSION_REUSE_UNSUPPORTED',
                  message: 'Extensions must omit sessionId until session ownership is persisted.',
                },
              }),
            )
            return
          }

          const sessionLease = runtime.sessionBusy
            ? await runtime.sessionBusy.acquireLeaseWithGrace(sessionId)
            : undefined
          if (runtime.sessionBusy && !sessionLease) {
            safeSend(
              JSON.stringify({
                type: 'error',
                error: {
                  code: 'BUSY',
                  message:
                    'This session is already processing another turn; it did not free up within the grace window. Wait a moment and retry.',
                },
              }),
            )
            return
          }

          try {
          const existingSession = await runtime.sessions.get(sessionId)
          const projectContext = await loadProjectContext(runtime, projectId)
          let requestCwd: string | undefined
          let requestWorkspaceRoot: string | undefined
          let requestWorkspaceIsolation: 'policy' | 'strict'
          try {
            const resolved = await resolveChatRequestWorkspace({
              cwd,
              workspaceRoot: msg.workspaceRoot,
              session: existingSession,
              projectDirectory: projectContext?.workingDirectory,
              surface: surfaceLabel,
            })
            requestWorkspaceIsolation = resolved.workspaceIsolation
            requestCwd = resolved.cwd
            requestWorkspaceRoot = resolved.workspaceRoot
          } catch (error) {
            if (error instanceof InvalidCwdError) {
              safeSend(
                JSON.stringify({
                  type: 'error',
                  ...invalidCwdResponse(error),
                }),
              )
              return
            }
            throw error
          }

          const customAgents =
            (await runtime.customDefs
              ?.agentsForCwd(requestCwd, requestWorkspaceRoot)
              .catch(() => [])) ?? []
          const storedPersonas = createPersonaRepo().list()
          const userDefinedPersonaIds = new Set([...storedPersonas.map((persona) => persona.id), ...customAgents.map((agent) => agent.id)])
          const persistedPersonas = withRoleBotPersonas(storedPersonas, runtime.roleBots)
          const personaSelection = resolveChatPersonaSelection(
            { persona: requestedPersona, personaIds: requestedPersonaIds },
            existingSession,
          )
          const unknownPersonaIds = findUnknownChatPersonaIds(
            personaSelection,
            (id) => Boolean(resolvePersona(id, customAgents, persistedPersonas)),
          )
          if (unknownPersonaIds.length > 0) {
            safeSend(
              JSON.stringify({
                type: 'error',
                ...unknownChatPersonaResponse(unknownPersonaIds),
              }),
            )
            return
          }
          let memoryContext: ReturnType<typeof resolvePersonaMemoryScope>
          try { memoryContext = resolvePersonaMemoryScope(baseScopeTags, personaSelection, persistedPersonas, existingSession) }
          catch (error) { safeSend(JSON.stringify({ type: 'error', error: { code: 'MEMORY_SPACE_MISMATCH', message: String(error) } })); return }
          const { scopeTags, memoryNamespace } = memoryContext
          const resolvedPanelPersonas = resolvePersonas(
            personaSelection.personaIds,
            customAgents,
            persistedPersonas,
          )

          void runtime.auditLogger?.log?.({
            timestamp: new Date().toISOString(),
            event: 'ws.chat.requested',
            device: runtime.config.device.name,
            clientKind: clientIdentity.kind,
            surface: surfaceLabel,
            sessionId,
            pairedDeviceId: clientIdentity.pairedDeviceId,
            extensionTokenId: clientIdentity.extensionTokenId,
          })

          const priorRunContractForSelection = reqSessionId
            ? await loadLatestRunContract(runtime.sessions, sessionId)
            : undefined
          const selection = selectChatProviderCandidates({
            runtime,
            message: userMessage,
            requestedProvider: reqProvider,
            requestedModel: model,
            existingSession,
            requireVision: requestMode === 'computer-use',
            preferVision: chatRequestPrefersVision(
              userMessage,
              requestMode,
              priorRunContractForSelection,
            ),
          })[0]
          if (!selection) {
            const failure = buildProviderSelectionFailure(runtime, reqProvider, model)
            safeSend(JSON.stringify({ type: 'error', error: failure.error }))
            return
          }
          const provider = selection.provider
          const selectedModel = selection.model

          const declaredSkillToolNames = new Set<string>()
          const autoLoadedSkillToolNames = new Set<string>()
          const loadedExecutionSkillIds = new Set<string>()
          try {
            await resolveSkillRefsContent(
              skillRefs,
              runtime.skillRegistry,
              runtime.toolRegistry,
              requestCwd,
              effectiveAutonomy,
              declaredSkillToolNames,
              requestWorkspaceRoot,
            )
            await resolveImageGeneratorAutoSkillContent(
              userMessage,
              msg.imageGenEnabled === true,
              skillRefs,
              runtime.skillRegistry,
              runtime.toolRegistry,
              requestCwd,
              effectiveAutonomy,
              autoLoadedSkillToolNames,
              requestWorkspaceRoot,
            )
          } catch (error) {
            if (error instanceof SkillNotFoundError || error instanceof SkillUnavailableError) {
              safeSend(
                JSON.stringify({
                  type: 'error',
                  error: {
                    code:
                      error instanceof SkillNotFoundError ? 'SKILL_NOT_FOUND' : error.code,
                    message: error.message,
                    name: error.skillName,
                  },
                }),
              )
              return
            }
            throw error
          }

            let runLease: RunLease | undefined
            try {
              runLease = await runtime.runLimiter?.acquire()
            } catch (error) {
              safeSend(
                JSON.stringify({
                  type: 'error',
                  error: toRunLimiterApiError(error),
                }),
              )
              return
            }

            let modeRouter: AgentModeRouter | null = null
            let agentInput = userMessage
            try {
              const promptGate = await applyPreUserPromptHook(userMessage, {
                hookRegistry: runtime.hookRegistry,
                sessionId,
                actor: clientLabel,
              })
              if (promptGate.aborted) {
                safeSend(
                  JSON.stringify({
                    type: 'error',
                    error: {
                      code: 'USER_PROMPT_REJECTED',
                      message: promptGate.reason ?? 'User prompt rejected by hook',
                    },
                  }),
                )
                return
              }
              const finalPrompt = promptGate.prompt
              const previousContext = await loadAutoCompactedSessionContextWithinBudget({
                sessionStore: runtime.sessions,
                sessionId,
                provider,
                model: selectedModel,
                hooks: runtime.hookRegistry,
                maxMessages: resolveSessionContextMaxMessages(),
              })
              const previousMessages = previousContext.messages
              const previousRunContract = previousContext.runContract
              const compactionNotice = formatContextCompactionNotice(previousContext)
              const presentationIntentText = [
                finalPrompt,
                ...attachments.map((attachment) => attachment.filename ?? ''),
              ].join('\n')
              const deferPptxTextExtraction = shouldDeferPptxAttachmentText(
                presentationIntentText,
                skillRefs,
                finalPrompt,
              )
              const officeAutoIntentText = deferPptxTextExtraction
                ? presentationIntentText
                : finalPrompt
              agentInput = finalPrompt
              let hasMultimodalContent = false
              if (attachments.length > 0) {
                const { buildAttachmentContentParts } = await import('../media/pipeline.js')
                const { encodeMultimodalInput } = await import('../agent/multimodal-input.js')
                const built = await buildAttachmentContentParts(attachments, {
                  allowedRoots: attachmentAllowedRoots({
                    dataDir: runtime.dataDir,
                    cwd: requestCwd,
                  }),
                  deferPptxTextExtraction,
                })
                for (const skip of built.skipped) {
                  wsSkippedAttachments.push({ source: skip.source, reason: skip.reason })
                }
                const parts: ContentPart[] = [{ type: 'text', text: finalPrompt }, ...built.parts]
                hasMultimodalContent = requiresNativeMultimodalInput(parts)
                agentInput = encodeMultimodalInput(parts)
              }

              let session = existingSession
              const sessionWorkspace = requestWorkspaceRoot ?? requestCwd
              if (!session) {
                session = await runtime.sessions.create({
                  id: sessionId,
                  title: fallbackSessionTitle(finalPrompt),
                  createdAt: new Date().toISOString(),
                  updatedAt: new Date().toISOString(),
                  provider: provider.id,
                  model: selectedModel,
                  device: runtime.config.device.name,
                  status: 'active',
                  cwd: sessionWorkspace,
                  workspaceIsolation: requestWorkspaceIsolation,
                  tags: sessionTags.data ?? [],
                  personaIds: personaSelection.personaIds ?? (personaSelection.persona ? [personaSelection.persona] : undefined),
                  memoryNamespace,
                })
              } else {
                const patch: { cwd?: string; workspaceIsolation?: 'policy' | 'strict'; personaIds?: string[]; memoryNamespace?: string } = {}
                if (memoryNamespace) patch.memoryNamespace = memoryNamespace
                if (sessionWorkspace && session.cwd !== sessionWorkspace) patch.cwd = sessionWorkspace
                if (!session.cwd && sessionWorkspace) patch.workspaceIsolation = requestWorkspaceIsolation
                if (requestedPersonaIds !== undefined || requestedPersona) patch.personaIds = requestedPersonaIds ?? [requestedPersona!]
                if (Object.keys(patch).length > 0) {
                  session = (await runtime.sessions.updateMeta?.(session.id, patch)) ?? session
                }
              }
              await runtime.sessions.appendEvent(sessionId, {
                type: 'user_message',
                id: randomUUID(),
                timestamp: new Date().toISOString(),
                content: finalPrompt,
                ...(sessionAttachments.length > 0 ? { attachments: sessionAttachments } : {}),
              })
              const userImageArtifacts = await extractAndStoreUserImageArtifacts(
                runtime,
                sessionId,
                finalPrompt,
              )
              if (userImageArtifacts.length > 0) {
                safeSend(
                  JSON.stringify({
                    type: 'chat.artifacts',
                    sessionId,
                    artifacts: userImageArtifacts,
                  }),
                )
              }
              if (wsSkippedAttachments.length > 0) {
                safeSend(
                  JSON.stringify({
                    type: 'warning',
                    sessionId,
                    code: 'ATTACHMENT_SKIPPED',
                    message: `${wsSkippedAttachments.length} attachment(s) were skipped: ${wsSkippedAttachments
                      .map((skip) => skip.reason)
                      .join('; ')}`,
                    skipped: wsSkippedAttachments,
                  }),
                )
              }

              const auxiliaryLlmBudget = createAuxiliaryLlmTurnBudget()
              const routerCfg = runtime.config.agent.intentRouter
              let intentRouter: IntentRouter | null = null
              if (routerCfg?.enabled) {
                try {
                  const routerProvider = routerCfg.provider
                    ? runtime.providerRegistry.get(routerCfg.provider)
                    : provider
                  const routerModel = resolveIntentRouterModel({
                    routerProvider,
                    configuredModel: routerCfg.model,
                    configuredProviderId: routerCfg.provider,
                    activeModel: routerProvider?.id === provider.id
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
                      activeRunContract: previousContext.runContract,
                      availableToolNames: runtime.toolRegistry.list().map((tool) => tool.name),
                    })
                  }
                } catch (error) {
                  logger.warn('intent_router.init_failed', {
                    err: error instanceof Error ? error.message : String(error),
                  })
                }
              }

              const panelLocked =
                requestMode === 'persona-panel'
                && (personaSelection.personaIds?.length ?? 0) > 0
              const personaLocked = panelLocked || Boolean(personaSelection.persona)
              const effectiveIntentRouting = personaLocked
                ? { enabled: false }
                : intentRouting
              const routingMode = requestMode
              const initialRouting = await applyIntentRouting({
                defaultMode: runtime.config.agent.mode,
                router: intentRouter,
                graphRegistry: runtime.graphRegistry,
                message: finalPrompt,
                prevMessages: previousMessages,
                body: {
                  mode: routingMode,
                  persona: personaSelection.persona,
                  skillRefs,
                  intentRouting: effectiveIntentRouting,
                },
                isMultimodal: hasMultimodalContent,
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

              logIntentRouterDecision(
                logger,
                routing,
                {
                  mode: requestMode,
                  persona: personaSelection.persona,
                  skillCount: skillRefs?.length ?? 0,
                },
                finalPrompt.length,
                sessionId,
                routerCfg?.model,
              )
              const memoryPolicy = resolveTurnMemoryPolicy(msg.memoryPolicy, routing.decision.memoryPolicy, routingLeavesMemoryIntentUnresolved(routing.decision))
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

              const botSkillRefs = await filterLoadableSkillRefs(
                roleBotSkillRefs(turnRoleBot, routing.effectiveSkillRefs),
                runtime.skillRegistry, requestCwd, effectiveAutonomy, requestWorkspaceRoot,
                routing.effectiveSkillRefs.length,
              )
              let skillPrefix = ''
              let botSkillPrefix = ''
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
                skillPrefix += resolveInitialExecutionMode(routing, runtime.config.agent.mode, false, surfaceLabel ?? undefined) === 'instant' ? '' : await resolveOfficeAutoSkillContent(
                  officeAutoIntentText,
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
                skillPrefix += await resolveImageGeneratorAutoSkillContent(
                  finalPrompt,
                  msg.imageGenEnabled === true,
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
                if (error instanceof SkillNotFoundError || error instanceof SkillUnavailableError) {
                  safeSend(
                    JSON.stringify({
                      type: 'error',
                      error: {
                        code:
                          error instanceof SkillNotFoundError ? 'SKILL_NOT_FOUND' : error.code,
                        message: error.message,
                        name: error.skillName,
                      },
                    }),
                  )
                  return
                }
                throw error
              }

              const effectiveMode = routing.effectiveMode as AgentMode | undefined
              const personaObj = resolvePersona(
                routing.effectivePersona,
                customAgents,
                persistedPersonas,
              )
              const agentsFiles = await collectAgentsMd({
                cwd: requestCwd,
                includeUserLevel: !excludesLocalContext(msg),
                boundaryRoot: requestWorkspaceRoot,
              })
              const agentsMemory = formatAgentsMdSection(agentsFiles) ?? undefined
              const relevantKnowledgeContext = memoryPolicy.read && memoryPolicy.automaticRead !== false ? await buildChatKnowledgeContext({
                registry: app.chatKnowledgeProviders,
                message: finalPrompt,
                previousMessages,
                allowLocalKnowledge: authContext?.kind !== 'extension' && !excludesLocalContext(msg),
              }) : undefined
              const directApiToolRegistry = withWritingDocTools(
                withSwarmToolsForSession(
                  withDirectApiMemoryToolset(
                    runtime.toolRegistry,
                    runtime.config.memory.directApiToolset,
                  ),
                  sessionId,
                ),
                writingDocId ?? getDocRegistry().getActiveId(),
              )
              const activeRemoteBrowser = remoteBrowserBridge(runtime.toolRegistry).list().some(connection => connection.sessionId === sessionId)
              const initialMode = resolveInitialExecutionMode(routing, runtime.config.agent.mode, activeRemoteBrowser, surfaceLabel ?? undefined)
              const turnToolAuthority = withTurnToolAuthority(directApiToolRegistry, {
                explicitToolNames: (toolNames ?? directApiToolRegistry.list().map(tool => tool.name)).filter((name: string) => memoryToolAllowed(name, memoryPolicy)), declaredSkillToolNames,
                personaAllowedTools: personaObj?.allowedTools, personaDeniedTools: personaObj?.deniedTools,
              })
              const delegationToolRegistry = withSelectedBrowserTarget(turnToolAuthority.delegation, activeRemoteBrowser)
              const authorizedToolRegistry = withSelectedBrowserTarget(turnToolAuthority.own, activeRemoteBrowser)
              const agentToolRegistry = withContextualToolExposure(authorizedToolRegistry, {
                previousMessages,
                surface: surfaceLabel,
                semanticRouting: true,
                activeRemoteBrowser,
                selectedGroups: routing.decision.toolGroups,
                botToolGroups: turnRoleBot?.toolGroups,
                routedMode: initialMode,
                routingConfidence: routing.decision.confidence,
                routingFallback: routing.decision.fallback,
                activeWritingDocument: Boolean(writingDocId ?? getDocRegistry().getActiveId()),
                swarmSession: sessionId.startsWith('swarm_'),
                supplementalToolNames: autoLoadedSkillToolNames,
                requestInput: finalPrompt,
                explicitToolNames: resolveInstantModeToolNames(
                  finalPrompt,
                  initialMode,
                  toolNames,
                  routing.decision.executionIntent,
                  routing.decision.mode,
                  activeRemoteBrowser,
                ),
              })
              const closedToolSurface = toolNames !== undefined || declaredSkillToolNames.size > 0
              const { prompt: baseSystemPrompt, systemToolGuidance } = await buildDeferredSystemPrompt({
                config: runtime.config,
                profile: initialMode === 'instant' ? 'instant' : 'agent',
                longTermInjectKb: roleBotLongTermInjectKb(turnRoleBot),
                skillInventory: turnRoleBot ? (turnRoleBot.skills ?? []) : undefined,
                tools: agentToolRegistry,
                skills: runtime.skillRegistry,
                fileMemory: !excludesLocalContext(msg) && memoryPolicy.read && memoryPolicy.automaticRead !== false ? resolveScopedFileMemoryReadView(
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
              const systemPrompt = [
                buildShellModePrefix(effectiveMode),
                buildWritingDocPrefix(effectiveMode, writingDocId, requestWorkspaceRoot),
                skillPrefix,
                botSkillPrefix,
                baseSystemPrompt,
              ].join('')
              const writingDocStart = snapshotWritingDocStart(
                effectiveMode,
                writingDocId,
                requestWorkspaceRoot,
              )
              const strictFinalAnswerProtocol =
                process.env.SEPILOTD_STRICT_ANSWER_PROTOCOL === '1'
                || skillPrefix.trim().length > 0

              const approvalCallback: ApprovalCallback = (_tc, requestId, options) => {
                return runtime.approvalRegistry.waitForApproval({
                  sessionId,
                  toolCall: _tc,
                  requestId,
                  recheckPolicy: options?.recheckPolicy,
                  forcePrompt: options?.forcePrompt,
                })
              }
              let emitQuestionRequestEvent: (event: AgentEvent) => Promise<void> = async () =>
                undefined
              const requestQuestion = createQuestionRequester(
                runtime.questions,
                async (question) => {
                  await emitQuestionRequestEvent({
                    type: 'question_request',
                    questionId: question.id,
                    prompt: question.prompt,
                    choices: question.choices,
                  })
                },
              )

              modeRouter = new AgentModeRouter({
                systemToolGuidance,
                provider,
                tools: agentToolRegistry,
                authorizedTools: authorizedToolRegistry,
                delegationTools: delegationToolRegistry,
                activeRemoteBrowser,
                policy: runtime.policyEngine,
                autonomy: effectiveAutonomy,
                auxiliaryLlmBudget,
                semanticIndex: excludesLocalContext(msg) ? undefined : runtime.semanticIndex,
                systemPrompt,
                previousMessages,
                maxIterations: resolveChatMaxIterations({
                  maxIterations: requestedMaxIterations ?? turnRoleBot?.maxIterations,
                }),
                ...roleBotModeRouterOptions(turnRoleBot),
                hardMaxIterations: requestedHardMaxIterations,
                auditLogger: runtime.auditLogger,
                usageTracker: runtime.usageTracker,
                spendBudget: runtime.config.limits,
                hookRegistry: runtime.hookRegistry,
                deviceName: runtime.config.device.name,
                thinkingLevel,
                maxTokens: typeof maxTokens === 'number' ? maxTokens : undefined,
                temperature,
                ragEnabled: ragEnabled ?? turnRoleBot?.documents,
                panelPersonas: resolvedPanelPersonas,
                panelStrategy,
                llmCache: runtime.llmCache,
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
                  runtime.approvalRegistry.tryAutoApproval({ sessionId, toolCall }),
                requestQuestion,
                textDeltaMode,
                ...createRuntimeBackedModeRouterOptions(runtime),
                strictFinalAnswerProtocol,
                reviewToollessFinals: true,
              })
              activeRouters.add(modeRouter)
              if (closed) {
                activeRouters.delete(modeRouter)
                await modeRouter.stop().catch(() => {})
                await persistAgentSessionEvent(runtime.sessions, sessionId, {
                  type: 'done',
                  usage: { inputTokens: 0, outputTokens: 0 },
                  stopReason: {
                    ...stopReasonUserAbort(),
                    summary: 'Run cancelled: the client disconnected before it started.',
                    detail: { layer: 'websocket_disconnect' },
                  },
                })
                return
              }

              safeSend(JSON.stringify({ type: 'chat.session', sessionId }))
              safeSend(
                JSON.stringify({
                  type: 'agent.router_decision',
                  sessionId,
                  id: randomUUID(),
                  decision: routing.decision,
                }),
              )
              const skillsLoaded = buildSkillsLoaded({
                strictPrefix: skillPrefix,
                callerSkillNames: (skillRefs ?? []).map((ref) => ref.name),
                botPrefix: botSkillPrefix,
                autoPrefix: '',
              })
              if (skillsLoaded) safeSend(JSON.stringify({ type: 'agent.skills_loaded', sessionId, skills: skillsLoaded }))
              if (compactionNotice) {
                safeSend(JSON.stringify({
                  type: 'agent.thinking',
                  sessionId,
                  content: compactionNotice,
                }))
              }

              const outputTracker = createAgentOutputTracker()
              const emitBufferedFinalTextDelta = async (content: string): Promise<void> => {
                if (textDeltaMode === 'live' || !content) return
                const deltaEvent: AgentEvent = { type: 'text_delta', text: content }
                outputTracker.consume(deltaEvent)
                safeSend(JSON.stringify({
                  ...deltaEvent,
                  type: 'agent.text_delta',
                  sessionId,
                }))
                await persistAgentSessionEvent(runtime.sessions, sessionId, deltaEvent)
              }
              const inactivityProbe = createAgentInactivityProbe()
              const inactivityMs = resolveAgentInactivityMs()
              let inactivityTripped = false
              let lastAgentEventAt = Date.now()
              const inactivityTimer = setInterval(
                () => {
                  if (closed || inactivityTripped) return
                  const pending = inactivityProbe.pendingDecision()
                  if (pending) {
                    const decisionTimeoutMs = resolvePendingDecisionTimeoutMs()
                    if (decisionTimeoutMs > 0 && Date.now() - pending.since > decisionTimeoutMs) {
                      inactivityTripped = true
                      void modeRouter?.stop().catch(() => {})
                    }
                    return
                  }
                  if (Date.now() - lastAgentEventAt > inactivityMs) {
                    inactivityTripped = true
                    void modeRouter?.stop().catch(() => {})
                  }
                },
                Math.max(100, Math.min(10_000, Math.floor(inactivityMs / 6))),
              )
              let terminalDoneEvent: Extract<AgentEvent, { type: 'done' }> | null = null
              let terminalPersisted = false
              let terminalStateChangeEvent: Extract<AgentEvent, { type: 'state_change' }> | null = null
              let providerError: ApiError | null = null
              emitQuestionRequestEvent = async (event: AgentEvent) => {
                if (isSubstantiveAgentActivity(event)) {
                  lastAgentEventAt = Date.now()
                  inactivityProbe.note(event)
                }
                outputTracker.consume(event)
                safeSend(JSON.stringify({ ...event, type: `agent.${event.type}`, sessionId }))
                await persistAgentSessionEvent(runtime.sessions, sessionId, event)
              }
              const liveViewTap = createLiveViewTap({
                views: liveViewRegistry(runtime.toolRegistry),
                sessionId,
                stills: runtime.config.views?.stills ?? DEFAULT_LIVE_VIEW_STILLS,
              })
              try {
                for await (const event of liveViewTap.wrap(modeRouter.run(
                  agentInput,
                  {
                    sessionId,
                    provider: provider.id,
                    model: selectedModel,
                    cwd: requestCwd,
                    workspaceRoot: requestWorkspaceRoot,
                    workspaceIsolation: requestWorkspaceIsolation,
                    executionPolicy: {
                      requestedAutonomy: autonomyResolution.requested,
                      configuredAutonomy: autonomyResolution.configured,
                      effectiveAutonomy: autonomyResolution.effective,
                      clamped: autonomyResolution.clamped,
                      clampReason: autonomyResolution.reason,
                      agentMode: effectiveMode ?? runtime.config.agent.mode ?? 'auto',
                      primaryAgentId: runtime.primaryAgents?.get(sessionId),
                      workspaceBoundary: requestWorkspaceRoot ? 'strict' : 'unrestricted',
                      freshApprovalRequired: requireToolApproval === true,
                    },
                    systemPrompt,
                    previousMessages,
                    toolAllowlist: authorizedToolRegistry.list().map((tool) => tool.name),
                    delegationToolAllowlist: delegationToolRegistry.list().map((tool) => tool.name),
                    ...resolveSkillExecutionContext(
                      loadedExecutionSkillIds,
                      new Set([...declaredSkillToolNames, ...autoLoadedSkillToolNames]),
                    ),
                    runContract: previousRunContract,
                    runContractScope: 'previous-turn',
                    completionRequest: completionRequest.data,
                    memoryPolicy,
                    memoryQuery: finalPrompt,
                    scopeTags,
                    writingDocId: effectiveMode === 'writing' ? writingDocId : undefined,
                    requireToolApproval,
                    ...(surfaceLabel ? { surface: surfaceLabel } : {}),
                    ...(turnRoleBot?.id ? { roleBotId: turnRoleBot.id } : {}),
                    ...(toolNames !== undefined ? { requestedToolNames: toolNames } : {}),
                    requestAuthorityComplete: true,
                  },
                  effectiveMode,
                ))) {
                  if (closed) {
                    void modeRouter.stop().catch(() => {})
                  }
                  if (isSubstantiveAgentActivity(event)) {
                    lastAgentEventAt = Date.now()
                    inactivityProbe.note(event)
                  }

                  outputTracker.consume(event)
                  if (event.type === 'error') {
                    providerError = event.error
                    continue
                  }
                  if (event.type === 'done') {
                    terminalDoneEvent = event
                    continue
                  }
                  if (event.type === 'state_change' && event.state === 'done') {
                    terminalStateChangeEvent = event
                    continue
                  }
                  if (event.type === 'text_delta' && textDeltaMode !== 'live') {
                    continue
                  }
                  if (event.type === 'message') {
                    continue
                  }

                  safeSend(JSON.stringify({ ...event, type: `agent.${event.type}`, sessionId }), {
                    dropIfBackpressured: DROPPABLE_AGENT_WS_EVENTS.has(event.type),
                  })
                  await persistAgentSessionEvent(runtime.sessions, sessionId, event)
                }

                if (inactivityTripped) {
                  throw new Error('Agent run stopped after an inactivity timeout.')
                }

                const finalProviderError = providerError as ApiError | null
                const retainedEvidenceIncomplete = finalProviderError && !terminalDoneEvent
                  ? outputTracker.providerFailureContent(finalProviderError)
                  : null
                if (finalProviderError && !terminalDoneEvent) {
                  const recoveryEvent: Extract<AgentEvent, { type: 'recovery' }> = {
                    type: 'recovery',
                    scope: 'output_synthesis',
                    kind: retainedEvidenceIncomplete
                      ? 'provider_failure_with_retained_evidence'
                      : 'provider_failure_without_evidence',
                    action: retainedEvidenceIncomplete
                      ? 'synthesize_from_retained_evidence'
                      : 'close_session_abandoned',
                    message: retainedEvidenceIncomplete
                      ? 'The provider failed after successful tool execution; closing the WebSocket turn with a deterministic incomplete response from retained evidence.'
                      : 'The provider failed before any successful tool evidence was available; closing the session as abandoned.',
                    recoverable: true,
                    details: {
                      provider: provider.id,
                      model: selectedModel,
                      errorCode: finalProviderError.code,
                    },
                  }
                  safeSend(JSON.stringify({
                    ...recoveryEvent,
                    type: 'agent.recovery',
                    sessionId,
                  }))
                  await persistAgentSessionEvent(runtime.sessions, sessionId, recoveryEvent)

                  if (!retainedEvidenceIncomplete) {
                    await persistAgentSessionEvent(runtime.sessions, sessionId, {
                      type: 'done',
                      usage: { inputTokens: 0, outputTokens: 0 },
                      stopReason: stopReasonProviderError({
                        layer: finalProviderError.code,
                        summary: finalProviderError.message,
                      }),
                    })
                    terminalPersisted = true
                    await runtime.sessions.updateMeta?.(sessionId, { status: 'abandoned' })
                    safeSend(JSON.stringify({
                      type: 'error',
                      error: finalProviderError,
                      sessionId,
                    }))
                    return
                  }
                  terminalDoneEvent = {
                    type: 'done',
                    usage: { inputTokens: 0, outputTokens: 0 },
                    stopReason: stopReasonProviderError({
                      layer: finalProviderError.code,
                      summary: finalProviderError.message,
                    }),
                  }
                }

                if (!closed) {
                  const canEnrichCompletion = !finalProviderError && terminalDoneEvent
                    && (!terminalDoneEvent.stopReason || terminalDoneEvent.stopReason.kind === 'completed')
                  const finalContent = retainedEvidenceIncomplete ?? outputTracker.finalContent()
                  let assistantChatContent = finalContent
                  if (finalContent) {
                    const fallback = applyWritingDocFallback(
                      effectiveMode,
                      writingDocStart,
                      finalContent,
                    )
                    assistantChatContent = fallback.chatContent ?? finalContent
                    const chatMessage: Extract<AgentEvent, { type: 'message' }> = {
                      type: 'message',
                      content: assistantChatContent,
                    }
                    await emitBufferedFinalTextDelta(chatMessage.content)
                    safeSend(
                      JSON.stringify({
                        ...chatMessage,
                        type: 'agent.message',
                        sessionId,
                      }),
                    )
                    await persistAgentSessionEvent(runtime.sessions, sessionId, chatMessage)
                    const liveViewRecords = liveViewTap.records()
                    await runtime.sessions.appendEvent(sessionId, {
                      type: 'assistant_message',
                      id: randomUUID(),
                      timestamp: new Date().toISOString(),
                      content: assistantChatContent,
                      ...(liveViewRecords.length > 0 ? { views: liveViewRecords } : {}),
                    })
                    if (canEnrichCompletion) {
                      app.sessionTitleUpdates?.schedule({
                        sessions: runtime.sessions,
                        session,
                        provider,
                        model: selectedModel,
                        firstMessage: finalPrompt,
                        assistantReply: assistantChatContent,
                      })
                    }
                  }

                  const artifacts = await extractAndStoreArtifacts(
                    runtime,
                    sessionId,
                    assistantChatContent,
                  )
                  if (artifacts.length > 0) {
                    safeSend(
                      JSON.stringify({
                        type: 'chat.artifacts',
                        sessionId,
                        artifacts,
                      }),
                    )
                  }

                  if (terminalDoneEvent && terminalStateChangeEvent) {
                    safeSend(JSON.stringify({
                      ...terminalStateChangeEvent,
                      type: 'agent.state_change',
                      sessionId,
                    }))
                    await persistAgentSessionEvent(
                      runtime.sessions,
                      sessionId,
                      terminalStateChangeEvent,
                    )
                  }

                  if (terminalDoneEvent) {
                    safeSend(
                      JSON.stringify({
                        ...terminalDoneEvent,
                        type: 'agent.done',
                        sessionId,
                      }),
                    )
                    await persistAgentSessionEvent(runtime.sessions, sessionId, terminalDoneEvent)
                    terminalPersisted = true
                  }

                  if (finalContent && canEnrichCompletion) {
                    if (!excludesLocalContext(msg)) triggerDreamingTurn(runtime.dreaming, sessionId, 'ws', scopeTags)
                  }
                }
              } catch (err) {
                if (inactivityTripped) {
                  await runtime.sessions.updateMeta?.(sessionId, { status: 'abandoned' })
                  safeSend(
                    JSON.stringify({
                      type: 'error',
                      error: inactivityProbe.describe({
                        inactivityMs,
                        provider: provider.id,
                        model: selectedModel,
                      }),
                      sessionId,
                    }),
                  )
                } else {
                  await runtime.sessions.updateMeta?.(sessionId, { status: 'abandoned' })
                  throw err
                }
              } finally {
                clearInterval(inactivityTimer)
                if (closed && !terminalPersisted) {
                  await persistAgentSessionEvent(runtime.sessions, sessionId, terminalDoneEvent ?? {
                    type: 'done',
                    usage: { inputTokens: 0, outputTokens: 0 },
                    stopReason: {
                      ...stopReasonUserAbort(),
                      summary: 'Run cancelled after the client disconnected.',
                      detail: { layer: 'websocket_disconnect' },
                    },
                  })
                }
              }
            } finally {
              if (modeRouter) {
                activeRouters.delete(modeRouter)
              }
              runLease?.release()
            }
          } finally {
            sessionLease?.release()
          }
        } else if (msg.type === 'ping') {
          if (!extensionHasScope('ws')) {
            safeSend(
              JSON.stringify({
                type: 'error',
                error: {
                  code: 'FORBIDDEN',
                  message: 'Token does not permit WebSocket access',
                },
              }),
            )
            return
          }

          safeSend(JSON.stringify({ type: 'pong' }))
        }
      } catch (error: unknown) {
        const message = error instanceof Error ? error.message : 'Unknown error'
        safeSend(JSON.stringify({ type: 'error', error: { code: 'INTERNAL_ERROR', message } }))
      }
    })
  })
}
