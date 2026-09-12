import { resolvePersonaMemoryScope } from '../../memory/persona-scope.js'
import { withSelectedBrowserTarget } from '../../tools/browser-target.js'
import { remoteBrowserBridge } from '../../tools/browser-remote.js'
import { createHash, randomUUID } from 'node:crypto'
import { resolveRequestedAutonomy } from '../../security/autonomy.js'
import type { AgentEvent, ApiError, ContentPart, RunStopReason, SessionEvent, ToolCall } from '@sepilotd/core'
import type { FastifyInstance, FastifyRequest } from 'fastify'
import { z } from 'zod'
import '../fastify-types.js'
import { AgentModeRouter, type AgentMode } from '../../agent/mode-router.js'
import { requiresNativeMultimodalInput } from '../../agent/multimodal-input.js'
import type { ApprovalCallback } from '../../agent/engine.js'
import { resolveInstantModeToolNames } from '../../agent/instant-mode-tool-intent.js'
import { resolveSkillExecutionContext } from '../../skills/execution-policy.js'
import {
  loadAutoCompactedSessionContext,
  loadLatestRunContract,
  resolveSessionContextMaxMessages,
} from '../../agent/auto-compaction.js'
import { fallbackSessionTitle, updateSessionTitleFromFirstTurn } from '../../agent/session-title.js'
import { createLogger } from '../../logger.js'
import { resolveApiActor } from '../request-actor.js'
import { registerSseDisconnectHandler } from '../sse-response.js'
import { triggerDreamingTurn } from '../../memory/dreaming.js'
import { createAgentOutputTracker } from '../../agent/event-output.js'
import { persistAgentSessionEvent } from '../session-events.js'
import { openApiComponentsFromZod } from '../openapi-zod.js'
import { toRunLimiterApiError, type RunLease } from '../runtime/run-limiter.js'
import { createSessionBusyLeaseLifecycle } from '../runtime/session-busy.js'
import { createRuntimeBackedModeRouterOptions } from '../runtime/mode-router-options.js'
import {
  chatRequestSchema,
  isDesktopExternalAgentMode,
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
import { extractAndStoreArtifacts } from './artifact-support.js'
import { loadProjectContext } from './project-context.js'
import { resolveFileIds, resolveSessionAttachmentRefs } from './file-registry.js'
import { AttachmentLimitError, assertChatAttachmentLimits } from '../../media/pipeline.js'
import { zodRequestValidation } from './utils.js'
import { createQuestionRequester } from '../../tools/question.js'
import { InvalidCwdError, invalidCwdResponse, resolveChatRequestWorkspace } from './request-cwd.js'
import { openApiJsonResponseRef, openApiSchemaRef, type OpenApiOverrideMap } from '../openapi.js'
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
  resolveIntentRouterModel,
  INTENT_ROUTER_CIRCUIT_BREAKER,
} from './chat-intent.js'
import {
  resolvePersona,
  resolvePersonaCatalog,
  resolvePersonas,
} from '../../agent/custom/persona-resolver.js'
import { createPersonaRepo } from '../../persona/repo.js'
import { publishChatCompletionNotification } from '../../notifications/publish.js'
import { resolveRequestSurface } from '../request-surface.js'
import { isExtensionSessionReuseDenied, readChatMemoryScope } from '../chat-memory-context.js'
import { resolveScopedFileMemoryReadView } from '../../memory/scoped-file-memory-read-view.js'
import { buildChatKnowledgeContext } from '../chat-knowledge.js'
import {
  findUnknownChatPersonaIds,
  resolveChatPersonaSelection,
  unknownChatPersonaResponse,
} from '../chat-personas.js'
import {
  hydrateBackgroundChatJobContent,
  JsonBackgroundChatJobStore,
  type BackgroundChatJob,
  type BackgroundChatProgress,
  type BackgroundChatJobStore,
} from '../background-chat-store.js'

const logger = createLogger('route.chat')
const BACKGROUND_CHAT_PARTIAL_CONTENT_LIMIT = 32_000
const BACKGROUND_CHAT_PARTIAL_WRITE_INTERVAL_MS = 500
const BACKGROUND_CHAT_ACTION_DETAIL_LIMIT = 240

const toolCallSchema = z.object({
  id: z.string(),
  name: z.string(),
  arguments: z.record(z.unknown()),
})

const tokenUsageSchema = z.object({
  inputTokens: z.number().int(),
  outputTokens: z.number().int(),
  thinkingTokens: z.number().int().optional(),
  cacheReadTokens: z.number().int().optional(),
  cacheCreationTokens: z.number().int().optional(),
  estimatedCost: z.number().optional(),
})

const routerDecisionSchema = z.object({
  mode: z.string().optional(),
  persona: z.string().optional(),
  skillIds: z.array(z.string()).optional(),
  confidence: z.string().optional(),
  reason: z.string().optional(),
  fallback: z.boolean().optional(),
  latencyMs: z.number().optional(),
})

const runStopReasonSchema = z.object({
  kind: z.enum(['completed', 'incomplete', 'blocked', 'cancelled', 'error']),
  code: z.string(),
  summary: z.string().optional(),
  detail: z.record(z.unknown()).optional(),
  resumable: z.boolean(),
  nextActions: z.array(z.string()),
})

const chatResponseSchema = z.object({
  data: z.object({
    sessionId: z.string(),
    messageId: z.string(),
    content: z.string(),
    toolCalls: z.array(toolCallSchema).optional(),
    usage: tokenUsageSchema,
    stopReason: runStopReasonSchema.optional(),
    routerDecision: routerDecisionSchema.optional(),
  }),
})

const chatBackgroundStatusValueSchema = z.enum(['running', 'completed', 'failed', 'cancelled'])

const chatBackgroundProgressActionSchema = z.union([
  z.object({
    type: z.literal('approval'),
    requestId: z.string(),
    toolName: z.string(),
    preview: z.string().optional(),
  }),
  z.object({
    type: z.literal('question'),
    questionId: z.string(),
    choices: z.array(z.string()).optional(),
  }),
])

const chatBackgroundProgressSchema = z.object({
  eventType: z.string(),
  label: z.string(),
  detail: z.string().optional(),
  action: chatBackgroundProgressActionSchema.optional(),
  eventCount: z.number().int().nonnegative(),
  updatedAt: z.string(),
  partialContent: z.string().optional(),
})

const chatBackgroundListProgressSchema = chatBackgroundProgressSchema.omit({
  partialContent: true,
})

const chatBackgroundJobStatusSchema = z.object({
  jobId: z.string(),
  sessionId: z.string(),
  status: chatBackgroundStatusValueSchema,
  messageId: z.string().optional(),
  content: z.string().optional(),
  progress: chatBackgroundProgressSchema.optional(),
  error: z
    .object({
      code: z.string().optional(),
      message: z.string(),
    })
    .optional(),
  createdAt: z.string(),
  updatedAt: z.string(),
})

const chatBackgroundListJobStatusSchema = chatBackgroundJobStatusSchema
  .omit({ content: true, progress: true })
  .extend({
    progress: chatBackgroundListProgressSchema.optional(),
  })

const chatBackgroundStartResponseSchema = z.object({
  data: z.object({
    jobId: z.string(),
    sessionId: z.string(),
    status: chatBackgroundStatusValueSchema,
  }),
})

const chatBackgroundStatusResponseSchema = z.object({
  data: chatBackgroundJobStatusSchema,
})

const chatBackgroundListResponseSchema = z.object({
  data: z.object({
    jobs: z.array(chatBackgroundListJobStatusSchema),
  }),
})

export const chatOpenApiComponents = openApiComponentsFromZod({
  schemas: {
    ToolCall: toolCallSchema,
    TokenUsage: tokenUsageSchema,
    ChatRequest: chatRequestSchema,
    ChatResponse: chatResponseSchema,
    ChatBackgroundStartResponse: chatBackgroundStartResponseSchema,
    ChatBackgroundStatusResponse: chatBackgroundStatusResponseSchema,
    ChatBackgroundListResponse: chatBackgroundListResponseSchema,
  },
})

export const chatOpenApiOverrides: OpenApiOverrideMap = {
  '/api/v1/chat': {
    post: {
      summary: 'Send message',
      tags: ['Chat'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('ChatRequest'),
          },
        },
      },
      responses: {
        200: openApiJsonResponseRef('ChatResponse'),
        400: { description: 'Invalid request' },
        503: { description: 'Service unavailable or agent run capacity exhausted' },
      },
    },
  },
  '/api/v1/chat/background': {
    get: {
      summary: 'List background chat jobs',
      tags: ['Chat'],
      responses: {
        200: openApiJsonResponseRef('ChatBackgroundListResponse'),
      },
    },
    post: {
      summary: 'Start background chat message',
      tags: ['Chat'],
      requestBody: {
        content: {
          'application/json': {
            schema: openApiSchemaRef('ChatRequest'),
          },
        },
      },
      responses: {
        202: openApiJsonResponseRef('ChatBackgroundStartResponse'),
        400: { description: 'Invalid request' },
        503: { description: 'Service unavailable or agent run capacity exhausted' },
      },
    },
  },
  '/api/v1/chat/background/{jobId}': {
    get: {
      summary: 'Get background chat status',
      tags: ['Chat'],
      parameters: [
        {
          name: 'jobId',
          in: 'path',
          required: true,
          schema: { type: 'string' },
        },
      ],
      responses: {
        200: openApiJsonResponseRef('ChatBackgroundStatusResponse'),
        404: { description: 'Background chat job not found' },
      },
    },
    delete: {
      summary: 'Cancel background chat',
      tags: ['Chat'],
      parameters: [
        {
          name: 'jobId',
          in: 'path',
          required: true,
          schema: { type: 'string' },
        },
      ],
      responses: {
        200: openApiJsonResponseRef('ChatBackgroundStatusResponse'),
        404: { description: 'Background chat job not found' },
      },
    },
  },
}

const backgroundChatAuthHeaderNames = [
  'authorization',
  'x-sepilotd-surface',
  'x-memory-scope-user-id',
  'x-memory-scope-channel-type',
  'x-memory-scope-channel-id',
  'x-memory-scope-session-id',
  'x-memory-scope-groups',
]

const BACKGROUND_CHAT_HEARTBEAT_MS = 10_000

const backgroundChatStores = new WeakMap<FastifyInstance, BackgroundChatJobStore>()
const backgroundChatRuns = new WeakMap<
  FastifyInstance,
  {
    cancelled: Set<string>
    active: Map<string, { stop: () => Promise<void> }>
  }
>()
const chatIdempotencyRuns = new WeakMap<FastifyInstance, Set<string>>()

function getBackgroundChatStore(app: FastifyInstance): BackgroundChatJobStore {
  const existing = backgroundChatStores.get(app)
  if (existing) return existing
  const store = new JsonBackgroundChatJobStore({
    dataDir: app.runtime?.dataDir,
  })
  backgroundChatStores.set(app, store)
  return store
}

function getBackgroundChatRunRegistry(app: FastifyInstance) {
  const existing = backgroundChatRuns.get(app)
  if (existing) return existing
  const registry = {
    cancelled: new Set<string>(),
    active: new Map<string, { stop: () => Promise<void> }>(),
  }
  backgroundChatRuns.set(app, registry)
  return registry
}

function getChatIdempotencyRunRegistry(app: FastifyInstance) {
  const existing = chatIdempotencyRuns.get(app)
  if (existing) return existing
  const registry = new Set<string>()
  chatIdempotencyRuns.set(app, registry)
  return registry
}

function isBackgroundChatCancelled(app: FastifyInstance, jobId: string | undefined): boolean {
  return Boolean(jobId && getBackgroundChatRunRegistry(app).cancelled.has(jobId))
}

function registerBackgroundChatRun(
  app: FastifyInstance,
  jobId: string | undefined,
  modeRouter: AgentModeRouter,
): () => void {
  if (!jobId) return () => {}
  const registry = getBackgroundChatRunRegistry(app)
  const registration = { stop: () => modeRouter.stop() }
  registry.active.set(jobId, registration)
  if (registry.cancelled.has(jobId)) {
    void registration.stop().catch((error) => {
      logger.warn('failed to stop already-cancelled background chat run', {
        err: error instanceof Error ? error.message : String(error),
        jobId,
      })
    })
  }
  return () => {
    if (registry.active.get(jobId) === registration) {
      registry.active.delete(jobId)
    }
  }
}

async function cancelBackgroundChatRun(app: FastifyInstance, jobId: string): Promise<void> {
  const registry = getBackgroundChatRunRegistry(app)
  registry.cancelled.add(jobId)
  const active = registry.active.get(jobId)
  if (!active) return
  await active.stop()
}

function clearBackgroundChatRunCancellation(app: FastifyInstance, jobId: string): void {
  const registry = getBackgroundChatRunRegistry(app)
  registry.cancelled.delete(jobId)
  registry.active.delete(jobId)
}

function backgroundChatJobListItem(job: BackgroundChatJob): Omit<BackgroundChatJob, 'content'> {
  const { content: _content, progress, ...summary } = job
  if (!progress) return summary
  const { partialContent: _partialContent, ...listProgress } = progress
  return {
    ...summary,
    progress: listProgress,
  }
}

function truncateBackgroundChatDetail(
  text: string,
  limit = BACKGROUND_CHAT_ACTION_DETAIL_LIMIT,
): string {
  return text.length > limit ? `${text.slice(0, limit - 3).trimEnd()}...` : text
}

function formatBackgroundToolCallPreview(toolCall: ToolCall): string {
  let input = '{}'
  try {
    input = JSON.stringify(toolCall.arguments ?? {})
  } catch {
    input = String(toolCall.arguments ?? {})
  }
  return truncateBackgroundChatDetail(`${toolCall.name}(${input})`)
}

function normalizeBackgroundChoices(choices: string[] | undefined): string[] | undefined {
  if (!choices?.length) return undefined
  return choices.slice(0, 8).map((choice) => truncateBackgroundChatDetail(choice, 120))
}

function backgroundChatProgressFromEvent(
  event: AgentEvent,
  eventCount: number,
): BackgroundChatProgress | null {
  const updatedAt = new Date().toISOString()
  if (event.type === 'state_change') {
    const labels: Record<string, string> = {
      acting: 'Running tools',
      done: 'Finalizing',
      error: 'Recovering from an error',
      idle: 'Preparing',
      observing: 'Reading tool output',
      thinking: 'Thinking',
    }
    return {
      eventType: event.type,
      label: labels[event.state] ?? event.state,
      eventCount,
      updatedAt,
    }
  }
  if (event.type === 'tool_call') {
    return {
      eventType: event.type,
      label: `Running ${event.toolCall.name}`,
      eventCount,
      updatedAt,
    }
  }
  if (event.type === 'approval_request') {
    const preview = formatBackgroundToolCallPreview(event.toolCall)
    return {
      eventType: event.type,
      label: `Waiting for approval: ${event.toolCall.name}`,
      detail: preview,
      action: {
        type: 'approval',
        requestId: event.requestId,
        toolName: event.toolCall.name,
        preview,
      },
      eventCount,
      updatedAt,
    }
  }
  if (event.type === 'question_request') {
    const choices = normalizeBackgroundChoices(event.choices)
    return {
      eventType: event.type,
      label: 'Waiting for answer',
      detail: event.prompt,
      action: {
        type: 'question',
        questionId: event.questionId,
        ...(choices ? { choices } : {}),
      },
      eventCount,
      updatedAt,
    }
  }
  if (event.type === 'auto_approval') {
    return {
      eventType: event.type,
      label: `Auto-approved ${event.toolCall.name}`,
      eventCount,
      updatedAt,
    }
  }
  if (event.type === 'reasoning_step') {
    return {
      eventType: event.type,
      label: event.label,
      ...(event.detail ? { detail: event.detail } : {}),
      eventCount,
      updatedAt,
    }
  }
  if (event.type === 'phase_change') {
    return {
      eventType: event.type,
      label: event.enteredPhase ? `Phase: ${event.enteredPhase}` : 'Finalizing phase',
      eventCount,
      updatedAt,
    }
  }
  if (event.type === 'cowork_task_start') {
    return {
      eventType: event.type,
      label: `Delegating to ${event.role}`,
      eventCount,
      updatedAt,
    }
  }
  if (event.type === 'cowork_synthesizing') {
    return {
      eventType: event.type,
      label: 'Synthesizing collaborator work',
      detail: event.summary,
      eventCount,
      updatedAt,
    }
  }
  if (event.type === 'panel_turn_start') {
    return {
      eventType: event.type,
      label: `${event.personaName} is responding`,
      eventCount,
      updatedAt,
    }
  }
  if (event.type === 'panel_synthesizing') {
    return {
      eventType: event.type,
      label: 'Synthesizing panel responses',
      eventCount,
      updatedAt,
    }
  }
  if (event.type === 'subagent_progress') {
    return {
      eventType: event.type,
      label: event.label ?? 'Subagent is working',
      eventCount,
      updatedAt,
    }
  }
  if (event.type === 'error') {
    return {
      eventType: event.type,
      label: 'Run hit an error',
      detail: event.error.message ?? event.error.code,
      eventCount,
      updatedAt,
    }
  }
  return null
}

function updateBackgroundChatProgress(
  store: BackgroundChatJobStore,
  jobId: string | undefined,
  event: AgentEvent,
  eventCount: number,
  partialContent?: string,
  forcePartialWrite = false,
): void {
  if (!jobId) return
  const progress =
    backgroundChatProgressFromEvent(event, eventCount) ??
    (forcePartialWrite && partialContent
      ? {
          eventType: 'text_delta',
          label: 'Answering',
          eventCount,
          updatedAt: new Date().toISOString(),
        }
      : null)
  if (!progress) return
  void store
    .updateProgress(jobId, {
      ...progress,
      ...(partialContent ? { partialContent } : {}),
    })
    .catch((error) => {
      logger.warn('failed to update background chat progress', {
        err: error instanceof Error ? error.message : String(error),
        jobId,
      })
    })
}

function toSingleHeader(value: string | string[] | undefined): string | undefined {
  if (Array.isArray(value)) return value.join(',')
  return value
}

type ChatResponseData = z.infer<typeof chatResponseSchema>['data']
type ChatIdempotencyState =
  | { status: 'missing' }
  | { status: 'pending' }
  | { status: 'completed'; result: ChatResponseData }

function resolveChatIdempotencyKey(request: FastifyRequest, body: ChatBody): string | undefined {
  const headerValue =
    toSingleHeader(request.headers['idempotency-key']) ??
    toSingleHeader(request.headers['x-idempotency-key'])
  const key = headerValue?.trim() || body.messageId?.trim()
  return key ? key.slice(0, 256) : undefined
}

function chatIdempotencyId(sessionId: string, key: string): string {
  const digest = createHash('sha256')
    .update(sessionId)
    .update('\0')
    .update(key)
    .digest('hex')
    .slice(0, 32)
  return `idem-${digest}`
}

function replayChatResultFromEvents(
  sessionId: string,
  messageId: string,
  events: readonly SessionEvent[],
): ChatIdempotencyState {
  const startIndex = events.findIndex(
    (event) => event.type === 'user_message' && event.id === messageId,
  )
  if (startIndex < 0) return { status: 'missing' }

  const laterEvents = events.slice(startIndex + 1)
  const assistant = laterEvents.find(
    (event): event is Extract<SessionEvent, { type: 'assistant_message' }> =>
      event.type === 'assistant_message',
  )
  const sessionEnd = laterEvents.find(
    (event): event is Extract<SessionEvent, { type: 'session_end' }> =>
      event.type === 'session_end',
  )
  if (!assistant || !sessionEnd) return { status: 'pending' }

  const toolCalls = laterEvents
    .filter(
      (event): event is Extract<SessionEvent, { type: 'tool_call' }> => event.type === 'tool_call',
    )
    .map((event) => ({
      id: event.id,
      name: event.tool,
      arguments: event.input,
    }))
  const routerDecision = laterEvents.find(
    (event): event is Extract<SessionEvent, { type: 'router_decision' }> =>
      event.type === 'router_decision',
  )?.decision
  const shouldSurfaceRouterDecision =
    routerDecision && !(routerDecision.fallback && routerDecision.reason === 'router disabled')

  return {
    status: 'completed',
    result: {
      sessionId,
      messageId,
      content: assistant.content,
      stopReason: sessionEnd.stopReason,
      usage: {
        inputTokens: sessionEnd.totalTokens.input,
        outputTokens: sessionEnd.totalTokens.output,
      },
      ...(toolCalls.length > 0 ? { toolCalls } : {}),
      ...(shouldSurfaceRouterDecision && routerDecision ? { routerDecision } : {}),
    },
  }
}

function internalChatHeaders(request: FastifyRequest, jobId?: string) {
  const headers: Record<string, string> = {}
  for (const name of backgroundChatAuthHeaderNames) {
    const value = toSingleHeader(request.headers[name])
    if (value) headers[name] = value
  }
  headers['x-sepilotd-background-chat'] = '1'
  if (jobId) headers['x-sepilotd-background-job-id'] = jobId
  return headers
}

function parseJsonObject(text: string): Record<string, unknown> | null {
  try {
    const parsed = JSON.parse(text) as unknown
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : null
  } catch {
    return null
  }
}

function parseBackgroundChatError(payload: Record<string, unknown> | null, fallback: string) {
  const error = payload?.error
  if (error && typeof error === 'object') {
    const code = (error as { code?: unknown }).code
    const message = (error as { message?: unknown }).message
    return {
      code: typeof code === 'string' ? code : undefined,
      message: typeof message === 'string' && message.trim() ? message : fallback,
    }
  }
  // Some established route contracts use a flat error envelope such as
  // `{ error: 'SKILL_AUTONOMY_REQUIRED', message: '...', ...details }`.
  // Background chat calls the foreground route internally, so discarding that
  // shape here turns an actionable preflight rejection into only
  // `403: Forbidden`. Preserve the bounded public contract without copying
  // arbitrary detail fields into the durable background-job record.
  if (typeof error === 'string' && error.trim()) {
    const code = /^[A-Z][A-Z0-9_]{1,79}$/.test(error.trim())
      ? error.trim()
      : undefined
    const message = payload?.message
    return {
      code,
      message: typeof message === 'string' && message.trim()
        ? message
        : code ?? error.trim(),
    }
  }
  return { message: fallback }
}

function parseBackgroundChatResult(payload: Record<string, unknown> | null) {
  const data = payload?.data
  if (!data || typeof data !== 'object') return null
  const result = data as { sessionId?: unknown; messageId?: unknown; content?: unknown; stopReason?: unknown }
  if (
    typeof result.sessionId !== 'string' ||
    typeof result.messageId !== 'string' ||
    typeof result.content !== 'string'
  ) {
    return null
  }
  return {
    sessionId: result.sessionId,
    messageId: result.messageId,
    content: result.content,
    stopReason: runStopReasonSchema.safeParse(result.stopReason).data,
  }
}

function startBackgroundChatHeartbeat(
  store: BackgroundChatJobStore,
  jobId: string,
  intervalMs = BACKGROUND_CHAT_HEARTBEAT_MS,
): () => void {
  const timer = setInterval(() => {
    void store.touchRunning(jobId).catch((error) => {
      logger.warn('failed to update background chat heartbeat', {
        err: error instanceof Error ? error.message : String(error),
        jobId,
      })
    })
  }, intervalMs)
  timer.unref?.()
  return () => clearInterval(timer)
}

async function runBackgroundChatJob(
  app: FastifyInstance,
  store: BackgroundChatJobStore,
  jobId: string,
  chatUrl: string,
  payload: ChatBody & { sessionId: string },
  headers: Record<string, string>,
  surface: string | null,
) {
  const stopHeartbeat = startBackgroundChatHeartbeat(store, jobId)
  try {
    const response = await app.inject({
      method: 'POST',
      url: chatUrl,
      headers,
      payload,
    })
    const parsed = parseJsonObject(response.body)
    const now = new Date().toISOString()
    const current = await store.get(jobId)
    if (!current) return
    if (current.status !== 'running') return
    const startedAt = Date.parse(current.createdAt)
    const durationMs = Number.isFinite(startedAt) ? Date.now() - startedAt : 0

    if (response.statusCode < 200 || response.statusCode >= 300) {
      const error = parseBackgroundChatError(
        parsed,
        `${response.statusCode}: ${response.statusMessage || 'Background chat failed'}`,
      )
      await store.set(jobId, {
        ...current,
        status: 'failed',
        error,
        updatedAt: now,
      })
      try {
        publishChatCompletionNotification(app.runtime?.config, {
          outcome: 'failed',
          sessionId: current.sessionId,
          durationMs,
          surface,
          errorMessage: error.message,
        })
      } catch {
        // best-effort notification only
      }
      return
    }

    const result = parseBackgroundChatResult(parsed)
    if (!result) {
      await store.set(jobId, {
        ...current,
        status: 'failed',
        error: {
          code: 'INVALID_BACKGROUND_CHAT_RESPONSE',
          message: 'Background chat returned an invalid response envelope.',
        },
        updatedAt: now,
      })
      try {
        publishChatCompletionNotification(app.runtime?.config, {
          outcome: 'failed',
          sessionId: current.sessionId,
          durationMs,
          surface,
          errorMessage: 'Background chat returned an invalid response envelope.',
        })
      } catch {
        // best-effort notification only
      }
      return
    }

    const session = app.runtime?.sessions?.get
      ? await app.runtime.sessions.get(result.sessionId).catch(() => null)
      : null
    const unsuccessful = result.stopReason && result.stopReason.kind !== 'completed'
    const outcome = unsuccessful ? 'failed' : 'completed'
    await store.set(jobId, {
      ...current,
      sessionId: result.sessionId,
      status: result.stopReason?.kind === 'cancelled' ? 'cancelled' : outcome,
      ...(unsuccessful ? { error: { code: result.stopReason!.code, message: result.stopReason!.summary ?? `Agent stopped: ${result.stopReason!.code}` } } : {}),
      messageId: result.messageId,
      content: result.content,
      updatedAt: now,
    })
    try {
      publishChatCompletionNotification(app.runtime?.config, {
        outcome,
        sessionId: result.sessionId,
        sessionTitle: session?.title,
        durationMs,
        surface,
      })
    } catch {
      // best-effort notification only
    }
  } catch (error) {
    const current = await store.get(jobId)
    if (!current) return
    if (current.status !== 'running') return
    const startedAt = Date.parse(current.createdAt)
    const durationMs = Number.isFinite(startedAt) ? Date.now() - startedAt : 0
    const message = error instanceof Error ? error.message : String(error)
    await store.set(jobId, {
      ...current,
      status: 'failed',
      error: {
        code: 'BACKGROUND_CHAT_FAILED',
        message,
      },
      updatedAt: new Date().toISOString(),
    })
    try {
      publishChatCompletionNotification(app.runtime?.config, {
        outcome: 'failed',
        sessionId: current.sessionId,
        durationMs,
        surface,
        errorMessage: message,
      })
    } catch {
      // best-effort notification only
    }
  } finally {
    stopHeartbeat()
    clearBackgroundChatRunCancellation(app, jobId)
  }
}

export async function chatRoutes(app: FastifyInstance) {
  app.post<{ Body: ChatBody }>(
    '/chat/background',
    {
      preValidation: zodRequestValidation({
        body: {
          schema: chatRequestSchema,
          message: 'Invalid chat request body',
        },
      }),
    },
    async (request, reply) => {
      const store = getBackgroundChatStore(app)
      await store.cleanup()
      const body = request.body
      if (isDesktopExternalAgentMode(body.mode)) {
        return reply.status(400).send({
          error: {
            code: 'UNSUPPORTED_MODE',
            message:
              'External CLI agent modes are desktop-only. Use the Electron desktop app so it can bridge the local CLI through a PTY.',
          },
        })
      }
      const sessionId = body.sessionId ?? randomUUID()
      const idempotencyKey = resolveChatIdempotencyKey(request, body)
      const jobId = idempotencyKey ? chatIdempotencyId(sessionId, idempotencyKey) : randomUUID()
      const existingJob = await store.get(jobId)
      if (existingJob) {
        return reply.status(202).send({
          data: {
            jobId: existingJob.jobId,
            sessionId: existingJob.sessionId,
            status: existingJob.status,
          },
        })
      }
      const now = new Date().toISOString()
      const job: BackgroundChatJob = {
        jobId,
        sessionId,
        status: 'running',
        createdAt: now,
        updatedAt: now,
      }
      await store.set(jobId, job)
      const payload = { ...body, sessionId, textDeltaMode: body.textDeltaMode ?? 'live' }
      const headers = internalChatHeaders(request, jobId)
      const surface = resolveRequestSurface(request)
      const chatUrl = request.url.replace(/\/chat\/background(?:\?.*)?$/, '/chat')
      setImmediate(() => {
        void runBackgroundChatJob(app, store, jobId, chatUrl, payload, headers, surface)
      })
      return reply.status(202).send({
        data: {
          jobId,
          sessionId,
          status: job.status,
        },
      })
    },
  )

  app.get('/chat/background', async () => {
    const store = getBackgroundChatStore(app)
    await store.cleanup()
    const jobs = (await store.entries())
      .sort((a, b) => Date.parse(b.updatedAt) - Date.parse(a.updatedAt))
      .map(backgroundChatJobListItem)
    return { data: { jobs } }
  })

  app.get<{ Params: { jobId: string } }>('/chat/background/:jobId', async (request, reply) => {
    const store = getBackgroundChatStore(app)
    await store.cleanup()
    const { jobId } = request.params
    const job = await store.get(jobId)
    if (!job) {
      return reply.status(404).send({
        error: {
          code: 'BACKGROUND_CHAT_NOT_FOUND',
          message: 'Background chat job not found',
        },
      })
    }
    return {
      data: await hydrateBackgroundChatJobContent(job, app.runtime?.sessions),
    }
  })

  app.delete<{ Params: { jobId: string } }>('/chat/background/:jobId', async (request, reply) => {
    const store = getBackgroundChatStore(app)
    await store.cleanup()
    const { jobId } = request.params
    const job = await store.get(jobId)
    if (!job) {
      return reply.status(404).send({
        error: {
          code: 'BACKGROUND_CHAT_NOT_FOUND',
          message: 'Background chat job not found',
        },
      })
    }

    if (job.status !== 'running') {
      return {
        data: await hydrateBackgroundChatJobContent(job, app.runtime?.sessions),
      }
    }

    const cancelledJob: BackgroundChatJob = {
      ...job,
      status: 'cancelled',
      error: {
        code: 'BACKGROUND_CHAT_CANCELLED',
        message: 'Background chat was cancelled by the user.',
      },
      updatedAt: new Date().toISOString(),
    }
    await store.set(jobId, cancelledJob)
    await cancelBackgroundChatRun(app, jobId).catch((error) => {
      logger.warn('failed to stop background chat run', {
        err: error instanceof Error ? error.message : String(error),
        jobId,
      })
    })
    return { data: cancelledJob }
  })

  app.post<{ Body: ChatBody }>(
    '/chat',
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
      const body = request.body
      const { message: rawMessage, sessionId: reqSessionId, model } = body
      const chatStartedAt = Date.now()
      const backgroundChatRequest =
        toSingleHeader(request.headers['x-sepilotd-background-chat']) === '1'
      const backgroundChatJobId = backgroundChatRequest
        ? toSingleHeader(request.headers['x-sepilotd-background-job-id'])
        : undefined
      const requestSurface = resolveRequestSurface(request)
      const baseScopeTags = readChatMemoryScope(request.headers, request.authContext)

      if (isExtensionSessionReuseDenied(request.authContext, reqSessionId)) {
        return reply.status(403).send({
          error: {
            code: 'EXTENSION_SESSION_REUSE_UNSUPPORTED',
            message: 'Extensions must omit sessionId until session ownership is persisted.',
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

      // No runtime available (daemon still starting up)
      if (!runtime) {
        return {
          data: {
            sessionId: reqSessionId ?? randomUUID(),
            messageId: randomUUID(),
            content: `[no runtime] Daemon is starting up. Try again shortly.`,
            usage: { inputTokens: 0, outputTokens: 0 },
          },
        }
      }

      if (isBackgroundChatCancelled(app, backgroundChatJobId)) {
        return reply.status(409).send({
          error: {
            code: 'BACKGROUND_CHAT_CANCELLED',
            message: 'Background chat was cancelled.',
          },
        })
      }

      const {
        SlashWorkspaceBoundaryError,
        tryExpandSlashInput,
      } = await import('../../agent/custom/slash-expand.js')
      const sessionId = reqSessionId ?? randomUUID()
      const idempotencyKey = resolveChatIdempotencyKey(request, body)
      const messageId = idempotencyKey ? chatIdempotencyId(sessionId, idempotencyKey) : randomUUID()
      const idempotencyRunKey = idempotencyKey ? `${sessionId}:${messageId}` : undefined
      const idempotencyRuns = idempotencyRunKey ? getChatIdempotencyRunRegistry(app) : undefined

      // Own the session before reading its workspace binding. Workspace PATCH,
      // JSON chat, SSE, and WS all share this lease, so a turn can never load
      // definitions from workspace A while the session is concurrently rebound
      // to workspace B. Response lifecycle cleanup covers every pre-flight
      // return before the main run-level finally is entered.
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
      if (idempotencyRunKey && idempotencyRuns?.has(idempotencyRunKey)) {
        return reply.status(409).send({
          error: {
            code: 'CHAT_IN_PROGRESS',
            message: 'A chat request with this Idempotency-Key is already running.',
          },
        })
      }
      if (idempotencyKey && runtime.sessions?.getEvents) {
        const idempotencyState = replayChatResultFromEvents(
          sessionId,
          messageId,
          await runtime.sessions.getEvents(sessionId).catch(() => []),
        )
        if (idempotencyState.status === 'completed') {
          return { data: idempotencyState.result }
        }
        if (idempotencyState.status === 'pending') {
          return reply.status(409).send({
            error: {
              code: 'CHAT_IN_PROGRESS',
              message: 'A chat request with this Idempotency-Key is already running.',
            },
          })
        }
      }
      const existingSession = runtime.sessions?.get ? await runtime.sessions.get(sessionId) : null
      const projectContext = await loadProjectContext(runtime, body.projectId)
      let requestCwd: string | undefined
      let requestWorkspaceRoot: string | undefined
      let requestWorkspaceIsolation: 'policy' | 'strict'
      try {
        const resolved = await resolveChatRequestWorkspace({
          cwd: body.cwd,
          workspaceRoot: body.workspaceRoot,
          session: existingSession,
          projectDirectory: projectContext?.workingDirectory,
          surface: requestSurface,
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

      const attachments = [
        ...(body.attachments ?? []),
        ...(body.fileIds?.length ? resolveFileIds(body.fileIds) : []),
      ]
      const sessionAttachments = resolveSessionAttachmentRefs(body.fileIds ?? [])
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

      // Explicit per-turn autonomy selection plus a structured trace. Real
      // ceilings are enforced by channel ACL and tool policy, not by treating
      // the configured local default as a security boundary.
      const autonomyResolution = resolveRequestedAutonomy(runtime.autonomy, body.autonomy)
      const effectiveAutonomy = autonomyResolution.effective
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
        requestedProvider: body.provider,
        requestedModel: model,
        existingSession,
        requireVision: body.mode === 'computer-use',
        preferVision: chatRequestPrefersVision(message, body.mode, priorRunContractForSelection),
      })
      const selection = selections[0]
      if (!selection) {
        const failure = buildProviderSelectionFailure(runtime, body.provider, model)
        return reply.status(failure.statusCode).send({ error: failure.error })
      }
      const provider = selection.provider
      const selectedModel = selection.model

      // Pre-flight skillRefs resolution: a typo'd ref must return 404
      // before we acquire a run lease or persist a session. The inner
      // run path resolves skillRefs again to build the system prompt;
      // the duplication is intentional — this validation only runs to
      // surface 4xx responses up front.
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
        await resolveImageGeneratorAutoSkillContent(
          message,
          body.imageGenEnabled,
          body.skillRefs,
          runtime.skillRegistry,
          runtime.toolRegistry,
          requestCwd,
          effectiveAutonomy,
          autoLoadedSkillToolNames,
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

      if (idempotencyRunKey) {
        idempotencyRuns?.add(idempotencyRunKey)
      }

      let runLease: RunLease | undefined
      try {
        runLease = await runtime.runLimiter?.acquire()
      } catch (error) {
        if (idempotencyRunKey) {
          idempotencyRuns?.delete(idempotencyRunKey)
        }
        return reply.status(503).send({
          error: toRunLimiterApiError(error),
        })
      }

      if (!sessionLeaseLifecycle.transferToRun()) {
        runLease?.release()
        if (idempotencyRunKey) {
          idempotencyRuns?.delete(idempotencyRunKey)
        }
        return reply
      }

      try {
        const sessionWorkspace = requestWorkspaceRoot ?? requestCwd
        let session = existingSession
        if (!session) {
          session = await runtime.sessions.create({
            id: sessionId,
            title: fallbackSessionTitle(message),
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
            provider: provider.id,
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
          } = {}
          if (memoryNamespace) patch.memoryNamespace = memoryNamespace
          if (sessionWorkspace && session.cwd !== sessionWorkspace) patch.cwd = sessionWorkspace
          if (!session.cwd && sessionWorkspace) patch.workspaceIsolation = requestWorkspaceIsolation
          if (body.personaIds !== undefined || body.persona) patch.personaIds = body.personaIds ?? [body.persona!]
          if (Object.keys(patch).length > 0) {
            session = (await runtime.sessions.updateMeta?.(session.id, patch)) ?? session
          }
        }
        // pre:user:prompt — Claude-Code parity hook. Fires before the user's
        // prompt is committed to the session journal or fed to the agent so a
        // handler can (a) abort the run with a 4xx, or (b) rewrite the prompt
        // via `modifiedPayload.data.prompt`. Default 'continue' is a no-op.
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

        let contentParts: ContentPart[] | undefined
        if (attachments.length) {
          const { buildAttachmentContentParts } = await import('../../media/pipeline.js')
          const { attachmentAllowedRoots } = await import('./file-registry.js')
          const built = await buildAttachmentContentParts(attachments, {
            allowedRoots: attachmentAllowedRoots({ dataDir: runtime.dataDir, cwd: requestCwd }),
            deferPptxTextExtraction,
          })
          contentParts = built.parts
          for (const skip of built.skipped) {
            logger.warn('attachment_skipped', {
              sessionId,
              messageId,
              source: skip.source,
              error: skip.reason,
            })
          }
        }

        let agentInput = finalPrompt
        if (contentParts?.length) {
          const { encodeMultimodalInput } = await import('../../agent/multimodal-input.js')
          agentInput = encodeMultimodalInput([{ type: 'text', text: finalPrompt }, ...contentParts])
        }

        const previousContext = reqSessionId
          ? await loadAutoCompactedSessionContext({
              sessionStore: runtime.sessions,
              sessionId,
              provider,
              model: selectedModel,
              hooks: runtime.hookRegistry,
              maxMessages: resolveSessionContextMaxMessages(),
            })
          : null
        const previousMessages = previousContext?.messages ?? []
        const previousRunContract = previousContext?.runContract

        await runtime.sessions.appendEvent(sessionId, {
          type: 'user_message',
          id: messageId,
          timestamp: new Date().toISOString(),
          content: finalPrompt,
          ...(sessionAttachments.length > 0 ? { attachments: sessionAttachments } : {}),
        })

        // ── Intent routing ────────────────────────────────────────────────
        // Build an IntentRouter when the config enables it and the router
        // provider/model are resolvable. A misconfigured router must never
        // block the chat turn — log and proceed with intentRouter=null so the
        // kill-switch path in applyIntentRouting takes effect.
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
                personaList: () => resolvePersonaCatalog(customAgents, persistedPersonas),
                currentAutonomy: effectiveAutonomy,
                cwd: requestCwd,
                workspaceRoot: requestWorkspaceRoot,
                timeoutMs: routerCfg.timeoutMs,
                maxPreviousMessages: routerCfg.maxPreviousMessages,
                perMessageCharLimit: routerCfg.perMessageCharLimit,
                reasonMaxChars: routerCfg.reasonMaxChars,
                // The module-level logger has typed params; cast to the
                // IntentRouterOptions.logger shape which uses spread unknown.
                logger: logger as unknown as IntentRouterOptions['logger'],
                circuitBreaker: INTENT_ROUTER_CIRCUIT_BREAKER,
                auxiliaryLlmBudget,
                activeRunContract: previousContext?.runContract,
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
          // Extracted text attachments are graph-safe context. Only provider-
          // native image/document parts bypass the plain-text intent router.
          isMultimodal: contentParts
            ? requiresNativeMultimodalInput(contentParts)
            : false,
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

        // Persist the routing decision immediately after user_message so the
        // session journal records what drove this turn's configuration.
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
          },
        })

        // ── Skill prefix ──────────────────────────────────────────────────
        // Resolved after routing so that the effective skill refs (which the
        // router may have augmented) are used. Unknown refs still cause a
        // 404 here — the run lease is already held so the error terminates
        // the request cleanly.
        let skillPrefix = ''
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
          skillPrefix += resolveInitialExecutionMode(routing, runtime.config.agent.mode) === 'instant' ? '' : await resolveOfficeAutoSkillContent(
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
          skillPrefix += await resolveImageGeneratorAutoSkillContent(
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

        const { buildSystemPrompt } = await import('../../agent/system-prompt.js')
        const { collectAgentsMd, formatAgentsMdSection } = await import('../../memory/agents-md.js')
        const { withContextualToolExposure, withAuthorizedToolExposure, withDirectApiMemoryToolset, withSwarmToolsForSession, withWritingDocTools } =
          await import('../../tools/role-filter.js')
        const { getDocRegistry } = await import('../../agent/doc/session.js')
        // Direct-API surfaces (CLI/web/desktop) default to a lean memory toolset;
        // the full ~40-tool set is the channel/auto-memory concept. Channels go
        // through the pipeline and are unaffected by this.
        const directApiToolRegistry = withWritingDocTools(
          withSwarmToolsForSession(
            withDirectApiMemoryToolset(
              runtime.toolRegistry,
              runtime.config.memory.directApiToolset,
            ),
            sessionId,
          ),
          // The same expression the doc.* tools resolve with; this route carries
          // no writingDocId of its own.
          getDocRegistry().getActiveId(),
        )
        const persona = resolvePersona(
          routing.effectivePersona,
          customAgents,
          persistedPersonas,
        )
        const activeRemoteBrowser = remoteBrowserBridge(runtime.toolRegistry).list().some(
          (connection) => connection.sessionId === sessionId,
        )
        const initialMode = resolveInitialExecutionMode(routing, runtime.config.agent.mode, activeRemoteBrowser)
        const authorizedToolRegistry = withSelectedBrowserTarget(withAuthorizedToolExposure(directApiToolRegistry, {
          explicitToolNames: body.toolNames, declaredSkillToolNames,
          personaAllowedTools: persona?.allowedTools, personaDeniedTools: persona?.deniedTools,
        }), activeRemoteBrowser)
        const agentToolRegistry = withContextualToolExposure(authorizedToolRegistry, {
          surface: requestSurface,
          semanticRouting: true,
          activeRemoteBrowser,
          selectedGroups: routing.decision.toolGroups,
          routedMode: initialMode,
          routingFallback: routing.decision.fallback,
          activeWritingDocument: Boolean(getDocRegistry().getActiveId()),
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
          customInstructions: persona?.systemPromptAddition,
          projectContext,
          cwd: requestCwd,
          workspaceRoot: requestWorkspaceRoot,
          sessionId,
          closedToolSurface,
          // Session history already supplies same-session context; journals
          // can aggregate unrelated conversations from the same day.
          includeDailyNotes: false,
        })
        const systemPrompt = skillPrefix + baseSystemPrompt
        const strictFinalAnswerProtocol =
          process.env.SEPILOTD_STRICT_ANSWER_PROTOCOL === '1' || skillPrefix.trim().length > 0

        const approvalCallback: ApprovalCallback = (toolCall, requestId, options) =>
          runtime.approvalRegistry.waitForApproval({
            sessionId,
            toolCall,
            requestId,
            runId: messageId,
            forcePrompt: options?.forcePrompt,
            signal: options?.signal,
          })
        const requestQuestion = createQuestionRequester(runtime.questions)

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
            systemPrompt,
            previousMessages,
            maxIterations: resolveChatMaxIterations(body),
            hardMaxIterations: body.maxIterations !== undefined,
            auditLogger: runtime.auditLogger,
            usageTracker: runtime.usageTracker,
            spendBudget: runtime.config.limits,
            hookRegistry: runtime.hookRegistry,
            deviceName: runtime.config.device.name,
            thinkingLevel: body.thinkingLevel,
            maxTokens: body.maxTokens,
            temperature: body.temperature,
            textDeltaMode: body.textDeltaMode,
            ragEnabled: body.ragEnabled,
            // 비-streaming /chat 경로에서도 persona-panel을 정상 동작시키려면
            // 검증된 세션/요청 로스터를 graph context에 넘겨야 한다.
            panelPersonas: resolvedPanelPersonas,
            panelStrategy: body.panelStrategy,
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
              runtime.approvalRegistry.tryAutoApproval({ sessionId, toolCall, runId: messageId }),
            requestQuestion,
            ...createRuntimeBackedModeRouterOptions(runtime),
            // recovery branch adds the strict-final-answer toggle; everything
            // else matches the shared helper above.
            strictFinalAnswerProtocol,
            reviewToollessFinals: true,
          })

        const outputTracker = createAgentOutputTracker()
        let totalUsage = { inputTokens: 0, outputTokens: 0 }
        let stopReason: RunStopReason | undefined
        const toolCalls: ToolCall[] = []
        let activeSelection = selection
        let terminalError: ApiError | null = null
        let backgroundProgressEventCount = 0
        let backgroundPartialContent = ''
        let backgroundPartialLastWriteAt = 0

        const requestMode = routing.effectiveMode as AgentMode | undefined
        const consumeEvent = async (event: AgentEvent): Promise<void> => {
          await persistAgentSessionEvent(runtime.sessions, sessionId, event)
          if (backgroundChatRequest) {
            backgroundProgressEventCount += 1
            let forcePartialWrite = false
            if (event.type === 'text_delta' && typeof event.text === 'string') {
              backgroundPartialContent = `${backgroundPartialContent}${event.text}`.slice(
                -BACKGROUND_CHAT_PARTIAL_CONTENT_LIMIT,
              )
              const now = Date.now()
              if (
                backgroundPartialContent &&
                (backgroundPartialLastWriteAt === 0 ||
                  now - backgroundPartialLastWriteAt >= BACKGROUND_CHAT_PARTIAL_WRITE_INTERVAL_MS)
              ) {
                backgroundPartialLastWriteAt = now
                forcePartialWrite = true
              }
            }
            updateBackgroundChatProgress(
              getBackgroundChatStore(app),
              backgroundChatJobId,
              event,
              backgroundProgressEventCount,
              backgroundPartialContent || undefined,
              forcePartialWrite,
            )
          }
          outputTracker.consume(event)
          if (event.type === 'done') {
            totalUsage = event.usage
            stopReason = event.stopReason
          }
          if (event.type === 'tool_call') toolCalls.push(event.toolCall)
          if (event.type === 'error') {
            terminalError = event.error
          }
        }

        // Non-streaming /chat had no cancellation path: if the client
        // disconnected mid-run, the agent kept running to completion and held
        // its run lease. Abort the active run on disconnect. Background chat is
        // exempt — it is designed to survive the client going away and be polled
        // later — so only wire this for a foreground request.
        let clientDisconnected = false
        let activeRunRouter: AgentModeRouter | null = null
        const releaseDisconnect = backgroundChatRequest
          ? () => {}
          : registerSseDisconnectHandler(request, reply, () => {
              clientDisconnected = true
              void activeRunRouter?.stop().catch(() => {})
            })

        try {
          for (let attemptIndex = 0; attemptIndex < selections.length; attemptIndex += 1) {
            const attemptSelection = selections[attemptIndex]!
            const nextSelection = selections[attemptIndex + 1]
            const attempt = attemptIndex + 1
            const modeRouter = createModeRouter(attemptSelection)
            activeRunRouter = modeRouter
            const buffer: AgentEvent[] = []
            let committed = false
            let retrying = false
            const unregisterBackgroundRun = registerBackgroundChatRun(
              app,
              backgroundChatJobId,
              modeRouter,
            )

            await appendChatProviderAttemptEvent(
              runtime.sessions,
              sessionId,
              attemptSelection,
              attempt,
              'started',
            )

            try {
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
                  ...(backgroundChatJobId ? { backgroundJobId: backgroundChatJobId } : {}),
                  ...(requestSurface ? { surface: requestSurface } : {}),
                  systemPrompt,
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
                },
                requestMode,
              )) {
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
                  await consumeEvent(
                    providerRetryThinkingEvent(attemptSelection, nextSelection, event.error),
                  )
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
                  for (const bufferedEvent of buffer) {
                    await consumeEvent(bufferedEvent)
                  }
                  buffer.length = 0
                  continue
                }

                await consumeEvent(event)
              }
            } finally {
              unregisterBackgroundRun()
            }

            if (clientDisconnected) {
              // The socket is gone; the run was already stopped. Stop iterating —
              // the response below will simply no-op on the closed connection.
              break
            }

            if (retrying) {
              continue
            }

            if (isBackgroundChatCancelled(app, backgroundChatJobId)) {
              terminalError = {
                code: 'CONFLICT',
                message: 'Background chat was cancelled.',
                details: { reason: 'BACKGROUND_CHAT_CANCELLED' },
              }
              break
            }

            if (!committed) {
              activeSelection = attemptSelection
              for (const bufferedEvent of buffer) {
                await consumeEvent(bufferedEvent)
              }
            }
            break
          }
        } finally {
          releaseDisconnect()
        }

        const finalTerminalError = terminalError as ApiError | null
        if (finalTerminalError) {
          recordChatProviderFailure(runtime, activeSelection, finalTerminalError)
          await appendChatProviderAttemptEvent(
            runtime.sessions,
            sessionId,
            activeSelection,
            activeSelection.rank + 1,
            'failed',
            { error: finalTerminalError, retryable: false },
          )
          if (!backgroundChatRequest) {
            try {
              publishChatCompletionNotification(runtime.config, {
                outcome: 'failed',
                sessionId,
                sessionTitle: session.title,
                durationMs: Date.now() - chatStartedAt,
                surface: requestSurface,
                errorMessage: finalTerminalError.message ?? finalTerminalError.code,
              })
            } catch {
              // best-effort notification only
            }
          }

          const retainedEvidenceIncomplete = backgroundChatRequest
            ? null
            : outputTracker.providerFailureContent(finalTerminalError)
          if (retainedEvidenceIncomplete) {
            await persistAgentSessionEvent(runtime.sessions, sessionId, {
              type: 'recovery',
              scope: 'output_synthesis',
              kind: 'provider_failure_with_retained_evidence',
              action: 'synthesize_from_retained_evidence',
              message: 'The provider failed after successful tool execution; closing the foreground turn with a deterministic incomplete response from retained evidence.',
              recoverable: true,
              details: {
                provider: activeSelection.provider.id,
                model: activeSelection.model,
                errorCode: finalTerminalError.code,
              },
            })
            await runtime.sessions.appendEvent(sessionId, {
              type: 'assistant_message',
              id: randomUUID(),
              timestamp: new Date().toISOString(),
              content: retainedEvidenceIncomplete,
            })
            await persistAgentSessionEvent(runtime.sessions, sessionId, {
              type: 'done',
              usage: totalUsage,
            })
            await updateSessionTitleFromFirstTurn({
              sessions: runtime.sessions,
              session,
              provider: activeSelection.provider,
              model: activeSelection.model,
              firstMessage: finalPrompt,
              assistantReply: retainedEvidenceIncomplete,
            })
            return {
              data: {
                sessionId,
                messageId,
                content: retainedEvidenceIncomplete,
                toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
                usage: totalUsage,
                routerDecision:
                  routing.decision.fallback && routing.decision.reason === 'router disabled'
                    ? undefined
                    : {
                        mode: routing.decision.mode,
                        persona: routing.decision.persona,
                        skillIds: routing.decision.skillIds,
                        toolGroups: routing.decision.toolGroups,
                        confidence: routing.decision.confidence,
                        reason: routing.decision.reason,
                        fallback: routing.decision.fallback,
                        latencyMs: routing.decision.latencyMs,
                      },
              },
            }
          }

          await runtime.sessions.updateMeta?.(sessionId, { status: 'abandoned' })
          return reply
            .status(finalTerminalError.code === 'SERVICE_UNAVAILABLE' ? 503 : 500)
            .send({ error: finalTerminalError })
        }
        recordChatProviderSuccess(runtime, activeSelection)
        await appendChatProviderAttemptEvent(
          runtime.sessions,
          sessionId,
          activeSelection,
          activeSelection.rank + 1,
          'succeeded',
        )
        await updateChatSessionProviderMeta(runtime.sessions, sessionId, activeSelection)

        const syntheticMessage = outputTracker.syntheticMessageEvent()
        if (syntheticMessage) {
          await persistAgentSessionEvent(runtime.sessions, sessionId, syntheticMessage)
        }
        const content = outputTracker.finalContent()

        await extractAndStoreArtifacts(runtime, sessionId, content)

        await runtime.sessions.appendEvent(sessionId, {
          type: 'assistant_message',
          id: randomUUID(),
          timestamp: new Date().toISOString(),
          content,
        })
        await updateSessionTitleFromFirstTurn({
          sessions: runtime.sessions,
          session,
          provider: activeSelection.provider,
          model: activeSelection.model,
          firstMessage: finalPrompt,
          assistantReply: content,
        })
        triggerDreamingTurn(runtime.dreaming, sessionId, 'chat', scopeTags)
        if (!backgroundChatRequest) {
          try {
            publishChatCompletionNotification(runtime.config, {
              outcome: 'completed',
              sessionId,
              sessionTitle: session.title,
              durationMs: Date.now() - chatStartedAt,
              surface: requestSurface,
            })
          } catch {
            // best-effort notification only
          }
        }

        return {
          data: {
            sessionId,
            messageId,
            content,
            toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
            usage: totalUsage,
            stopReason,
            // SSE callers see this via the `router_decision` event; surface
            // the same payload to JSON callers so automation / tests can
            // observe when the router overrode the caller's `mode` / `persona`
            // / `skillRefs` hint. Omitted when routing was skipped entirely.
            routerDecision:
              routing.decision.fallback && routing.decision.reason === 'router disabled'
                ? undefined
                : {
                    mode: routing.decision.mode,
                    persona: routing.decision.persona,
                    skillIds: routing.decision.skillIds,
                    toolGroups: routing.decision.toolGroups,
                    confidence: routing.decision.confidence,
                    reason: routing.decision.reason,
                    fallback: routing.decision.fallback,
                    latencyMs: routing.decision.latencyMs,
                  },
          },
        }
      } catch (error) {
        const failedSession = await runtime.sessions.get(sessionId).catch(() => null)
        if (failedSession?.status === 'active') {
          await runtime.sessions.updateMeta?.(sessionId, { status: 'abandoned' })
        }
        throw error
      } finally {
        runLease?.release()
        sessionLeaseLifecycle.releaseRun()
        if (idempotencyRunKey) {
          idempotencyRuns?.delete(idempotencyRunKey)
        }
      }
    },
  )
}
