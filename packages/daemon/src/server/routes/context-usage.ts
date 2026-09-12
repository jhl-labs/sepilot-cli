import { resolvePersonaMemoryScope } from '../../memory/persona-scope.js'
import { createPersonaRepo } from '../../persona/repo.js'
import type { FastifyInstance } from 'fastify'
import '../fastify-types.js'
import {
  loadAutoCompactedSessionContext,
  resolveSessionContextMaxMessages,
} from '../../agent/auto-compaction.js'
import { DEFAULT_UNKNOWN_MODEL_CONTEXT_WINDOW } from '../../agent/context-manager.js'
import { estimateContextUsage } from '../../agent/context-usage.js'
import { getDocRegistry } from '../../agent/doc/session.js'
import { gatherEnvironmentInfo } from '../../agent/environment.js'
import { resolveInstantModeToolNames } from '../../agent/instant-mode-tool-intent.js'
import { resolveClosedExactToolNames } from '../../agent/tool-call-budget.js'
import { buildSystemPrompt } from '../../agent/system-prompt.js'
import { collectAgentsMd, formatAgentsMdSection } from '../../memory/agents-md.js'
import { resolveScopedFileMemoryReadView } from '../../memory/scoped-file-memory-read-view.js'
import { tokenCalibration } from '../../providers/token-calibration.js'
import {
  withContextualToolExposure,
  withDirectApiMemoryToolset,
  withSwarmToolsForSession,
  withWritingDocTools,
} from '../../tools/role-filter.js'
import { readChatMemoryScope } from '../chat-memory-context.js'
import { buildShellModePrefix, buildWritingDocPrefix } from './chat-mode-prefixes.js'
import { z } from 'zod'

const contextUsageRequestSchema = z.object({
  sessionId: z.string().trim().min(1).max(200).optional(),
  mode: z.string().trim().min(1).max(80).optional(),
  provider: z.string().trim().min(1).max(80).optional(),
  model: z.string().trim().min(1).max(160).optional(),
  maxOutputTokens: z.number().int().positive().optional(),
  input: z.string().max(200_000).optional(),
}).strict()

const legacyContextUsageQuerySchema = contextUsageRequestSchema.omit({ input: true }).extend({
  maxOutputTokens: z.coerce.number().int().positive().optional(),
})

type ContextUsageRequest = z.infer<typeof contextUsageRequestSchema>

/** Read-only context inspector used by trusted local operator surfaces. */
export async function contextUsageRoutes(app: FastifyInstance) {
  app.route<{ Body: ContextUsageRequest; Querystring: ContextUsageRequest }>({
    method: ['GET', 'POST'],
    url: '/context-usage',
    handler: async (request, reply) => {
    if (request.authContext?.kind === 'extension') {
      return reply.status(403).send({
        error: {
          code: 'CONTEXT_USAGE_ADMIN_DENIED',
          message: 'Context usage inspection requires the daemon master token.',
        },
      })
    }

    const parsed = request.method === 'GET'
      ? legacyContextUsageQuerySchema.safeParse(request.query)
      : contextUsageRequestSchema.safeParse(request.body)
    if (!parsed.success) {
      return reply.status(400).send({
        error: { code: 'BAD_REQUEST', message: 'Invalid context usage request' },
      })
    }
    const body = parsed.data
    const input = 'input' in body && typeof body.input === 'string'
      ? body.input
      : undefined
    const runtime = app.runtime
    if (
      !runtime?.sessions
      || !runtime.config
      || !runtime.toolRegistry
      || !runtime.skillRegistry
    ) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
      })
    }

    const sessionId = body.sessionId
    const session = sessionId ? await runtime.sessions.get(sessionId) : null
    if (sessionId && !session) {
      return reply.status(404).send({
        error: { code: 'NOT_FOUND', message: 'Session not found' },
      })
    }

    const requestedProviderId = body.provider
    const requestedModelId = body.model
    const provider = (requestedProviderId
      ? runtime.providerRegistry?.get(requestedProviderId)
      : session?.provider
        ? runtime.providerRegistry?.get(session.provider)
        : runtime.providerRegistry?.getDefault())
      ?? runtime.providerRegistry?.getDefault()
    const providerId = requestedProviderId || session?.provider || provider?.id || 'unknown'
    const modelId = requestedModelId || session?.model || provider?.models[0]?.id || 'unknown'
    const modelInfo = provider?.models.find((candidate) => candidate.id === modelId)
    // The inspector must report the window the run loops actually plan
    // against; a friendlier guess here only produced a usage bar that
    // disagreed with the engine.
    const contextWindow = modelInfo?.contextWindow ?? DEFAULT_UNKNOWN_MODEL_CONTEXT_WINDOW
    const reservedOutputTokens = body.maxOutputTokens
      ? Math.min(body.maxOutputTokens, contextWindow)
      : Math.min(modelInfo?.maxOutputTokens ?? 4_096, contextWindow)
    const charsPerToken = tokenCalibration.charsPerToken(providerId, modelId)
    const messages = session
      ? (await loadAutoCompactedSessionContext({
          sessionStore: runtime.sessions,
          sessionId: session.id,
          provider,
          model: modelId,
          hooks: runtime.hookRegistry,
          maxMessages: resolveSessionContextMaxMessages(),
          charsPerToken,
          // Describing the context must not change it. Without this the
          // inspector would summarize the session with an LLM call and persist
          // a `context_compact` event as a side effect of being asked how full
          // the window is.
          inspectOnly: true,
        })).messages
      : []

    const activeWritingDocumentId = getDocRegistry().getActiveId()
    const activeWritingDocument = Boolean(activeWritingDocumentId)
    const mode = body.mode
    const explicitToolNames = resolveInstantModeToolNames(
      input ?? '',
      mode,
      undefined,
    )
    const directApiTools = withWritingDocTools(
      withSwarmToolsForSession(
        withDirectApiMemoryToolset(
          runtime.toolRegistry,
          runtime.config.memory.directApiToolset,
        ),
        session?.id,
      ),
      activeWritingDocumentId,
    )
    const contextualTools = withContextualToolExposure(directApiTools, {
      routedMode: mode && mode !== 'auto' ? mode : undefined,
      routingFallback: false,
      activeWritingDocument,
      swarmSession: session?.id.startsWith('swarm_') ?? false,
      requestInput: input,
      explicitToolNames,
    })
    const closedToolSurface = input
      ? resolveClosedExactToolNames(input, directApiTools.list()) !== null
      : false

    let scopeTags: string[]
    try { scopeTags = resolvePersonaMemoryScope(readChatMemoryScope(request.headers, request.authContext), { personaIds: session?.personaIds }, createPersonaRepo().list(), session).scopeTags }
    catch (error) { return reply.status(400).send({ error: { code: 'MEMORY_SPACE_MISMATCH', message: String(error) } }) }
    const scopedFileMemory = resolveScopedFileMemoryReadView(
      runtime.fileMemory,
      runtime.fileMemoryRegistry,
      scopeTags,
    )
    const now = new Date()
    const [promptMemoryContext, skillList, environment, agentsFiles] = await Promise.all([
      scopedFileMemory?.getPromptContext(now),
      runtime.skillRegistry.listForCwd(session?.cwd, session?.cwd),
      gatherEnvironmentInfo({ cwd: session?.cwd, workspaceRoot: session?.cwd, now }),
      session?.cwd
        ? collectAgentsMd({ cwd: session.cwd, boundaryRoot: session.cwd })
        : Promise.resolve([]),
    ])
    const cachedMemory = promptMemoryContext
      ? { getPromptContext: async () => promptMemoryContext }
      : undefined
    const cachedSkills = { listForCwd: async () => skillList }
    const emptySkills = { listForCwd: async () => [] }
    const agentsMemory = formatAgentsMdSection(agentsFiles) ?? undefined
    const shellPrefix = buildShellModePrefix(mode)
    const documentPrefix = buildWritingDocPrefix(
      mode,
      activeWritingDocumentId ?? undefined,
      session?.cwd,
    )
    const basePromptOptions = {
      config: runtime.config,
      tools: contextualTools,
      skills: cachedSkills,
      fileMemory: cachedMemory,
      agentsMemory,
      cwd: session?.cwd,
      workspaceRoot: session?.cwd,
      sessionId: session?.id,
      closedToolSurface,
      includeDailyNotes: false,
      now,
      environment,
    }
    const [baseSystemPrompt, baseWithoutMemory, baseWithoutSkills, baseWithoutWorkspace] =
      await Promise.all([
        buildSystemPrompt(basePromptOptions),
        buildSystemPrompt({ ...basePromptOptions, fileMemory: undefined }),
        buildSystemPrompt({ ...basePromptOptions, skills: emptySkills }),
        buildSystemPrompt({ ...basePromptOptions, agentsMemory: undefined }),
      ])
    const fullSystemPrompt = shellPrefix + documentPrefix + baseSystemPrompt
    const systemPromptWithoutMemory = shellPrefix + documentPrefix + baseWithoutMemory
    const systemPromptWithoutSkills = shellPrefix + documentPrefix + baseWithoutSkills
    const systemPromptWithoutWorkspace = shellPrefix + documentPrefix + baseWithoutWorkspace
    const systemPromptWithoutDocument = shellPrefix + baseSystemPrompt
    const tools = contextualTools.toToolDefinitions()
    const estimate = estimateContextUsage({
      fullSystemPrompt,
      systemPromptWithoutMemory,
      systemPromptWithoutSkills,
      systemPromptWithoutWorkspace,
      systemPromptWithoutDocument,
      messages,
      tools,
      charsPerToken,
      contextWindow,
      reservedOutputTokens,
      toolCount: tools.length,
    })

    return {
      estimatedAt: new Date().toISOString(),
      model: {
        providerId,
        modelId,
        contextWindow,
        contextWindowSource: modelInfo ? 'catalog' : 'fallback',
      },
      estimator: {
        charsPerToken,
        calibrated: charsPerToken !== 4,
      },
      ...estimate,
      dynamicContext: [
        ...(mode === 'auto' ? ['자동 모드의 요청별 도구 그룹'] : []),
        '최신 입력을 사용한 의미 메모리·RAG 검색 결과',
        '선택한 스킬·페르소나·첨부 파일 본문',
      ],
    }
    },
  })
}
