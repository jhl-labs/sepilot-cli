import { modelCatalogCost, modelMeetsRoutingRequirements, resolveModelRoutingHints, type ModelRoutingHints } from '../../providers/model-routing-policy.js'
import { randomUUID } from 'node:crypto'
import type {
  AgentEvent,
  AgentRunContract,
  ApiError,
  ILLMProvider,
  ISessionStore,
  SessionMeta,
} from '@sepilotd/core'
import { isProviderFallbackError } from '../../providers/model-fallback.js'
import {
  isProviderModelImageInputRejected,
  markProviderModelImageInputRejected,
} from '../../providers/vision-capability-state.js'
import {
  contractRequiresRenderedUiValidation,
  inputLooksLikeContextualFollowup,
  inputRequiresRenderedUiValidation,
} from '../../agent/task-contract.js'
import type { RuntimeServices } from '../runtime/types.js'

export type ChatProviderSelectionSource =
  | 'request_provider'
  | 'session'
  | 'model_router'
  | 'default_provider'

export interface ChatProviderSelection {
  provider: ILLMProvider
  model: string
  source: ChatProviderSelectionSource
  rank: number
  reason?: string
}

export type ChatProviderAttemptStatus = 'started' | 'failed' | 'succeeded'

export interface NoProviderError {
  code: 'NO_PROVIDER_CONFIGURED'
  message: string
}

export interface ProviderSelectionFailure {
  statusCode: 404 | 422 | 503
  error: NoProviderError | {
    code: 'PROVIDER_NOT_FOUND' | 'MODEL_NOT_FOUND' | 'MODEL_REQUIREMENTS_UNSATISFIED'
    message: string
    providerId?: string
    modelId?: string
    availableModels?: string[]
  }
}

export function chatRequestPrefersVision(
  message: string,
  mode?: string,
  runContract?: AgentRunContract,
): boolean {
  if (mode === 'computer-use') return true
  if (
    contractRequiresRenderedUiValidation(runContract)
    && inputLooksLikeContextualFollowup(message)
  ) {
    return true
  }
  if (inputRequiresRenderedUiValidation(message)) return true
  return /(?:browser\.screenshot|screenshot|screen shot|visual(?:ly)?\s+inspect|inspect(?:ing)?\s+(?:the\s+)?(?:actual\s+)?(?:image|screenshot)|rendered\s+ui|desktop\s+\d{3,4}x\d{3,4}|mobile\s+\d{3,4}x\d{3,4}|스크린샷|브라우저.*(?:검증|검사)|화면\s*(?:검증|검사)|뷰포트)/iu.test(message)
}

function isImageInputUnsupportedApiError(error: ApiError): boolean {
  const message = error.message.toLowerCase()
  return (
    message.includes('does not support image input') ||
    message.includes('image input is not supported') ||
    message.includes('image input not supported') ||
    (message.includes('unsupported') && message.includes('image')) ||
    (message.includes('vision') && message.includes('not supported'))
  )
}

export function buildNoProviderError(
  runtime: { config?: Pick<RuntimeServices['config'], 'providers'> },
): NoProviderError {
  const hasOllamaEntry = runtime.config?.providers
    ?.some((provider) => provider.type === 'ollama') ?? false
  const lines = [
    'No LLM provider is configured, so this request cannot be answered.',
    'Add at least one provider under `providers:` in ~/.sepilotd/config.yaml, then restart the daemon.',
    hasOllamaEntry
      ? 'An Ollama provider is present but exposed no usable model — start Ollama and `ollama pull` a model (e.g. `ollama pull qwen3`), then retry.'
      : 'For a local setup: install Ollama (https://ollama.com), run `ollama pull qwen3`, and add a providers entry `{ type: ollama, baseUrl: http://127.0.0.1:11434 }`.',
    'For a hosted provider, add its entry with the API key (or set the matching provider environment variable).',
  ]
  return { code: 'NO_PROVIDER_CONFIGURED', message: lines.join(' ') }
}

function chatModelIds(provider: ILLMProvider): string[] {
  return provider.models
    .filter((model) => model.capabilities.embedding !== true || model.capabilities.toolUse === true)
    .map((model) => model.id)
}

export function buildProviderSelectionFailure(
  runtime: Pick<RuntimeServices, 'providerRegistry'> & {
    config?: RuntimeServices['config']
  },
  requestedProvider?: string,
  requestedModel?: string,
  routing?: ModelRoutingHints,
): ProviderSelectionFailure {
  if (requestedProvider && !runtime.providerRegistry.get(requestedProvider)) {
    return {
      statusCode: 404,
      error: {
        code: 'PROVIDER_NOT_FOUND',
        message: `Provider "${requestedProvider}" is not registered. Run \`sepilot providers\` and choose a configured provider. No provider request was sent.`,
        providerId: requestedProvider,
      },
    }
  }

  if (requestedModel) {
    const candidates = requestedProvider
      ? [runtime.providerRegistry.get(requestedProvider)].filter(
          (provider): provider is ILLMProvider => Boolean(provider),
        )
      : runtime.providerRegistry.list()
    const authoritative = candidates.filter(
      (provider) => provider.modelCatalogAuthority === 'endpoint',
    )
    if (
      authoritative.length > 0
      && !candidates.some((provider) => provider.models.some((model) => model.id === requestedModel))
    ) {
      const providerId = requestedProvider ?? authoritative[0]!.id
      const availableModels = Array.from(new Set(
        authoritative.flatMap((provider) => chatModelIds(provider)),
      )).slice(0, 20)
      return {
        statusCode: 404,
        error: {
          code: 'MODEL_NOT_FOUND',
          message: `Model "${requestedModel}" is not present in the current endpoint catalog for provider "${providerId}". Run \`sepilot providers\` and choose an available model. No provider request was sent.`,
          providerId,
          modelId: requestedModel,
          availableModels,
        },
      }
    }
  }

  const requirements = resolveModelRoutingHints(runtime.config?.agent?.modelRouting?.defaults, routing)
  if (runtime.providerRegistry.list().length && (requirements.requireToolUse || requirements.minContextWindow !== undefined || requirements.maxInputCostPer1k !== undefined || requirements.maxOutputCostPer1k !== undefined)) {
    return { statusCode: 422, error: { code: 'MODEL_REQUIREMENTS_UNSATISFIED', message: 'No available selected model satisfies the configured capabilities and catalog cost limits. Review modelRouting and provider catalog metadata.', providerId: requestedProvider, modelId: requestedModel } }
  }
  return { statusCode: 503, error: buildNoProviderError(runtime) }
}

export function selectChatProviderCandidates(args: {
  runtime: RuntimeServices
  message: string
  requestedProvider?: string
  requestedModel?: string
  existingSession?: SessionMeta | null
  requireVision?: boolean
  preferVision?: boolean
  routing?: ModelRoutingHints
}): ChatProviderSelection[] {
  const { runtime, message, requestedProvider, requestedModel, existingSession, requireVision, preferVision } =
    args
  const routing = resolveModelRoutingHints(runtime.config?.agent?.modelRouting?.defaults, args.routing)
  const shouldPreferVision = Boolean(preferVision && !requestedModel && !requireVision)

  const modelInfoHasRequiredCapabilities = (
    provider: ILLMProvider,
    modelInfo: ILLMProvider['models'][number],
  ): boolean => {
    if (modelInfo.capabilities.embedding === true && modelInfo.capabilities.toolUse !== true) {
      return false
    }
    if (!modelMeetsRoutingRequirements(modelInfo, routing)) return false
    if (requireVision && !modelInfo.capabilities.vision) {
      return false
    }
    if (
      (requireVision || shouldPreferVision)
      && modelInfo.capabilities.vision === true
      && isProviderModelImageInputRejected(provider.id, modelInfo.id)
    ) {
      return false
    }
    return true
  }

  const modelHasRequiredCapabilities = (provider: ILLMProvider, model: string): boolean => {
    const knownModel = provider.models.find((candidate) => candidate.id === model)
    if (!knownModel) return !requireVision && !routing.requireToolUse && routing.minContextWindow === undefined && routing.maxInputCostPer1k === undefined && routing.maxOutputCostPer1k === undefined
    return modelInfoHasRequiredCapabilities(provider, knownModel)
  }
  const modelHasVision = (provider: ILLMProvider, model: string): boolean =>
    provider.models.find((candidate) => candidate.id === model)?.capabilities.vision === true

  const visionPreferredModel = (provider: ILLMProvider): string | null => {
    if (!shouldPreferVision) return null
    return provider.models.find(
      (modelInfo) =>
        modelInfo.capabilities.vision === true && modelInfoHasRequiredCapabilities(provider, modelInfo),
    )?.id ?? null
  }

  const selectModelForProvider = (
    provider: ILLMProvider,
    preferredModel?: string,
  ): string | null => {
    const visionModel = visionPreferredModel(provider)
    if (visionModel) return visionModel
    const knownPreferredModel = preferredModel
      ? provider.models.find((candidate) => candidate.id === preferredModel)
      : undefined
    if (knownPreferredModel && !modelMeetsRoutingRequirements(knownPreferredModel, routing)) return null
    if (knownPreferredModel && modelInfoHasRequiredCapabilities(provider, knownPreferredModel)) {
      return knownPreferredModel.id
    }
    if (preferredModel && !knownPreferredModel) {
      if (requestedModel && runtime.providerRegistry.list().some(candidate => candidate.models.some(model => model.id === preferredModel))) return null
      if (provider.modelCatalogAuthority === 'endpoint') return null
      if (modelHasRequiredCapabilities(provider, preferredModel)) return preferredModel
    }
    const eligible = provider.models.filter(modelInfo => modelInfoHasRequiredCapabilities(provider, modelInfo))
    if (routing.preference === 'lowest-cost') eligible.sort((a, b) => modelCatalogCost(a) - modelCatalogCost(b))
    return eligible[0]?.id ?? null
  }
  const sortVisionPreferred = (selections: ChatProviderSelection[]): ChatProviderSelection[] => {
    if (!shouldPreferVision) return selections
    return selections
      .map((selection, index) => ({ selection, index }))
      .sort((a, b) => {
        const aVision = modelHasVision(a.selection.provider, a.selection.model)
        const bVision = modelHasVision(b.selection.provider, b.selection.model)
        if (aVision !== bVision) return aVision ? -1 : 1
        return a.index - b.index
      })
      .map(({ selection }, rank) => ({ ...selection, rank }))
  }

  if (requestedProvider) {
    const provider = runtime.providerRegistry.get(requestedProvider)
    if (!provider) return []
    const candidateModel = selectModelForProvider(provider, requestedModel)
    if (!candidateModel) return []
    return [
      {
        provider,
        model: candidateModel,
        source: 'request_provider',
        rank: 0,
      },
    ]
  }

  const isCandidateAvailable = (providerId: string, model: string): boolean =>
    runtime.modelFallbackState?.isAvailable({ providerId, model }) ?? true

  if (existingSession) {
    const pinnedProvider =
      runtime.providerRegistry.get(existingSession.provider) ??
      runtime.providerRegistry.getDefault()
    const pinnedModelName = requestedModel ?? existingSession.model
    const results: ChatProviderSelection[] = []
    const seen = new Set<string>()

    if (pinnedProvider) {
      const pinnedModel = selectModelForProvider(pinnedProvider, pinnedModelName)
      if (pinnedModel && isCandidateAvailable(pinnedProvider.id, pinnedModel)) {
        results.push({ provider: pinnedProvider, model: pinnedModel, source: 'session', rank: 0 })
        seen.add(`${pinnedProvider.id}:${pinnedModel}`)
      }
    }

    const fallbackTail =
      runtime.modelRouter?.selectCandidates(message, pinnedModelName, {
        ...routing,
        includeFallbacksForExplicitModel: !requestedModel,
        isCandidateAvailable: (candidate) =>
          modelHasRequiredCapabilities(candidate.provider, candidate.model) &&
          isCandidateAvailable(candidate.providerId, candidate.model),
      }) ?? []
    for (const candidate of fallbackTail) {
      const key = `${candidate.providerId}:${candidate.model}`
      if (seen.has(key)) continue
      seen.add(key)
      results.push({
        provider: candidate.provider,
        model: candidate.model,
        source: 'model_router',
        reason: `${routing.taskType ?? 'general'}:${candidate.reason}`,
        rank: results.length,
      })
    }

    if (results.length > 0) {
      return sortVisionPreferred(results)
    }
    if (requestedModel) return []
  }

  const configuredDefaultModel =
    !requestedModel && !existingSession
      ? runtime.config?.agent.defaultModel
      : undefined
  const hasTaskRoute = runtime.config?.agent?.modelRouting?.routes?.some(route => route.taskType === routing.taskType)
  const routedPreferredModel = requestedModel ?? (hasTaskRoute || routing.preference === 'lowest-cost' ? undefined : configuredDefaultModel)
  const routed = runtime.modelRouter
    ? runtime.modelRouter.selectCandidates(message, routedPreferredModel, {
        ...routing,
        includeFallbacksForExplicitModel:
          Boolean(configuredDefaultModel) && !requestedModel,
        isCandidateAvailable: (candidate) =>
          modelHasRequiredCapabilities(candidate.provider, candidate.model) &&
          (runtime.modelFallbackState?.isAvailable({
            providerId: candidate.providerId,
            model: candidate.model,
          }) ??
            true),
      })
    : null
  if (routed?.length) {
    return sortVisionPreferred(routed.map((candidate, index) => ({
      provider: candidate.provider,
      model: candidate.model,
      source: 'model_router',
      reason: `${routing.taskType ?? 'general'}:${candidate.reason}`,
      rank: index,
    })))
  }

  if (routedPreferredModel && runtime.modelRouter && !routed?.length) return []
  const provider = runtime.providerRegistry.getDefault()
  if (!provider) return []
  const candidateModel = selectModelForProvider(provider, requestedModel)
  if (!candidateModel) return []
  return sortVisionPreferred([
    {
      provider,
      model: candidateModel,
      source: 'default_provider',
      rank: 0,
    },
  ])
}

export function recordChatProviderSuccess(
  runtime: RuntimeServices,
  selection: ChatProviderSelection,
): void {
  runtime.modelFallbackState?.recordSuccess({
    providerId: selection.provider.id,
    model: selection.model,
  })
}

export function recordChatProviderFailure(
  runtime: RuntimeServices,
  selection: ChatProviderSelection,
  error: ApiError,
): void {
  if (isImageInputUnsupportedApiError(error)) {
    markProviderModelImageInputRejected(selection.provider.id, selection.model, error.message)
  }
  if (!isProviderFallbackError(error)) return
  runtime.modelFallbackState?.recordFailure(
    {
      providerId: selection.provider.id,
      model: selection.model,
    },
    new Error(error.message),
  )
}

export function canRetryChatProviderAttempt(
  selection: ChatProviderSelection,
  error: ApiError,
  committed: boolean,
  hasNextCandidate: boolean,
): boolean {
  return (
    (selection.source === 'model_router' || selection.source === 'session') &&
    hasNextCandidate &&
    !committed &&
    (isProviderFallbackError(error) || isImageInputUnsupportedApiError(error))
  )
}

export function isAgentAttemptCommitted(event: AgentEvent): boolean {
  return (
    event.type !== 'state_change' &&
    event.type !== 'thinking' &&
    event.type !== 'memory_context' &&
    event.type !== 'llm_request' &&
    event.type !== 'context_usage' &&
    event.type !== 'mode_route_decision' &&
    event.type !== 'quality_gate_verdict' &&
    event.type !== 'backtrack' &&
    event.type !== 'node_trace' &&
    event.type !== 'recovery' &&
    event.type !== 'run_contract' &&
    event.type !== 'view_attached' &&
    event.type !== 'view_detached' &&
    event.type !== 'view_still' &&
    event.type !== 'error'
  )
}

export function providerRetryThinkingEvent(
  failed: ChatProviderSelection,
  next: ChatProviderSelection,
  error: ApiError,
): Extract<AgentEvent, { type: 'thinking' }> {
  return {
    type: 'thinking',
    content: [
      `Provider ${failed.provider.id}/${failed.model} failed before producing output.`,
      `Retrying with ${next.provider.id}/${next.model}.`,
      `Reason: ${error.message}`,
    ].join(' '),
  }
}

export async function appendChatProviderAttemptEvent(
  sessions: ISessionStore,
  sessionId: string,
  selection: ChatProviderSelection,
  attempt: number,
  status: ChatProviderAttemptStatus,
  options: {
    error?: ApiError
    retryable?: boolean
    nextSelection?: ChatProviderSelection
  } = {},
): Promise<void> {
  await sessions.appendEvent(sessionId, {
    type: 'provider_attempt',
    id: randomUUID(),
    timestamp: new Date().toISOString(),
    provider: selection.provider.id,
    model: selection.model,
    source: selection.source,
    rank: selection.rank,
    selectionReason: selection.reason ?? selection.source,
    attempt,
    status,
    retryable: options.retryable,
    errorCode: options.error?.code,
    errorMessage: options.error?.message,
    nextProvider: options.nextSelection?.provider.id,
    nextModel: options.nextSelection?.model,
  })
}

export async function updateChatSessionProviderMeta(
  sessions: ISessionStore,
  sessionId: string,
  selection: ChatProviderSelection,
): Promise<void> {
  await sessions.updateMeta?.(sessionId, {
    provider: selection.provider.id,
    model: selection.model,
  })
}
