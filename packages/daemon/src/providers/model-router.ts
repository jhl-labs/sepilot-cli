import type { ILLMProvider } from '@sepilotd/core'
import type { ProviderRegistry } from './registry.js'

export type TaskType = 'simple' | 'complex' | 'code' | 'creative' | 'analysis'
export type ModelSelectionReason = 'explicit_model' | 'preferred_model' | 'default_provider'

interface ModelRoute {
  taskType: TaskType
  preferredModels: string[]
}

export interface ModelCandidate {
  provider: ILLMProvider
  providerId: string
  model: string
  taskType: TaskType
  rank: number
  reason: ModelSelectionReason
}

export interface ModelCandidateOptions {
  includeFallbacksForExplicitModel?: boolean
  isCandidateAvailable?: (candidate: ModelCandidate) => boolean
}

const DEFAULT_ROUTES: ModelRoute[] = [
  { taskType: 'simple', preferredModels: ['gpt-4o-mini', 'llama3.3', 'gemini-2.0-flash', 'llama-3.1-8b-instant', 'claude-haiku'] },
  { taskType: 'complex', preferredModels: ['gpt-4o', 'claude-sonnet-4-20250514', 'gemini-2.5-pro', 'deepseek-chat', 'qwen3.5:35b'] },
  { taskType: 'code', preferredModels: ['gpt-4o', 'claude-sonnet-4-20250514', 'deepseek-chat', 'gemini-2.5-pro', 'llama-3.3-70b-versatile', 'qwen3-coder:30b'] },
  { taskType: 'creative', preferredModels: ['gpt-4o', 'claude-sonnet-4-20250514', 'gemini-2.5-pro'] },
  { taskType: 'analysis', preferredModels: ['gpt-4o', 'claude-sonnet-4-20250514', 'gemini-2.5-pro', 'deepseek-chat', 'qwen3.5:35b'] },
]

function modelIsChatCapable(model: ILLMProvider['models'][number]): boolean {
  return model.capabilities.embedding !== true || model.capabilities.toolUse === true
}

const DOWNSIZE_SUFFIXES = new Set(['mini', 'nano', 'small', 'lite', 'tiny', 'micro'])

function isDownsizedVariant(id: string, base: string): boolean {
  const firstSegment = id.slice(base.length + 1).split(/[-.:]/, 1)[0]?.toLowerCase()
  return firstSegment ? DOWNSIZE_SUFFIXES.has(firstSegment) : false
}

function matchModelId(
  models: ILLMProvider['models'],
  modelId: string,
): ILLMProvider['models'][number] | undefined {
  const exact = models.find((m) => m.id === modelId)
  if (exact) return exact
  return models.find(
    (m) => m.id.startsWith(modelId + '-') && !isDownsizedVariant(m.id, modelId),
  )
}

export class ModelRouter {
  private registry: ProviderRegistry
  private routes: ModelRoute[]

  constructor(registry: ProviderRegistry, routes?: ModelRoute[]) {
    this.registry = registry
    this.routes = routes ?? DEFAULT_ROUTES
  }

  classifyTask(message: string): TaskType {
    const lower = message.toLowerCase()

    if (/\b(code|function|class|implement|debug|fix|refactor|test|api|endpoint|component)\b/.test(lower)) return 'code'
    if (/```|\bdef\b|\bconst\b|\bimport\b/.test(message)) return 'code'

    if (/\b(analyze|compare|evaluate|review|assess|explain.*detail|pros.*cons)\b/.test(lower)) return 'analysis'

    if (/\b(write|create|generate|design|story|poem|email|blog|article)\b/.test(lower)) return 'creative'

    if (message.length > 500 || /\b(step.by.step|plan|architecture|strategy|comprehensive)\b/.test(lower)) return 'complex'

    return 'simple'
  }

  selectCandidates(
    message: string,
    explicitModel?: string,
    options: ModelCandidateOptions = {},
  ): ModelCandidate[] {
    const taskType = this.classifyTask(message)
    const candidates: ModelCandidate[] = []
    const seen = new Set<string>()

    const pushCandidate = (
      provider: ILLMProvider,
      model: string,
      reason: ModelSelectionReason,
    ) => {
      const candidate: ModelCandidate = {
        provider,
        providerId: provider.id,
        model,
        taskType,
        rank: candidates.length,
        reason,
      }
      const key = `${candidate.providerId}:${candidate.model}`
      if (seen.has(key)) return
      if (options.isCandidateAvailable && !options.isCandidateAvailable(candidate)) return
      seen.add(key)
      candidates.push(candidate)
    }

    if (explicitModel) {
      const defaultProvider = this.registry.getDefault()
      if (defaultProvider?.models.some(m => m.id === explicitModel)) {
        pushCandidate(defaultProvider, explicitModel, 'explicit_model')
      }
      for (const p of this.registry.list()) {
        if (p === defaultProvider) continue
        if (p.models.some(m => m.id === explicitModel)) {
          pushCandidate(p, explicitModel, 'explicit_model')
          break
        }
      }
      if (!options.includeFallbacksForExplicitModel) {
        return candidates
      }
    }

    const route = this.routes.find(r => r.taskType === taskType) ?? this.routes[0]

    for (const modelId of route.preferredModels) {
      for (const provider of this.registry.list()) {
        const model = matchModelId(provider.models, modelId)
        if (model) pushCandidate(provider, model.id, 'preferred_model')
      }
    }

    const defaultProvider = this.registry.getDefault()
    const fallbackProviders = defaultProvider
      ? [defaultProvider, ...this.registry.list().filter((p) => p !== defaultProvider)]
      : this.registry.list()
    for (const provider of fallbackProviders) {
      if (provider.models.length === 0) continue
      const defaultModel = provider.models.find(modelIsChatCapable) ?? provider.models[0]
      pushCandidate(provider, defaultModel.id, 'default_provider')
    }

    return candidates
  }

  selectModel(
    message: string,
    explicitModel?: string,
    options?: ModelCandidateOptions,
  ): { provider: ILLMProvider; model: string } | null {
    const candidate = this.selectCandidates(message, explicitModel, options)[0]
    return candidate ? { provider: candidate.provider, model: candidate.model } : null
  }

  getRoutingInfo(message: string): { taskType: TaskType; selectedModel?: string; candidateCount: number } {
    const taskType = this.classifyTask(message)
    const candidates = this.selectCandidates(message)
    return { taskType, selectedModel: candidates[0]?.model, candidateCount: candidates.length }
  }
}
