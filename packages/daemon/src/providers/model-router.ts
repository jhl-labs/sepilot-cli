import type { ILLMProvider } from '@sepilotd/core'
import {
  modelCatalogCost,
  modelMeetsRoutingRequirements,
  type ModelRoutingHints,
} from './model-routing-policy.js'
import type { ProviderRegistry } from './registry.js'

export type TaskType = 'general' | 'simple' | 'complex' | 'code' | 'creative' | 'analysis'
export type ModelSelectionReason =
  | 'explicit_model'
  | 'preferred_model'
  | 'default_provider'
  | 'lowest_catalog_cost'
export interface ModelRoute {
  taskType: TaskType
  preferredModels: string[]
  providerId?: string
}
export interface ModelCandidate {
  provider: ILLMProvider
  providerId: string
  model: string
  taskType: TaskType
  rank: number
  reason: ModelSelectionReason
}
export interface ModelCandidateOptions extends ModelRoutingHints {
  includeFallbacksForExplicitModel?: boolean
  isCandidateAvailable?: (candidate: ModelCandidate) => boolean
  taskType?: TaskType
  minContextWindow?: number
  requireToolUse?: boolean
}
function modelIsChatCapable(model: ILLMProvider['models'][number]): boolean {
  return model.capabilities.embedding !== true || model.capabilities.toolUse === true
}

export class ModelRouter {
  constructor(
    private registry: ProviderRegistry,
    private routes: ModelRoute[] = [],
  ) {}

  classifyTask(_message: string): TaskType {
    return 'general'
  }

  selectCandidates(
    message: string,
    explicitModel?: string,
    options: ModelCandidateOptions = {},
  ): ModelCandidate[] {
    const taskType = options.taskType ?? this.classifyTask(message)
    const candidates: ModelCandidate[] = []
    const seen = new Set<string>()
    const eligible = (model: ILLMProvider['models'][number]) =>
      modelIsChatCapable(model) && modelMeetsRoutingRequirements(model, options)
    const pushCandidate = (
      provider: ILLMProvider,
      model: ILLMProvider['models'][number],
      reason: ModelSelectionReason,
    ) => {
      if (!eligible(model)) return
      const key = JSON.stringify([provider.id, model.id])
      const candidate: ModelCandidate = {
        provider,
        providerId: provider.id,
        model: model.id,
        taskType,
        rank: candidates.length,
        reason,
      }
      if (
        seen.has(key) ||
        (options.isCandidateAvailable && !options.isCandidateAvailable(candidate))
      )
        return
      seen.add(key)
      candidates.push(candidate)
    }
    const primary = this.registry.getDefault()
    const providers = primary
      ? [primary, ...this.registry.list().filter((provider) => provider !== primary)]
      : this.registry.list()
    if (explicitModel) {
      for (const provider of providers) {
        const model = provider.models.find((entry) => entry.id === explicitModel)
        if (model) {
          pushCandidate(provider, model, 'explicit_model')
          if (candidates.length) break
        }
      }
      if (!options.includeFallbacksForExplicitModel) return candidates
    }
    const route = this.routes.find((entry) => entry.taskType === taskType)
    for (const modelId of route?.preferredModels ?? []) {
      for (const provider of providers) {
        if (route?.providerId && route.providerId !== provider.id) continue
        const model = provider.models.find((entry) => entry.id === modelId)
        if (model) pushCandidate(provider, model, 'preferred_model')
      }
    }
    for (const provider of providers) {
      for (const model of provider.models) {
        const before = candidates.length
        pushCandidate(provider, model, 'default_provider')
        if (
          options.preference !== 'lowest-cost' &&
          (candidates.length > before || seen.has(JSON.stringify([provider.id, model.id])))
        )
          break
      }
    }
    if (options.preference === 'lowest-cost') {
      const pinned = explicitModel
        ? candidates.filter((candidate) => candidate.reason === 'explicit_model')
        : []
      const others = candidates
        .filter((candidate) => !pinned.includes(candidate))
        .sort(
          (a, b) =>
            modelCatalogCost(a.provider.models.find((model) => model.id === a.model)!) -
            modelCatalogCost(b.provider.models.find((model) => model.id === b.model)!),
        )
      return [...pinned, ...others].map((candidate, rank) => ({
        ...candidate,
        rank,
        reason: candidate.reason === 'explicit_model' ? candidate.reason : 'lowest_catalog_cost',
      }))
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
  getRoutingInfo(message: string): {
    taskType: TaskType
    selectedModel?: string
    candidateCount: number
  } {
    const candidates = this.selectCandidates(message)
    return {
      taskType: this.classifyTask(message),
      selectedModel: candidates[0]?.model,
      candidateCount: candidates.length,
    }
  }
}
