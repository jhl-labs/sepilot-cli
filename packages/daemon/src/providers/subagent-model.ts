import type { ProviderRegistry } from './registry.js'

export interface SubagentModelSelectionInput {
  providerId?: string
  model: string
  explicitModel: boolean
  explicitProvider: boolean
}

export function resolveSubagentModel(
  registry: ProviderRegistry,
  input: SubagentModelSelectionInput,
): { providerId: string; model: string } {
  const inherited = input.providerId ? registry.get(input.providerId) : registry.getDefault()
  if (!inherited)
    throw new Error('SUBAGENT_PROVIDER_UNAVAILABLE: the selected provider is not configured')
  const generates = (model: (typeof inherited.models)[number]) =>
    model.capabilities.embedding !== true || model.capabilities.toolUse === true
  if (
    input.explicitModel &&
    !input.explicitProvider &&
    !inherited.models.some((model) => model.id === input.model)
  ) {
    const matches = registry
      .list()
      .filter((provider) =>
        provider.models.some((model) => model.id === input.model && generates(model)),
      )
    if (matches.length !== 1)
      throw new Error(
        'SUBAGENT_MODEL_UNAVAILABLE: select an unambiguous configured provider and generation model',
      )
    return { providerId: matches[0]!.id, model: input.model }
  }
  const known = inherited.models.find((model) => model.id === input.model)
  if (known && !generates(known))
    throw new Error('SUBAGENT_MODEL_UNAVAILABLE: embedding-only models cannot run a child')
  return { providerId: inherited.id, model: input.model }
}
