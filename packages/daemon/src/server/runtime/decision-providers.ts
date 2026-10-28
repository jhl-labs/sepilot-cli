import type { IDecisionProvider } from '@sepilotd/core'
import type { SepilotdConfig } from '../../config/schema.js'
import { OpenAIDecisionsProvider } from '../../decisions/openai-decisions-provider.js'
import { DecisionProviderRegistry } from '../../decisions/registry.js'
import {
  AI_GATEWAY_DECISIONS_PATH,
  SystemOneDecisionProvider,
} from '../../decisions/systemone-provider.js'
import { createLogger } from '../../logger.js'
import {
  extractMissingConfigEnvReferences,
  extractMissingConfigHeaderEnvReferences,
  resolveConfigEnvReference,
  resolveConfigHeaderReferences,
} from './providers.js'

const log = createLogger('runtime.decision-providers')

export type DecisionProviderConfig = SepilotdConfig['decisionProviders'][number]

export const OLLAMA_DECISIONS_DEFAULT_BASE_URL = 'http://localhost:11434'

export interface BuildDecisionProviderResult {
  provider?: IDecisionProvider
  skipReason?: string
}

export function buildDecisionProviderInstance(
  config: DecisionProviderConfig,
  env: Record<string, string | undefined> = process.env,
): BuildDecisionProviderResult {
  const missing = [
    ...extractMissingConfigEnvReferences(config.apiKey, env),
    ...extractMissingConfigEnvReferences(config.baseUrl, env),
    ...extractMissingConfigHeaderEnvReferences(config.headers, env),
  ]
  if (missing.length > 0) {
    return { skipReason: `unresolved environment reference: ${missing.join(', ')}` }
  }
  const apiKey = resolveConfigEnvReference(config.apiKey, env)
  const baseUrl = resolveConfigEnvReference(config.baseUrl, env)
  const headers = resolveConfigHeaderReferences(config.headers, env)
  switch (config.type) {
    case 'ollama':
      return {
        provider: new SystemOneDecisionProvider({
          id: config.id,
          name: 'Ollama decisions',
          baseUrl: baseUrl ?? OLLAMA_DECISIONS_DEFAULT_BASE_URL,
          apiKey,
          headers,
          models: config.models,
          timeoutMs: config.timeoutMs,
        }),
      }
    case 'systemone':
      if (!baseUrl) return { skipReason: 'baseUrl is required' }
      return {
        provider: new SystemOneDecisionProvider({
          id: config.id,
          baseUrl,
          apiKey,
          headers,
          models: config.models,
          timeoutMs: config.timeoutMs,
        }),
      }
    case 'ai-gateway':
      if (!baseUrl) return { skipReason: 'baseUrl is required' }
      if (!apiKey) return { skipReason: 'apiKey is required' }
      return {
        provider: new SystemOneDecisionProvider({
          id: config.id,
          name: 'ai-gateway decisions',
          baseUrl,
          apiKey,
          headers,
          models: config.models,
          timeoutMs: config.timeoutMs,
          endpointPath: AI_GATEWAY_DECISIONS_PATH,
        }),
      }
    case 'openai':
      if (!apiKey) return { skipReason: 'apiKey is required' }
      return {
        provider: new OpenAIDecisionsProvider({
          id: config.id,
          apiKey,
          baseUrl,
          headers,
          models: config.models,
          timeoutMs: config.timeoutMs,
        }),
      }
  }
}

export function buildDecisionProviderRegistry(
  config: Pick<SepilotdConfig, 'decisionProviders'>,
  env: Record<string, string | undefined> = process.env,
): DecisionProviderRegistry {
  const registry = new DecisionProviderRegistry()
  const defaultId = config.decisionProviders.find((entry) => entry.default)?.id
    ?? config.decisionProviders[0]?.id
  for (const entry of config.decisionProviders) {
    const result = buildDecisionProviderInstance(entry, env)
    if (!result.provider) {
      log.warn(`Skipping decision provider ${entry.id}`, { type: entry.type, reason: result.skipReason })
      continue
    }
    registry.register(result.provider, { default: entry.id === defaultId })
  }
  return registry
}
