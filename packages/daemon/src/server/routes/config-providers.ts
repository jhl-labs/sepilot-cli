import type { FastifyInstance } from 'fastify'
import '../fastify-types.js'
import { z } from 'zod'
import {
  buildProviderInstance,
  extractMissingConfigHeaderEnvReferences,
  extractMissingConfigEnvReferences,
  resolveConfigEnvReference,
  resolveConfigHeaderReferences,
} from '../runtime/providers.js'
import { restoreProviderApiKey, type ConfigUpdateKey } from '../runtime/config-mutations.js'
import {
  applyAndPersistRuntimeUpdate,
  ConfigValidationError,
} from '../runtime/config-runtime.js'
import type { ProviderFactoryCapabilities } from '../runtime/capabilities.js'
import { MODEL_PROBE_TIMEOUT_MS, probeProviderModel } from '../../providers/model-probe.js'
import { CodexProvider } from '../../providers/codex.js'
import { OpencodeProvider } from '../../providers/opencode.js'
import { zodRequestValidation } from './utils.js'
import {
  configProviderDiscoverModelsRequestSchema,
  configProviderValidationRequestSchema,
  type ConfigProviderDiscoverModelsBody,
  type configProviderValidationResponseSchema,
  type ConfigProviderValidationBody,
} from './config-schema.js'
import { providerSchema, type SepilotdConfig } from '../../config/schema.js'
import { validateConfig } from '../../config/validator.js'

const configProviderIdParamsSchema = z.object({
  id: z.string().trim().min(1),
})
const configProviderMutationQuerySchema = z.object({
  mode: z.enum(['create', 'update']),
})

type ConfigProviderIdParams = z.infer<typeof configProviderIdParamsSchema>
type ConfigProviderMutationQuery = z.infer<typeof configProviderMutationQuerySchema>
type ConfigProvider = SepilotdConfig['providers'][number]

class ProviderConfigConflictError extends Error {
  readonly code = 'PROVIDER_CONFIG_CHANGED' as const
}

function providerConnectionFingerprint(provider: ConfigProvider): string {
  return JSON.stringify([
    provider.type,
    provider.baseUrl ?? '',
    provider.apiKey ?? '',
    Object.entries(provider.headers ?? {}).sort(([left], [right]) => left.localeCompare(right)),
  ])
}

function configuredDefaultTarget(config: SepilotdConfig): { providerId: string; modelId: string } | null {
  const provider =
    (config.agent.defaultProvider
      ? config.providers.find((candidate) => candidate.id === config.agent.defaultProvider)
      : undefined) ??
    config.providers.find((candidate) => candidate.default) ??
    config.providers[0]
  if (!provider) return null
  const modelId = config.agent.defaultModel ?? provider.models[0]
  return modelId ? { providerId: provider.id, modelId } : null
}

function assertNoNewConfigErrors(config: SepilotdConfig, previousErrors: Set<string>): void {
  const newErrors = validateConfig(config).errors.filter((error) => !previousErrors.has(error))
  if (newErrors.length > 0) {
    throw new ConfigValidationError(newErrors.join(' '), newErrors)
  }
}

export function collectProviderValidationMissingEnvVars(
  provider: ConfigProviderValidationBody['provider'],
  env: Record<string, string | undefined> = process.env,
): string[] {
  return Array.from(new Set([
    ...extractMissingConfigEnvReferences(provider.apiKey, env),
    ...extractMissingConfigEnvReferences(provider.baseUrl, env),
    ...extractMissingConfigHeaderEnvReferences(provider.headers, env),
  ])).sort()
}

export function collectConfiguredProviderModelIds(
  provider: Pick<SepilotdConfig['providers'][number], 'models'>,
): string[] {
  return Array.isArray(provider.models)
    ? provider.models
      .filter((model): model is string => typeof model === 'string' && model.trim().length > 0)
      .map((model) => model.trim())
    : []
}

export function extractValidationErrorMessage(error: unknown): string {
  if (error instanceof Error && error.message) {
    return error.message
  }
  return String(error)
}

export async function validateProviderConfiguration(
  runtime: ProviderFactoryCapabilities | undefined,
  body: ConfigProviderValidationBody,
): Promise<z.infer<typeof configProviderValidationResponseSchema>['data']> {
  if (!runtime?.providerFactoryRegistry) {
    throw new Error('Provider factory registry is not initialized.')
  }

  const validationEnv = {
    ...process.env,
    ...(body.env ?? {}),
  }

  const missingEnvVars = collectProviderValidationMissingEnvVars(body.provider, validationEnv)
  if (missingEnvVars.length > 0) {
    throw new Error(
      `Missing environment variables: ${missingEnvVars.join(', ')}.`,
    )
  }

  const buildResult = buildProviderInstance(
    body.provider,
    runtime.providerFactoryRegistry,
    validationEnv,
  )
  if (!buildResult.provider) {
    throw new Error(
      `Cannot initialize provider ${body.provider.id}: ${buildResult.skipReason ?? 'unknown error'}.`,
    )
  }

  const model = body.model?.trim() || buildResult.resolvedProvider.models[0]
  if (!model) {
    throw new Error('Provider validation requires a model.')
  }

  const timeoutMs = body.timeoutMs ?? MODEL_PROBE_TIMEOUT_MS
  const probe = await probeProviderModel(
    buildResult.provider,
    { providerId: body.provider.id, modelId: model },
    timeoutMs,
  )
  if (!probe.ok) {
    throw new Error(
      `Validation failed for ${body.provider.id}/${model}: ${probe.reason ?? 'self-test failed'}`,
    )
  }

  return {
    ok: true,
    providerId: body.provider.id,
    providerType: body.provider.type,
    model,
    latencyMs: probe.latencyMs,
    message: `Validated ${body.provider.id}/${model}.`,
  }
}

interface DiscoverModelsResult {
  models: string[]
  source: 'remote' | 'static'
  endpoint?: string
  compatibility?: {
    profile: 'ollama-openai'
    capabilities: {
      thinkingControl: 'reasoning-effort'
    }
  }
}

function normalizeBaseUrl(url: string): string {
  return url.replace(/\/+$/, '')
}

async function fetchJsonWithTimeout(
  url: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<unknown> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const response = await fetch(url, { ...init, signal: controller.signal })
    if (!response.ok) {
      const detail = await response.text().catch(() => '')
      const trimmed = detail.length > 200 ? `${detail.slice(0, 200)}…` : detail
      throw new Error(
        `${response.status} ${response.statusText}${trimmed ? ` — ${trimmed}` : ''}`,
      )
    }
    return await response.json()
  } finally {
    clearTimeout(timer)
  }
}

function openAiCompatibilityMetadataEndpoint(baseUrl: string): string | null {
  try {
    const url = new URL(baseUrl)
    const normalizedPath = url.pathname.replace(/\/+$/, '')
    const rootPath = normalizedPath.endsWith('/v1')
      ? normalizedPath.slice(0, -3)
      : normalizedPath
    url.pathname = `${rootPath}/api/version`.replace(/\/{2,}/g, '/')
    url.search = ''
    url.hash = ''
    return url.toString().replace(/\/$/, '')
  } catch {
    return null
  }
}

async function detectOpenAiCompatibilityProfile(
  baseUrl: string,
  headers: HeadersInit,
  timeoutMs: number,
): Promise<DiscoverModelsResult['compatibility'] | undefined> {
  const endpoint = openAiCompatibilityMetadataEndpoint(baseUrl)
  if (!endpoint) return undefined

  try {
    const metadata = await fetchJsonWithTimeout(
      endpoint,
      { headers },
      Math.min(timeoutMs, 2_000),
    )
    if (
      metadata
      && typeof metadata === 'object'
      && !Array.isArray(metadata)
      && typeof (metadata as { version?: unknown }).version === 'string'
      && Object.keys(metadata).every((key) => key === 'version')
    ) {
      return {
        profile: 'ollama-openai',
        capabilities: { thinkingControl: 'reasoning-effort' },
      }
    }
  } catch {
    // Compatibility metadata is optional. A conforming /models response is
    // still useful when an endpoint does not advertise a known profile.
  }
  return undefined
}

function applyDiscoveredCompatibility(
  provider: ConfigProvider,
  compatibility: DiscoverModelsResult['compatibility'],
): ConfigProvider {
  const thinkingControl = compatibility?.capabilities.thinkingControl
  if (!thinkingControl) return provider

  return {
    ...provider,
    capabilities: {
      ...provider.capabilities,
      thinkingControl,
    },
    modelOverrides: (provider.modelOverrides ?? []).map((override) => ({
      ...override,
      capabilities: override.capabilities
        ? { ...override.capabilities, thinkingControl }
        : override.capabilities,
    })),
  }
}

export async function discoverProviderModels(
  body: ConfigProviderDiscoverModelsBody,
): Promise<DiscoverModelsResult> {
  const type = body.type.toLowerCase()
  const timeoutMs = body.timeoutMs ?? 10_000
  const extraHeaders: HeadersInit = { ...(body.headers ?? {}) }

  if (type === 'opencode') {
    return {
      models: await OpencodeProvider.listAvailableModels(),
      source: 'remote',
      endpoint: 'opencode models',
    }
  }

  if (type === 'codex') {
    return {
      models: await CodexProvider.listAvailableModels(),
      source: 'remote',
      endpoint: 'codex debug models',
    }
  }

  if (type === 'openai' || type === 'openai-compat' || type === 'openai-compatible' || type === 'custom') {
    if (type === 'custom' && !body.baseUrl) {
      throw new Error('Model discovery for provider type "custom" requires baseUrl.')
    }
    const base = body.baseUrl ? normalizeBaseUrl(body.baseUrl) : 'https://api.openai.com/v1'
    const endpoint = `${base}/models`
    const headers: Record<string, string> = { ...extraHeaders }
    if (body.apiKey) headers.authorization = `Bearer ${body.apiKey}`
    const json = (await fetchJsonWithTimeout(endpoint, { headers }, timeoutMs)) as {
      data?: Array<{ id?: string }>
    }
    const ids = (json.data ?? [])
      .map((m) => (typeof m?.id === 'string' ? m.id : ''))
      .filter((id) => id.length > 0)
    const shouldDetectCompatibility = Boolean(body.baseUrl)
    const compatibility = shouldDetectCompatibility
      ? await detectOpenAiCompatibilityProfile(base, headers, timeoutMs)
      : undefined
    return {
      models: Array.from(new Set(ids)).sort(),
      source: 'remote',
      endpoint,
      ...(compatibility ? { compatibility } : {}),
    }
  }

  if (type === 'anthropic') {
    const base = body.baseUrl ? normalizeBaseUrl(body.baseUrl) : 'https://api.anthropic.com/v1'
    const endpoint = `${base}/models`
    const headers: Record<string, string> = {
      'anthropic-version': '2023-06-01',
      ...extraHeaders,
    }
    if (body.apiKey) headers['x-api-key'] = body.apiKey
    const json = (await fetchJsonWithTimeout(endpoint, { headers }, timeoutMs)) as {
      data?: Array<{ id?: string }>
    }
    const ids = (json.data ?? [])
      .map((m) => (typeof m?.id === 'string' ? m.id : ''))
      .filter((id) => id.length > 0)
    return { models: Array.from(new Set(ids)).sort(), source: 'remote', endpoint }
  }

  if (type === 'ollama') {
    const base = body.baseUrl ? normalizeBaseUrl(body.baseUrl) : 'http://localhost:11434'
    const endpoint = `${base}/api/tags`
    const json = (await fetchJsonWithTimeout(
      endpoint,
      { headers: extraHeaders },
      timeoutMs,
    )) as { models?: Array<{ name?: string }> }
    const ids = (json.models ?? [])
      .map((m) => (typeof m?.name === 'string' ? m.name : ''))
      .filter((id) => id.length > 0)
    return { models: Array.from(new Set(ids)).sort(), source: 'remote', endpoint }
  }

  throw new Error(`Model discovery not supported for provider type "${body.type}".`)
}

export function registerConfigProvidersRoutes(app: FastifyInstance): void {
  const runtime = app.runtime

  app.get('/config/providers', async (_request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const runtimeProviders = new Map(
      runtime.providerRegistry.list().map((provider) => [provider.id, provider] as const),
    )
    const providers = runtime.config.providers.map((configuredProvider) => {
      const configuredModelIds = collectConfiguredProviderModelIds(configuredProvider)
      const runtimeProvider = runtimeProviders.get(configuredProvider.id)
      const missingEnvVars = collectProviderValidationMissingEnvVars(configuredProvider)

      if (runtimeProvider) {
        const embeddingModelIds = runtimeProvider.models
          .filter((model) => model.capabilities.embedding)
          .map((model) => model.id)
        const runtimeModelIds = new Set(runtimeProvider.models.map((model) => model.id))
        const unavailableConfiguredModelIds = runtimeProvider.modelCatalogAuthority === 'endpoint'
          ? configuredModelIds.filter((modelId) => !runtimeModelIds.has(modelId))
          : []

        return {
          id: runtimeProvider.id,
          name: runtimeProvider.name,
          models: runtimeProvider.models,
          supportsEmbedding: embeddingModelIds.length > 0 || typeof runtimeProvider.embed === 'function',
          embeddingModelIds,
          configuredModelIds,
          modelCatalogAuthority: runtimeProvider.modelCatalogAuthority,
          unavailableConfiguredModelIds,
          health: {
            status: 'ready' as const,
            missingEnvVars: [],
          },
        }
      }

      if (missingEnvVars.length > 0) {
        return {
          id: configuredProvider.id,
          name: configuredProvider.id,
          models: [],
          supportsEmbedding: false,
          embeddingModelIds: [],
          configuredModelIds,
          unavailableConfiguredModelIds: [],
          health: {
            status: 'env_missing' as const,
            message: `Missing environment variables: ${missingEnvVars.join(', ')}.`,
            missingEnvVars,
          },
        }
      }

      try {
        const buildResult = buildProviderInstance(
          configuredProvider,
          runtime.providerFactoryRegistry,
        )

        if (buildResult.provider) {
          const embeddingModelIds = buildResult.provider.models
            .filter((model) => model.capabilities.embedding)
            .map((model) => model.id)

          return {
            id: buildResult.provider.id,
            name: buildResult.provider.name,
            models: buildResult.provider.models,
            supportsEmbedding: embeddingModelIds.length > 0 || typeof buildResult.provider.embed === 'function',
            embeddingModelIds,
            configuredModelIds,
            modelCatalogAuthority: buildResult.provider.modelCatalogAuthority,
            unavailableConfiguredModelIds: [],
            health: {
              status: 'ready' as const,
              missingEnvVars: [],
            },
          }
        }

        return {
          id: configuredProvider.id,
          name: configuredProvider.id,
          models: [],
          supportsEmbedding: false,
          embeddingModelIds: [],
          configuredModelIds,
          unavailableConfiguredModelIds: [],
          health: {
            status: 'unavailable' as const,
            message: buildResult.skipReason ?? 'Provider could not be initialized.',
            missingEnvVars: [],
          },
        }
      } catch (error) {
        return {
          id: configuredProvider.id,
          name: configuredProvider.id,
          models: [],
          supportsEmbedding: false,
          embeddingModelIds: [],
          configuredModelIds,
          unavailableConfiguredModelIds: [],
          health: {
            status: 'unavailable' as const,
            message: extractValidationErrorMessage(error),
            missingEnvVars: [],
          },
        }
      }
    })
    return { data: providers }
  })

  app.post<{ Body: ConfigProviderDiscoverModelsBody }>('/config/providers/discover-models', {
    preValidation: zodRequestValidation({
      body: {
        schema: configProviderDiscoverModelsRequestSchema,
        message: 'Invalid provider discover-models request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
      })
    }
    try {
      const body = request.body
      const resolvedBody: ConfigProviderDiscoverModelsBody = {
        ...body,
        baseUrl: resolveConfigEnvReference(body.baseUrl),
        apiKey: resolveConfigEnvReference(body.apiKey),
        headers: body.headers
          ? resolveConfigHeaderReferences(body.headers)
          : undefined,
      }
      const result = await discoverProviderModels(resolvedBody)
      return { data: result }
    } catch (error) {
      const message = extractValidationErrorMessage(error)
      const code = message.startsWith('Model discovery not supported')
        ? 'PROVIDER_TYPE_UNSUPPORTED'
        : 'PROVIDER_DISCOVERY_FAILED'
      return reply.status(code === 'PROVIDER_TYPE_UNSUPPORTED' ? 400 : 502).send({
        error: { code, message },
      })
    }
  })

  app.put<{
    Params: ConfigProviderIdParams
    Querystring: ConfigProviderMutationQuery
    Body: ConfigProvider
  }>(
    '/config/providers/:id',
    {
      preValidation: zodRequestValidation({
        params: {
          schema: configProviderIdParamsSchema,
          message: 'Invalid provider params',
        },
        query: {
          schema: configProviderMutationQuerySchema,
          message: 'Invalid provider mutation query',
        },
        body: {
          schema: providerSchema,
          message: 'Invalid provider body',
        },
      }),
    },
    async (request, reply) => {
      if (!runtime) {
        return reply.status(503).send({
          error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
        })
      }
      const params = configProviderIdParamsSchema.parse(request.params)
      const query = configProviderMutationQuerySchema.parse(request.query)
      const body = providerSchema.parse(request.body)
      if (params.id !== body.id) {
        return reply.status(400).send({
          error: { code: 'PROVIDER_ID_MISMATCH', message: 'Provider path and body IDs must match.' },
        })
      }

      let action: 'created' | 'updated' = 'created'
      try {
        await runtime.configMutationService.apply('config.providers.upsert', async () => {
          const previousErrors = new Set(validateConfig(runtime.config).errors)
          const index = runtime.config.providers.findIndex(
            (provider) => provider.id === params.id,
          )
          if (query.mode === 'create' && index >= 0) {
            throw new ProviderConfigConflictError(
              `Provider "${params.id}" was created by another client. Reload before editing it.`,
            )
          }
          if (query.mode === 'update' && index < 0) {
            throw new ProviderConfigConflictError(
              `Provider "${params.id}" no longer exists. Reload before saving.`,
            )
          }
          action = index >= 0 ? 'updated' : 'created'
          const restored = restoreProviderApiKey(runtime.config.providers, body)
          runtime.config.providers =
            index >= 0
              ? runtime.config.providers.map((provider, providerIndex) =>
                  providerIndex === index ? restored : provider,
                )
              : [...runtime.config.providers, restored]
          assertNoNewConfigErrors(runtime.config, previousErrors)
          await applyAndPersistRuntimeUpdate(
            runtime,
            new Set<ConfigUpdateKey>(['providers']),
            { providers: runtime.config.providers },
          )
        })
      } catch (error) {
        if (error instanceof ProviderConfigConflictError) {
          return reply.status(409).send({
            error: { code: error.code, message: error.message },
          })
        }
        if (error instanceof ConfigValidationError) {
          return reply.status(409).send({
            error: { code: error.code, message: error.message, errors: error.errors },
          })
        }
        throw error
      }

      return { data: { providerId: params.id, action } }
    },
  )

  app.delete<{ Params: ConfigProviderIdParams }>(
    '/config/providers/:id',
    {
      preValidation: zodRequestValidation({
        params: {
          schema: configProviderIdParamsSchema,
          message: 'Invalid provider params',
        },
      }),
    },
    async (request, reply) => {
      if (!runtime) {
        return reply.status(503).send({
          error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
        })
      }
      const params = configProviderIdParamsSchema.parse(request.params)
      if (!runtime.config.providers.some((provider) => provider.id === params.id)) {
        return reply.status(404).send({
          error: {
            code: 'PROVIDER_NOT_FOUND',
            message: `Provider "${params.id}" is not configured.`,
          },
        })
      }

      try {
        await runtime.configMutationService.apply('config.providers.delete', async () => {
          const previousErrors = new Set(validateConfig(runtime.config).errors)
          runtime.config.providers = runtime.config.providers.filter(
            (provider) => provider.id !== params.id,
          )
          assertNoNewConfigErrors(runtime.config, previousErrors)
          await applyAndPersistRuntimeUpdate(
            runtime,
            new Set<ConfigUpdateKey>(['providers']),
            { providers: runtime.config.providers },
          )
        })
      } catch (error) {
        if (error instanceof ConfigValidationError) {
          return reply.status(409).send({
            error: { code: error.code, message: error.message, errors: error.errors },
          })
        }
        throw error
      }

      return reply.status(204).send()
    },
  )

  // Server-side model rediscovery for a *stored* provider. Clients only ever
  // see redacted apiKeys, so they cannot call discover-models themselves for
  // key-protected providers; this route runs discovery with the daemon's own
  // (unredacted) stored credentials, persists the refreshed model list into
  // the provider's config entry, and returns the models. Raw credentials are
  // never included in the response.
  app.post<{ Params: ConfigProviderIdParams }>('/config/providers/:id/refresh-models', {
    preValidation: zodRequestValidation({
      params: {
        schema: configProviderIdParamsSchema,
        message: 'Invalid provider params',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
      })
    }

    const params = request.params
    const providerId = params.id
    const provider = runtime.config.providers.find((entry) => entry.id === providerId)
    if (!provider) {
      return reply.status(404).send({
        error: { code: 'PROVIDER_NOT_FOUND', message: `Provider "${providerId}" is not configured.` },
      })
    }

    let result: DiscoverModelsResult
    try {
      const headers = provider.headers
        ? resolveConfigHeaderReferences(provider.headers)
        : undefined
      result = await discoverProviderModels({
        type: provider.type,
        baseUrl: resolveConfigEnvReference(provider.baseUrl),
        apiKey: resolveConfigEnvReference(provider.apiKey),
        headers,
        timeoutMs: 10_000,
      })
    } catch (error) {
      // Discovery failed: leave the stored config untouched.
      const message = extractValidationErrorMessage(error)
      const code = message.startsWith('Model discovery not supported')
        ? 'PROVIDER_TYPE_UNSUPPORTED'
        : 'PROVIDER_DISCOVERY_FAILED'
      return reply.status(code === 'PROVIDER_TYPE_UNSUPPORTED' ? 400 : 502).send({
        error: { code, message },
      })
    }

    const sortedResult = [...result.models].sort()
    const connectionFingerprint = providerConnectionFingerprint(provider)
    const initialModels = collectConfiguredProviderModelIds(provider).sort()
    const discoveredProvider = applyDiscoveredCompatibility(provider, result.compatibility)
    const compatibilityChanged = JSON.stringify(discoveredProvider) !== JSON.stringify(provider)
    let changed = compatibilityChanged || (
      result.models.length > 0 &&
      (initialModels.length !== sortedResult.length ||
        initialModels.some((model, index) => model !== sortedResult[index]))
    )
    if (changed) {
      try {
        await runtime.configMutationService.apply('config.providers.refresh-models', async () => {
          const currentProvider = runtime.config.providers.find((entry) => entry.id === providerId)
          if (
            !currentProvider ||
            providerConnectionFingerprint(currentProvider) !== connectionFingerprint
          ) {
            throw new ProviderConfigConflictError(
              `Provider "${providerId}" changed while models were being discovered.`,
            )
          }

          const previousErrors = new Set(validateConfig(runtime.config).errors)
          const previousModels = collectConfiguredProviderModelIds(currentProvider)
          const sortedPrevious = [...previousModels].sort()
          const nextProvider = applyDiscoveredCompatibility(currentProvider, result.compatibility)
          const currentCompatibilityChanged = JSON.stringify(nextProvider) !== JSON.stringify(currentProvider)
          changed = currentCompatibilityChanged ||
            sortedPrevious.length !== sortedResult.length ||
            sortedPrevious.some((model, index) => model !== sortedResult[index])
          if (!changed) return

          const active = configuredDefaultTarget(runtime.config)
          if (
            active?.providerId === providerId &&
            !result.models.includes(active.modelId)
          ) {
            throw new ProviderConfigConflictError(
              `Discovered models do not include the active model "${active.modelId}". Switch the default model before refreshing.`,
            )
          }
          runtime.config.providers = runtime.config.providers.map((entry) => {
            if (entry.id !== providerId) return entry
            const compatible = applyDiscoveredCompatibility(entry, result.compatibility)
            return result.models.length > 0
              ? { ...compatible, models: [...result.models] }
              : compatible
          })
          assertNoNewConfigErrors(runtime.config, previousErrors)
          await applyAndPersistRuntimeUpdate(
            runtime,
            new Set<ConfigUpdateKey>(['providers']),
            { providers: runtime.config.providers },
          )
        })
      } catch (error) {
        if (error instanceof ProviderConfigConflictError) {
          return reply.status(409).send({
            error: { code: error.code, message: error.message },
          })
        }
        if (error instanceof ConfigValidationError) {
          return reply.status(409).send({
            error: { code: error.code, message: error.message, errors: error.errors },
          })
        }
        throw error
      }
    }

    return {
      data: {
        providerId,
        models: result.models,
        source: result.source,
        endpoint: result.endpoint,
        ...(result.compatibility ? { compatibility: result.compatibility } : {}),
        updated: changed,
      },
    }
  })

  app.post<{ Body: ConfigProviderValidationBody }>('/config/providers/validate', {
    preValidation: zodRequestValidation({
      body: {
        schema: configProviderValidationRequestSchema,
        message: 'Invalid provider validation request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) {
      return reply.status(503).send({
        error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' },
      })
    }

    try {
      const body = request.body
      const result = await validateProviderConfiguration(runtime, {
        ...body,
        provider: restoreProviderApiKey(
          runtime.config.providers,
          body.provider,
        ),
      })
      return { data: result }
    } catch (error) {
      return reply.status(400).send({
        error: {
          code: 'PROVIDER_VALIDATION_FAILED',
          message: extractValidationErrorMessage(error),
        },
      })
    }
  })
}
