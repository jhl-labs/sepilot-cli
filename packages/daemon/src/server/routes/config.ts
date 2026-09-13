import type { FastifyInstance } from 'fastify'
import '../fastify-types.js'
import type { SepilotdConfig } from '../../config/schema.js'
import { isRuntimeManagedChannel } from '../../config/runtime-channel-env.js'
import { resolveChannelPipelineHealthConfig } from '../runtime/channel-pipeline-health.js'
import { resolveWebhookSecurityHealthConfig } from '../runtime/webhook-security-health.js'
import { resolveWebhookSecurityPolicy } from '../runtime/webhook-security-policy.js'
import {
  applyAndPersistRuntimeUpdate,
  ConfigValidationError,
} from '../runtime/config-runtime.js'
import { validateConfig } from '../../config/validator.js'
import { ProviderNetworkConfigurationError } from '../../providers/http-timeout.js'
import {
  applyConfigUpdate,
  isConfigUpdateKey,
  restoreMcpServerSecrets,
  restoreProviderSecrets,
  type ConfigUpdateKey,
} from '../runtime/config-mutations.js'
import { zodRequestValidation } from './utils.js'
import {
  recordChannelPipelineHealthConfigAuditEvent,
  recordWebhookSecurityHealthConfigAuditEvent,
  recordWebhookSecurityPolicyConfigAuditEvent,
} from './config-audit-events.js'
import {
  redactChannelConfigValue,
  redactProviderApiKey,
} from './config-channels-internals.js'
import { registerConfigProvidersRoutes } from './config-providers.js'
import { registerConfigModelRoutes } from './config-model.js'
import { registerConfigEnvRoutes } from './config-env.js'
import { registerConfigMcpRoutes } from './config-mcp.js'
import { registerConfigChannelWebhooksRoutes } from './config-channel-webhooks.js'
import { registerConfigChannelSlackRoutes } from './config-channel-slack.js'
import { registerConfigChannelDiscordRoutes } from './config-channel-discord.js'
import { registerConfigChannelMattermostRoutes } from './config-channel-mattermost.js'
import { registerConfigChannelTelegramRoutes } from './config-channel-telegram.js'
import { registerConfigHooksOutboundRoutes } from './config-hooks-outbound.js'
import { registerConfigObservabilityRoutes } from './config-observability.js'
import { registerConfigSecurityWebhookRoutes } from './config-security-webhooks.js'
import {
  CHANNEL_PIPELINE_HEALTH_CONFIG_AUDIT_EVENT,
  OUTBOUND_WEBHOOK_DEAD_LETTER_ACK_AUDIT_EVENT,
  WEBHOOK_SECURITY_HEALTH_CONFIG_AUDIT_EVENT,
  WEBHOOK_SECURITY_POLICY_CONFIG_AUDIT_EVENT,
  configUpdateRequestSchema,
  type ConfigUpdateBody,
} from './config-schema.js'

export {
  CHANNEL_PIPELINE_HEALTH_CONFIG_AUDIT_EVENT,
  OUTBOUND_WEBHOOK_DEAD_LETTER_ACK_AUDIT_EVENT,
  WEBHOOK_SECURITY_HEALTH_CONFIG_AUDIT_EVENT,
  WEBHOOK_SECURITY_POLICY_CONFIG_AUDIT_EVENT,
}

export { configOpenApiComponents, configOpenApiOverrides } from './config-openapi.js'

export async function configRoutes(app: FastifyInstance) {
  const runtime = app.runtime

  const redactRecordValues = (record: Record<string, string> | undefined) =>
    Object.fromEntries(
      Object.keys(record ?? {}).map((key) => [key, '***redacted***']),
    )
  const redactOptionalSecret = (value: string | undefined) =>
    value ? '***redacted***' : undefined
  // Both callers below already narrow `value` to non-undefined via
  // the surrounding ternary, so redactMemoryAuth is typed strictly:
  // the previous `T | undefined => T | undefined` shape forced
  // every spread of the result to widen back into a possibly-
  // undefined object, which broke the strict response type for
  // memory.rag.rerank.
  const redactMemoryAuth = <T extends { auth?: { password?: string } }>(
    value: T,
  ): T => {
    if (!value.auth) return value
    return {
      ...value,
      auth: {
        ...value.auth,
        password: redactOptionalSecret(value.auth.password),
      },
    }
  }
  const redactMemoryConfig = (
    memory: SepilotdConfig['memory'],
  ): SepilotdConfig['memory'] => ({
    ...memory,
    qdrant: memory.qdrant
      ? {
          ...memory.qdrant,
          apiKey: redactOptionalSecret(memory.qdrant.apiKey),
        }
      : undefined,
    opensearch: memory.opensearch
      ? {
          ...memory.opensearch,
          apiKey: redactOptionalSecret(memory.opensearch.apiKey),
          password: redactOptionalSecret(memory.opensearch.password),
        }
      : undefined,
    elasticsearch: memory.elasticsearch
      ? {
          ...memory.elasticsearch,
          apiKey: redactOptionalSecret(memory.elasticsearch.apiKey),
          password: redactOptionalSecret(memory.elasticsearch.password),
        }
      : undefined,
    meilisearch: memory.meilisearch
      ? {
          ...memory.meilisearch,
          apiKey: redactOptionalSecret(memory.meilisearch.apiKey),
        }
      : undefined,
    customApi: memory.customApi
      ? {
          ...redactMemoryAuth(memory.customApi),
          apiKey: redactOptionalSecret(memory.customApi.apiKey),
          headers: redactRecordValues(memory.customApi.headers),
        }
      : undefined,
    rag: memory.rag
      ? {
          ...memory.rag,
          rerank: {
            ...redactMemoryAuth(memory.rag.rerank),
            apiKey: redactOptionalSecret(memory.rag.rerank.apiKey),
            headers: redactRecordValues(memory.rag.rerank.headers),
          },
        }
      : undefined,
  })

  registerConfigProvidersRoutes(app)
  registerConfigModelRoutes(app)
  registerConfigEnvRoutes(app)
  registerConfigMcpRoutes(app)
  registerConfigChannelWebhooksRoutes(app)
  registerConfigChannelSlackRoutes(app)
  registerConfigChannelDiscordRoutes(app)
  registerConfigChannelMattermostRoutes(app)
  registerConfigChannelTelegramRoutes(app)
  await registerConfigHooksOutboundRoutes(app)
  registerConfigObservabilityRoutes(app)
  registerConfigSecurityWebhookRoutes(app)

  // GET /config — get current config (redact sensitive fields)
  app.get('/config', async (_request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })
    const providers = runtime.config.providers.map(p => ({
      ...p,
      apiKey: redactProviderApiKey(p.apiKey),
      headers: Object.fromEntries(
        Object.entries(p.headers ?? {}).map(([key, value]) => [
          key,
          redactProviderApiKey(value) ?? '***redacted***',
        ]),
      ),
    }))
    const channels = (runtime.config.channels ?? []).map((channel) => ({
      ...channel,
      config: redactChannelConfigValue(channel.config),
    }))
    const mcp = {
      servers: (runtime.config.mcp?.servers ?? []).map((server) => ({
        ...server,
        ...((
          'env' in server
          && server.env
          && typeof server.env === 'object'
          && !Array.isArray(server.env)
        )
          ? {
              env: redactRecordValues(server.env),
            }
          : {}),
        ...((
          'headers' in server
          && server.headers
          && typeof server.headers === 'object'
          && !Array.isArray(server.headers)
        )
          ? {
              headers: redactRecordValues(server.headers),
            }
          : {}),
      })),
      client: runtime.config.mcp?.client,
    }
    const hooks = {
      commandHooks: (runtime.config.hooks?.commandHooks ?? []).map((hook) => ({
        event: hook.event, enabled: hook.enabled !== false, async: hook.async === true,
        timeoutMs: hook.timeoutMs ?? 10_000, toolMatcher: hook.toolMatcher,
      })),
      outboundWebhooks: (runtime.config.hooks?.outboundWebhooks ?? []).map((webhook) => ({
        ...webhook,
        secret: webhook.secret ? '***redacted***' : undefined,
        headers: Object.fromEntries(
          Object.keys(webhook.headers ?? {}).map((key) => [key, '***redacted***']),
        ),
      })),
    }
    return {
      data: {
        ...runtime.config,
        providers,
        channels,
        hooks,
        mcp,
        memory: redactMemoryConfig(runtime.config.memory),
        // Older persisted snapshots predate this section entirely.
        ...(runtime.config.webSearch
          ? {
              webSearch: {
                ...runtime.config.webSearch,
                apiKey: redactOptionalSecret(runtime.config.webSearch.apiKey),
              },
            }
          : {}),
      },
    }
  })

  // PUT /config — update runtime config values
  app.put<{ Body: ConfigUpdateBody }>('/config', {
    preValidation: zodRequestValidation({
      body: {
        schema: configUpdateRequestSchema,
        message: 'Invalid config update request body',
      },
    }),
  }, async (request, reply) => {
    if (!runtime) return reply.status(503).send({ error: { code: 'SERVICE_UNAVAILABLE', message: 'Runtime not initialized' } })

    const updates = request.body

    if (
      updates.channels !== undefined
      && isRuntimeManagedChannel(runtime.config, 'mattermost')
    ) {
      return reply.status(409).send({
        error: {
          code: 'CHANNEL_MANAGED_BY_ENVIRONMENT',
          message:
            'Mattermost is managed by the daemon runtime environment. Remove SEPILOTD_MATTERMOST_FROM_ENV and restart before replacing the channel catalog.',
        },
      })
    }

    let updated: Set<ConfigUpdateKey>
    try {
      updated = await runtime.configMutationService.apply(
      'config.update',
      async () => {
        const requestedChannelPipelineHealth =
          updates['observability.channelPipelineHealth']
        const previousChannelPipelineHealth = requestedChannelPipelineHealth
          ? resolveChannelPipelineHealthConfig(runtime.config)
          : null
        const requestedWebhookSecurityHealth =
          updates['observability.webhookSecurityHealth']
        const previousWebhookSecurityHealth = requestedWebhookSecurityHealth
          ? resolveWebhookSecurityHealthConfig(runtime.config)
          : null
        const requestedWebhookSecurityPolicy = updates['security.webhooks']
        const previousWebhookSecurityPolicy = requestedWebhookSecurityPolicy
          ? resolveWebhookSecurityPolicy(runtime.config)
          : null
        // Snapshot pre-mutation validation errors so the gate below only
        // rejects errors THIS request introduces. A pre-existing latent
        // incoherence (e.g. a stale embeddingProvider) must not block an
        // unrelated field update or prevent the user from fixing config
        // through the API.
        const preValidationErrors = new Set(validateConfig(runtime.config).errors)
        const updatedKeys = new Set<ConfigUpdateKey>()
        const updatedValues: Partial<Record<ConfigUpdateKey, unknown>> = {}
        for (const [key, value] of Object.entries(updates)) {
          if (!isConfigUpdateKey(key))
            continue
          const normalizedValue =
            key === 'providers' && Array.isArray(value)
              ? restoreProviderSecrets(
                runtime.config.providers,
                value as SepilotdConfig['providers'],
              )
              : key === 'mcp.servers' && Array.isArray(value)
                ? restoreMcpServerSecrets(
                  runtime.config.mcp.servers,
                  value as SepilotdConfig['mcp']['servers'],
                )
                : value
          if (applyConfigUpdate(runtime.config, key, normalizedValue)) {
            updatedKeys.add(key)
            updatedValues[key] = normalizedValue
          }
        }

        // Cross-field validation on the post-mutation config, mirroring the
        // full settings-json replace path. Granular PUT applies fields
        // independently, so an individually-valid field can still leave the
        // config incoherent (e.g. vectorBackend=qdrant with no qdrant.url,
        // an embeddingProvider that is not configured, a dangling
        // defaultProvider). Persisting that would fail on the next boot.
        // Only errors this request newly introduces are rejected — see the
        // preValidationErrors snapshot above. Throwing here rolls back the
        // in-memory config via the mutation service before anything is
        // written to disk.
        if (updatedKeys.size > 0) {
          const validation = validateConfig(runtime.config)
          const newErrors = validation.errors.filter(
            (error) => !preValidationErrors.has(error),
          )
          if (newErrors.length > 0) {
            throw new ConfigValidationError(newErrors.join(' '), newErrors)
          }
        }

        await applyAndPersistRuntimeUpdate(runtime, updatedKeys, updatedValues)
        if (
          requestedChannelPipelineHealth
          && previousChannelPipelineHealth
          && updatedKeys.has('observability.channelPipelineHealth')
        ) {
          await recordChannelPipelineHealthConfigAuditEvent(runtime, {
            route: '/api/v1/config',
            requested: requestedChannelPipelineHealth,
            previous: previousChannelPipelineHealth,
            current: resolveChannelPipelineHealthConfig(runtime.config),
          })
        }
        if (
          requestedWebhookSecurityHealth
          && previousWebhookSecurityHealth
          && updatedKeys.has('observability.webhookSecurityHealth')
        ) {
          await recordWebhookSecurityHealthConfigAuditEvent(runtime, {
            route: '/api/v1/config',
            requested: requestedWebhookSecurityHealth,
            previous: previousWebhookSecurityHealth,
            current: resolveWebhookSecurityHealthConfig(runtime.config),
          })
        }
        if (
          requestedWebhookSecurityPolicy
          && previousWebhookSecurityPolicy
          && updatedKeys.has('security.webhooks')
        ) {
          await recordWebhookSecurityPolicyConfigAuditEvent(runtime, {
            route: '/api/v1/config',
            requested: requestedWebhookSecurityPolicy,
            previous: previousWebhookSecurityPolicy,
            current: resolveWebhookSecurityPolicy(runtime.config),
          })
        }

        return updatedKeys
      },
    )
    } catch (error) {
      if (error instanceof ConfigValidationError) {
        return reply.status(400).send({
          error: {
            code: 'CONFIG_VALIDATION_FAILED',
            message: error.message,
            details: error.errors,
          },
        })
      }
      if (error instanceof ProviderNetworkConfigurationError) {
        return reply.status(400).send({
          error: {
            code: error.code,
            message: error.message,
          },
        })
      }
      throw error
    }

    return { data: { updated: Array.from(updated) } }
  })
}
