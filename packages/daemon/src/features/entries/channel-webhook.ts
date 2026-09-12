import { WebhookChannel, type WebhookEndpointConfig } from '../../channels/webhook.js'
import { webhookEndpointConfigSchema } from '../../channels/webhook-validation.js'
import { createLogger } from '../../logger.js'
import type { ChannelFactoryRegistry } from '../../server/runtime/channels.js'

const log = createLogger('runtime.channels')

function webhookEndpointValidationReason(error: unknown): string {
  if (error && typeof error === 'object' && 'issues' in error && Array.isArray(error.issues)) {
    return error.issues
      .map((issue) => {
        if (!issue || typeof issue !== 'object') {
          return String(issue)
        }
        const path = 'path' in issue && Array.isArray(issue.path) ? issue.path.join('.') : ''
        const message = 'message' in issue ? String(issue.message) : String(issue)
        return path ? `${path}: ${message}` : message
      })
      .join('; ')
  }
  return String(error)
}

function parseWebhookEndpoints(rawEndpoints: unknown): WebhookEndpointConfig[] {
  if (!Array.isArray(rawEndpoints)) {
    return []
  }

  return rawEndpoints.flatMap((rawEndpoint): WebhookEndpointConfig[] => {
    const parsed = webhookEndpointConfigSchema.safeParse(rawEndpoint)
    if (parsed.success) {
      return [parsed.data]
    }

    log.warn('Skipping invalid webhook endpoint config', {
      reason: webhookEndpointValidationReason(parsed.error),
    })
    return []
  })
}

export function registerChannelFactory(registry: ChannelFactoryRegistry): void {
  registry.register('webhook', (channel) => {
    const channelConfig = channel.config as Record<string, unknown> | undefined
    const endpoints = parseWebhookEndpoints(channelConfig?.endpoints)
    const enabledEndpoints = endpoints.filter((endpoint) => endpoint.enabled !== false)
    return enabledEndpoints.length > 0
      ? {
          channel: new WebhookChannel({
            endpoints: endpoints as WebhookEndpointConfig[],
            rateLimitPerMinute:
              typeof channelConfig?.rateLimitPerMinute === 'number'
                ? channelConfig.rateLimitPerMinute
                : undefined,
          }),
        }
      : { skipReason: 'at least one endpoint is required' }
  })
}
