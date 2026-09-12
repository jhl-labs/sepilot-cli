import type { WebhookSecurityHealthConfig } from '../../config/schema.js'
import {
  DEFAULT_WEBHOOK_SECURITY_HEALTH_CONFIG,
} from '../../config/schema.js'

interface RuntimeConfigShape {
  observability?: {
    webhookSecurityHealth?: Partial<WebhookSecurityHealthConfig>
  }
}

export function resolveWebhookSecurityHealthConfig(
  config?: RuntimeConfigShape,
): WebhookSecurityHealthConfig {
  const raw = config?.observability?.webhookSecurityHealth

  return {
    degradeWhenUnreadyEndpointsAtLeast:
      raw?.degradeWhenUnreadyEndpointsAtLeast
      ?? DEFAULT_WEBHOOK_SECURITY_HEALTH_CONFIG.degradeWhenUnreadyEndpointsAtLeast,
    detailTopMissingRequirements:
      raw?.detailTopMissingRequirements
      ?? DEFAULT_WEBHOOK_SECURITY_HEALTH_CONFIG.detailTopMissingRequirements,
    detailTopUnreadyRoutes:
      raw?.detailTopUnreadyRoutes
      ?? DEFAULT_WEBHOOK_SECURITY_HEALTH_CONFIG.detailTopUnreadyRoutes,
  }
}
