import type { AuditQueryCapabilities } from './capabilities.js'
import { z } from 'zod'
import { WEBHOOK_SECURITY_POLICY_CONFIG_AUDIT_EVENT } from '../routes/config.js'
import { getLatestAuditChange } from './latest-audit-change.js'

export const latestWebhookSecurityPolicyChangeSchema = z.object({
  timestamp: z.string().datetime(),
  route: z.enum([
    '/api/v1/config',
    '/api/v1/config/security/webhooks',
  ]),
  device: z.string(),
})

export async function getLatestWebhookSecurityPolicyChange(
  runtime: AuditQueryCapabilities | undefined,
): Promise<z.infer<typeof latestWebhookSecurityPolicyChangeSchema> | null> {
  return getLatestAuditChange(
    runtime,
    WEBHOOK_SECURITY_POLICY_CONFIG_AUDIT_EVENT,
    latestWebhookSecurityPolicyChangeSchema,
  )
}
