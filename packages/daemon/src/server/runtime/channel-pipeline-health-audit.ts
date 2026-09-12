import type { AuditQueryCapabilities } from './capabilities.js'
import { z } from 'zod'
import { CHANNEL_PIPELINE_HEALTH_CONFIG_AUDIT_EVENT } from '../routes/config.js'
import { getLatestAuditChange } from './latest-audit-change.js'

export const latestChannelPipelineHealthChangeSchema = z.object({
  timestamp: z.string().datetime(),
  route: z.enum([
    '/api/v1/config',
    '/api/v1/config/observability/channel-pipeline-health',
  ]),
  device: z.string(),
})

export async function getLatestChannelPipelineHealthPolicyChange(
  runtime: AuditQueryCapabilities | undefined,
): Promise<z.infer<typeof latestChannelPipelineHealthChangeSchema> | null> {
  return getLatestAuditChange(
    runtime,
    CHANNEL_PIPELINE_HEALTH_CONFIG_AUDIT_EVENT,
    latestChannelPipelineHealthChangeSchema,
  )
}
