import type { AuditQueryCapabilities } from './capabilities.js'
import type { ZodType } from 'zod'

type LatestAuditChange = {
  timestamp: string
}

export async function getLatestAuditChange<TChange extends LatestAuditChange>(
  runtime: AuditQueryCapabilities | undefined,
  event: string,
  schema: ZodType<TChange>,
): Promise<TChange | null> {
  if (!runtime?.auditLogger) {
    return null
  }

  const auditEvents = await runtime.auditLogger.query({
    event,
    limit: 20,
  })

  let latest: TChange | null = null

  for (const auditEvent of auditEvents) {
    const parsed = schema.safeParse(pickAuditChangeFields(auditEvent))
    if (!parsed.success) {
      continue
    }

    if (!latest || parsed.data.timestamp.localeCompare(latest.timestamp) > 0) {
      latest = parsed.data
    }
  }

  return latest
}

function pickAuditChangeFields(event: unknown): Record<string, unknown> {
  if (!event || typeof event !== 'object') {
    return {}
  }

  return {
    timestamp: readField(event, 'timestamp'),
    route: readField(event, 'route'),
    device: readField(event, 'device'),
  }
}

function readField(event: object, field: string): unknown {
  return field in event ? (event as Record<string, unknown>)[field] : undefined
}
