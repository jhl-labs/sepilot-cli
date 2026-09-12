import type { AuditQueryCapabilities } from '../runtime/capabilities.js'
import type { z } from 'zod'
import {
  DEFAULT_OUTBOUND_WEBHOOK_RETRY_CONFIG,
  type SepilotdConfig,
} from '../../config/schema.js'
import {
  outboundWebhookIdFromUrl,
} from '../../hooks/outbound-webhook.js'
import {
  OUTBOUND_WEBHOOK_DEAD_LETTER_ACK_AUDIT_EVENT,
  outboundWebhookDeadLetterAckAuditRecordSchema,
  outboundWebhookDeadLetterAckHistoryCursorPayloadSchema,
  outboundWebhookDeadLetterAckHistoryEntrySchema,
  type outboundWebhookDeadLetterAckHistoryResponseSchema,
  outboundWebhookDeadLetterCursorPayloadSchema,
  type outboundWebhookDeadLetterListResponseSchema,
  outboundWebhookDeadLetterSchema,
  outboundWebhookDeadLetterTimelineAcknowledgementEntrySchema,
  outboundWebhookDeadLetterTimelineCursorPayloadSchema,
  outboundWebhookDeadLetterTimelineDeliveryEntrySchema,
  type outboundWebhookDeadLetterTimelineEntrySchema,
  type outboundWebhookDeadLetterTimelineResponseSchema,
  outboundWebhookDeliveryAuditRecordSchema,
  outboundWebhookDeliveryCursorPayloadSchema,
  outboundWebhookDeliveryDetailSchema,
  type outboundWebhookDeliveryLineageEntrySchema,
  type outboundWebhookDeliveryListResponseSchema,
  type outboundWebhookDeliveryQuerySchema,
  type outboundWebhookDeliverySchema,
  type outboundWebhookListResponseSchema,
  type OutboundWebhookDeadLetter,
  type OutboundWebhookDeadLetterAckHistoryQuery,
  type OutboundWebhookDeadLetterQuery,
  type OutboundWebhookDeadLetterTimelineQuery,
} from './config-schema.js'

export function outboundWebhookDeliveryDetailApiPath(deliveryId: string): string {
  return `/api/v1/config/hooks/outbound-webhooks/deliveries/${encodeURIComponent(deliveryId)}`
}

export function outboundWebhookDeadLetterApiPath(rootDeliveryId: string): string {
  return `/api/v1/config/hooks/outbound-webhooks/dead-letters/${encodeURIComponent(rootDeliveryId)}`
}

export function outboundWebhookDeadLetterTimelineApiPath(rootDeliveryId: string): string {
  return `${outboundWebhookDeadLetterApiPath(rootDeliveryId)}/timeline`
}

export function outboundWebhookDeadLetterAcknowledgementsApiPath(
  rootDeliveryId: string,
): string {
  const searchParams = new URLSearchParams({
    rootDeliveryId,
  })
  return `/api/v1/config/hooks/outbound-webhooks/dead-letters/acknowledgements?${searchParams.toString()}`
}

export function listOutboundWebhookSummaries(
  config: SepilotdConfig,
): z.infer<typeof outboundWebhookListResponseSchema>['data'] {
  return (config.hooks?.outboundWebhooks ?? []).map((webhook) => ({
    id: outboundWebhookIdFromUrl(webhook.url),
    enabled: webhook.enabled !== false,
    url: webhook.url,
    events: [...webhook.events],
    hasSecret: Boolean(webhook.secret),
    headerKeys: Object.keys(webhook.headers ?? {}).sort(),
    retry: {
      maxAttempts: webhook.retry?.maxAttempts
        ?? DEFAULT_OUTBOUND_WEBHOOK_RETRY_CONFIG.maxAttempts,
      backoffMs: webhook.retry?.backoffMs
        ?? DEFAULT_OUTBOUND_WEBHOOK_RETRY_CONFIG.backoffMs,
    },
  }))
}

export function normalizeOutboundWebhookDeliveries(
  auditEvents: unknown[],
): z.infer<typeof outboundWebhookDeliverySchema>[] {
  return auditEvents
    .map((event) => outboundWebhookDeliveryAuditRecordSchema.safeParse(event))
    .filter((result): result is z.SafeParseSuccess<z.infer<typeof outboundWebhookDeliveryAuditRecordSchema>> => result.success)
    .map((result) => ({
      timestamp: result.data.timestamp,
      deliveryId: result.data.deliveryId,
      webhookId: result.data.webhookId,
      url: result.data.url,
      hookEvent: result.data.hookEvent,
      deliveryStatus: result.data.deliveryStatus,
      attemptCount: result.data.attemptCount,
      statusCode: result.data.statusCode,
      durationMs: result.data.durationMs,
      error: result.data.error,
      sessionId: result.data.session,
      ...(result.data.replayedFromDeliveryId
        ? { replayedFromDeliveryId: result.data.replayedFromDeliveryId }
        : {}),
    }))
    .sort((left, right) => left.timestamp.localeCompare(right.timestamp))
}

export function normalizeOutboundWebhookDeliveryAuditRecords(
  auditEvents: unknown[],
): z.infer<typeof outboundWebhookDeliveryAuditRecordSchema>[] {
  return auditEvents
    .map((event) => outboundWebhookDeliveryAuditRecordSchema.safeParse(event))
    .filter(
      (
        result,
      ): result is z.SafeParseSuccess<
        z.infer<typeof outboundWebhookDeliveryAuditRecordSchema>
      > => result.success,
    )
    .map((result) => result.data)
    .sort((left, right) => left.timestamp.localeCompare(right.timestamp))
}

export function summarizeOutboundWebhookReplayLineageEntry(
  delivery: z.infer<typeof outboundWebhookDeliveryAuditRecordSchema>,
  depth: number,
): z.infer<typeof outboundWebhookDeliveryLineageEntrySchema> {
  return {
    deliveryId: delivery.deliveryId,
    timestamp: delivery.timestamp,
    device: delivery.device,
    deliveryStatus: delivery.deliveryStatus,
    attemptCount: delivery.attemptCount,
    statusCode: delivery.statusCode,
    error: delivery.error,
    sessionId: delivery.session,
    replayedFromDeliveryId: delivery.replayedFromDeliveryId,
    depth,
    deliveryDetailPath: outboundWebhookDeliveryDetailApiPath(delivery.deliveryId),
  }
}

export function listOutboundWebhookReplayAncestors(
  deliveriesById: ReadonlyMap<
    string,
    z.infer<typeof outboundWebhookDeliveryAuditRecordSchema>
  >,
  target: z.infer<typeof outboundWebhookDeliveryAuditRecordSchema>,
): z.infer<typeof outboundWebhookDeliveryLineageEntrySchema>[] {
  const ancestors: z.infer<typeof outboundWebhookDeliveryLineageEntrySchema>[] = []
  let current = target
  let depth = 1

  while (current.replayedFromDeliveryId) {
    const parent = deliveriesById.get(current.replayedFromDeliveryId)
    if (!parent) {
      break
    }
    ancestors.push(summarizeOutboundWebhookReplayLineageEntry(parent, depth))
    current = parent
    depth += 1
  }

  return ancestors
}

export function listOutboundWebhookReplayDescendants(
  deliveries: readonly z.infer<typeof outboundWebhookDeliveryAuditRecordSchema>[],
  targetDeliveryId: string,
): z.infer<typeof outboundWebhookDeliveryLineageEntrySchema>[] {
  const childrenByParent = new Map<
    string,
    z.infer<typeof outboundWebhookDeliveryAuditRecordSchema>[]
  >()

  for (const delivery of deliveries) {
    if (!delivery.replayedFromDeliveryId) {
      continue
    }
    const children = childrenByParent.get(delivery.replayedFromDeliveryId)
    if (children) {
      children.push(delivery)
    } else {
      childrenByParent.set(delivery.replayedFromDeliveryId, [delivery])
    }
  }

  const descendants: z.infer<typeof outboundWebhookDeliveryLineageEntrySchema>[] = []
  const queue = (childrenByParent.get(targetDeliveryId) ?? [])
    .map((delivery) => ({ delivery, depth: 1 }))

  while (queue.length > 0) {
    const current = queue.shift()
    if (!current) {
      break
    }

    descendants.push(
      summarizeOutboundWebhookReplayLineageEntry(
        current.delivery,
        current.depth,
      ),
    )

    const children = childrenByParent.get(current.delivery.deliveryId) ?? []
    for (const child of children) {
      queue.push({
        delivery: child,
        depth: current.depth + 1,
      })
    }
  }

  return descendants
}

export function listOutboundWebhookDeliveries(
  auditEvents: unknown[],
  query: z.infer<typeof outboundWebhookDeliveryQuerySchema>,
): z.infer<typeof outboundWebhookDeliveryListResponseSchema> {
  const parsed = normalizeOutboundWebhookDeliveries(auditEvents)

  const filtered = parsed
    .filter((entry) => {
      if (query.id && entry.webhookId !== query.id) {
        return false
      }
      if (query.status && entry.deliveryStatus !== query.status) {
        return false
      }
      return true
    })
    .sort(
      (left, right) =>
        right.timestamp.localeCompare(left.timestamp)
        || right.deliveryId.localeCompare(left.deliveryId)
        || right.webhookId.localeCompare(left.webhookId),
    )

  let startIndex = 0
  if (query.cursor) {
    let decodedCursor: z.infer<
      typeof outboundWebhookDeliveryCursorPayloadSchema
    > | null = null
    try {
      const parsedCursor = JSON.parse(
        Buffer.from(query.cursor, 'base64url').toString('utf-8'),
      )
      decodedCursor = outboundWebhookDeliveryCursorPayloadSchema.parse(
        parsedCursor,
      )
    } catch {
      throw new Error('INVALID_OUTBOUND_WEBHOOK_DELIVERY_CURSOR')
    }

    const cursorIndex = filtered.findIndex(
      (entry) =>
        entry.timestamp === decodedCursor.timestamp
        && entry.deliveryId === decodedCursor.deliveryId
        && entry.webhookId === decodedCursor.webhookId,
    )
    if (cursorIndex < 0) {
      throw new Error('INVALID_OUTBOUND_WEBHOOK_DELIVERY_CURSOR')
    }
    startIndex = cursorIndex + 1
  }

  const limit = query.limit ?? 50
  const data = filtered.slice(startIndex, startIndex + limit)
  const hasMore = startIndex + data.length < filtered.length
  const nextCursor = hasMore && data.length > 0
    ? Buffer.from(
        JSON.stringify({
          timestamp: data[data.length - 1]!.timestamp,
          deliveryId: data[data.length - 1]!.deliveryId,
          webhookId: data[data.length - 1]!.webhookId,
        } satisfies z.infer<typeof outboundWebhookDeliveryCursorPayloadSchema>),
        'utf-8',
      ).toString('base64url')
    : null

  return {
    data,
    meta: {
      limit,
      returned: data.length,
      nextCursor,
    },
  }
}

export function findOutboundWebhookDeliveryDetail(
  auditEvents: unknown[],
  deliveryId: string,
): z.infer<typeof outboundWebhookDeliveryDetailSchema> | null {
  const deliveries = normalizeOutboundWebhookDeliveryAuditRecords(auditEvents)
  const deliveriesById = new Map(
    deliveries.map((delivery) => [delivery.deliveryId, delivery] as const),
  )
  const target = deliveriesById.get(deliveryId)
  if (!target) {
    return null
  }

  const replayChildren = deliveries
    .filter((delivery) => delivery.replayedFromDeliveryId === deliveryId)
    .map((delivery) => ({
      deliveryId: delivery.deliveryId,
      timestamp: delivery.timestamp,
      deliveryStatus: delivery.deliveryStatus,
      attemptCount: delivery.attemptCount,
      statusCode: delivery.statusCode,
      error: delivery.error,
      deliveryDetailPath: outboundWebhookDeliveryDetailApiPath(delivery.deliveryId),
    }))
  const replayAncestors = listOutboundWebhookReplayAncestors(
    deliveriesById,
    target,
  )
  const replayDescendants = listOutboundWebhookReplayDescendants(
    deliveries,
    target.deliveryId,
  )

  return outboundWebhookDeliveryDetailSchema.parse({
    timestamp: target.timestamp,
    deliveryId: target.deliveryId,
    selfPath: outboundWebhookDeliveryDetailApiPath(target.deliveryId),
    webhookId: target.webhookId,
    url: target.url,
    hookEvent: target.hookEvent,
    deliveryStatus: target.deliveryStatus,
    attemptCount: target.attemptCount,
    statusCode: target.statusCode,
    durationMs: target.durationMs,
    error: target.error,
    sessionId: target.session,
    replayedFromDeliveryId: target.replayedFromDeliveryId,
    device: target.device,
    rootDeliveryId: resolveOutboundWebhookDeliveryRootId(
      deliveriesById,
      target.deliveryId,
    ),
    hasPayloadSnapshot: target.payload != null,
    ...(target.payload ? { payloadSnapshot: target.payload } : {}),
    replayChildren,
    replayAncestors,
    replayDescendants,
  })
}

export function resolveOutboundWebhookDeliveryRootId(
  deliveriesById: ReadonlyMap<string, z.infer<typeof outboundWebhookDeliverySchema>>,
  deliveryId: string,
): string {
  const visited = new Set<string>()
  let currentId = deliveryId

  while (!visited.has(currentId)) {
    visited.add(currentId)
    const current = deliveriesById.get(currentId)
    if (!current?.replayedFromDeliveryId) {
      return currentId
    }
    if (!deliveriesById.has(current.replayedFromDeliveryId)) {
      return currentId
    }
    currentId = current.replayedFromDeliveryId
  }

  return currentId
}

export function hasOutboundWebhookReplayChildren(
  deliveries: readonly z.infer<typeof outboundWebhookDeliverySchema>[],
  deliveryId: string,
): boolean {
  return deliveries.some((delivery) => delivery.replayedFromDeliveryId === deliveryId)
}

export function normalizeOutboundWebhookDeadLetterAcks(
  auditEvents: unknown[],
): z.infer<typeof outboundWebhookDeadLetterAckAuditRecordSchema>[] {
  return auditEvents
    .map((event) => outboundWebhookDeadLetterAckAuditRecordSchema.safeParse(event))
    .filter((result): result is z.SafeParseSuccess<z.infer<typeof outboundWebhookDeadLetterAckAuditRecordSchema>> => result.success)
    .map((result) => result.data)
    .sort((left, right) => left.timestamp.localeCompare(right.timestamp))
}

export function listOutboundWebhookDeadLetterAcknowledgements(
  auditEvents: unknown[],
  query: OutboundWebhookDeadLetterAckHistoryQuery,
): z.infer<typeof outboundWebhookDeadLetterAckHistoryResponseSchema> {
  const entries = normalizeOutboundWebhookDeadLetterAcks(auditEvents)
    .filter((entry) => !query.id || entry.webhookId === query.id)
    .filter(
      (entry) =>
        !query.rootDeliveryId || entry.rootDeliveryId === query.rootDeliveryId,
    )
    .filter(
      (entry) =>
        !query.latestDeliveryId
        || entry.latestDeliveryId === query.latestDeliveryId,
    )
    .filter((entry) => !query.device || entry.device === query.device)
    .filter((entry) => !query.since || entry.timestamp >= query.since)
    .sort(
      (left, right) =>
        right.timestamp.localeCompare(left.timestamp)
        || right.rootDeliveryId.localeCompare(left.rootDeliveryId)
        || right.latestDeliveryId.localeCompare(left.latestDeliveryId)
        || right.device.localeCompare(left.device),
    )
    .map((entry) =>
      outboundWebhookDeadLetterAckHistoryEntrySchema.parse({
        ...entry,
        deadLetterPath: outboundWebhookDeadLetterApiPath(entry.rootDeliveryId),
        latestDeliveryDetailPath: outboundWebhookDeliveryDetailApiPath(
          entry.latestDeliveryId,
        ),
      }),
    )

  let startIndex = 0
  if (query.cursor) {
    let decodedCursor: z.infer<
      typeof outboundWebhookDeadLetterAckHistoryCursorPayloadSchema
    > | null = null
    try {
      const parsedCursor = JSON.parse(
        Buffer.from(query.cursor, 'base64url').toString('utf-8'),
      )
      decodedCursor =
        outboundWebhookDeadLetterAckHistoryCursorPayloadSchema.parse(
          parsedCursor,
        )
    } catch {
      throw new Error('INVALID_OUTBOUND_WEBHOOK_DEAD_LETTER_ACK_CURSOR')
    }

    const cursorIndex = entries.findIndex(
      (entry) =>
        entry.timestamp === decodedCursor.timestamp
        && entry.rootDeliveryId === decodedCursor.rootDeliveryId
        && entry.latestDeliveryId === decodedCursor.latestDeliveryId
        && entry.device === decodedCursor.device,
    )
    if (cursorIndex < 0) {
      throw new Error('INVALID_OUTBOUND_WEBHOOK_DEAD_LETTER_ACK_CURSOR')
    }
    startIndex = cursorIndex + 1
  }

  const data = entries.slice(startIndex, startIndex + query.limit)
  const hasMore = startIndex + data.length < entries.length
  const nextCursor = hasMore && data.length > 0
    ? Buffer.from(
        JSON.stringify({
          timestamp: data[data.length - 1]!.timestamp,
          rootDeliveryId: data[data.length - 1]!.rootDeliveryId,
          latestDeliveryId: data[data.length - 1]!.latestDeliveryId,
          device: data[data.length - 1]!.device,
        } satisfies z.infer<
          typeof outboundWebhookDeadLetterAckHistoryCursorPayloadSchema
        >),
        'utf-8',
      ).toString('base64url')
    : null

  return {
    data,
    meta: {
      limit: query.limit,
      returned: data.length,
      nextCursor,
    },
  }
}

export function outboundWebhookDeadLetterTimelineEntryKey(
  entry: z.infer<typeof outboundWebhookDeadLetterTimelineEntrySchema>,
): string {
  if (entry.type === 'delivery') {
    return entry.deliveryId
  }
  return `${entry.rootDeliveryId}:${entry.latestDeliveryId}:${entry.device}`
}

export function listOutboundWebhookDeadLetterTimeline(
  auditEvents: unknown[],
  rootDeliveryId: string,
  query: OutboundWebhookDeadLetterTimelineQuery,
): z.infer<typeof outboundWebhookDeadLetterTimelineResponseSchema> {
  const deliveries = normalizeOutboundWebhookDeliveries(auditEvents)
  const deliveriesById = new Map(
    deliveries.map((delivery) => [delivery.deliveryId, delivery] as const),
  )
  const deliveryEntries = deliveries
    .filter(
      (delivery) =>
        resolveOutboundWebhookDeliveryRootId(
          deliveriesById,
          delivery.deliveryId,
        ) === rootDeliveryId,
    )
    .map((delivery) =>
      outboundWebhookDeadLetterTimelineDeliveryEntrySchema.parse({
        type: 'delivery',
        timestamp: delivery.timestamp,
        rootDeliveryId,
        deliveryId: delivery.deliveryId,
        deliveryDetailPath: outboundWebhookDeliveryDetailApiPath(
          delivery.deliveryId,
        ),
        webhookId: delivery.webhookId,
        url: delivery.url,
        hookEvent: delivery.hookEvent,
        deliveryStatus: delivery.deliveryStatus,
        attemptCount: delivery.attemptCount,
        statusCode: delivery.statusCode,
        durationMs: delivery.durationMs,
        error: delivery.error,
        sessionId: delivery.sessionId,
        replayedFromDeliveryId: delivery.replayedFromDeliveryId,
      }),
    )
  const acknowledgementEntries = normalizeOutboundWebhookDeadLetterAcks(
    auditEvents,
  )
    .filter((entry) => entry.rootDeliveryId === rootDeliveryId)
    .map((entry) =>
      outboundWebhookDeadLetterTimelineAcknowledgementEntrySchema.parse({
        ...entry,
        deadLetterPath: outboundWebhookDeadLetterApiPath(entry.rootDeliveryId),
        latestDeliveryDetailPath: outboundWebhookDeliveryDetailApiPath(
          entry.latestDeliveryId,
        ),
        type: 'acknowledgement',
      }),
    )

  const entries = [...deliveryEntries, ...acknowledgementEntries].sort(
    (left, right) =>
      left.timestamp.localeCompare(right.timestamp)
      || left.type.localeCompare(right.type)
      || outboundWebhookDeadLetterTimelineEntryKey(left).localeCompare(
        outboundWebhookDeadLetterTimelineEntryKey(right),
      ),
  )

  let startIndex = 0
  if (query.cursor) {
    let decodedCursor: z.infer<
      typeof outboundWebhookDeadLetterTimelineCursorPayloadSchema
    > | null = null
    try {
      const parsedCursor = JSON.parse(
        Buffer.from(query.cursor, 'base64url').toString('utf-8'),
      )
      decodedCursor = outboundWebhookDeadLetterTimelineCursorPayloadSchema.parse(
        parsedCursor,
      )
    } catch {
      throw new Error('INVALID_OUTBOUND_WEBHOOK_DEAD_LETTER_TIMELINE_CURSOR')
    }

    const cursorIndex = entries.findIndex(
      (entry) =>
        entry.timestamp === decodedCursor.timestamp
        && entry.type === decodedCursor.type
        && outboundWebhookDeadLetterTimelineEntryKey(entry) === decodedCursor.key,
    )
    if (cursorIndex < 0) {
      throw new Error('INVALID_OUTBOUND_WEBHOOK_DEAD_LETTER_TIMELINE_CURSOR')
    }
    startIndex = cursorIndex + 1
  }

  const data = entries.slice(startIndex, startIndex + query.limit)
  const hasMore = startIndex + data.length < entries.length
  const nextCursor = hasMore && data.length > 0
    ? Buffer.from(
        JSON.stringify({
          timestamp: data[data.length - 1]!.timestamp,
          type: data[data.length - 1]!.type,
          key: outboundWebhookDeadLetterTimelineEntryKey(
            data[data.length - 1]!,
          ),
        } satisfies z.infer<
          typeof outboundWebhookDeadLetterTimelineCursorPayloadSchema
        >),
        'utf-8',
      ).toString('base64url')
    : null

  return {
    data,
    meta: {
      limit: query.limit,
      returned: data.length,
      nextCursor,
    },
  }
}

export function listOutboundWebhookDeadLetters(
  auditEvents: unknown[],
  query: OutboundWebhookDeadLetterQuery,
): z.infer<typeof outboundWebhookDeadLetterListResponseSchema> {
  const deliveries = normalizeOutboundWebhookDeliveries(auditEvents)
  const acknowledgements = normalizeOutboundWebhookDeadLetterAcks(auditEvents)
  const deliveriesById = new Map(
    deliveries.map((delivery) => [delivery.deliveryId, delivery] as const),
  )
  const acknowledgementsByRootDeliveryId = new Map(
    acknowledgements.map((ack) => [ack.rootDeliveryId, ack] as const),
  )
  const chains = new Map<string, z.infer<typeof outboundWebhookDeliverySchema>[]>()

  for (const delivery of deliveries) {
    const rootDeliveryId = resolveOutboundWebhookDeliveryRootId(
      deliveriesById,
      delivery.deliveryId,
    )
    const chain = chains.get(rootDeliveryId)
    if (chain) {
      chain.push(delivery)
    } else {
      chains.set(rootDeliveryId, [delivery])
    }
  }

  const deadLetters = Array.from(chains.entries())
    .map<OutboundWebhookDeadLetter | null>(([rootDeliveryId, chain]) => {
      if (chain.some((delivery) => delivery.deliveryStatus === 'success')) {
        return null
      }

      const latest = [...chain].reverse().find((delivery) => delivery.deliveryStatus === 'error')
      if (!latest) {
        return null
      }

      const firstFailed = chain.find((delivery) => delivery.deliveryStatus === 'error') ?? latest
      const latestAcknowledgement = acknowledgementsByRootDeliveryId.get(rootDeliveryId)
      const isAcknowledged = latestAcknowledgement != null
        && latestAcknowledgement.timestamp >= latest.timestamp
      return outboundWebhookDeadLetterSchema.parse({
        rootDeliveryId,
        latestDeliveryId: latest.deliveryId,
        selfPath: outboundWebhookDeadLetterApiPath(rootDeliveryId),
        timelinePath: outboundWebhookDeadLetterTimelineApiPath(rootDeliveryId),
        latestDeliveryDetailPath: outboundWebhookDeliveryDetailApiPath(
          latest.deliveryId,
        ),
        acknowledgementsPath: outboundWebhookDeadLetterAcknowledgementsApiPath(
          rootDeliveryId,
        ),
        webhookId: latest.webhookId,
        url: latest.url,
        hookEvent: latest.hookEvent,
        firstFailedAt: firstFailed.timestamp,
        lastAttemptAt: latest.timestamp,
        replayCount: chain.filter((delivery) => delivery.replayedFromDeliveryId).length,
        attemptCount: latest.attemptCount,
        statusCode: latest.statusCode,
        durationMs: latest.durationMs,
        error: latest.error,
        sessionId: latest.sessionId,
        state: isAcknowledged ? 'acknowledged' : 'open',
        ...(latestAcknowledgement
          ? {
              latestAcknowledgement: {
                timestamp: latestAcknowledgement.timestamp,
                device: latestAcknowledgement.device,
                ...(latestAcknowledgement.note
                  ? { note: latestAcknowledgement.note }
                  : {}),
              },
            }
          : {}),
        ...(isAcknowledged
          ? {
              acknowledgedAt: latestAcknowledgement.timestamp,
              acknowledgedByDevice: latestAcknowledgement.device,
              acknowledgmentNote: latestAcknowledgement.note,
            }
          : {}),
      })
    })
    .filter((entry): entry is OutboundWebhookDeadLetter => entry !== null)
    .filter((entry) => !query.id || entry.webhookId === query.id)
    .filter((entry) => {
      if (!query.state || query.state === 'open') {
        return entry.state === 'open'
      }
      if (query.state === 'acknowledged') {
        return entry.state === 'acknowledged'
      }
      return true
    })
    .sort(
      (left, right) =>
        right.lastAttemptAt.localeCompare(left.lastAttemptAt)
        || right.rootDeliveryId.localeCompare(left.rootDeliveryId)
        || right.latestDeliveryId.localeCompare(left.latestDeliveryId),
    )

  let startIndex = 0
  if (query.cursor) {
    let decodedCursor: z.infer<
      typeof outboundWebhookDeadLetterCursorPayloadSchema
    > | null = null
    try {
      const parsedCursor = JSON.parse(
        Buffer.from(query.cursor, 'base64url').toString('utf-8'),
      )
      decodedCursor = outboundWebhookDeadLetterCursorPayloadSchema.parse(
        parsedCursor,
      )
    } catch {
      throw new Error('INVALID_OUTBOUND_WEBHOOK_DEAD_LETTER_CURSOR')
    }

    const cursorIndex = deadLetters.findIndex(
      (entry) =>
        entry.lastAttemptAt === decodedCursor.lastAttemptAt
        && entry.rootDeliveryId === decodedCursor.rootDeliveryId
        && entry.latestDeliveryId === decodedCursor.latestDeliveryId,
    )
    if (cursorIndex < 0) {
      throw new Error('INVALID_OUTBOUND_WEBHOOK_DEAD_LETTER_CURSOR')
    }
    startIndex = cursorIndex + 1
  }

  const limit = query.limit ?? 50
  const data = deadLetters.slice(startIndex, startIndex + limit)
  const hasMore = startIndex + data.length < deadLetters.length
  const nextCursor = hasMore && data.length > 0
    ? Buffer.from(
        JSON.stringify({
          lastAttemptAt: data[data.length - 1]!.lastAttemptAt,
          rootDeliveryId: data[data.length - 1]!.rootDeliveryId,
          latestDeliveryId: data[data.length - 1]!.latestDeliveryId,
        } satisfies z.infer<typeof outboundWebhookDeadLetterCursorPayloadSchema>),
        'utf-8',
      ).toString('base64url')
    : null

  return {
    data,
    meta: {
      limit,
      returned: data.length,
      nextCursor,
    },
  }
}

export function findOutboundWebhookDeadLetter(
  auditEvents: unknown[],
  rootDeliveryId: string,
): OutboundWebhookDeadLetter | null {
  return listOutboundWebhookDeadLetters(auditEvents, {
    state: 'all',
    limit: Number.MAX_SAFE_INTEGER,
  }).data.find((deadLetter) => deadLetter.rootDeliveryId === rootDeliveryId) ?? null
}

export function findOutboundWebhookIndexById(
  config: SepilotdConfig,
  id: string,
): number {
  return config.hooks.outboundWebhooks.findIndex(
    (webhook) => outboundWebhookIdFromUrl(webhook.url) === id,
  )
}

export function findOutboundWebhookDeliveryAuditRecord(
  auditEvents: unknown[],
  deliveryId: string,
): z.infer<typeof outboundWebhookDeliveryAuditRecordSchema> | null {
  for (const event of auditEvents) {
    const parsed = outboundWebhookDeliveryAuditRecordSchema.safeParse(event)
    if (!parsed.success) {
      continue
    }
    if (parsed.data.deliveryId === deliveryId) {
      return parsed.data
    }
  }
  return null
}

export async function queryOutboundWebhookAuditEvents(
  runtime: AuditQueryCapabilities | undefined,
): Promise<unknown[]> {
  if (!runtime?.auditLogger) {
    return []
  }

  return runtime.auditLogger.query({})
}

export async function queryOutboundWebhookDeadLetterAcknowledgementAuditEvents(
  runtime: AuditQueryCapabilities | undefined,
  since?: string,
): Promise<unknown[]> {
  if (!runtime?.auditLogger) {
    return []
  }

  return runtime.auditLogger.query({
    event: OUTBOUND_WEBHOOK_DEAD_LETTER_ACK_AUDIT_EVENT,
    ...(since ? { since } : {}),
  })
}
