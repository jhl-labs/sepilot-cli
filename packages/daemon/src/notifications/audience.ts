import { normalizeSurfaceLabel } from '../server/request-surface.js'

export const ALL_NOTIFICATION_AUDIENCES = '*'

export type NotificationAudience = string[] | null

export function normalizeNotificationAudience(value: unknown): NotificationAudience {
  if (value == null) return null
  if (!Array.isArray(value)) return null

  const seen = new Set<string>()
  const out: string[] = []
  for (const raw of value) {
    if (typeof raw !== 'string') continue
    const trimmed = raw.trim()
    if (!trimmed) continue
    const isDeny = trimmed.startsWith('!')
    const body = isDeny ? trimmed.slice(1) : trimmed
    const normalized = body === ALL_NOTIFICATION_AUDIENCES
      ? ALL_NOTIFICATION_AUDIENCES
      : normalizeSurfaceLabel(body)
    if (!normalized) continue
    const token = isDeny && normalized !== ALL_NOTIFICATION_AUDIENCES
      ? `!${normalized}`
      : normalized
    if (seen.has(token)) continue
    seen.add(token)
    out.push(token)
  }
  return out
}

export function notificationVisibleToSurface(
  audience: NotificationAudience | undefined,
  surface: string | null | undefined,
): boolean {
  // No audience restriction → visible to everyone.
  if (audience == null) return true
  // `undefined` = no surface filter requested (internal/admin list-all): show
  // everything. `null` = a client resolved to no surface label (unlabeled
  // client) — that is an identified request context, so it is fail-closed below.
  if (surface === undefined) return true
  const normalizedSurface = normalizeSurfaceLabel(surface)
  if (!normalizedSurface) {
    // Fail-CLOSED for an unlabeled/unrecognized surface: a restricted
    // notification must not leak to a client we cannot identify. Only a
    // wildcard (broadcast) audience is visible without a known surface.
    return audience.includes(ALL_NOTIFICATION_AUDIENCES)
  }
  if (audience.includes(`!${normalizedSurface}`)) return false
  if (audience.includes(ALL_NOTIFICATION_AUDIENCES)) return true
  return audience.includes(normalizedSurface)
}

export function updateNotificationAudienceSubscription(
  audience: NotificationAudience | undefined,
  surface: string,
  subscribed: boolean,
): string[] {
  const normalizedSurface = normalizeSurfaceLabel(surface)
  if (!normalizedSurface) {
    return normalizeNotificationAudience(audience ?? [ALL_NOTIFICATION_AUDIENCES])
      ?? [ALL_NOTIFICATION_AUDIENCES]
  }

  const current = normalizeNotificationAudience(audience) ?? [ALL_NOTIFICATION_AUDIENCES]
  const next = current.filter((entry) =>
    entry !== normalizedSurface && entry !== `!${normalizedSurface}`)

  if (subscribed) {
    if (!next.includes(ALL_NOTIFICATION_AUDIENCES)) {
      next.push(normalizedSurface)
    }
  } else if (next.includes(ALL_NOTIFICATION_AUDIENCES)) {
    next.push(`!${normalizedSurface}`)
  }

  return normalizeNotificationAudience(next) ?? [ALL_NOTIFICATION_AUDIENCES]
}
