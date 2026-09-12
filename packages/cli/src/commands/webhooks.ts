import chalk from 'chalk'
import type { DaemonWebhookEndpointSummary } from '@sepilotd/api-client'
import { DaemonClient } from '../client/http.js'
import { output } from '../output/formatter.js'
import { detectCliLocale } from '../utils/locale.js'

const WEBHOOKS_COPY = {
  en: {
    invalidPathDotDot: (path: string) => `Invalid webhook path: ${path} (contains '..')`,
    invalidPathDelimiter: (path: string) => `Invalid webhook path: ${path} (contains URL delimiter or escape char)`,
    invalidPathWhitespace: (path: string) => `Invalid webhook path: ${path} (contains whitespace or control chars)`,
    pathEmpty: 'Webhook path cannot be empty',
    invalidPathReserved: (path: string, seg: string) => `Invalid webhook path: ${path} (reserved route segment: ${seg})`,
    enabled: 'enabled',
    disabled: 'disabled',
    noEndpoints: 'No generic webhook endpoints configured.\nUse `sepilot webhooks add custom --header x-secret --secret shared-secret` to add one.',
    updated: (path: string) => `Generic webhook endpoint updated: ${path}`,
    updatedWith: (path: string, summary: string) => `Generic webhook endpoint updated: ${path}\n${summary}`,
    removed: (id: string) => `Generic webhook endpoint removed: ${id}`,
    endpointEnabled: (id: string) => `Generic webhook endpoint enabled: ${id}`,
    endpointEnabledWith: (id: string, summary: string) => `Generic webhook endpoint enabled: ${id}\n${summary}`,
    endpointDisabled: (id: string) => `Generic webhook endpoint disabled: ${id}`,
    endpointDisabledWith: (id: string, summary: string) => `Generic webhook endpoint disabled: ${id}\n${summary}`,
  },
  ko: {
    invalidPathDotDot: (path: string) => `잘못된 웹훅 경로: ${path} ('..' 포함)`,
    invalidPathDelimiter: (path: string) => `잘못된 웹훅 경로: ${path} (URL 구분자 또는 이스케이프 문자 포함)`,
    invalidPathWhitespace: (path: string) => `잘못된 웹훅 경로: ${path} (공백 또는 제어 문자 포함)`,
    pathEmpty: '웹훅 경로는 비어 있을 수 없습니다',
    invalidPathReserved: (path: string, seg: string) => `잘못된 웹훅 경로: ${path} (예약된 라우트 세그먼트: ${seg})`,
    enabled: '활성화됨',
    disabled: '비활성화됨',
    noEndpoints: '구성된 일반 웹훅 엔드포인트가 없습니다.\n`sepilot webhooks add custom --header x-secret --secret shared-secret`로 추가하세요.',
    updated: (path: string) => `일반 웹훅 엔드포인트 업데이트됨: ${path}`,
    updatedWith: (path: string, summary: string) => `일반 웹훅 엔드포인트 업데이트됨: ${path}\n${summary}`,
    removed: (id: string) => `일반 웹훅 엔드포인트 제거됨: ${id}`,
    endpointEnabled: (id: string) => `일반 웹훅 엔드포인트 활성화됨: ${id}`,
    endpointEnabledWith: (id: string, summary: string) => `일반 웹훅 엔드포인트 활성화됨: ${id}\n${summary}`,
    endpointDisabled: (id: string) => `일반 웹훅 엔드포인트 비활성화됨: ${id}`,
    endpointDisabledWith: (id: string, summary: string) => `일반 웹훅 엔드포인트 비활성화됨: ${id}\n${summary}`,
  },
} as const

function webhooksCopy() {
  return WEBHOOKS_COPY[detectCliLocale()] ?? WEBHOOKS_COPY.en
}

const RESERVED_GENERIC_WEBHOOK_ROUTE_SEGMENTS = new Set([
  'discord',
  'line',
  'security',
  'slack',
  'teams',
  'whatsapp',
])

export interface WebhooksMutationOptions {
  url?: string
}

export interface WebhooksAddOptions extends WebhooksMutationOptions {
  header: string
  secret: string
  event?: string[]
  ip?: string[]
  disabled?: boolean
}

function assertSafeWebhookPath(path: string): void {
  const copy = webhooksCopy()
  // Reject path-traversal segments and other characters that have no
  // legitimate use in a public webhook route. Without this guard,
  // `webhooks add "../../etc/passwd"` would round-trip to the daemon
  // as `/webhooks/../../etc/passwd` — a route the user definitely
  // didn't intend to expose.
  if (/(?:^|\/)\.\.(?:\/|$)/.test(path)) {
    throw new Error(copy.invalidPathDotDot(path))
  }
  if (/[?#%]/.test(path)) {
    throw new Error(copy.invalidPathDelimiter(path))
  }
  if (/[\s\x00-\x1f]/.test(path)) {
    throw new Error(copy.invalidPathWhitespace(path))
  }
  if (path.length === 0 || path.trim().length === 0) {
    throw new Error(copy.pathEmpty)
  }
  const firstSegment = publicWebhookRoutePath(path.trim())
    .replace(/^\/+/, '')
    .split('/', 1)[0]!
    .toLowerCase()
  if (RESERVED_GENERIC_WEBHOOK_ROUTE_SEGMENTS.has(firstSegment)) {
    throw new Error(copy.invalidPathReserved(path, firstSegment))
  }
}

function publicWebhookRoutePath(path: string): string {
  const withLeadingSlash = path.startsWith('/') ? path : `/${path}`
  if (withLeadingSlash.startsWith('/api/v1/webhooks/')) {
    return withLeadingSlash.slice('/api/v1/webhooks/'.length)
  }
  if (withLeadingSlash.startsWith('/webhooks/')) {
    return withLeadingSlash.slice('/webhooks/'.length)
  }
  if (withLeadingSlash.startsWith('/hook/')) {
    return withLeadingSlash.slice('/hook/'.length)
  }
  return withLeadingSlash.replace(/^\/+/, '')
}

function normalizeWebhookPath(path: string): string {
  assertSafeWebhookPath(path)
  const trimmed = path.trim()
  const withLeadingSlash = trimmed.startsWith('/') ? trimmed : `/${trimmed}`
  if (withLeadingSlash.startsWith('/api/v1/webhooks/')) {
    return `/hook/${withLeadingSlash.slice('/api/v1/webhooks/'.length)}`
  }
  if (withLeadingSlash.startsWith('/webhooks/')) {
    return `/hook/${withLeadingSlash.slice('/webhooks/'.length)}`
  }
  if (withLeadingSlash.startsWith('/hook/')) {
    return withLeadingSlash.replace(/\/+/g, '/')
  }
  return `/hook/${withLeadingSlash.replace(/^\/+/, '')}`.replace(/\/+/g, '/')
}

function webhookEndpointId(path: string): string {
  return `webhook-endpoint:${normalizeWebhookPath(path).slice('/hook/'.length)}`
    .replace(/[^a-zA-Z0-9:/._-]/g, '_')
}

function formatWebhookEndpointSummary(
  endpoint: DaemonWebhookEndpointSummary,
): string {
  const copy = webhooksCopy()
  const status = endpoint.enabled ? chalk.green(copy.enabled) : chalk.gray(copy.disabled)
  const extras: string[] = [`header=${endpoint.secretHeader}`]
  if (endpoint.allowedEvents.length > 0) {
    extras.push(`events=${endpoint.allowedEvents.join(',')}`)
  }
  if (endpoint.allowedIps.length > 0) {
    extras.push(`ips=${endpoint.allowedIps.join(',')}`)
  }

  const head = `  ${chalk.bold(endpoint.id)}  ${status}`
  const tail = `      ${endpoint.publicRoute}  ${chalk.gray(`[${extras.join(' ')}]`)}`
  return `${head}\n${tail}`
}

export async function webhooksListCommand(options: WebhooksMutationOptions) {
  const copy = webhooksCopy()
  const client = new DaemonClient(options.url)
  const data = await client.webhookEndpoints()
  output(data, (endpoints) => {
    if (!endpoints.length) {
      return copy.noEndpoints
    }
    return endpoints.map(formatWebhookEndpointSummary).join('\n')
  })
}

export async function webhooksAddCommand(path: string, options: WebhooksAddOptions) {
  const copy = webhooksCopy()
  const client = new DaemonClient(options.url)
  const endpointId = webhookEndpointId(path)
  await client.upsertWebhookEndpoint({
    enabled: options.disabled ? false : true,
    path,
    secretHeader: options.header,
    secretValue: options.secret,
    allowedEvents: options.event ?? [],
    allowedIps: options.ip ?? [],
  })

  const endpoint = (await client.webhookEndpoints()).find(
    (entry) => entry.id === endpointId,
  )
  output(
    {
      ok: true,
      path,
      action: 'upserted',
      endpoint,
    },
    (result) => {
      if (!result.endpoint) {
        return copy.updated(result.path)
      }
      return copy.updatedWith(result.path, formatWebhookEndpointSummary(result.endpoint))
    },
  )
}

export async function webhooksRemoveCommand(id: string, options: WebhooksMutationOptions) {
  const copy = webhooksCopy()
  const client = new DaemonClient(options.url)
  await client.deleteWebhookEndpoint(id)
  output(
    {
      ok: true,
      id,
      action: 'removed',
    },
    (result) => copy.removed(result.id),
  )
}

export async function webhooksEnableCommand(id: string, options: WebhooksMutationOptions) {
  const copy = webhooksCopy()
  const client = new DaemonClient(options.url)
  await client.setWebhookEndpointEnabled(id, true)
  const endpoint = (await client.webhookEndpoints()).find((entry) => entry.id === id)
  output(
    {
      ok: true,
      id,
      action: 'enabled',
      endpoint,
    },
    (result) => {
      if (!result.endpoint) {
        return copy.endpointEnabled(result.id)
      }
      return copy.endpointEnabledWith(result.id, formatWebhookEndpointSummary(result.endpoint))
    },
  )
}

export async function webhooksDisableCommand(id: string, options: WebhooksMutationOptions) {
  const copy = webhooksCopy()
  const client = new DaemonClient(options.url)
  await client.setWebhookEndpointEnabled(id, false)
  const endpoint = (await client.webhookEndpoints()).find((entry) => entry.id === id)
  output(
    {
      ok: true,
      id,
      action: 'disabled',
      endpoint,
    },
    (result) => {
      if (!result.endpoint) {
        return copy.endpointDisabled(result.id)
      }
      return copy.endpointDisabledWith(result.id, formatWebhookEndpointSummary(result.endpoint))
    },
  )
}

export const __testables = {
  assertSafeWebhookPath,
  formatWebhookEndpointSummary,
  normalizeWebhookPath,
  publicWebhookRoutePath,
  webhookEndpointId,
}
