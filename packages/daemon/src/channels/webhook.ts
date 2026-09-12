import type { IChannel, ChannelType, ChannelTarget, ChannelMessage, ChannelStatus, IncomingMessage, Disposable } from '@sepilotd/core'
import { createHmac, timingSafeEqual } from 'node:crypto'
import { isIP } from 'node:net'

export interface WebhookEndpointConfig {
  enabled?: boolean
  path: string
  secretHeader: string
  secretValue: string
  allowedIps?: string[]
  allowedEvents?: string[]
  /** Opt-in replay protection; defaults to off for back-compat. */
  requireTimestamp?: boolean
  timestampHeader?: string
  timestampMaxSkewSeconds?: number
}

export interface WebhookChannelConfig {
  endpoints: WebhookEndpointConfig[]
  rateLimitPerMinute?: number
}

export interface WebhookEndpointStatus {
  path: string
  verificationReady: boolean
}

export function normalizeWebhookEndpointPath(path: string): string {
  const trimmed = path.trim()
  if (trimmed.length === 0) {
    return '/hook'
  }

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

export function webhookEndpointIdFromPath(path: string): string {
  return `webhook-endpoint:${normalizeWebhookEndpointPath(path).slice('/hook/'.length)}`
    .replace(/[^a-zA-Z0-9:/._-]/g, '_')
}

export function webhookPublicRouteFromPath(path: string): string {
  return `/api/v1/webhooks/${normalizeWebhookEndpointPath(path).slice('/hook/'.length)}`
}

function parseIpv4Bytes(input: string): number[] | null {
  if (isIP(input) !== 4) {
    return null
  }

  return input.split('.').map((part) => Number.parseInt(part, 10))
}

function parseIpv6Bytes(input: string): number[] | null {
  const value = input.split('%', 1)[0]!.toLowerCase()
  if (isIP(value) !== 6) {
    return null
  }

  const parseSegment = (segment: string): number[] | null => {
    if (!segment) {
      return []
    }

    const hextets: number[] = []
    for (const piece of segment.split(':')) {
      if (!piece) {
        return null
      }

      if (piece.includes('.')) {
        const ipv4Bytes = parseIpv4Bytes(piece)
        if (!ipv4Bytes) {
          return null
        }
        hextets.push((ipv4Bytes[0]! << 8) + ipv4Bytes[1]!)
        hextets.push((ipv4Bytes[2]! << 8) + ipv4Bytes[3]!)
        continue
      }

      const hextet = Number.parseInt(piece, 16)
      if (!Number.isInteger(hextet) || hextet < 0 || hextet > 0xffff) {
        return null
      }
      hextets.push(hextet)
    }
    return hextets
  }

  const parts = value.split('::')
  if (parts.length > 2) {
    return null
  }

  const head = parseSegment(parts[0] ?? '')
  const tail = parseSegment(parts[1] ?? '')
  if (!head || !tail) {
    return null
  }

  const missing = parts.length === 2 ? 8 - head.length - tail.length : 0
  if (missing < 0 || (parts.length === 1 && head.length !== 8)) {
    return null
  }

  const hextets = [...head, ...Array.from({ length: missing }, () => 0), ...tail]
  if (hextets.length !== 8) {
    return null
  }

  return hextets.flatMap((hextet) => [hextet >> 8, hextet & 0xff])
}

function parseIpBytes(input: string): number[] | null {
  const trimmed = input.trim()
  if (trimmed.startsWith('::ffff:')) {
    const mappedIpv4 = trimmed.slice('::ffff:'.length)
    const mappedBytes = parseIpv4Bytes(mappedIpv4)
    if (mappedBytes) {
      return mappedBytes
    }
  }

  return parseIpv4Bytes(trimmed) ?? parseIpv6Bytes(trimmed)
}

function bytesEqual(left: readonly number[], right: readonly number[]): boolean {
  return left.length === right.length
    && left.every((byte, index) => byte === right[index])
}

function bytesInCidr(
  candidate: readonly number[],
  network: readonly number[],
  prefixLength: number,
): boolean {
  if (candidate.length !== network.length) {
    return false
  }

  const totalBits = candidate.length * 8
  if (!Number.isInteger(prefixLength) || prefixLength < 0 || prefixLength > totalBits) {
    return false
  }

  const fullBytes = Math.floor(prefixLength / 8)
  const remainingBits = prefixLength % 8
  for (let index = 0; index < fullBytes; index += 1) {
    if (candidate[index] !== network[index]) {
      return false
    }
  }

  if (remainingBits === 0) {
    return true
  }

  const mask = (0xff << (8 - remainingBits)) & 0xff
  return (candidate[fullBytes]! & mask) === (network[fullBytes]! & mask)
}

function matchesAllowedIp(remoteIp: string | undefined, allowedEntry: string): boolean {
  if (!remoteIp) {
    return false
  }

  const candidate = parseIpBytes(remoteIp)
  if (!candidate) {
    return false
  }

  const parts = allowedEntry.trim().split('/')
  if (parts.length > 2) {
    return false
  }

  const [networkText, prefixText] = parts
  const network = parseIpBytes(networkText ?? '')
  if (!network) {
    return false
  }

  if (prefixText == null) {
    return bytesEqual(candidate, network)
  }

  if (!/^\d+$/.test(prefixText)) {
    return false
  }

  const prefixLength = Number.parseInt(prefixText, 10)
  return bytesInCidr(candidate, network, prefixLength)
}

function signatureMatches(headerValue: string, expectedHex: string): boolean {
  const expected = Buffer.from(expectedHex, 'hex')
  const candidates = headerValue
    .split(',')
    .map((part) => part.trim())
    .flatMap((part) => [
      part,
      part.startsWith('sha256=') ? part.slice('sha256='.length) : '',
    ])
    .filter((part) => /^[a-f0-9]{64}$/i.test(part))

  return candidates.some((candidate) => {
    const actual = Buffer.from(candidate, 'hex')
    return actual.length === expected.length && timingSafeEqual(actual, expected)
  })
}

export class WebhookChannel implements IChannel {
  readonly id = 'webhook'
  readonly type: ChannelType = 'webhook'
  private config: WebhookChannelConfig
  private status: ChannelStatus = 'disconnected'
  private handlers: Array<(msg: IncomingMessage) => Promise<void>> = []

  constructor(config: WebhookChannelConfig) {
    this.config = config
  }

  async start(): Promise<void> {
    this.status = 'connected'
  }

  async stop(): Promise<void> {
    this.status = 'disconnected'
  }

  getStatus(): ChannelStatus {
    return this.status
  }

  listConfiguredEndpoints(): WebhookEndpointStatus[] {
    return this.config.endpoints
      .filter((endpoint) => endpoint.enabled !== false)
      .map((endpoint) => ({
        path: normalizeWebhookEndpointPath(endpoint.path),
        verificationReady: this.isEndpointVerificationReady(endpoint),
      }))
  }

  listConfiguredPaths(): string[] {
    return this.listConfiguredEndpoints()
      .map((endpoint) => endpoint.path)
  }

  getEndpointVerificationState(path: string): 'missing' | 'verification-unavailable' | 'ready' {
    const normalizedPath = normalizeWebhookEndpointPath(path)
    const endpoint = this.config.endpoints.find((entry) =>
      entry.enabled !== false
      && normalizeWebhookEndpointPath(entry.path) === normalizedPath,
    )
    if (!endpoint) {
      return 'missing'
    }
    return this.isEndpointVerificationReady(endpoint)
      ? 'ready'
      : 'verification-unavailable'
  }

  onMessage(handler: (msg: IncomingMessage) => Promise<void>): Disposable {
    this.handlers.push(handler)
    return {
      dispose: () => {
        const i = this.handlers.indexOf(handler)
        if (i >= 0) this.handlers.splice(i, 1)
      },
    }
  }

  async sendMessage(_target: ChannelTarget, _msg: ChannelMessage): Promise<void> {
    // Webhooks are receive-only by nature
  }

  /** Process incoming webhook request */
  async handleWebhook(
    path: string,
    body: Record<string, unknown>,
    headers: Record<string, string>,
    ip?: string,
    rawBody?: string,
  ): Promise<boolean> {
    const normalizedPath = normalizeWebhookEndpointPath(path)
    const endpoint = this.config.endpoints.find((entry) =>
      entry.enabled !== false
      && normalizeWebhookEndpointPath(entry.path) === normalizedPath,
    )
    if (!endpoint) return false
    if (!this.isEndpointVerificationReady(endpoint)) return false

    // Verify IP allowlist
    if (endpoint.allowedIps?.length && !endpoint.allowedIps.some(allowed => matchesAllowedIp(ip, allowed))) {
      return false
    }

    // Opt-in replay protection: when the endpoint declares
    // `requireTimestamp`, refuse deliveries older than the configured skew
    // window AND bind the timestamp into the signed payload. Skew alone is
    // ineffective because a replayer can simply resend a captured body with a
    // fresh timestamp; binding the timestamp into the HMAC forces the sender
    // to sign `${timestamp}.${rawBody}`, so a replay with a new timestamp no
    // longer matches the captured signature.
    let boundTimestamp: string | undefined
    if (endpoint.requireTimestamp) {
      const headerName = (endpoint.timestampHeader ?? 'x-webhook-timestamp').toLowerCase()
      const tsValue = headers[headerName]
      if (typeof tsValue !== 'string' || !tsValue.trim()) return false
      const tsNum = Number(tsValue)
      if (!Number.isFinite(tsNum) || tsNum <= 0) return false
      // Accept either seconds or millis since epoch.
      const tsMs = tsNum < 1e12 ? tsNum * 1000 : tsNum
      const skewSeconds = endpoint.timestampMaxSkewSeconds ?? 300
      const ageSeconds = Math.abs(Date.now() - tsMs) / 1000
      if (ageSeconds > skewSeconds) return false
      boundTimestamp = tsValue
    }

    // Verify HMAC signature
    const signature = headers[endpoint.secretHeader.toLowerCase()]
    if (!signature) return false
    const signedBody = rawBody ?? JSON.stringify(body)
    const signingPayload = boundTimestamp !== undefined
      ? `${boundTimestamp}.${signedBody}`
      : signedBody
    const expected = createHmac('sha256', endpoint.secretValue)
      .update(signingPayload)
      .digest('hex')
    if (!signatureMatches(signature, expected)) return false

    // Check event type
    const eventType = headers['x-github-event'] ?? headers['x-webhook-event'] ?? (body.type as string) ?? 'unknown'
    if (endpoint.allowedEvents?.length && !endpoint.allowedEvents.includes(eventType)) return false

    const msg: IncomingMessage = {
      channelType: 'webhook',
      channelId: normalizedPath,
      messageId: headers['x-delivery'] ?? headers['x-request-id'] ?? Date.now().toString(),
      text: typeof body === 'string' ? body : JSON.stringify(body),
      sender: { id: 'webhook', name: normalizedPath, type: 'bot' },
      timestamp: new Date().toISOString(),
      raw: body,
    }

    for (const h of this.handlers) {
      try { await h(msg) } catch { /* handler errors are silently ignored */ }
    }
    return true
  }

  private isEndpointVerificationReady(endpoint: WebhookEndpointConfig): boolean {
    return typeof endpoint.secretHeader === 'string'
      && endpoint.secretHeader.trim().length > 0
      && typeof endpoint.secretValue === 'string'
      && endpoint.secretValue.trim().length > 0
  }
}
