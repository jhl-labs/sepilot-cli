import type { IChannel, ChannelType, ChannelTarget, ChannelMessage, ChannelStatus, IncomingMessage, Disposable } from '@sepilotd/core'
import { createPublicKey, verify as verifySignature } from 'node:crypto'
import { createLogger } from '../logger.js'

const log = createLogger('teams')
const BOTFRAMEWORK_OPENID_CONFIG_URL = 'https://login.botframework.com/v1/.well-known/openidconfiguration'

type TeamsJsonWebKey = {
  kid?: string
  kty?: string
  alg?: string
  n?: string
  e?: string
  x5c?: string[]
  [property: string]: unknown
}

export interface TeamsChannelConfig {
  /** Microsoft App ID */
  appId: string
  /** Microsoft App Password */
  appPassword: string
  /** Allowed tenant IDs (empty = allow all) */
  allowedTenants?: string[]
  rateLimitPerMinute?: number
  /**
   * Outbound service host allowlist for the Bot Framework connector. Suffix
   * entries (`.botframework.com`) match any subdomain; bare entries match the
   * exact host. Defaults to the Bot Framework connector + regional traffic
   * manager host. Used to stop bot bearer tokens being relayed to an
   * attacker-chosen serviceUrl (SSRF).
   */
  allowedServiceHosts?: string[]
}

/** Default Bot Framework connector hosts the bot token may be sent to. */
const DEFAULT_TEAMS_SERVICE_HOSTS = ['.botframework.com', 'smba.trafficmanager.net']

/**
 * Whether a Bot Framework serviceUrl points at an allowed connector host.
 * Suffix entries starting with '.' match any subdomain; other entries match
 * the exact hostname. Anything unparseable is rejected (fail-closed).
 */
export function isAllowedTeamsServiceHost(
  serviceUrl: string,
  allowed: string[] = DEFAULT_TEAMS_SERVICE_HOSTS,
): boolean {
  let host: string
  try {
    const parsed = new URL(serviceUrl)
    if (parsed.protocol !== 'https:') return false
    host = parsed.hostname.toLowerCase()
  } catch {
    return false
  }
  return allowed.some((entry) => {
    const candidate = entry.toLowerCase()
    return candidate.startsWith('.')
      ? host === candidate.slice(1) || host.endsWith(candidate)
      : host === candidate
  })
}

/** Normalize a serviceUrl for equality comparison (lowercase, no trailing slash). */
function normalizeServiceUrl(value: string): string {
  return value.trim().toLowerCase().replace(/\/+$/, '')
}

/**
 * Microsoft Teams channel via Bot Framework REST API.
 *
 * Uses the Bot Framework v4 REST API directly (no SDK dependency).
 * Receives messages via webhook (POST /webhooks/teams).
 * Sends responses via Bot Framework connector service.
 */
export class TeamsChannel implements IChannel {
  readonly id = 'teams'
  readonly type: ChannelType = 'teams'
  private config: TeamsChannelConfig
  private status: ChannelStatus = 'disconnected'
  private handlers: Array<(msg: IncomingMessage) => Promise<void>> = []
  private allowedTenants: Set<string>
  private allowedServiceHosts: string[]
  private messageCount = new Map<string, { count: number; resetAt: number }>()
  private accessToken: string | null = null
  private tokenExpiry = 0
  private openIdConfig?: { issuer: string; jwksUri: string; expiresAt: number }
  private signingKeys?: { keys: TeamsJsonWebKey[]; expiresAt: number }

  constructor(config: TeamsChannelConfig) {
    this.config = config
    this.allowedTenants = new Set(config.allowedTenants ?? [])
    this.allowedServiceHosts = config.allowedServiceHosts && config.allowedServiceHosts.length > 0
      ? config.allowedServiceHosts
      : DEFAULT_TEAMS_SERVICE_HOSTS
  }

  async start(): Promise<void> {
    this.status = 'connected'
    log.info('Teams channel started (webhook mode)')
  }

  async stop(): Promise<void> {
    this.status = 'disconnected'
    this.accessToken = null
  }

  getStatus(): ChannelStatus {
    return this.status
  }

  onMessage(handler: (msg: IncomingMessage) => Promise<void>): Disposable {
    this.handlers.push(handler)
    return {
      dispose: () => {
        const idx = this.handlers.indexOf(handler)
        if (idx >= 0) this.handlers.splice(idx, 1)
      },
    }
  }

  async sendMessage(target: ChannelTarget, msg: ChannelMessage): Promise<void> {
    const [serviceUrl, conversationId] = target.id.split('|')

    // Never relay the bot bearer token to an unvetted host (SSRF / token relay).
    if (!serviceUrl || !isAllowedTeamsServiceHost(serviceUrl, this.allowedServiceHosts)) {
      log.error('Teams send blocked: serviceUrl host not in allowlist', { serviceUrl })
      throw new Error('Teams serviceUrl host not allowed')
    }

    const token = await this.getToken()
    const url = `${serviceUrl}/v3/conversations/${conversationId}/activities`
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        type: 'message',
        text: msg.text,
        textFormat: msg.format === 'markdown' ? 'markdown' : 'plain',
      }),
    })

    if (!response.ok) {
      const error = await response.text()
      log.error('Teams send failed', { error })
      throw new Error(`Teams API error: ${response.status}`)
    }
  }

  canValidateRequest(): boolean {
    return typeof this.config.appId === 'string'
      && this.config.appId.trim().length > 0
      && typeof this.config.appPassword === 'string'
      && this.config.appPassword.trim().length > 0
  }

  /**
   * Whether a JWT `serviceurl` claim matches the inbound activity's serviceUrl.
   * When the token carries a serviceurl claim it MUST match the activity
   * (fail-closed) so a caller cannot bind a valid token to an attacker URL.
   * When no expected serviceUrl is supplied, we cannot bind and fall through
   * to the outbound host allowlist enforced in sendMessage.
   */
  private serviceUrlClaimMatches(claim: unknown, expected: string | undefined): boolean {
    if (typeof claim !== 'string' || claim.length === 0) {
      // No claim to bind against — nothing to reject here.
      return true
    }
    if (typeof expected !== 'string' || expected.length === 0) {
      // Token asserts a serviceUrl but we have nothing to compare it to.
      return false
    }
    return normalizeServiceUrl(claim) === normalizeServiceUrl(expected)
  }

  async validateRequest(
    headers: Record<string, string>,
    expectedServiceUrl?: string,
  ): Promise<boolean> {
    try {
      if (!this.canValidateRequest()) {
        return false
      }

      const authHeader = headers.authorization
      if (!authHeader?.startsWith('Bearer ')) {
        return false
      }

      const token = authHeader.slice(7)
      const parsed = this.parseJwt(token)
      if (!parsed || parsed.header.alg !== 'RS256' || typeof parsed.header.kid !== 'string') {
        return false
      }

      // Bind the token's serviceurl claim to the inbound activity's serviceUrl.
      if (!this.serviceUrlClaimMatches(parsed.payload.serviceurl, expectedServiceUrl)) {
        log.warn('Teams serviceUrl claim mismatch', { expectedServiceUrl })
        return false
      }

      const nowSeconds = Math.floor(Date.now() / 1000)
      if (typeof parsed.payload.exp === 'number' && parsed.payload.exp <= nowSeconds) {
        return false
      }
      if (typeof parsed.payload.nbf === 'number' && parsed.payload.nbf > nowSeconds) {
        return false
      }
      if (parsed.payload.aud !== this.config.appId) {
        return false
      }

      const openIdConfig = await this.getOpenIdConfig()
      const allowedIssuers = new Set([
        'https://api.botframework.com',
        openIdConfig.issuer,
      ].filter((issuer): issuer is string => typeof issuer === 'string' && issuer.length > 0))
      if (typeof parsed.payload.iss !== 'string' || !allowedIssuers.has(parsed.payload.iss)) {
        return false
      }

      const signingKeys = await this.getSigningKeys(openIdConfig.jwksUri)
      const jwk = signingKeys.find(
        (key) => key.kid === parsed.header.kid,
      )
      if (!jwk) {
        return false
      }

      const publicKey = createPublicKey({ key: jwk, format: 'jwk' } as Parameters<typeof createPublicKey>[0])
      return verifySignature(
        'RSA-SHA256',
        Buffer.from(parsed.signingInput, 'utf8'),
        publicKey,
        parsed.signature,
      )
    } catch (err) {
      log.warn('Teams request validation failed', { error: String(err) })
      return false
    }
  }

  /** Handle incoming Bot Framework activity */
  async handleActivity(activity: TeamsActivity): Promise<void> {
    if (activity.type !== 'message' || !activity.text) return

    // Tenant check
    if (this.allowedTenants.size > 0) {
      const tenantId = activity.channelData?.tenant?.id
      if (tenantId && !this.allowedTenants.has(tenantId)) {
        log.warn('Teams message from non-allowed tenant', { tenantId })
        return
      }
    }

    const userId = activity.from?.id ?? 'unknown'

    // Rate limit
    if (!this.checkRateLimit(userId)) return

    const incoming: IncomingMessage = {
      channelType: 'teams',
      channelId: activity.conversation?.id ?? '',
      messageId: activity.id ?? '',
      sender: {
        id: userId,
        name: activity.from?.name ?? 'Unknown',
        type: 'user',
      },
      text: activity.text,
      timestamp: activity.timestamp ?? new Date().toISOString(),
      raw: {
        serviceUrl: activity.serviceUrl,
        conversationId: activity.conversation?.id,
        isGroup: activity.conversation?.isGroup,
        conversationType: activity.conversation?.conversationType,
      },
    }

    for (const handler of this.handlers) {
      try {
        await handler(incoming)
      } catch (err) {
        log.error('Teams handler error', { error: String(err) })
      }
    }
  }

  /** Get Bot Framework access token (cached) */
  private async getToken(): Promise<string> {
    if (this.accessToken && Date.now() < this.tokenExpiry) {
      return this.accessToken
    }

    const response = await fetch('https://login.microsoftonline.com/botframework.com/oauth2/v2.0/token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        grant_type: 'client_credentials',
        client_id: this.config.appId,
        client_secret: this.config.appPassword,
        scope: 'https://api.botframework.com/.default',
      }),
    })

    if (!response.ok) {
      throw new Error(`Teams token error: ${response.status}`)
    }

    const data = await response.json() as { access_token: string; expires_in: number }
    this.accessToken = data.access_token
    this.tokenExpiry = Date.now() + (data.expires_in - 60) * 1000
    return this.accessToken
  }

  private checkRateLimit(userId: string): boolean {
    const limit = this.config.rateLimitPerMinute ?? 30
    const now = Date.now()
    const entry = this.messageCount.get(userId)

    if (!entry || now > entry.resetAt) {
      this.messageCount.set(userId, { count: 1, resetAt: now + 60000 })
      return true
    }

    if (entry.count >= limit) return false
    entry.count++
    return true
  }

  private parseJwt(token: string): ParsedJwt | null {
    const parts = token.split('.')
    if (parts.length !== 3) {
      return null
    }

    try {
      const header = JSON.parse(Buffer.from(parts[0]!, 'base64url').toString('utf8')) as JwtHeader
      const payload = JSON.parse(Buffer.from(parts[1]!, 'base64url').toString('utf8')) as JwtPayload
      return {
        header,
        payload,
        signingInput: `${parts[0]}.${parts[1]}`,
        signature: Buffer.from(parts[2]!, 'base64url'),
      }
    } catch {
      return null
    }
  }

  private async getOpenIdConfig(): Promise<{ issuer: string; jwksUri: string }> {
    if (this.openIdConfig && this.openIdConfig.expiresAt > Date.now()) {
      return {
        issuer: this.openIdConfig.issuer,
        jwksUri: this.openIdConfig.jwksUri,
      }
    }

    const response = await fetch(BOTFRAMEWORK_OPENID_CONFIG_URL)
    if (!response.ok) {
      throw new Error(`Teams OpenID config error: ${response.status}`)
    }

    const data = await response.json() as {
      issuer?: string
      jwks_uri?: string
    }
    const issuer = data.issuer ?? 'https://api.botframework.com'
    const jwksUri = data.jwks_uri
    if (!jwksUri) {
      throw new Error('Teams OpenID config missing jwks_uri')
    }

    this.openIdConfig = {
      issuer,
      jwksUri,
      expiresAt: Date.now() + 60 * 60 * 1000,
    }
    return { issuer, jwksUri }
  }

  private async getSigningKeys(jwksUri: string): Promise<TeamsJsonWebKey[]> {
    if (this.signingKeys && this.signingKeys.expiresAt > Date.now()) {
      return this.signingKeys.keys
    }

    const response = await fetch(jwksUri)
    if (!response.ok) {
      throw new Error(`Teams signing key error: ${response.status}`)
    }

    const data = await response.json() as { keys?: TeamsJsonWebKey[] }
    const keys = Array.isArray(data.keys) ? data.keys : []
    this.signingKeys = {
      keys,
      expiresAt: Date.now() + 60 * 60 * 1000,
    }
    return keys
  }
}

interface TeamsActivity {
  type: string
  id?: string
  text?: string
  timestamp?: string
  serviceUrl?: string
  from?: { id: string; name?: string }
  conversation?: { id: string; isGroup?: boolean; conversationType?: string }
  channelData?: { tenant?: { id: string } }
}

interface JwtHeader {
  alg?: string
  kid?: string
}

interface JwtPayload {
  aud?: string
  iss?: string
  exp?: number
  nbf?: number
  serviceurl?: string
}

interface ParsedJwt {
  header: JwtHeader
  payload: JwtPayload
  signingInput: string
  signature: Buffer
}
