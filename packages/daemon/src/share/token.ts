import { createHmac, timingSafeEqual } from 'node:crypto'

/**
 * Upper bound on share-token TTL. Public /share URLs cannot be
 * revoked — once a token is minted, anyone holding the URL has read
 * access to the session content for the full TTL. Capping at 24h
 * limits the blast radius of an accidentally-pasted URL (browser
 * history, referrer, CDN logs).
 */
export const SHARE_TOKEN_MAX_TTL_SECONDS = 24 * 60 * 60

export interface ShareTokenInput {
  sessionId: string
  ttlSeconds: number
}

export interface VerifyShareTokenResult {
  valid: boolean
  sessionId?: string
  reason?: 'malformed' | 'signature' | 'expired' | 'payload'
}

interface ShareTokenPayload {
  sid: string
  exp: number
}

function base64urlEncode(buf: Buffer): string {
  return buf.toString('base64').replace(/=+$/g, '').replace(/\+/g, '-').replace(/\//g, '_')
}

function base64urlDecode(s: string): Buffer {
  const pad = (4 - (s.length % 4)) % 4
  const normalized = s.replace(/-/g, '+').replace(/_/g, '/') + '='.repeat(pad)
  return Buffer.from(normalized, 'base64')
}

function sign(body: string, secret: string): string {
  return base64urlEncode(createHmac('sha256', secret).update(body).digest())
}

function constantTimeEquals(a: string, b: string): boolean {
  if (a.length !== b.length) return false
  const ab = Buffer.from(a)
  const bb = Buffer.from(b)
  if (ab.length !== bb.length) return false
  return timingSafeEqual(ab, bb)
}

export function createShareToken(input: ShareTokenInput, secret: string): string {
  if (!secret) throw new Error('share token secret is required')
  if (input.ttlSeconds <= 0) throw new Error('share token ttl must be positive')
  // Cap server-side regardless of operator config — verifyShareToken
  // mirrors this so a previously-minted longer token is also rejected
  // after the cap is in place.
  const cappedTtl = Math.min(input.ttlSeconds, SHARE_TOKEN_MAX_TTL_SECONDS)
  const payload: ShareTokenPayload = {
    sid: input.sessionId,
    exp: Math.floor(Date.now() / 1000) + cappedTtl,
  }
  const body = base64urlEncode(Buffer.from(JSON.stringify(payload), 'utf8'))
  const sig = sign(body, secret)
  return `${body}.${sig}`
}

export function verifyShareToken(token: string, secret: string): VerifyShareTokenResult {
  if (!secret) return { valid: false, reason: 'signature' }
  const parts = token.split('.')
  if (parts.length !== 2) return { valid: false, reason: 'malformed' }
  const [body, sig] = parts
  if (!body || !sig) return { valid: false, reason: 'malformed' }
  const expected = sign(body, secret)
  if (!constantTimeEquals(expected, sig)) return { valid: false, reason: 'signature' }
  try {
    const parsed = JSON.parse(base64urlDecode(body).toString('utf8')) as ShareTokenPayload
    if (typeof parsed.sid !== 'string' || typeof parsed.exp !== 'number') {
      return { valid: false, reason: 'payload' }
    }
    const nowSeconds = Math.floor(Date.now() / 1000)
    if (parsed.exp < nowSeconds) {
      return { valid: false, reason: 'expired' }
    }
    // Refuse tokens whose remaining lifetime exceeds the cap, even if
    // they were minted before the cap was added or by a misconfigured
    // operator. A 30-day token suddenly becomes a 24-hour token on
    // upgrade, matching the security contract documented above.
    if (parsed.exp - nowSeconds > SHARE_TOKEN_MAX_TTL_SECONDS) {
      return { valid: false, reason: 'expired' }
    }
    return { valid: true, sessionId: parsed.sid }
  } catch {
    return { valid: false, reason: 'payload' }
  }
}
