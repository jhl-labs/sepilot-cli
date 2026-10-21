import { MobilePairingTickets } from './mobile-pairing.js'
import type { FastifyInstance, FastifyRequest, FastifyReply } from 'fastify'
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises'
import { randomBytes } from 'node:crypto'
import { timingSafeEqual } from 'node:crypto'
import { dirname, isAbsolute, join } from 'node:path'
import { homedir } from 'node:os'
import type { ExtensionAccessTokenScope } from './runtime/extension-tokens.js'
import type { AuthCapabilities } from './runtime/capabilities.js'
import { skipOverride } from './skip-override.js'
import { isNodeFsError } from '../utils/fs-error.js'

export type RequestAuthContext =
  | {
      kind: 'master'
    }
  | {
      kind: 'extension'
      tokenId: string
      label: string
      scopes: ExtensionAccessTokenScope[]
    }

type AuthPluginApp = FastifyInstance & {
  runtime?: AuthCapabilities
  authToken?: string | null
}

type AuthPluginRequest = FastifyRequest & {
  mobilePairingToken?: string
  authContext?: RequestAuthContext
}

const LOOPBACK_HOSTS = new Set(['127.0.0.1', 'localhost', '::1', '[::1]'])

export type DaemonAuthTokenResolver = () => Promise<string | null>

export interface AuthPluginOptions {
  resolveMasterToken?: DaemonAuthTokenResolver
  masterTokenRequired?: boolean
}

interface FileTokenResolverOptions {
  initialToken: string
  refreshIntervalMs?: number
  now?: () => number
}

export function timingSafeTokenEqual(
  a: string | null | undefined,
  b: string | null | undefined,
): boolean {
  if (typeof a !== 'string' || typeof b !== 'string') return false
  const bufA = Buffer.from(a, 'utf8')
  const bufB = Buffer.from(b, 'utf8')
  if (bufA.length !== bufB.length) return false
  return timingSafeEqual(bufA, bufB)
}

export function isDaemonLoopbackHost(host: string): boolean {
  return LOOPBACK_HOSTS.has(host.trim().toLowerCase())
}

export function assertDaemonAuthPolicy(options: {
  host: string
  tokenConfigured: boolean
  allowUnauthenticatedExternal: boolean
}): void {
  if (options.tokenConfigured || isDaemonLoopbackHost(options.host)) {
    return
  }

  if (options.allowUnauthenticatedExternal) {
    const ack = process.env.SEPILOTD_INSECURE_BIND_ACK
    if (ack !== 'YES_I_UNDERSTAND') {
      throw new Error(
        'SEPILOTD_ALLOW_UNAUTHENTICATED_EXTERNAL=1 also requires ' +
          'SEPILOTD_INSECURE_BIND_ACK=YES_I_UNDERSTAND. ' +
          'This is intentional: an unauthenticated external bind exposes ' +
          'every protected daemon route to anyone on the LAN.',
      )
    }
    return
  }

  throw new Error(
    'Daemon bearer token is required when binding to a non-loopback host. ' +
      'Restore <SEPILOTD_DATA_DIR>/security/daemon.token ' +
      'or set SEPILOTD_ALLOW_UNAUTHENTICATED_EXTERNAL=1 along with ' +
      'SEPILOTD_INSECURE_BIND_ACK=YES_I_UNDERSTAND only for a trusted test network.',
  )
}

function decodedPathname(url: string): string | null {
  try {
    return decodeURIComponent(url.split('?', 1)[0]!)
  } catch {
    return null
  }
}

function isPublicWebhookReceiverRequest(url: string, method: string): boolean {
  const pathname = decodedPathname(url)
  if (pathname === null) return false

  if (method === 'GET') {
    return pathname === '/api/v1/webhooks/whatsapp'
  }

  if (method !== 'POST') {
    return false
  }

  return (
    pathname.startsWith('/api/v1/webhooks/') &&
    pathname !== '/api/v1/webhooks/security' &&
    !pathname.startsWith('/api/v1/webhooks/security/')
  )
}

function isPublicHealthRequest(url: string): boolean {
  const pathname = url.split('?', 1)[0]
  return (
    pathname === '/api/v1/health' ||
    pathname === '/api/v1/health/live' ||
    pathname === '/api/v1/health/ready'
  )
}

function isReadinessHealthRequest(url: string): boolean {
  return url.split('?', 1)[0] === '/api/v1/health/ready'
}

function isPublicShareRequest(url: string): boolean {
  const pathname = url.split('?', 1)[0]
  return pathname.startsWith('/share/')
}

function isPublicA2AAgentCardRequest(url: string, method: string): boolean {
  if (method !== 'GET') return false
  const pathname = url.split('?', 1)[0]
  return pathname === '/.well-known/agent-card.json' || pathname === '/.well-known/agent.json'
}

function isPublicGitHubOAuthCallbackRequest(url: string, method: string): boolean {
  if (method !== 'GET') return false
  const pathname = url.split('?', 1)[0]
  return pathname === '/github/oauth/callback'
}

function isWebSocketRequest(url: string): boolean {
  const pathname = url.split('?', 1)[0]
  return pathname === '/api/v1/ws' || pathname === '/api/v1/voice/realtime/ws'
}

function isExtensionSessionPlaneDenied(url: string): boolean {
  const pathname = url.split('?', 1)[0]
  return pathname === '/api/v1/sessions'
    || pathname.startsWith('/api/v1/sessions/')
    || pathname === '/api/v1/chat/background'
    || pathname.startsWith('/api/v1/chat/background/')
    || pathname === '/api/v1/approvals'
    || pathname.startsWith('/api/v1/approvals/')
    || pathname === '/api/v1/projects'
    || pathname.startsWith('/api/v1/projects/')
}

const MEMORY_SCOPE_HEADER_NAMES = [
  'x-memory-scope-user-id',
  'x-memory-scope-channel-type',
  'x-memory-scope-channel-id',
  'x-memory-scope-session-id',
  'x-memory-scope-groups',
] as const

const MEMORY_SCOPE_QUERY_NAMES = [
  'scopeUserId',
  'scopeChannelType',
  'scopeChannelId',
  'scopeSessionId',
  'scopeGroups',
] as const

function hasCallerControlledMemoryScope(request: FastifyRequest): boolean {
  if (MEMORY_SCOPE_HEADER_NAMES.some((name) => request.headers[name] !== undefined)) {
    return true
  }
  const queryStart = request.url.indexOf('?')
  if (queryStart < 0) return false
  const query = new URLSearchParams(request.url.slice(queryStart + 1))
  return MEMORY_SCOPE_QUERY_NAMES.some((name) => query.has(name))
}

function requiresSwarmControlToken(url: string, method: string): boolean {
  if (process.env.SEPILOTD_SWARM_REQUIRE_TOKEN !== '1') return false
  const pathname = url.split('?', 1)[0]
  if (method === 'GET') {
    return /^\/api\/v1\/swarm\/runs\/[^/]+\/agents\/[^/]+\/capture$/.test(pathname)
  }
  if (method === 'POST') {
    return /^\/api\/v1\/swarm\/runs\/[^/]+\/agents\/[^/]+\/(?:keys|interrupt)$/.test(pathname)
  }
  return false
}

function acceptsBrowserQueryToken(request: FastifyRequest): boolean {
  if (request.method !== 'GET') return false
  const pathname = request.url.split('?', 1)[0]
  return (
    (request.headers.accept ?? '').includes('text/event-stream') ||
    pathname === '/image-gen/events' ||
    pathname === '/extensions/events' ||
    pathname.startsWith('/image-gen/files/')
  )
}

export function daemonAuthTokenPath(dataDir?: string): string {
  const dir = dataDir ?? join(homedir(), '.sepilotd')
  const override = process.env.SEPILOTD_DAEMON_TOKEN_FILE?.trim()
  if (!override) return join(dir, 'security', 'daemon.token')
  if (!isAbsolute(override)) {
    throw new Error('SEPILOTD_DAEMON_TOKEN_FILE must be an absolute path')
  }
  return override
}

export async function ensureDaemonAuthToken(dataDir?: string): Promise<string | null> {
  const existing = await loadToken(dataDir)
  if (existing) return existing
  if (process.env.SEPILOTD_DAEMON_TOKEN_FILE?.trim()) return null

  const tokenPath = daemonAuthTokenPath(dataDir)
  const token = randomBytes(32).toString('hex')
  await mkdir(dirname(tokenPath), { recursive: true, mode: 0o700 })
  const tempPath = `${tokenPath}.${randomBytes(6).toString('hex')}.tmp`
  try {
    await writeFile(tempPath, `${token}\n`, { mode: 0o600, flag: 'wx' })
    await rename(tempPath, tokenPath)
  } catch (err) {
    throw new Error(
      `Failed to create daemon auth token at ${tokenPath}: ${
        err instanceof Error ? err.message : String(err)
      }`,
      { cause: err },
    )
  }
  return (await loadToken(dataDir)) ?? token
}

export async function loadToken(dataDir?: string): Promise<string | null> {
  const tokenPath = daemonAuthTokenPath(dataDir)
  const explicitlyConfigured = Boolean(process.env.SEPILOTD_DAEMON_TOKEN_FILE?.trim())
  try {
    const loaded = (await readFile(tokenPath, 'utf-8')).trim()
    if (!loaded) throw new Error(`Daemon auth token file is empty: ${tokenPath}`)
    return loaded
  } catch (err) {
    if (isNodeFsError(err, 'ENOENT') && !explicitlyConfigured) {
      return null
    }
    throw new Error(
      `Failed to read daemon auth token at ${tokenPath}: ${
        err instanceof Error ? err.message : String(err)
      }`,
      { cause: err },
    )
  }
}

export function createFileDaemonAuthTokenResolver(
  tokenPath: string,
  options: FileTokenResolverOptions,
): DaemonAuthTokenResolver {
  const refreshIntervalMs = options.refreshIntervalMs ?? 1_000
  if (!Number.isFinite(refreshIntervalMs) || refreshIntervalMs < 0) {
    throw new Error('refreshIntervalMs must be a non-negative finite number')
  }
  const now = options.now ?? Date.now
  let lastCheckedAt = now()
  let cached:
    | { kind: 'value'; token: string }
    | { kind: 'error'; error: Error } = { kind: 'value', token: options.initialToken }
  let pending: Promise<string> | undefined

  const fromCache = (): string => {
    if (cached.kind === 'error') throw cached.error
    return cached.token
  }

  return async () => {
    if (pending) return pending
    if (now() - lastCheckedAt < refreshIntervalMs) return fromCache()

    pending = (async () => {
      lastCheckedAt = now()
      try {
        const nextToken = (await readFile(tokenPath, 'utf-8')).trim()
        if (!nextToken) throw new Error(`Daemon auth token file is empty: ${tokenPath}`)
        cached = { kind: 'value', token: nextToken }
        return nextToken
      } catch (error) {
        const wrapped = error instanceof Error
          ? new Error(`Failed to refresh daemon auth token: ${error.message}`, { cause: error })
          : new Error('Failed to refresh daemon auth token')
        cached = { kind: 'error', error: wrapped }
        throw wrapped
      } finally {
        pending = undefined
      }
    })()
    return pending
  }
}

export function createAuthPlugin(token: string | null, options: AuthPluginOptions = {}) {
  async function authPlugin(app: FastifyInstance) {
    const runtimeApp = app as AuthPluginApp
    const masterTokenRequired = options.masterTokenRequired
      ?? Boolean(token || options.resolveMasterToken)
    let lastRefreshErrorLogAt = 0
    const mobilePairing = new MobilePairingTickets()

    app.addHook('onRequest', async (request: FastifyRequest, reply: FastifyReply) => {
      const authRequest = request as AuthPluginRequest
      if (request.method === 'OPTIONS') return
      const readinessRequest = isReadinessHealthRequest(request.url)
      if (isPublicHealthRequest(request.url) && !readinessRequest) return
      if (isPublicShareRequest(request.url)) return
      if (isPublicA2AAgentCardRequest(request.url, request.method)) return
      if (isPublicGitHubOAuthCallbackRequest(request.url, request.method)) return
      if (isPublicWebhookReceiverRequest(request.url, request.method)) return
      let currentToken = token
      if (options.resolveMasterToken) {
        try {
          currentToken = await options.resolveMasterToken()
          runtimeApp.authToken = currentToken
          lastRefreshErrorLogAt = 0
        } catch (error) {
          runtimeApp.authToken = null
          const now = Date.now()
          if (lastRefreshErrorLogAt === 0 || now - lastRefreshErrorLogAt >= 30_000) {
            lastRefreshErrorLogAt = now
            request.log.error({ err: error }, 'daemon auth token refresh failed')
          }
          return reply.status(503).send({
            error: {
              code: 'AUTH_TOKEN_UNAVAILABLE',
              message: 'Daemon authentication is temporarily unavailable',
            },
          })
        }
      }
      if (masterTokenRequired && !currentToken) {
        runtimeApp.authToken = null
        return reply.status(503).send({
          error: {
            code: 'AUTH_TOKEN_UNAVAILABLE',
            message: 'Daemon authentication is temporarily unavailable',
          },
        })
      }
      if (readinessRequest) return
      const extensionTokenStore = runtimeApp.runtime?.extensionTokenStore
      const hasPresentableMasterToken = Boolean(currentToken)
      const bearerPresented = (request.headers.authorization ?? '').startsWith('Bearer ')
      if (
        !hasPresentableMasterToken
        && !bearerPresented
        && !requiresSwarmControlToken(request.url, request.method)
      ) {
        return
      }

      const authHeader = request.headers.authorization
      if (isWebSocketRequest(request.url) && !authHeader?.startsWith('Bearer ')) {
        return
      }
      let bearerToken: string | null = null
      if (authHeader?.startsWith('Bearer ')) {
        bearerToken = authHeader.slice(7)
      } else if (acceptsBrowserQueryToken(request)) {
        const queryToken = (request.query as { token?: unknown } | undefined)?.token
        if (typeof queryToken === 'string' && queryToken.length > 0) {
          bearerToken = queryToken
        }
      }
      if (!bearerToken) {
        return reply
          .status(401)
          .send({ error: { code: 'UNAUTHORIZED', message: 'Bearer token required' } })
      }
      if (request.method === 'POST' && request.url.split('?')[0] === '/api/v1/mobile-pairing/redeem') {
        if (!currentToken || !mobilePairing.consume(bearerToken, currentToken)) {
          return reply.status(401).send({ error: 'Pairing code is invalid, expired, or already used.' })
        }
        authRequest.mobilePairingToken = currentToken
        return
      }
      if (currentToken && timingSafeTokenEqual(bearerToken, currentToken)) {
        authRequest.authContext = { kind: 'master' }
        return
      }

      if (!extensionTokenStore) {
        return reply.status(401).send({ error: { code: 'UNAUTHORIZED', message: 'Invalid token' } })
      }

      const authorization = extensionTokenStore.authorize(bearerToken, {
        method: request.method,
        url: request.url,
      })

      if (authorization.ok) {
        authRequest.authContext = authorization.principal
        if (isExtensionSessionPlaneDenied(request.routeOptions.url ?? request.url)) {
          return reply.status(403).send({
            error: {
              code: 'EXTENSION_SESSION_ISOLATION_UNSUPPORTED',
              message: 'Extension session history, background jobs, approvals, and projects are disabled until principal ownership is persisted.',
            },
          })
        }
        if (hasCallerControlledMemoryScope(request)) {
          return reply.status(403).send({
            error: {
              code: 'MEMORY_SCOPE_OVERRIDE_DENIED',
              message: 'Extension memory scope is bound to the authenticated token and cannot be overridden.',
            },
          })
        }
        return
      }

      if (authorization.reason === 'forbidden') {
        return reply.status(403).send({
          error: {
            code: 'FORBIDDEN',
            message: 'Token does not permit this route',
          },
        })
      }

      return reply.status(401).send({ error: { code: 'UNAUTHORIZED', message: 'Invalid token' } })
    })
    app.post('/api/v1/mobile-pairing', async (request, reply) => {
      if ((request as AuthPluginRequest).authContext?.kind !== 'master') {
        return reply.code(403).send({ error: 'Only the daemon owner can connect a mobile device.' })
      }
      const currentToken = runtimeApp.authToken ?? token
      if (!currentToken) return reply.code(503).send({ error: 'Daemon token unavailable.' })
      reply.header('Cache-Control', 'no-store')
      return mobilePairing.issue(currentToken)
    })
    app.post('/api/v1/mobile-pairing/redeem', async (request, reply) => {
      const pairedToken = (request as AuthPluginRequest).mobilePairingToken
      if (!pairedToken) return reply.code(401).send({ error: 'Pairing code required.' })
      reply.header('Cache-Control', 'no-store')
      return { token: pairedToken }
    })

  }

  skipOverride(authPlugin)
  return authPlugin
}
