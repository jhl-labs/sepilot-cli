import { createHash, randomBytes, randomUUID } from 'node:crypto'
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { createLogger } from '../../logger.js'
import { isNodeFsError } from '../../utils/fs-error.js'
import type { JsonlAuditLogger } from '../../security/audit-logger.js'

const log = createLogger('extension-tokens')

export const EXTENSION_ACCESS_TOKEN_SCOPES = [
  'all',
  'inspect',
  'chat',
  'ws',
  'sessions',
  'memory',
  'files',
  'skills',
  'projects',
  'approvals',
  'extensions',
  'personas',
  'artifacts',
  'browser',
] as const

export type ExtensionAccessTokenScope =
  (typeof EXTENSION_ACCESS_TOKEN_SCOPES)[number]

interface PersistedExtensionAccessTokenRecord {
  id: string
  label: string
  scopes: ExtensionAccessTokenScope[]
  tokenHash: string
  createdAt: string
  expiresAt?: string
  revokedAt?: string
}

export interface ExtensionAccessTokenSummary {
  id: string
  label: string
  scopes: ExtensionAccessTokenScope[]
  createdAt: string
  expiresAt?: string
  revokedAt?: string
  active: boolean
}

export interface IssuedExtensionAccessToken
  extends ExtensionAccessTokenSummary {
  token: string
}

export interface IssueExtensionAccessTokenInput {
  label: string
  scopes: ExtensionAccessTokenScope[]
  expiresAt?: string
}

export interface ExtensionAccessTokenPrincipal {
  kind: 'extension'
  tokenId: string
  label: string
  scopes: ExtensionAccessTokenScope[]
}

export type ExtensionAccessTokenAuthorizeResult =
  | {
    ok: true
    principal: ExtensionAccessTokenPrincipal
  }
  | {
    ok: false
    reason: 'invalid' | 'forbidden'
  }

const TOKEN_FILE_NAME = 'extension-tokens.json'
const NON_ALL_SCOPES = EXTENSION_ACCESS_TOKEN_SCOPES.filter(
  (scope) => scope !== 'all',
)
const READ_ONLY_METHODS = new Set(['GET', 'HEAD'])

function hashToken(token: string): string {
  return createHash('sha256').update(token).digest('hex')
}

function normalizePath(url: string): string {
  const [path] = url.split('?', 1)
  return path || url
}

function matchesPrefix(path: string, prefix: string): boolean {
  return path === prefix || path.startsWith(`${prefix}/`)
}

function isInspectRoute(method: string, path: string): boolean {
  if (!READ_ONLY_METHODS.has(method)) {
    return false
  }

  return path === '/api/v1/openapi.json'
    || path === '/api/v1/config'
    || path === '/api/v1/config/providers'
    || path === '/api/v1/providers'
    || path === '/api/v1/devices'
    || path === '/api/v1/channels'
    || path === '/api/v1/plugins'
    || path === '/api/v1/mcp/servers'
    || path === '/api/v1/usage'
    || path === '/api/v1/agents'
}

function scopeAllowsRoute(
  scope: ExtensionAccessTokenScope,
  method: string,
  url: string,
): boolean {
  const normalizedMethod = method.toUpperCase()
  const path = normalizePath(url)

  if (scope === 'all') {
    return NON_ALL_SCOPES.some((candidate) =>
      scopeAllowsRoute(candidate, normalizedMethod, path))
  }

  if (scope === 'browser') {
    return normalizedMethod === 'POST' && [
      '/api/v1/browser/extension/connect',
      '/api/v1/browser/extension/exchange',
      '/api/v1/browser/extension/disconnect',
    ].includes(path)
  }

  if (scope === 'inspect') {
    return isInspectRoute(normalizedMethod, path)
  }

  if (scope === 'chat') {
    return (
      normalizedMethod === 'POST'
      && (
        path === '/api/v1/chat'
        || path === '/api/v1/chat/stream'
        || path === '/api/v1/chat/background'
        || path === '/api/v1/cowork/run'
      )
    ) || (
      normalizedMethod === 'GET'
      && matchesPrefix(path, '/api/v1/chat/background')
    ) || (
      normalizedMethod === 'DELETE'
      && matchesPrefix(path, '/api/v1/chat/background')
    )
  }

  if (scope === 'ws') {
    return normalizedMethod === 'GET' && path === '/api/v1/ws'
  }

  if (scope === 'sessions') {
    return matchesPrefix(path, '/api/v1/sessions')
  }

  if (scope === 'memory') {
    return matchesPrefix(path, '/api/v1/memory')
  }

  if (scope === 'files') {
    return matchesPrefix(path, '/api/v1/files')
  }

  if (scope === 'skills') {
    return matchesPrefix(path, '/api/v1/skills')
      || matchesPrefix(path, '/api/v1/skill-store')
  }

  if (scope === 'projects') {
    return matchesPrefix(path, '/api/v1/projects')
  }

  if (scope === 'approvals') {
    return matchesPrefix(path, '/api/v1/approvals')
  }

  if (scope === 'extensions') {
    // MCP server config and outbound-webhook config WRITES are
    // RCE-equivalent: registering an MCP stdio server runs an arbitrary local
    // command, and outbound-webhook config can exfiltrate to any URL. These
    // are master-token only. Non-master extension tokens keep read access so
    // clients can inspect current config, but never mutate it.
    if (
      matchesPrefix(path, '/api/v1/config/mcp/servers')
      || matchesPrefix(path, '/api/v1/config/hooks/outbound-webhooks')
    ) {
      return READ_ONLY_METHODS.has(normalizedMethod)
    }
    return matchesPrefix(path, '/api/v1/config/channels/webhook/endpoints')
      || matchesPrefix(path, '/api/v1/config/channels/telegram')
      || matchesPrefix(path, '/api/v1/config/channels/discord')
      || matchesPrefix(path, '/api/v1/config/channels/mattermost')
      || path === '/api/v1/mcp/servers'
      || path === '/api/v1/channels'
      || path === '/api/v1/plugins'
  }

  if (scope === 'personas') {
    return matchesPrefix(path, '/api/v1/personas')
  }

  if (scope === 'artifacts') {
    return matchesPrefix(path, '/api/v1/artifacts')
  }

  return false
}

function normalizeScopes(
  scopes: ExtensionAccessTokenScope[],
): ExtensionAccessTokenScope[] {
  const seen = new Set<ExtensionAccessTokenScope>()
  const requested = scopes.filter((scope) => {
    if (seen.has(scope)) {
      return false
    }
    seen.add(scope)
    return true
  })

  return EXTENSION_ACCESS_TOKEN_SCOPES.filter((scope) =>
    requested.includes(scope))
}

function summarizeRecord(
  record: PersistedExtensionAccessTokenRecord,
): ExtensionAccessTokenSummary {
  const now = Date.now()
  const expired = record.expiresAt
    ? Date.parse(record.expiresAt) <= now
    : false

  return {
    id: record.id,
    label: record.label,
    scopes: [...record.scopes],
    createdAt: record.createdAt,
    expiresAt: record.expiresAt,
    revokedAt: record.revokedAt,
    active: !record.revokedAt && !expired,
  }
}

export class ExtensionAccessTokenStore {
  private readonly filePath: string
  private readonly auditLogger?: JsonlAuditLogger
  private readonly deviceName?: string
  private records: PersistedExtensionAccessTokenRecord[] = []

  constructor(
    dataDir: string,
    auditLogger?: JsonlAuditLogger,
    deviceName?: string,
  ) {
    this.filePath = join(dataDir, 'security', TOKEN_FILE_NAME)
    this.auditLogger = auditLogger
    this.deviceName = deviceName
  }

  async init(): Promise<void> {
    let raw: string
    try {
      raw = await readFile(this.filePath, 'utf-8')
    } catch (err) {
      if (isNodeFsError(err, 'ENOENT')) {
        return
      }
      log.error('failed to read extension token store', {
        path: this.filePath,
        error: err instanceof Error ? err.message : String(err),
      })
      throw err
    }

    let parsed: unknown
    try {
      parsed = JSON.parse(raw)
    } catch (err) {
      // Corrupt token file silently turning into "no extensions
      // registered" would 401 every existing extension with no
      // operator-visible reason. Rotate the bad file aside so
      // tokens can be reissued and the original is preserved for
      // forensic review.
      const aside = `${this.filePath}.broken-${Date.now()}`
      log.error('extension token store unparseable; rotating aside', {
        path: this.filePath,
        rotated: aside,
        error: err instanceof Error ? err.message : String(err),
      })
      await rename(this.filePath, aside)
      return
    }

    if (Array.isArray(parsed)) {
      this.records = parsed.filter((value): value is PersistedExtensionAccessTokenRecord => {
        return typeof value === 'object'
          && value !== null
          && typeof value.id === 'string'
          && typeof value.label === 'string'
          && Array.isArray(value.scopes)
          && typeof value.tokenHash === 'string'
          && typeof value.createdAt === 'string'
      })
    } else {
      log.warn('extension token store is not an array; treating as empty', {
        path: this.filePath,
      })
    }
  }

  list(): ExtensionAccessTokenSummary[] {
    return [...this.records]
      .sort((left, right) => right.createdAt.localeCompare(left.createdAt))
      .map((record) => summarizeRecord(record))
  }

  async issue(
    input: IssueExtensionAccessTokenInput,
  ): Promise<IssuedExtensionAccessToken> {
    const label = input.label.trim()
    if (!label) {
      throw new Error('Token label is required')
    }

    const scopes = normalizeScopes(input.scopes)
    if (scopes.length === 0) {
      throw new Error('At least one scope is required')
    }

    if (input.expiresAt && Date.parse(input.expiresAt) <= Date.now()) {
      throw new Error('expiresAt must be in the future')
    }

    const token = `sep_ext_${randomBytes(24).toString('base64url')}`
    const record: PersistedExtensionAccessTokenRecord = {
      id: randomUUID(),
      label,
      scopes,
      tokenHash: hashToken(token),
      createdAt: new Date().toISOString(),
      ...(input.expiresAt ? { expiresAt: input.expiresAt } : {}),
    }

    this.records.push(record)
    await this.persist()
    await this.auditLogger?.log({
      timestamp: new Date().toISOString(),
      event: 'auth.extension_token.issued',
      device: this.deviceName ?? 'unknown-device',
      tokenId: record.id,
      label: record.label,
      scopes: record.scopes,
      expiresAt: record.expiresAt,
    })

    return {
      ...summarizeRecord(record),
      token,
    }
  }

  async revoke(id: string): Promise<ExtensionAccessTokenSummary | null> {
    const record = this.records.find((candidate) => candidate.id === id)
    if (!record || record.revokedAt) {
      return null
    }

    record.revokedAt = new Date().toISOString()
    await this.persist()
    await this.auditLogger?.log({
      timestamp: record.revokedAt,
      event: 'auth.extension_token.revoked',
      device: this.deviceName ?? 'unknown-device',
      tokenId: record.id,
      label: record.label,
      scopes: record.scopes,
    })

    return summarizeRecord(record)
  }

  authorize(
    rawToken: string,
    request: { method: string; url: string },
  ): ExtensionAccessTokenAuthorizeResult {
    const tokenHash = hashToken(rawToken)
    const record = this.records.find((candidate) => candidate.tokenHash === tokenHash)
    if (!record || record.revokedAt) {
      return { ok: false, reason: 'invalid' }
    }

    if (record.expiresAt && Date.parse(record.expiresAt) <= Date.now()) {
      return { ok: false, reason: 'invalid' }
    }

    const allowed = record.scopes.some((scope) =>
      scopeAllowsRoute(scope, request.method, request.url))
    if (!allowed) {
      return { ok: false, reason: 'forbidden' }
    }

    return {
      ok: true,
      principal: {
        kind: 'extension',
        tokenId: record.id,
        label: record.label,
        scopes: [...record.scopes],
      },
    }
  }

  private async persist(): Promise<void> {
    await mkdir(dirname(this.filePath), { recursive: true })
    const tempPath = `${this.filePath}.tmp`
    await writeFile(
      tempPath,
      JSON.stringify(this.records, null, 2),
      { mode: 0o600 },
    )
    await rename(tempPath, this.filePath)
  }
}

export async function buildExtensionAccessTokenStore(
  dataDir: string,
  auditLogger?: JsonlAuditLogger,
  deviceName?: string,
): Promise<ExtensionAccessTokenStore> {
  const store = new ExtensionAccessTokenStore(
    dataDir,
    auditLogger,
    deviceName,
  )
  await store.init()
  return store
}

export function extensionScopeAllowsRoute(
  scope: ExtensionAccessTokenScope,
  method: string,
  url: string,
): boolean {
  return scopeAllowsRoute(scope, method, url)
}
