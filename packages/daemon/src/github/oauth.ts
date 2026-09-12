import {
  existsSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs'
import { join } from 'node:path'
import { sepilotdHome } from '../storage/home.js'
import { secureDir, secureFile } from '../utils/secure-file.js'

interface OAuthState {
  state: string
  createdAt: number
}

interface OAuthStored {
  state: OAuthState | null
  token: string | null
  login: string | null
  scopes?: string[]
  connectedAt?: number | null
  updatedAt?: number | null
  lastError?: string | null
}

export interface OAuthStatus {
  connected: boolean
  login: string | null
  configured: boolean
  pending: boolean
  lastError: string | null
}

interface GitHubTokenResponse {
  access_token?: string
  token_type?: string
  scope?: string
  error?: string
  error_description?: string
}

interface GitHubUserResponse {
  login?: string
}

const STATE_TTL_MS = 10 * 60_000

function stateFile(): string {
  return join(sepilotdHome(), 'security', 'github-oauth.json')
}

function emptyStore(): OAuthStored {
  return {
    state: null,
    token: null,
    login: null,
    scopes: [],
    connectedAt: null,
    updatedAt: null,
    lastError: null,
  }
}

function load(): OAuthStored {
  if (!existsSync(stateFile())) {
    return emptyStore()
  }
  try {
    return { ...emptyStore(), ...JSON.parse(readFileSync(stateFile(), 'utf-8')) as OAuthStored }
  } catch {
    return emptyStore()
  }
}

function save(next: OAuthStored): void {
  const dir = join(sepilotdHome(), 'security')
  mkdirSync(dir, { recursive: true, mode: 0o700 })
  secureDir(dir)
  writeFileSync(stateFile(), JSON.stringify(next, null, 2), 'utf-8')
  secureFile(stateFile())
}

function oauthConfigured(): boolean {
  return Boolean(
    process.env.GITHUB_OAUTH_CLIENT_ID
    && process.env.GITHUB_OAUTH_CLIENT_SECRET,
  )
}

function githubHeaders(token?: string): HeadersInit {
  return {
    accept: 'application/vnd.github+json',
    ...(token ? { authorization: `Bearer ${token}` } : {}),
    'content-type': 'application/json',
    'user-agent': 'sepilotd',
    'x-github-api-version': '2022-11-28',
  }
}

async function fetchGitHubLogin(token: string): Promise<string> {
  const response = await fetch('https://api.github.com/user', {
    headers: githubHeaders(token),
  })
  if (!response.ok) {
    const text = await response.text().catch(() => '')
    throw new Error(`GitHub user lookup failed (${response.status}): ${text || response.statusText}`)
  }
  const user = await response.json() as GitHubUserResponse
  if (!user.login) throw new Error('GitHub user response did not include login')
  return user.login
}

function setError(message: string): void {
  const stored = load()
  stored.lastError = message
  stored.updatedAt = Date.now()
  save(stored)
}

export function startOAuth(input: {
  clientId: string
  redirectUri: string
  scopes: string[]
}): { url: string } {
  const state = crypto.randomUUID()
  const stored = load()
  stored.state = { state, createdAt: Date.now() }
  stored.lastError = null
  stored.updatedAt = Date.now()
  save(stored)
  const url = `https://github.com/login/oauth/authorize?client_id=${encodeURIComponent(
    input.clientId,
  )}&redirect_uri=${encodeURIComponent(input.redirectUri)}&state=${state}&scope=${encodeURIComponent(
    input.scopes.join(' '),
  )}`
  return { url }
}

export async function completeOAuth(input: {
  code: string
  state: string
  clientId: string
  clientSecret: string
  redirectUri: string
}): Promise<OAuthStatus> {
  const stored = load()
  if (!stored.state) {
    throw new Error('GitHub OAuth session was not started from this daemon.')
  }
  if (stored.state.state !== input.state) {
    setError('GitHub OAuth state mismatch.')
    throw new Error('GitHub OAuth state mismatch.')
  }
  if (Date.now() - stored.state.createdAt > STATE_TTL_MS) {
    setError('GitHub OAuth state expired.')
    throw new Error('GitHub OAuth state expired.')
  }

  const response = await fetch('https://github.com/login/oauth/access_token', {
    method: 'POST',
    headers: githubHeaders(),
    body: JSON.stringify({
      client_id: input.clientId,
      client_secret: input.clientSecret,
      code: input.code,
      redirect_uri: input.redirectUri,
      state: input.state,
    }),
  })
  if (!response.ok) {
    const text = await response.text().catch(() => '')
    setError(`GitHub OAuth token exchange failed (${response.status}).`)
    throw new Error(`GitHub OAuth token exchange failed (${response.status}): ${text || response.statusText}`)
  }
  const token = await response.json() as GitHubTokenResponse
  if (token.error || !token.access_token) {
    const message = token.error_description ?? token.error ?? 'GitHub OAuth token response did not include access_token.'
    setError(message)
    throw new Error(message)
  }

  return connectWithToken(token.access_token, {
    scopes: token.scope?.split(',').map((scope) => scope.trim()).filter(Boolean) ?? [],
  })
}

export async function connectWithToken(
  token: string,
  options: { scopes?: string[] } = {},
): Promise<OAuthStatus> {
  const trimmed = token.trim()
  if (!trimmed) throw new Error('GitHub token is required.')
  const login = await fetchGitHubLogin(trimmed)
  save({
    ...load(),
    state: null,
    token: trimmed,
    login,
    scopes: options.scopes ?? [],
    connectedAt: Date.now(),
    updatedAt: Date.now(),
    lastError: null,
  })
  return getStatus()
}

export function disconnect(): OAuthStatus {
  try {
    rmSync(stateFile(), { force: true })
  } catch {
    save(emptyStore())
  }
  return getStatus()
}

export function getStatus(): OAuthStatus {
  const s = load()
  return {
    connected: Boolean(s.token),
    login: s.login,
    configured: oauthConfigured(),
    pending: Boolean(s.state),
    lastError: s.lastError ?? null,
  }
}

export function getAccessToken(): string | null {
  return load().token
}
