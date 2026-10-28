import { readFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'
import { resolveLocalDaemonEndpoint } from '@sepilotd/api-client/node'

function isNodeFsError(error: unknown, code: string): boolean {
  return Boolean(
    error
      && typeof error === 'object'
      && 'code' in error
      && (error as { code?: unknown }).code === code,
  )
}

export function resolveDaemonBaseUrl(explicit?: string): string {
  const trimmed = explicit?.trim()
  if (trimmed) return trimmed
  const fromEnv = process.env.SEPILOTD_URL?.trim()
  if (fromEnv) return fromEnv
  return resolveLocalDaemonEndpoint({ dataDir: resolveDaemonDataDir() }).origin
}

export type DaemonEndpointScope = 'loopback' | 'remote'

export interface LocalDaemonListenTarget {
  baseUrl: string
  host: string
  port: number
}

export const LOCAL_DAEMON_START_ERROR =
  'Automatic local daemon startup requires a root HTTP loopback URL. '
  + 'Start the remote daemon separately or use a loopback --url.'

export function resolveLocalDaemonListenTarget(
  explicit?: string,
): LocalDaemonListenTarget | null {
  const resolved = resolveDaemonBaseUrl(explicit)
  let url: URL
  try {
    url = new URL(resolved)
  } catch {
    return null
  }

  if (
    url.protocol !== 'http:'
    || url.username.length > 0
    || url.password.length > 0
    || url.pathname !== '/'
    || url.search.length > 0
    || url.hash.length > 0
    || url.href !== `${url.origin}/`
  ) {
    return null
  }

  const hostname = url.hostname.toLowerCase()
  const host = hostname === 'localhost'
    ? '127.0.0.1'
    : hostname === '[::1]'
      ? '::1'
      : hostname
  if (host !== '127.0.0.1' && host !== '::1') return null

  const port = Number(url.port || '80')
  if (!Number.isInteger(port) || port <= 0 || port > 65_535) return null

  const renderedHost = host === '::1' ? '[::1]' : host
  return {
    baseUrl: `http://${renderedHost}:${port}`,
    host,
    port,
  }
}

export function resolveDaemonEndpointScope(explicit?: string): DaemonEndpointScope {
  const resolved = resolveDaemonBaseUrl(explicit)
  const hostname = new URL(resolved).hostname.toLowerCase()
  return ['localhost', '127.0.0.1', '::1', '[::1]'].includes(hostname)
    ? 'loopback'
    : 'remote'
}

function targetsLoopbackDaemon(baseUrl: string | undefined): boolean {
  try {
    return resolveDaemonEndpointScope(baseUrl) === 'loopback'
  } catch {
    return false
  }
}

export function resolveGatewayBaseUrl(explicit?: string): string | undefined {
  const trimmed = explicit?.trim()
  if (trimmed) return trimmed
  const fromEnv = process.env.SEPILOTD_GATEWAY_URL?.trim()
  return fromEnv && fromEnv.length > 0 ? fromEnv : undefined
}

export function resolveDaemonDataDir(): string {
  return process.env.SEPILOTD_DATA_DIR?.trim() || join(homedir(), '.sepilotd')
}

export function loadDaemonToken(baseUrl?: string): string | null {
  const directExtension = process.env.SEPILOT_DAEMON_TOKEN?.trim()
  if (directExtension) return directExtension

  const directCli = process.env.SEPILOTD_TOKEN?.trim()
  if (directCli) return directCli

  const customPath =
    process.env.SEPILOT_DAEMON_TOKEN_FILE?.trim()
    || process.env.SEPILOTD_TOKEN_FILE?.trim()
  if (!customPath && !targetsLoopbackDaemon(baseUrl)) return null
  const path = customPath && customPath.length > 0
    ? customPath
    : join(resolveDaemonDataDir(), 'security', 'daemon.token')

  try {
    return readFileSync(path, 'utf-8').trim() || null
  } catch (error) {
    if (isNodeFsError(error, 'ENOENT')) {
      return null
    }
    throw new Error(
      `Failed to read daemon auth token at ${path}: ${
        error instanceof Error ? error.message : String(error)
      }`,
      { cause: error },
    )
  }
}
