import { readFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'
import { DEFAULT_DAEMON_BASE_URL } from '@sepilotd/api-client'

function isNodeFsError(error: unknown, code: string): boolean {
  return Boolean(
    error
      && typeof error === 'object'
      && 'code' in error
      && (error as { code?: unknown }).code === code,
  )
}

/**
 * Resolve the daemon base URL for the CLI's `DaemonClient`.
 *
 * Precedence:
 * 1. Explicit `--url` flag (passed via `options.url`)
 * 2. `SEPILOTD_URL` env (recognized by other CLI commands like `acp`)
 * 3. `undefined` so the shared `DaemonClient` falls back to
 *    `DEFAULT_DAEMON_BASE_URL` (`http://127.0.0.1:17600`).
 */
export function resolveDaemonBaseUrl(explicit?: string): string | undefined {
  const trimmed = explicit?.trim()
  if (trimmed) return trimmed
  const fromEnv = process.env.SEPILOTD_URL?.trim()
  return fromEnv && fromEnv.length > 0 ? fromEnv : undefined
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

/**
 * Resolve the one local listen target shared by daemon probes and child
 * process startup. Only a root HTTP loopback origin is safe to translate into
 * SEPILOTD_HOST / SEPILOTD_PORT; remote, TLS-terminating, credential-bearing,
 * or path-routed URLs belong to an operator-managed daemon.
 */
export function resolveLocalDaemonListenTarget(
  explicit?: string,
): LocalDaemonListenTarget | null {
  const resolved = resolveDaemonBaseUrl(explicit) ?? DEFAULT_DAEMON_BASE_URL
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

/**
 * Classify the daemon endpoint for recovery paths whose authority differs
 * between a user-owned loopback daemon and a remote/operator-owned runtime.
 * Invalid URLs remain caller-visible errors instead of being treated as local.
 */
export function resolveDaemonEndpointScope(explicit?: string): DaemonEndpointScope {
  const resolved = resolveDaemonBaseUrl(explicit) ?? DEFAULT_DAEMON_BASE_URL
  const hostname = new URL(resolved).hostname.toLowerCase()
  return ['localhost', '127.0.0.1', '::1', '[::1]'].includes(hostname)
    ? 'loopback'
    : 'remote'
}

/**
 * Resolve the gateway base URL for the CLI's `GatewayClient`.
 *
 * Precedence:
 * 1. Explicit argument (e.g. `--gateway-url`)
 * 2. `SEPILOTD_GATEWAY_URL` env
 * 3. `undefined` so the shared `GatewayClient` falls back to its
 *    bundled `DEFAULT_GATEWAY_BASE_URL` constant.
 */
export function resolveGatewayBaseUrl(explicit?: string): string | undefined {
  const trimmed = explicit?.trim()
  if (trimmed) return trimmed
  const fromEnv = process.env.SEPILOTD_GATEWAY_URL?.trim()
  return fromEnv && fromEnv.length > 0 ? fromEnv : undefined
}

/**
 * Resolve the daemon data directory the CLI should agree with.
 *
 * Mirrors the daemon's own resolution (`bootstrap.ts` → `resolveDaemonDataDir`)
 * and the auto-spawn path (`ensure-daemon.ts`): the `SEPILOTD_DATA_DIR` env var
 * wins when set, otherwise `~/.sepilotd`. The standalone single-file binary
 * passes `SEPILOTD_DATA_DIR` through to its self-launched `__daemon` child, so
 * the daemon writes its PID file at `<dataDir>/sepilotd.pid` — `sepilot
 * start/stop/restart` must read the same path.
 */
export function resolveDaemonDataDir(): string {
  return process.env.SEPILOTD_DATA_DIR?.trim() || join(homedir(), '.sepilotd')
}

/**
 * Resolve the daemon bearer token for the CLI's `DaemonClient`.
 *
 * Precedence (first wins):
 * 1. `SEPILOT_DAEMON_TOKEN` env — matches extension template convention
 * 2. `SEPILOTD_TOKEN` env — matches the short CLI prefix used by
 *    `SEPILOTD_URL` / `SEPILOTD_DATA_DIR` / `SEPILOTD_HOST` and by
 *    the shell smoke-test scripts. Accepted as an alias so users do
 *    not have to remember two conventions.
 * 3. `SEPILOT_DAEMON_TOKEN_FILE` env — custom token file path
 * 4. `SEPILOTD_TOKEN_FILE` env — alias for the same
 * 5. `<SEPILOTD_DATA_DIR>/security/daemon.token`, or
 *    `~/.sepilotd/security/daemon.token` when SEPILOTD_DATA_DIR is unset
 *
 * Returns `null` only when the resolved token file is missing or empty.
 * Other read errors are surfaced so a permission flap or a bad custom
 * path does not silently downgrade the CLI to unauthenticated requests.
 * Callers pass the result to `DaemonClient` / `DaemonWsClient`, which
 * then send it as `Authorization: Bearer <token>` when non-null.
 */
export function loadDaemonToken(): string | null {
  const directExtension = process.env.SEPILOT_DAEMON_TOKEN?.trim()
  if (directExtension) return directExtension

  const directCli = process.env.SEPILOTD_TOKEN?.trim()
  if (directCli) return directCli

  const customPath =
    process.env.SEPILOT_DAEMON_TOKEN_FILE?.trim()
    || process.env.SEPILOTD_TOKEN_FILE?.trim()
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
