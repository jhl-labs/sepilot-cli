import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { parse } from 'yaml'

export const DEFAULT_LOCAL_DAEMON_HOST = '127.0.0.1'
export const DEFAULT_LOCAL_DAEMON_PORT = 17600

export interface LocalDaemonEndpointOptions {
  dataDir: string
  env?: NodeJS.ProcessEnv
}

export interface LocalDaemonEndpoint {
  origin: string
  host: string
  port: number
}

interface ConfiguredDaemonBind {
  host?: string
  port?: number
}

function parsePort(value: unknown): number | undefined {
  const port = typeof value === 'string' && value.trim() !== '' ? Number(value) : value
  return typeof port === 'number' && Number.isInteger(port) && port > 0 && port <= 65_535
    ? port
    : undefined
}

function readConfiguredBind(dataDir: string): ConfiguredDaemonBind {
  let raw: string
  try {
    raw = readFileSync(join(dataDir, 'config.yaml'), 'utf-8')
  } catch {
    return {}
  }
  try {
    const daemon = (parse(raw) as { daemon?: unknown } | null)?.daemon
    if (!daemon || typeof daemon !== 'object') return {}
    const { host, port } = daemon as { host?: unknown; port?: unknown }
    return {
      host: typeof host === 'string' && host.trim() ? host.trim() : undefined,
      port: parsePort(port),
    }
  } catch {
    return {}
  }
}

function connectHost(bindHost: string): string {
  const host = bindHost.replace(/^\[(.*)\]$/u, '$1').toLowerCase()
  if (host === '0.0.0.0' || host === '::' || host === '') return DEFAULT_LOCAL_DAEMON_HOST
  return host
}

export function resolveLocalDaemonEndpoint(
  options: LocalDaemonEndpointOptions,
): LocalDaemonEndpoint {
  const env = options.env ?? process.env
  const configured = readConfiguredBind(options.dataDir)
  const port = parsePort(env.SEPILOTD_PORT) ?? configured.port ?? DEFAULT_LOCAL_DAEMON_PORT
  const host = connectHost(env.SEPILOTD_HOST?.trim() || configured.host || DEFAULT_LOCAL_DAEMON_HOST)
  const renderedHost = host.includes(':') ? `[${host}]` : host
  return { origin: `http://${renderedHost}:${port}`, host, port }
}
